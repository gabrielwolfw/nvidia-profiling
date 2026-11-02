/*
 *  Copyright 2024 NVIDIA Corporation. All rights reserved
 *
 * This collector uses the PM sampling feature in the CUDA Profiling Tools Interface (CUPTI).
 * There are two supported operations:
 * 1) Querying the available metrics for a chip and their properties.
 * 2) Collecting device-wide PM sampling data while an external workload runs.
 *
 * The launcher or benchmark script starts the external workload only after this
 * process reports that PM sampling is active. A decode thread continuously
 * drains hardware samples into counter data images while the main thread
 * controls the collection duration. With more than one image, an evaluation
 * thread turns decoded samples into metric values and streams them to the CSV,
 * so that metric evaluation does not delay draining the hardware buffer.
 *
 */

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <thread>

// CUDA headers
#include <cuda.h>

#include "efimon/gpu/nvidia-gpu/pm-sampling.hpp"

std::atomic<bool> stopDecodeThread(false);
volatile std::sig_atomic_t stopCollection = 0;

void RequestCollectionStop(int)
{
    stopCollection = 1;
}

// NVIDIA samples conventionally use status 2 when profiling is unsupported
// by the selected device or environment. Keep that outcome distinct from a
// program failure without depending on samples/common/helper_cupti.h.
constexpr int kExitWaived = 2;

// Each counter data image holds up to --maxsamples samples.
constexpr size_t kMaxDecodeImages = 8;

struct ParsedArgs
{
    bool isDeviceIndexSet = false;
    bool isChipNameSet = false;
    int deviceIndex = 0;
    int queryBaseMetrics = 0;
    int queryMetricProperties = 0;
    std::string chipName;
    uint64_t samplingInterval = 10000; // GPU sysclk cycles between samples.
    size_t hardwareBufferSize = 512 * 1024 * 1024; // 512MB
    uint64_t maxSamples = 100000;
    // Counter data images rotated between decoding and evaluation. One image
    // evaluates inside the decode thread, as the original sample did.
    size_t decodeImages = 2;
    double durationSeconds = 10.0;
    std::string outputPath = "pm_sampling.csv";
    std::vector<std::string> metricNames =
    {
        "gr__cycles_active.avg",
        "gr__cycles_elapsed.max",
        "gpu__time_duration.sum",
        "sm__inst_executed_realtime.avg.per_cycle_active",
        "sm__cycles_active.avg",
        "tpc__warps_inactive_sm_idle_realtime.avg.pct_of_peak_sustained_elapsed",
        "sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed"
    };

    std::vector<const char*> MetricPointers() const
    {
        std::vector<const char*> metrics;
        metrics.reserve(metricNames.size());
        for (const std::string& metric : metricNames)
        {
            metrics.push_back(metric.c_str());
        }
        return metrics;
    }
};

ParsedArgs parseArgs(int argc, char *argv[]);
void PmSamplingDeviceSupportStatus(CUdevice device);
int PmSamplingCollection(std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args);
int PmSamplingQueryMetrics(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args);

// Samples decoded into one counter data image and waiting for evaluation.
struct DecodedBatch
{
    size_t imageIndex = 0;
    std::vector<SamplerRange> ranges;
};

// Hands counter data images between the decode and evaluation threads. Only
// the decode thread calls the PM sampling object, including image resets.
struct DecodePipeline
{
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<size_t> readyImages;   // Initialized and ready for decoding.
    std::deque<size_t> usedImages;    // Evaluated; must be reset first.
    std::deque<DecodedBatch> pendingBatches;
    bool decodeFinished = false;
    bool evaluateFailed = false;
    std::exception_ptr decodeError;
    std::exception_ptr evaluateError;
};

// Collection diagnostics. Each field has one writer thread and is read only
// after both threads join.
struct DecodeStatistics
{
    uint64_t decodeCalls = 0;
    uint64_t overflowReports = 0;
    uint64_t stopReasons[CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNT] = {};
    size_t maxSamplesPerDecode = 0;
    bool hardwareBufferOverflow = false;
    std::chrono::nanoseconds decodeTime{0};
    std::chrono::nanoseconds imageWaitTime{0};
    std::chrono::nanoseconds evaluateTime{0};
};

// Set when decoding cannot continue, so the main thread stops waiting.
std::atomic<bool> decodeStopped(false);

void DecodeCounterData(
    std::vector<std::vector<uint8_t>>& counterDataImages,
    std::vector<const char*> metricsList,
    CuptiPmSampling& cuptiPmSamplingTarget,
    CuptiProfilerHost& pmSamplingHost,
    DecodePipeline& pipeline,
    DecodeStatistics& statistics
);
void EvaluateDecodedSamples(
    std::vector<std::vector<uint8_t>>& counterDataImages,
    std::vector<const char*> metricsList,
    CuptiProfilerHost& pmSamplingHost,
    DecodePipeline& pipeline,
    DecodeStatistics& statistics
);
void PrintDecodeStatistics(const DecodeStatistics& statistics, size_t imageCount);

int main(int argc, char *argv[])
{
    try
    {
        ParsedArgs args = parseArgs(argc, argv);
        DRIVER_API_CALL(cuInit(0));

        std::string chipName = args.chipName;
        std::vector<uint8_t> counterAvailibilityImage;
        if ((args.isDeviceIndexSet && args.deviceIndex >= 0) ||
            !args.isChipNameSet)
        {
            CUdevice cuDevice;
            DRIVER_API_CALL(cuDeviceGet(&cuDevice, args.deviceIndex));
            PmSamplingDeviceSupportStatus(cuDevice);

            CuptiPmSampling::GetChipName(args.deviceIndex, chipName);
            CuptiPmSampling::GetCounterAvailabilityImage(
                args.deviceIndex, counterAvailibilityImage);
        }

        if (args.queryBaseMetrics || args.queryMetricProperties)
        {
            return PmSamplingQueryMetrics(
                chipName, counterAvailibilityImage, args);
        }
        return PmSamplingCollection(counterAvailibilityImage, args);
    }
    catch (const std::exception& error)
    {
        std::cerr << "PM sampler failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}

int PmSamplingQueryMetrics(std::string chipName, std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args)
{
    CuptiProfilerHost pmSamplingHost;
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);

    if (args.queryBaseMetrics)
    {
        std::vector<std::string> baseMetrics;
        CUPTI_API_CALL(pmSamplingHost.GetSupportedBaseMetrics(baseMetrics));
        printf("Base Metrics:\n");
        for (const auto& metric : baseMetrics)
        {
            printf("  %s\n", metric.c_str());
        }
        return 0;
    }

    if (args.queryMetricProperties)
    {
        for (const auto& metricName : args.metricNames)
        {
            std::vector<std::string> subMetrics;
            CUPTI_API_CALL(pmSamplingHost.GetSubMetrics(metricName, subMetrics));
            printf("Sub Metrics for %s:\n", metricName.c_str());
            for (const auto& metric : subMetrics) {
                printf("  %s\n", metric.c_str());
            }

            std::string metricDescription;
            CUpti_MetricType metricType;
            CUPTI_API_CALL(pmSamplingHost.GetMetricProperties(metricName, metricType,metricDescription));

            printf("Metric Description: %s\n", metricDescription.c_str());
            printf("Metric Type: %s\n", metricType == CUPTI_METRIC_TYPE_COUNTER ? "Counter" : (metricType == CUPTI_METRIC_TYPE_RATIO) ? "Ratio" : "Throughput");
            printf("\n");
        }
        return 0;
    }

    pmSamplingHost.TearDown();
    return 0;
}

int PmSamplingCollection(std::vector<uint8_t>& counterAvailibilityImage, ParsedArgs& args)
{
    const std::vector<const char*> metrics = args.MetricPointers();
    std::string chipName;
    CuptiPmSampling::GetChipName(args.deviceIndex, chipName);

    CuptiProfilerHost pmSamplingHost;
    pmSamplingHost.SetUp(chipName, counterAvailibilityImage);

    std::vector<uint8_t> configImage;
    CUPTI_API_CALL(pmSamplingHost.CreateConfigImage(metrics, configImage));

    // Rows are streamed while sampling runs, so open the CSV before starting.
    if (!pmSamplingHost.OpenSamplesCsv(args.outputPath, metrics))
    {
        pmSamplingHost.TearDown();
        return EXIT_FAILURE;
    }

    CuptiPmSampling cuptiPmSamplingTarget;
    cuptiPmSamplingTarget.SetUp(args.deviceIndex);

    // 1. Enable PM sampling and set config for the PM sampling data collection.
    CUPTI_API_CALL(cuptiPmSamplingTarget.EnablePmSampling(args.deviceIndex));
    CUPTI_API_CALL(cuptiPmSamplingTarget.SetConfig(configImage, args.hardwareBufferSize, args.samplingInterval));
    std::cout << "Hardware buffer: " << args.hardwareBufferSize << " bytes\n";

    // 2. Create the counter data images rotated between decode and evaluation.
    std::vector<std::vector<uint8_t>> counterDataImages(args.decodeImages);
    for (std::vector<uint8_t>& counterDataImage : counterDataImages)
    {
        CUPTI_API_CALL(cuptiPmSamplingTarget.CreateCounterDataImage(args.maxSamples, metrics, counterDataImage));
    }
    std::cout << "Counter data images: " << counterDataImages.size() << " x "
              << counterDataImages.front().size() << " bytes\n";

    stopCollection = 0;
    if (std::signal(SIGINT, RequestCollectionStop) == SIG_ERR ||
        std::signal(SIGTERM, RequestCollectionStop) == SIG_ERR)
    {
        throw std::runtime_error("failed to install PM sampler signal handlers");
    }

    // 3. Start PM sampling before launching the decode and evaluation threads.
    CUPTI_API_CALL(cuptiPmSamplingTarget.StartPmSampling());
    stopDecodeThread = false;
    decodeStopped = false;
    DecodePipeline pipeline;
    for (size_t imageIndex = 0; imageIndex < counterDataImages.size(); ++imageIndex)
    {
        pipeline.readyImages.push_back(imageIndex);
    }
    DecodeStatistics statistics;
    std::thread decodeThread(
        DecodeCounterData, std::ref(counterDataImages), metrics,
        std::ref(cuptiPmSamplingTarget), std::ref(pmSamplingHost),
        std::ref(pipeline), std::ref(statistics));
    std::thread evaluateThread;
    if (counterDataImages.size() > 1)
    {
        evaluateThread = std::thread(
            EvaluateDecodedSamples, std::ref(counterDataImages), metrics,
            std::ref(pmSamplingHost), std::ref(pipeline),
            std::ref(statistics));
    }

    // 4. Wait while the external workload runs.
    std::cout << "PM Sampling active for " << args.durationSeconds
              << " seconds with a " << args.maxSamples
              << "-sample counter data image..." << std::endl;
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration<double>(args.durationSeconds);
    while (!stopCollection && !decodeStopped &&
           std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // 5. Stop the PM sampling and join the decode and evaluation threads
    const CUptiResult stopResult = cuptiPmSamplingTarget.StopPmSampling();
    stopDecodeThread = true;
    decodeThread.join();
    if (evaluateThread.joinable())
    {
        evaluateThread.join();
    }
    const bool csvSaved = pmSamplingHost.CloseSamplesCsv();
    CUPTI_API_CALL(stopResult);
    if (pipeline.decodeError)
    {
        std::rethrow_exception(pipeline.decodeError);
    }
    if (pipeline.evaluateError)
    {
        std::rethrow_exception(pipeline.evaluateError);
    }

    // 6. Report PM sampling continuity and decode diagnostics before printing
    // the collected ranges.
    pmSamplingHost.PrintSamplingGapSummary();
    PrintDecodeStatistics(statistics, counterDataImages.size());

    // 7. Print the sample ranges for the collected metrics
    pmSamplingHost.PrintSampleRanges();

    // 8. Disable PM sampling for release all the resources allocated in CUPTI
    CUPTI_API_CALL(cuptiPmSamplingTarget.DisablePmSampling());

    // 9. Clean up
    cuptiPmSamplingTarget.TearDown();
    pmSamplingHost.TearDown();

    if (!csvSaved)
    {
        return EXIT_FAILURE;
    }
    if (statistics.hardwareBufferOverflow)
    {
        std::cerr << "PM sampler: the hardware buffer overflowed and samples "
                     "were lost; increase --hardwareBufferSize or "
                     "--samplingInterval\n";
        return EXIT_FAILURE;
    }
    return 0;
}

// Returns an image ready for decoding in imageIndex, resetting one that the
// evaluation thread released. Returns false if evaluation failed.
bool AcquireCounterDataImage(
    std::vector<std::vector<uint8_t>>& counterDataImages,
    CuptiPmSampling& cuptiPmSamplingTarget,
    DecodePipeline& pipeline,
    DecodeStatistics& statistics,
    size_t& imageIndex)
{
    bool needsReset = false;
    {
        const auto waitStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(pipeline.mutex);
        pipeline.changed.wait(lock, [&pipeline]() {
            return !pipeline.readyImages.empty() ||
                !pipeline.usedImages.empty() || pipeline.evaluateFailed;
        });
        statistics.imageWaitTime += std::chrono::steady_clock::now() - waitStart;
        if (pipeline.evaluateFailed)
        {
            return false;
        }
        if (!pipeline.readyImages.empty())
        {
            imageIndex = pipeline.readyImages.front();
            pipeline.readyImages.pop_front();
        }
        else
        {
            imageIndex = pipeline.usedImages.front();
            pipeline.usedImages.pop_front();
            needsReset = true;
        }
    }
    if (needsReset)
    {
        CUPTI_API_CALL(cuptiPmSamplingTarget.ResetCounterDataImage(counterDataImages[imageIndex]));
    }
    return true;
}

void DecodeCounterData( std::vector<std::vector<uint8_t>>& counterDataImages,
                        std::vector<const char*> metricsList,
                        CuptiPmSampling& cuptiPmSamplingTarget,
                        CuptiProfilerHost& pmSamplingHost,
                        DecodePipeline& pipeline,
                        DecodeStatistics& statistics)
{
    const bool evaluateInline = counterDataImages.size() == 1;
    std::vector<double> metricValues(metricsList.size());
    try
    {
        size_t imageIndex = 0;
        bool hasImage = AcquireCounterDataImage(
            counterDataImages, cuptiPmSamplingTarget, pipeline, statistics,
            imageIndex);
        while (hasImage)
        {
            std::vector<uint8_t>& counterDataImage = counterDataImages[imageIndex];
            CUpti_PmSampling_DecodeStopReason stopReason =
                CUPTI_PM_SAMPLING_DECODE_STOP_REASON_OTHER;
            bool overflow = false;
            const auto decodeStart = std::chrono::steady_clock::now();
            const CUptiResult decodeResult =
                cuptiPmSamplingTarget.DecodePmSamplingData(
                    counterDataImage, stopReason, overflow);
            statistics.decodeTime += std::chrono::steady_clock::now() - decodeStart;
            ++statistics.decodeCalls;

            // CUPTI reports a hardware buffer overflow as out of memory.
            if (decodeResult == CUPTI_ERROR_OUT_OF_MEMORY)
            {
                std::cerr << "PM sampler: hardware buffer overflow reported "
                             "by cuptiPmSamplingDecodeData; stopping\n";
                statistics.hardwareBufferOverflow = true;
                break;
            }
            CUPTI_API_CALL(decodeResult);
            if (stopReason < CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNT)
            {
                ++statistics.stopReasons[stopReason];
            }
            if (overflow)
            {
                if (statistics.overflowReports == 0)
                {
                    std::cerr << "PM sampler: hardware buffer overflow; "
                                 "samples are being lost\n";
                }
                ++statistics.overflowReports;
                statistics.hardwareBufferOverflow = true;
            }

            CUpti_PmSampling_GetCounterDataInfo_Params counterDataInfo {
                CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
            counterDataInfo.pCounterDataImage = counterDataImage.data();
            counterDataInfo.counterDataImageSize = counterDataImage.size();
            CUPTI_API_CALL(
                cuptiPmSamplingGetCounterDataInfo(&counterDataInfo));

            const size_t completedSamples = counterDataInfo.numCompletedSamples;
            if (completedSamples == 0)
            {
                // PM sampling has stopped and a decode returned no ranges, so
                // all samples pending in the hardware buffer have been drained.
                if (stopDecodeThread)
                {
                    break;
                }

                // Avoid reinitializing an empty image in a tight loop.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            statistics.maxSamplesPerDecode =
                std::max(statistics.maxSamplesPerDecode, completedSamples);

            DecodedBatch batch;
            batch.imageIndex = imageIndex;
            batch.ranges.resize(completedSamples);
            for (size_t sampleIndex = 0; sampleIndex < completedSamples; ++sampleIndex)
            {
                CUPTI_API_CALL(CuptiProfilerHost::GetSampleRange(
                    cuptiPmSamplingTarget.GetPmSamplerObject(),
                    counterDataImage, sampleIndex, batch.ranges[sampleIndex]));
            }

            if (evaluateInline)
            {
                const auto evaluateStart = std::chrono::steady_clock::now();
                for (size_t sampleIndex = 0; sampleIndex < completedSamples; ++sampleIndex)
                {
                    CUPTI_API_CALL(pmSamplingHost.EvaluateSample(
                        counterDataImage, sampleIndex, metricsList, metricValues));
                    pmSamplingHost.AppendSample(batch.ranges[sampleIndex], metricValues);
                }
                statistics.evaluateTime +=
                    std::chrono::steady_clock::now() - evaluateStart;
                CUPTI_API_CALL(
                    cuptiPmSamplingTarget.ResetCounterDataImage(counterDataImage));
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(pipeline.mutex);
                pipeline.pendingBatches.push_back(std::move(batch));
            }
            pipeline.changed.notify_all();
            hasImage = AcquireCounterDataImage(
                counterDataImages, cuptiPmSamplingTarget, pipeline,
                statistics, imageIndex);
        }
    }
    catch (...)
    {
        pipeline.decodeError = std::current_exception();
    }

    if (pipeline.decodeError || statistics.hardwareBufferOverflow)
    {
        decodeStopped = true;
    }
    {
        std::lock_guard<std::mutex> lock(pipeline.mutex);
        pipeline.decodeFinished = true;
    }
    pipeline.changed.notify_all();
}

void EvaluateDecodedSamples(std::vector<std::vector<uint8_t>>& counterDataImages,
                            std::vector<const char*> metricsList,
                            CuptiProfilerHost& pmSamplingHost,
                            DecodePipeline& pipeline,
                            DecodeStatistics& statistics)
{
    std::vector<double> metricValues(metricsList.size());
    try
    {
        while (true)
        {
            DecodedBatch batch;
            {
                std::unique_lock<std::mutex> lock(pipeline.mutex);
                pipeline.changed.wait(lock, [&pipeline]() {
                    return !pipeline.pendingBatches.empty() ||
                        pipeline.decodeFinished;
                });
                if (pipeline.pendingBatches.empty())
                {
                    return;
                }
                batch = std::move(pipeline.pendingBatches.front());
                pipeline.pendingBatches.pop_front();
            }

            const auto evaluateStart = std::chrono::steady_clock::now();
            const std::vector<uint8_t>& counterDataImage =
                counterDataImages[batch.imageIndex];
            for (size_t sampleIndex = 0; sampleIndex < batch.ranges.size(); ++sampleIndex)
            {
                CUPTI_API_CALL(pmSamplingHost.EvaluateSample(
                    counterDataImage, sampleIndex, metricsList, metricValues));
                pmSamplingHost.AppendSample(batch.ranges[sampleIndex], metricValues);
            }
            statistics.evaluateTime +=
                std::chrono::steady_clock::now() - evaluateStart;

            {
                std::lock_guard<std::mutex> lock(pipeline.mutex);
                pipeline.usedImages.push_back(batch.imageIndex);
            }
            pipeline.changed.notify_all();
        }
    }
    catch (...)
    {
        {
            std::lock_guard<std::mutex> lock(pipeline.mutex);
            pipeline.evaluateError = std::current_exception();
            pipeline.evaluateFailed = true;
        }
        pipeline.changed.notify_all();
        decodeStopped = true;
    }
}

void PrintDecodeStatistics(const DecodeStatistics& statistics, size_t imageCount)
{
    using Milliseconds = std::chrono::duration<double, std::milli>;
    std::cout << "\n--- PM Sampling decode diagnostics ---\n";
    std::cout << "Counter data images: " << imageCount << "\n";
    std::cout << "Decode calls: " << statistics.decodeCalls << "\n";
    std::cout << "Decode stop reasons (other/counter data full/end of records): "
              << statistics.stopReasons[CUPTI_PM_SAMPLING_DECODE_STOP_REASON_OTHER] << "/"
              << statistics.stopReasons[CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL] << "/"
              << statistics.stopReasons[CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS] << "\n";
    std::cout << "Hardware buffer overflow reports: " << statistics.overflowReports << "\n";
    std::cout << "Hardware buffer overflow: "
              << (statistics.hardwareBufferOverflow ? "yes" : "no") << "\n";
    std::cout << "Largest decode: " << statistics.maxSamplesPerDecode << " samples\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Decode time: "
              << Milliseconds(statistics.decodeTime).count() << " ms\n";
    std::cout << "Evaluation time: "
              << Milliseconds(statistics.evaluateTime).count() << " ms\n";
    std::cout << "Decode wait for a free image: "
              << Milliseconds(statistics.imageWaitTime).count() << " ms\n";
}

void PrintHelp()
{
    printf("Usage:\n");
    printf("  Query Metrics:\n");
    printf("    List Base Metrics : ./pm_sampling --device/-d <deviceIndex> --chip/-c <chipname> --queryBaseMetrics/-q\n");
    printf("    List submetrics   : ./pm_sampling --device/-d <deviceIndex> --chip/-c <chipname> --metrics/-m <metric1,metric2,...> --queryMetricsProp/-p\n");
    printf("  Note: when device index flag is passed, the chip name flag will be ignored.\n");
    printf("  PM Sampling:\n");
    printf("    Collection: ./pm_sampling --device/-d <deviceIndex> --duration/-t <seconds> --samplingInterval/-i <samplingInterval> --maxsamples/-s <maxSamples in CounterDataImage> --hardwareBufferSize/-b <hardware buffer size> --decodeImages/-k <counter data images> --metrics/-m <metric1,metric2,...> [--output <csv path>]\n");
    printf("    Default collection duration: 10 seconds.\n");
    printf("    --samplingInterval is in GPU sysclk cycles.\n");
    printf("    --decodeImages 1 evaluates metrics in the decode thread; 2 to %zu overlap evaluation with decoding (default 2).\n", kMaxDecodeImages);
}

[[noreturn]] void ArgumentError(const std::string& message)
{
    fprintf(stderr, "%s\n", message.c_str());
    exit(EXIT_FAILURE);
}

const char* RequireValue(int argc, char* argv[], int& index,
                         const std::string& option)
{
    if (index + 1 >= argc)
    {
        ArgumentError("Missing value for " + option);
    }
    return argv[++index];
}

uint64_t ParsePositiveInteger(const char* value, const std::string& option)
{
    if (value[0] == '\0' || value[0] == '-')
    {
        ArgumentError(option + " must be a positive integer");
    }

    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<uint64_t>::max())
    {
        ArgumentError(option + " must be a positive integer");
    }
    return static_cast<uint64_t>(parsed);
}

int ParseDeviceIndex(const char* value, const std::string& option)
{
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max())
    {
        ArgumentError(option + " must be a non-negative integer");
    }
    return static_cast<int>(parsed);
}

double ParseDuration(const char* value, const std::string& option)
{
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if (errno == ERANGE || end == value || *end != '\0' ||
        !std::isfinite(parsed) || parsed <= 0.0)
    {
        ArgumentError(option + " must be a finite number greater than zero");
    }
    return parsed;
}

ParsedArgs parseArgs(int argc, char *argv[])
{
    ParsedArgs args;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--device" || arg == "-d")
        {
            args.deviceIndex = ParseDeviceIndex(
                RequireValue(argc, argv, i, arg), arg);
            args.isDeviceIndexSet = true;
        }
        else if (arg == "--samplingInterval" || arg == "-i")
        {
            args.samplingInterval = ParsePositiveInteger(
                RequireValue(argc, argv, i, arg), arg);
        }
        else if (arg == "--duration" || arg == "-t")
        {
            args.durationSeconds = ParseDuration(
                RequireValue(argc, argv, i, arg), arg);
        }
        else if (arg == "--maxsamples" || arg == "-s")
        {
            args.maxSamples = ParsePositiveInteger(
                RequireValue(argc, argv, i, arg), arg);
        }
        else if (arg == "--hardwareBufferSize" || arg == "-b")
        {
            const uint64_t size = ParsePositiveInteger(
                RequireValue(argc, argv, i, arg), arg);
            if (size > std::numeric_limits<size_t>::max())
            {
                ArgumentError(arg + " exceeds the platform size limit");
            }
            args.hardwareBufferSize = static_cast<size_t>(size);
        }
        else if (arg == "--decodeImages" || arg == "-k")
        {
            const uint64_t images = ParsePositiveInteger(
                RequireValue(argc, argv, i, arg), arg);
            if (images > kMaxDecodeImages)
            {
                ArgumentError(arg + " must be at most " + std::to_string(kMaxDecodeImages));
            }
            args.decodeImages = static_cast<size_t>(images);
        }
        else if (arg == "--output")
        {
            args.outputPath = RequireValue(argc, argv, i, arg);
            if (args.outputPath.empty())
            {
                ArgumentError(arg + " cannot be empty");
            }
        }
        else if (arg == "--chip" || arg == "-c")
        {
            args.chipName = RequireValue(argc, argv, i, arg);
            if (args.chipName.empty())
            {
                ArgumentError(arg + " cannot be empty");
            }
            args.isChipNameSet = true;
        }
        else if (arg == "--queryBaseMetrics" || arg == "-q")
        {
            args.queryBaseMetrics = 1;
        }
        else if (arg == "--queryMetricsProp" || arg == "-p")
        {
            args.queryMetricProperties = 1;
        }
        else if (arg == "--metrics" || arg == "-m")
        {
            const std::string value = RequireValue(argc, argv, i, arg);
            if (value.empty() || value.front() == ',' || value.back() == ',' ||
                value.find(",,") != std::string::npos)
            {
                ArgumentError(arg + " requires a comma-separated list of metrics");
            }
            std::stringstream ss(value);
            std::string metric;
            args.metricNames.clear();
            while (std::getline(ss, metric, ','))
            {
                if (metric.empty())
                {
                    ArgumentError(arg + " contains an empty metric name");
                }
                args.metricNames.push_back(metric);
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            PrintHelp();
            exit(EXIT_SUCCESS);
        }
        else
        {
            fprintf(stderr, "Invalid argument: %s\n", arg.c_str());
            PrintHelp();
            exit(EXIT_FAILURE);
        }
    }
    return args;
}

void PmSamplingDeviceSupportStatus(CUdevice device)
{
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = device;
    params.api = CUPTI_PROFILER_PM_SAMPLING;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << device << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }

        exit(kExitWaived);
    }
}
