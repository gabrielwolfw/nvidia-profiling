project(
  'nvidia-process-metrics',
  'cpp',
  version : '0.1.0',
  # The collector runs inside the profiled process, so build it optimized by
  # default; debugoptimized keeps the debug symbols.
  default_options : [
    'cpp_std=c++17',
    'warning_level=3',
    'buildtype=debugoptimized',
  ],
)

install_prefix = get_option('prefix')
installed_libdir = join_paths(install_prefix, get_option('libdir'))
profiler_libdir = join_paths(installed_libdir, 'nvidia-process-metrics')

path_configuration = configuration_data()
path_configuration.set_quoted(
  'NVIDIA_PROCESS_METRICS_LIBRARY_PATH',
  join_paths(profiler_libdir, 'libnvidia-process-metrics.so'),
)
path_configuration.set_quoted(
  'NVIDIA_PROCESS_METRICS_SAMPLER_PATH',
  join_paths(profiler_libdir, 'pm_sampling_simple'),
)
path_configuration.set(
  'NVIDIA_PROCESS_METRICS_PM_HARDWARE_BUFFER_BYTES',
  '@0@ULL'.format(get_option('pm_hardware_buffer_mb') * 1024 * 1024),
)
path_configuration.set(
  'NVIDIA_PROCESS_METRICS_PM_MAX_SAMPLES',
  '@0@ULL'.format(get_option('pm_max_samples')),
)
configure_file(
  output : 'nvidia-process-metrics-config.hpp',
  configuration : path_configuration,
)

project_include = include_directories('include')

# The kernel journal has no CUDA dependency, so its test always builds.
kernel_journal_sources = files('src/efimon/gpu/nvidia-gpu/kernel-journal.cpp')
test(
  'kernel-journal',
  executable(
    'kernel-journal-test',
    kernel_journal_sources +
      files('src/efimon/gpu/nvidia-gpu/tests/kernel-journal-test.cpp'),
    include_directories : project_include,
  ),
)

if get_option('build_cuda_components')
  cuda_prefix = get_option('nvidia_toolkit_root')
  cuda_target = get_option('nvidia_toolkit_target')
  if cuda_target == ''
    cupti_prefix = join_paths(cuda_prefix, 'extras', 'CUPTI')
    cuda_include_dir = join_paths(cuda_prefix, 'include')
    cupti_include_dir = join_paths(cupti_prefix, 'include')
    cuda_libdir = join_paths(cuda_prefix, 'lib64')
    cupti_libdir = join_paths(cupti_prefix, 'lib64')
  else
    cuda_include_dir = join_paths(cuda_target, 'include')
    cupti_include_dir = cuda_include_dir
    cuda_libdir = join_paths(cuda_target, 'lib')
    cupti_libdir = cuda_libdir
  endif
  cuda_include = include_directories(cuda_include_dir)
  cupti_include = include_directories(cupti_include_dir)
  cuda_stub_libdir = join_paths(cuda_libdir, 'stubs')
  profiler_install_dir = join_paths(get_option('libdir'), 'nvidia-process-metrics')

  subdir('src/efimon/gpu/nvidia-gpu')
endif

subdir('src/tools')
