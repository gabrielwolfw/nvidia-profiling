project(
  'nvidia-process-metrics',
  'cpp',
  version : '0.1.0',
  default_options : ['cpp_std=c++17', 'warning_level=3'],
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
configure_file(
  output : 'nvidia-process-metrics-config.hpp',
  configuration : path_configuration,
)

if get_option('build_cuda_components')
  cuda_prefix = get_option('nvidia_toolkit_root')
  cupti_prefix = join_paths(cuda_prefix, 'extras', 'CUPTI')
  cuda_include = include_directories(join_paths(cuda_prefix, 'include'))
  cupti_include = include_directories(join_paths(cupti_prefix, 'include'))
  cuda_libdir = join_paths(cuda_prefix, 'lib64')
  cuda_stub_libdir = join_paths(cuda_libdir, 'stubs')
  cupti_libdir = join_paths(cupti_prefix, 'lib64')
  profiler_install_dir = join_paths(get_option('libdir'), 'nvidia-process-metrics')

  shared_library(
    'nvidia-process-metrics',
    files('nvidia/nvidia-process-metrics.cpp'),
    include_directories : [cuda_include, cupti_include],
    dependencies : dependency('threads'),
    link_args : ['-L' + cuda_libdir, '-L' + cuda_stub_libdir,
                 '-L' + cupti_libdir,
                 '-lcupti', '-lnvidia-ml', '-lcuda'],
    install_rpath : cupti_libdir,
    install : true,
    install_dir : profiler_install_dir,
  )

  if get_option('enable_pm_sampling')
    nvcc = find_program(join_paths(cuda_prefix, 'bin', 'nvcc'))
    custom_target(
      'pm-sampling-simple',
      input : files('nvidia/pm_sampling_simple.cu'),
      output : 'pm_sampling_simple',
      command : [nvcc, '-std=c++17', '@INPUT@',
                 '-I' + join_paths(cuda_prefix, 'include'),
                 '-I' + join_paths(cupti_prefix, 'include'),
                 '-L' + cupti_libdir, '-L' + cuda_stub_libdir,
                 '-Xlinker', '-rpath', '-Xlinker', cupti_libdir,
                 '-lcupti', '-lcuda', '-Xcompiler', '-pthread',
                 '-o', '@OUTPUT@'],
      depend_files : files('nvidia/pm_sampling.h', 'nvidia/cuda_api_checks.hpp'),
      build_by_default : true,
      install : true,
      install_dir : profiler_install_dir,
    )
  endif
endif

subdir('tools')
