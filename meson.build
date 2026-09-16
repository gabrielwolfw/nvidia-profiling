project(
  'nvidia-process-metrics',
  'cpp',
  version : '0.1.0',
  default_options : ['cpp_std=c++17', 'warning_level=3'],
)

if get_option('build_cuda_components')
  cuda_prefix = get_option('nvidia_toolkit_root')
  cuda_include = include_directories(join_paths(cuda_prefix, 'include'))
  cupti_include = include_directories(
    join_paths(cuda_prefix, 'extras', 'CUPTI', 'samples', 'common'),
  )
  cuda_libdir = join_paths(cuda_prefix, 'lib64')
  cupti_libdir = cuda_libdir
  profiler_libdir = join_paths(get_option('libdir'), 'nvidia-process-metrics')

  shared_library(
    'nvidia-process-metrics',
    files('nvidia/nvidia-process-metrics.cpp'),
    include_directories : [cuda_include, cupti_include],
    dependencies : dependency('threads'),
    link_args : ['-L' + cuda_libdir, '-L' + cupti_libdir,
                 '-lcupti', '-lnvidia-ml', '-lcuda'],
    install : true,
    install_dir : profiler_libdir,
  )

  nvcc = find_program(join_paths(cuda_prefix, 'bin', 'nvcc'))
  custom_target(
    'pm-sampling-simple',
    input : files('nvidia/pm_sampling_simple.cu'),
    output : 'pm_sampling_simple',
    command : [nvcc, '-std=c++17', '@INPUT@',
               '-I' + join_paths(cuda_prefix, 'include'),
               '-I' + join_paths(cuda_prefix, 'extras', 'CUPTI', 'samples', 'common'),
               '-L' + cupti_libdir, '-lcupti', '-lcuda', '-Xcompiler', '-pthread',
               '-o', '@OUTPUT@'],
    depend_files : files('nvidia/pm_sampling.h'),
    build_by_default : true,
    install : true,
    install_dir : profiler_libdir,
  )
endif

subdir('tools')
