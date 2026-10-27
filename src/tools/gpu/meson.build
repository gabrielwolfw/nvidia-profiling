gpu_daemon_sources = files('session.cpp', 'worker.cpp')

if get_option('build_cuda_components') and get_option('enable_pm_sampling')
  nvcc = find_program(join_paths(cuda_prefix, 'bin', 'nvcc'))
  custom_target(
    'pm-sampling-simple',
    input : files('pm-sampling.cu'),
    output : 'pm_sampling_simple',
    command : [nvcc, '-std=c++17', '@INPUT@',
               '-I' + join_paths(meson.project_source_root(), 'include'),
               '-I' + cuda_include_dir,
               '-I' + cupti_include_dir,
               '-L' + cupti_libdir, '-L' + cuda_stub_libdir,
               '-Xlinker', '-rpath', '-Xlinker', cupti_libdir,
               '-lcupti', '-lcuda', '-Xcompiler', '-pthread',
               '-o', '@OUTPUT@'],
    depend_files : files(
      '../../../include/efimon/gpu/nvidia-gpu/pm-sampling.hpp',
      '../../../include/efimon/gpu/nvidia-gpu/cuda-api-checks.hpp',
    ),
    build_by_default : true,
    install : true,
    install_dir : profiler_install_dir,
  )
endif
