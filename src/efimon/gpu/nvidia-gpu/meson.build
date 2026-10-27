shared_library(
  'nvidia-process-metrics',
  files('process-metrics.cpp'),
  include_directories : [project_include, cuda_include, cupti_include],
  dependencies : dependency('threads'),
  link_args : ['-L' + cuda_libdir, '-L' + cuda_stub_libdir,
               '-L' + cupti_libdir,
               '-lcupti', '-lnvidia-ml', '-lcuda'],
  install_rpath : cupti_libdir,
  install : true,
  install_dir : profiler_install_dir,
)
