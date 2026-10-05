daemon_sources = files(
  'nvidia-process-metrics-daemon.cpp',
  'nvidia-process-metrics-session.cpp',
  'nvidia-process-metrics-worker.cpp',
)

linux_cpp_args = [
  '-D_GNU_SOURCE',
  '-DNVIDIA_PROCESS_METRICS_ENABLE_PM_SAMPLING=' +
    (get_option('enable_pm_sampling') ? '1' : '0'),
]

executable(
  'nvidia-process-metrics-daemon',
  daemon_sources,
  cpp_args : linux_cpp_args,
  dependencies : dependency('threads'),
  install : true,
  install_dir : get_option('sbindir'),
)

executable(
  'nvidia-process-metrics-launcher',
  files('nvidia-process-metrics-launcher.cpp'),
  cpp_args : linux_cpp_args,
  dependencies : dependency('threads'),
  install : true,
)

install_data(
  'nvidia-process-metrics-daemon.service',
  install_dir : join_paths(get_option('libdir'), 'systemd', 'system'),
)
