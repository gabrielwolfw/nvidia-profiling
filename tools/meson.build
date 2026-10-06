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

generated_config_include = include_directories('..')

executable(
  'nvidia-process-metrics-daemon',
  daemon_sources,
  cpp_args : linux_cpp_args,
  include_directories : generated_config_include,
  dependencies : dependency('threads'),
  install : true,
  install_dir : get_option('sbindir'),
)

executable(
  'nvidia-process-metrics-launcher',
  files('nvidia-process-metrics-launcher.cpp'),
  cpp_args : linux_cpp_args,
  include_directories : generated_config_include,
  dependencies : dependency('threads'),
  install : true,
)

service_configuration = configuration_data()
service_configuration.set('SBINDIR', join_paths(get_option('prefix'), get_option('sbindir')))
configured_service = configure_file(
  input : 'nvidia-process-metrics-daemon.service.in',
  output : 'nvidia-process-metrics-daemon.service',
  configuration : service_configuration,
)

systemd_unit_dir = get_option('systemd_unit_dir')
if systemd_unit_dir == ''
  systemd_unit_dir = join_paths(get_option('libdir'), 'systemd', 'system')
endif
install_data(configured_service, install_dir : systemd_unit_dir)
