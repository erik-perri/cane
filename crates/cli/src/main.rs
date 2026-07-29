mod repl;

use anyhow::Context;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt};

const CANE_VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_MAX_TOKENS: u32 = 32 * 1024;

#[cfg(not(windows))]
const HOME_DIRECTORY_VARIABLE: &str = "HOME";
#[cfg(windows)]
const HOME_DIRECTORY_VARIABLE: &str = "USERPROFILE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CliCommand {
    Chat { shell_mode: CliShellMode },
    Doctor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CliShellMode {
    Disabled,
    Sandboxed,
    Unsafe,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .with_writer(std::io::stderr)
        .init();

    let shell_mode = match parse_cli_command(std::env::args_os().skip(1))? {
        CliCommand::Doctor => {
            if !run_doctor()? {
                std::process::exit(1);
            }
            return Ok(());
        }
        CliCommand::Chat { shell_mode } => shell_mode,
    };

    let api_key = std::env::var("CANE_API_KEY").context("CANE_API_KEY not set")?;
    let base_url = std::env::var("CANE_BASE_URL").context("CANE_BASE_URL not set")?;
    let model = std::env::var("CANE_MODEL").context("CANE_MODEL not set")?;
    let max_tokens: u32 = std::env::var("CANE_MAX_TOKENS")
        .unwrap_or_else(|_| DEFAULT_MAX_TOKENS.to_string())
        .parse()
        .context("CANE_MAX_TOKENS must be an integer")?;
    let path = std::env::current_dir()?;

    let provider = cane_core::ProviderConfig {
        base_url,
        api_key,
        max_tokens,
        model,
    };
    let workspace = cane_core::Workspace::new(path)?;
    let cane_home = resolve_cane_home(
        std::env::var_os("CANE_HOME"),
        std::env::var_os(HOME_DIRECTORY_VARIABLE),
    )
    .context("could not locate Cane's home directory")?;
    let sessions_directory = cane_home.join("sessions");
    let shell = configure_shell(shell_mode, &workspace)?;
    let sessions = cane_core::SessionConfig::new(CANE_VERSION, "", sessions_directory);

    if shell_mode == CliShellMode::Unsafe {
        eprintln!("{}", unsafe_shell_warning());
    }

    let mut agent = cane_core::spawn_agent_with_shell(provider, workspace, sessions, shell).await?;

    // Esc-to-interrupt stand-in: Ctrl-C cancels active work or exits an idle session.
    tokio::spawn({
        let cancel = agent.cancel.clone();
        async move {
            tokio::signal::ctrl_c().await.ok();
            cancel.cancel();
        }
    });

    let repl_result = repl::run_stdio(&mut agent).await;
    let join_result = agent.join().await.context("agent task failed");

    repl_result?;
    join_result?;

    println!();

    Ok(())
}

fn parse_cli_command(arguments: impl IntoIterator<Item = OsString>) -> anyhow::Result<CliCommand> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    match arguments.as_slice() {
        [] => Ok(CliCommand::Chat {
            shell_mode: CliShellMode::Sandboxed,
        }),
        [switch] if switch == "--no-shell" => Ok(CliCommand::Chat {
            shell_mode: CliShellMode::Disabled,
        }),
        [switch] if switch == "--unsafe-shell" => Ok(CliCommand::Chat {
            shell_mode: CliShellMode::Unsafe,
        }),
        [switch] if switch == "--doctor" => Ok(CliCommand::Doctor),
        [first, second]
            if (first == "--no-shell" && second == "--unsafe-shell")
                || (first == "--unsafe-shell" && second == "--no-shell") =>
        {
            Err(anyhow::anyhow!(
                "--no-shell and --unsafe-shell are mutually exclusive"
            ))
        }
        _ => Err(anyhow::anyhow!(
            "usage: cane [--no-shell | --unsafe-shell | --doctor]"
        )),
    }
}

fn run_doctor() -> anyhow::Result<bool> {
    let workspace = cane_core::Workspace::new(std::env::current_dir()?)?;
    let report = cane_core::command::diagnose_sandbox(sandbox_diagnostic_input(&workspace));

    print!("{}", report.render());
    Ok(report.is_supported())
}

fn sandbox_diagnostic_input(
    workspace: &cane_core::Workspace,
) -> cane_core::command::SandboxDiagnosticInput {
    let docker_endpoint = discover_docker_endpoint(
        std::env::var_os("DOCKER_HOST"),
        std::env::var_os("XDG_RUNTIME_DIR"),
    );
    sandbox_diagnostic_input_with_docker(workspace, docker_endpoint.as_ref())
}

fn sandbox_diagnostic_input_with_docker(
    workspace: &cane_core::Workspace,
    docker_endpoint: Option<&cane_core::command::DockerEndpoint>,
) -> cane_core::command::SandboxDiagnosticInput {
    let inherited_path = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect())
        .unwrap_or_default();
    let user_home = non_empty_path(std::env::var_os(HOME_DIRECTORY_VARIABLE));
    let pass_through_roots = user_home
        .as_deref()
        .map(user_executable_roots)
        .unwrap_or_default();
    cane_core::command::SandboxDiagnosticInput {
        docker_endpoint: docker_endpoint.map(|endpoint| endpoint.path().to_path_buf()),
        inherited_path,
        pass_through_roots,
        toolchain_roots: Vec::new(),
        workspace: workspace.root().to_path_buf(),
    }
}

#[cfg(target_os = "linux")]
fn configure_shell(
    mode: CliShellMode,
    workspace: &cane_core::Workspace,
) -> anyhow::Result<cane_core::AgentShellConfig> {
    match mode {
        CliShellMode::Disabled => Ok(cane_core::AgentShellConfig::disabled()),
        CliShellMode::Sandboxed => {
            let docker_endpoint = discover_docker_endpoint(
                std::env::var_os("DOCKER_HOST"),
                std::env::var_os("XDG_RUNTIME_DIR"),
            );
            let (executor, report) = cane_core::command::establish_bubblewrap_executor(
                sandbox_diagnostic_input_with_docker(workspace, docker_endpoint.as_ref()),
            )
            .map_err(|report| {
                anyhow::anyhow!(
                    "sandboxed shell startup failed closed:\n{}\nRun `cane --doctor` for the same capability report, use `--no-shell` to start without command execution, or explicitly accept host access with `--unsafe-shell`.",
                    report.render()
                )
            })?;
            let policy = report
                .policy
                .as_ref()
                .context("successful sandbox diagnostics did not produce a command policy")?;
            let environment = command_environment(
                policy.private_home(),
                policy.executable_path(),
                policy.private_temp(),
            )?;

            cane_core::AgentShellConfig::sandboxed(
                environment,
                Arc::new(executor),
                report.bubblewrap_version.clone(),
                policy,
            )
            .map(|config| match docker_endpoint {
                Some(endpoint) => {
                    config.with_integration(cane_core::ShellIntegration::Docker(endpoint))
                }
                None => config,
            })
            .map_err(Into::into)
        }
        CliShellMode::Unsafe => {
            let environment =
                command_environment(Path::new("/tmp"), &unsafe_visible_path(), Path::new("/tmp"))?;
            let executor = cane_core::command::UnsafeExecutor::new(workspace.root().to_path_buf());
            let docker_endpoint = discover_docker_endpoint(
                std::env::var_os("DOCKER_HOST"),
                std::env::var_os("XDG_RUNTIME_DIR"),
            );

            cane_core::AgentShellConfig::unsafe_host(
                environment,
                Arc::new(executor),
                workspace.root().to_path_buf(),
            )
            .map(|config| match docker_endpoint {
                Some(endpoint) => {
                    config.with_integration(cane_core::ShellIntegration::Docker(endpoint))
                }
                None => config,
            })
            .map_err(Into::into)
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn configure_shell(
    mode: CliShellMode,
    _workspace: &cane_core::Workspace,
) -> anyhow::Result<cane_core::AgentShellConfig> {
    match mode {
        CliShellMode::Disabled => Ok(cane_core::AgentShellConfig::disabled()),
        CliShellMode::Sandboxed | CliShellMode::Unsafe => Err(anyhow::anyhow!(
            "shell execution is currently supported only on Linux; use `--no-shell`"
        )),
    }
}

fn command_environment(
    home: &Path,
    path: &[PathBuf],
    temp_directory: &Path,
) -> anyhow::Result<cane_core::command::CommandEnvironmentConfig> {
    let home = utf8_path(home, "command HOME")?;
    let temp_directory = utf8_path(temp_directory, "command TMPDIR")?;
    let path = path
        .iter()
        .map(|entry| utf8_path(entry, "command PATH entry").map(str::to_string))
        .collect::<anyhow::Result<Vec<_>>>()?;

    cane_core::command::CommandEnvironmentConfig::new(home, path, temp_directory)
        .map_err(Into::into)
}

fn utf8_path<'a>(path: &'a Path, name: &str) -> anyhow::Result<&'a str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("{name} is not valid UTF-8: '{}'", path.display()))
}

fn unsafe_visible_path() -> Vec<PathBuf> {
    let inherited = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .unwrap_or_default();
    let mut allowed_roots = [
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
        "/run/current-system/sw/bin",
        "/nix/store",
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect::<Vec<_>>();
    if let Some(home) = non_empty_path(std::env::var_os(HOME_DIRECTORY_VARIABLE)) {
        allowed_roots.extend([home.join(".local").join("bin"), home.join("bin")]);
    }

    let mut visible = inherited
        .into_iter()
        .filter(|entry| {
            entry.is_absolute()
                && entry.is_dir()
                && allowed_roots
                    .iter()
                    .any(|root| entry == root || entry.starts_with(root))
        })
        .collect::<Vec<_>>();
    if visible.is_empty() {
        visible.extend(
            allowed_roots
                .into_iter()
                .filter(|path| path.is_dir() && path != Path::new("/nix/store")),
        );
    }
    visible
}

fn unsafe_shell_warning() -> &'static str {
    "WARNING: --unsafe-shell runs model-requested commands as unsandboxed host processes.\n\
     Commands may read host files and credential stores, access host IPC and networks, and inspect \
     Cane or other same-user processes. One-time command approval still applies."
}

fn user_executable_roots(home: &Path) -> Vec<PathBuf> {
    [home.join(".local").join("bin"), home.join("bin")]
        .into_iter()
        .filter(|path| path.is_dir())
        .collect()
}

fn discover_docker_endpoint(
    docker_host: Option<OsString>,
    xdg_runtime_directory: Option<OsString>,
) -> Option<cane_core::command::DockerEndpoint> {
    discover_docker_endpoint_from(
        docker_host,
        xdg_runtime_directory,
        Path::new("/var/run/docker.sock"),
    )
}

fn discover_docker_endpoint_from(
    docker_host: Option<OsString>,
    xdg_runtime_directory: Option<OsString>,
    system_socket: &Path,
) -> Option<cane_core::command::DockerEndpoint> {
    if let Some(docker_host) = docker_host.and_then(|value| value.into_string().ok())
        && let Some(path) = docker_host.strip_prefix("unix://")
        && let Ok(endpoint) = cane_core::command::DockerEndpoint::validate(path)
    {
        return Some(endpoint);
    }

    if let Ok(endpoint) = cane_core::command::DockerEndpoint::validate(system_socket) {
        return Some(endpoint);
    }

    non_empty_path(xdg_runtime_directory)
        .map(|directory| directory.join("docker.sock"))
        .and_then(|path| cane_core::command::DockerEndpoint::validate(path).ok())
}

fn resolve_cane_home(cane_home: Option<OsString>, user_home: Option<OsString>) -> Option<PathBuf> {
    non_empty_path(cane_home).or_else(|| non_empty_path(user_home).map(|path| path.join(".cane")))
}

fn non_empty_path(value: Option<OsString>) -> Option<PathBuf> {
    value.filter(|value| !value.is_empty()).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::fs;

    #[test]
    fn cane_home_overrides_the_platform_home_directory() {
        // Arrange
        let cane_home = Some(OsString::from("/configured/cane"));
        let user_home = Some(OsString::from("/users/example"));

        // Act
        let resolved = resolve_cane_home(cane_home, user_home);

        // Assert
        assert_eq!(resolved, Some(PathBuf::from("/configured/cane")));
    }

    #[test]
    fn platform_home_uses_the_dot_cane_default() {
        // Arrange
        let user_home = Some(OsString::from("/users/example"));

        // Act
        let resolved = resolve_cane_home(None, user_home);

        // Assert
        assert_eq!(resolved, Some(PathBuf::from("/users/example/.cane")));
    }

    #[test]
    fn empty_cane_home_falls_back_to_the_platform_home_directory() {
        // Arrange
        let cane_home = Some(OsString::new());
        let user_home = Some(OsString::from("/users/example"));

        // Act
        let resolved = resolve_cane_home(cane_home, user_home);

        // Assert
        assert_eq!(resolved, Some(PathBuf::from("/users/example/.cane")));
    }

    #[test]
    fn doctor_is_a_standalone_switch() {
        // Arrange
        let doctor = [OsString::from("--doctor")];
        let positional = [OsString::from("doctor")];
        let extra = [OsString::from("--doctor"), OsString::from("--json")];

        // Act
        let doctor_result = parse_cli_command(doctor);
        let positional_result = parse_cli_command(positional);
        let extra_result = parse_cli_command(extra);

        // Assert
        assert_eq!(doctor_result.unwrap(), CliCommand::Doctor);
        assert!(positional_result.is_err());
        assert!(extra_result.is_err());
    }

    #[test]
    fn shell_switches_select_explicit_mutually_exclusive_modes() {
        // Arrange
        let default = [];
        let disabled = [OsString::from("--no-shell")];
        let unsafe_host = [OsString::from("--unsafe-shell")];
        let both_orders = [
            [
                OsString::from("--no-shell"),
                OsString::from("--unsafe-shell"),
            ],
            [
                OsString::from("--unsafe-shell"),
                OsString::from("--no-shell"),
            ],
        ];

        // Act
        let default_result = parse_cli_command(default);
        let disabled_result = parse_cli_command(disabled);
        let unsafe_result = parse_cli_command(unsafe_host);
        let conflict_results = both_orders.map(parse_cli_command);

        // Assert
        assert_eq!(
            default_result.unwrap(),
            CliCommand::Chat {
                shell_mode: CliShellMode::Sandboxed,
            }
        );
        assert_eq!(
            disabled_result.unwrap(),
            CliCommand::Chat {
                shell_mode: CliShellMode::Disabled,
            }
        );
        assert_eq!(
            unsafe_result.unwrap(),
            CliCommand::Chat {
                shell_mode: CliShellMode::Unsafe,
            }
        );
        assert!(conflict_results.iter().all(Result::is_err));
    }

    #[test]
    fn unsafe_shell_warning_states_the_host_risk() {
        // Arrange
        let expected_risk = "unsandboxed host processes";

        // Act
        let warning = unsafe_shell_warning();

        // Assert
        assert!(warning.contains(expected_risk));
    }

    #[cfg(unix)]
    #[test]
    fn docker_discovery_validates_and_canonicalizes_unix_endpoints_from_the_environment() {
        // Arrange
        use std::os::unix::fs::symlink;
        use std::os::unix::net::UnixListener;
        use tempfile::TempDir;

        let root = TempDir::new().unwrap();
        let socket = root.path().join("daemon.sock");
        let link = root.path().join("docker.sock");
        let _listener = UnixListener::bind(&socket).unwrap();
        symlink(&socket, &link).unwrap();
        let unix = Some(OsString::from(format!("unix://{}", link.display())));
        let tcp = Some(OsString::from("tcp://localhost:2375"));
        let relative = Some(OsString::from("unix://docker.sock"));
        let missing_system_socket = root.path().join("system.sock");

        // Act
        let unix_endpoint = discover_docker_endpoint_from(unix, None, &missing_system_socket);
        let tcp_endpoint = discover_docker_endpoint_from(tcp, None, &missing_system_socket);
        let relative_endpoint =
            discover_docker_endpoint_from(relative, None, &missing_system_socket);

        // Assert
        assert_eq!(
            unix_endpoint.unwrap().path(),
            fs::canonicalize(socket).unwrap()
        );
        assert_eq!(tcp_endpoint, None);
        assert_eq!(relative_endpoint, None);
    }

    #[cfg(unix)]
    #[test]
    fn docker_discovery_falls_back_to_system_then_rootless_sockets() {
        // Arrange
        use std::os::unix::net::UnixListener;
        use tempfile::TempDir;

        let root = TempDir::new().unwrap();
        let system_socket = root.path().join("system.sock");
        let missing_system_socket = root.path().join("missing-system.sock");
        let runtime_directory = root.path().join("runtime");
        fs::create_dir(&runtime_directory).unwrap();
        let rootless_socket = runtime_directory.join("docker.sock");
        let _system_listener = UnixListener::bind(&system_socket).unwrap();
        let _rootless_listener = UnixListener::bind(&rootless_socket).unwrap();

        // Act
        let system = discover_docker_endpoint_from(
            Some(OsString::from("tcp://localhost:2375")),
            Some(runtime_directory.clone().into_os_string()),
            &system_socket,
        )
        .unwrap();
        let rootless = discover_docker_endpoint_from(
            None,
            Some(runtime_directory.into_os_string()),
            &missing_system_socket,
        )
        .unwrap();

        // Assert
        assert_eq!(system.path(), fs::canonicalize(system_socket).unwrap());
        assert_eq!(rootless.path(), fs::canonicalize(rootless_socket).unwrap());
    }
}
