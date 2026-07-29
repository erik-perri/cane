mod repl;

use anyhow::Context;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
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
    Chat,
    Doctor,
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

    match parse_cli_command(std::env::args_os().skip(1))? {
        CliCommand::Doctor => {
            if !run_doctor()? {
                std::process::exit(1);
            }
            return Ok(());
        }
        CliCommand::Chat => {}
    }

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
    let sessions = cane_core::SessionConfig::new(CANE_VERSION, "", sessions_directory);

    let mut agent = cane_core::spawn_agent(provider, workspace, sessions).await?;

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
        [] => Ok(CliCommand::Chat),
        [switch] if switch == "--doctor" => Ok(CliCommand::Doctor),
        _ => Err(anyhow::anyhow!("usage: cane [--doctor]")),
    }
}

fn run_doctor() -> anyhow::Result<bool> {
    let workspace = cane_core::Workspace::new(std::env::current_dir()?)?;
    let inherited_path = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect())
        .unwrap_or_default();
    let user_home = non_empty_path(std::env::var_os(HOME_DIRECTORY_VARIABLE));
    let pass_through_roots = user_home
        .as_deref()
        .map(user_executable_roots)
        .unwrap_or_default();
    let docker_endpoint = discover_docker_endpoint(
        std::env::var_os("DOCKER_HOST"),
        std::env::var_os("XDG_RUNTIME_DIR"),
    );
    let report = cane_core::command::diagnose_sandbox(cane_core::command::SandboxDiagnosticInput {
        docker_endpoint,
        inherited_path,
        pass_through_roots,
        toolchain_roots: Vec::new(),
        workspace: workspace.root().to_path_buf(),
    });

    print!("{}", report.render());
    Ok(report.is_supported())
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
) -> Option<PathBuf> {
    if let Some(docker_host) = docker_host.and_then(|value| value.into_string().ok())
        && let Some(path) = docker_host.strip_prefix("unix://")
    {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            return Some(path);
        }
    }

    let system_socket = PathBuf::from("/var/run/docker.sock");
    if system_socket.exists() {
        return Some(system_socket);
    }

    non_empty_path(xdg_runtime_directory)
        .map(|directory| directory.join("docker.sock"))
        .filter(|path| path.exists())
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

    #[cfg(unix)]
    #[test]
    fn docker_discovery_accepts_only_absolute_unix_endpoints_from_the_environment() {
        // Arrange
        let unix = Some(OsString::from("unix:///run/user/1000/docker.sock"));
        let tcp = Some(OsString::from("tcp://localhost:2375"));
        let relative = Some(OsString::from("unix://docker.sock"));

        // Act
        let unix_endpoint = discover_docker_endpoint(unix, None);
        let tcp_endpoint = discover_docker_endpoint(tcp, None);
        let relative_endpoint = discover_docker_endpoint(relative, None);

        // Assert
        assert_eq!(
            unix_endpoint,
            Some(PathBuf::from("/run/user/1000/docker.sock"))
        );
        assert_ne!(tcp_endpoint, Some(PathBuf::from("localhost:2375")));
        assert_ne!(relative_endpoint, Some(PathBuf::from("docker.sock")));
    }
}
