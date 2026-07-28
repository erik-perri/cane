mod repl;

use anyhow::Context;
use std::ffi::OsString;
use std::path::PathBuf;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt};

const CANE_VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_MAX_TOKENS: u32 = 32 * 1024;

#[cfg(not(windows))]
const HOME_DIRECTORY_VARIABLE: &str = "HOME";
#[cfg(windows)]
const HOME_DIRECTORY_VARIABLE: &str = "USERPROFILE";

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
}
