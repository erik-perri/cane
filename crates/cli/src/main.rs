mod repl;

use anyhow::Context;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt};

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
        .unwrap_or_else(|_| "8192".into())
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

    let agent = cane_core::spawn_agent(provider, workspace);

    // Esc-to-interrupt stand-in: Ctrl-C trips the cancellation token
    tokio::spawn({
        let cancel = agent.cancel.clone();
        async move {
            tokio::signal::ctrl_c().await.ok();
            cancel.cancel();
        }
    });

    let stdin = std::io::stdin();
    let input = stdin.lock();

    let stdout = std::io::stdout();
    let output = stdout.lock();

    repl::run(agent, input, output).await?;

    println!();

    Ok(())
}
