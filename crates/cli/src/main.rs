use anyhow::Context;
use cane_core::AgentCommand;
use std::io::{Write, stdout};
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

    let prompt = std::env::args().nth(1).context("usage: cane <prompt>")?;
    let api_key = std::env::var("CANE_API_KEY").context("CANE_API_KEY not set")?;
    let base_url = std::env::var("CANE_BASE_URL").context("CANE_BASE_URL not set")?;
    let model = std::env::var("CANE_MODEL").context("CANE_MODEL not set")?;
    let max_tokens: u32 = std::env::var("CANE_MAX_TOKENS")
        .unwrap_or_else(|_| "8192".into())
        .parse()
        .context("CANE_MAX_TOKENS must be an integer")?;

    let mut agent = cane_core::spawn_agent(cane_core::ProviderConfig {
        base_url,
        api_key,
        max_tokens,
        model,
    });

    // Esc-to-interrupt stand-in: Ctrl-C trips the cancellation token
    tokio::spawn({
        let cancel = agent.cancel.clone();
        async move {
            tokio::signal::ctrl_c().await.ok();
            cancel.cancel();
        }
    });

    agent.commands.send(AgentCommand::UserInput(prompt)).await?;

    loop {
        let ev = agent
            .events
            .recv()
            .await
            .context("agent stopped before completing the turn")?;

        match ev {
            cane_core::AgentEvent::TextDelta(t) => {
                print!("{t}");
                stdout().flush()?;
            }
            cane_core::AgentEvent::ToolStarted { name, input } => {
                println!("\n[tool: {name} {input}]")
            }
            cane_core::AgentEvent::ToolFinished { .. } => {}
            cane_core::AgentEvent::TurnComplete { .. } => break,
            cane_core::AgentEvent::Error(e) => {
                // Terminate any partially streamed line; anyhow prints the
                // error itself when main returns Err.
                eprintln!();
                return Err(anyhow::anyhow!(e));
            }
        }
    }

    println!();

    Ok(())
}
