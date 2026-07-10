use anyhow::Context;
use cane_core::{AgentCommand, AgentEvent, AgentHandle, TurnOutcome};
use std::io::{BufRead, Write};

pub(crate) async fn run(
    mut agent: AgentHandle,
    mut input: impl BufRead,
    mut output: impl Write,
) -> anyhow::Result<()> {
    loop {
        let Some(line) = read_input(&mut input, &mut output)? else {
            break;
        };

        if line == "/quit" {
            break;
        }

        if agent
            .commands
            .send(AgentCommand::UserInput(line))
            .await
            .is_err()
        {
            // Exit cleanly if the agent task disappears.
            break;
        }

        loop {
            let ev = agent
                .events
                .recv()
                .await
                .context("agent stopped before completing the turn")?;

            match ev {
                AgentEvent::TextDelta(text) => {
                    write!(output, "{text}")?;
                    output.flush()?;
                }

                AgentEvent::ToolStarted { name, input } => {
                    let message = format!("\n[tool: {name} {input}]");
                    output.write_all(message.as_bytes())?;
                }

                AgentEvent::ToolFinished { .. } => {}

                AgentEvent::TurnComplete { outcome } => {
                    writeln!(output)?;
                    if matches!(outcome, TurnOutcome::Cancelled) {
                        return Ok(());
                    }
                    break;
                }

                AgentEvent::Error(e) => eprintln!("\nerror: {e}"),
            }
        }
    }

    Ok(())
}

fn read_input(input: &mut impl BufRead, output: &mut impl Write) -> anyhow::Result<Option<String>> {
    write!(output, "> ")?;
    output.flush()?;

    let mut line = String::new();

    match input.read_line(&mut line) {
        Ok(0) => Ok(None),
        Ok(_) => Ok(Some(line.trim_end().to_owned())),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cane_core::StopReason;
    use std::io::Cursor;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn run_displays_tool_errors_but_hides_successful_tool_output() {
        // Arrange
        let (commands, mut command_rx) = mpsc::channel(1);
        let (event_tx, events) = mpsc::channel(8);
        let agent = AgentHandle {
            cancel: Default::default(),
            commands,
            events,
        };
        let frontend = tokio::spawn(async move {
            command_rx.recv().await.unwrap();
            event_tx
                .send(AgentEvent::ToolStarted {
                    name: "read_file".to_string(),
                    input: Default::default(),
                })
                .await
                .unwrap();
            event_tx
                .send(AgentEvent::ToolFinished {
                    name: "read_file".to_string(),
                    output: "secret successful output".to_string(),
                    is_error: false,
                })
                .await
                .unwrap();
            event_tx
                .send(AgentEvent::ToolStarted {
                    name: "read_file".to_string(),
                    input: Default::default(),
                })
                .await
                .unwrap();
            event_tx
                .send(AgentEvent::ToolFinished {
                    name: "read_file".to_string(),
                    output: "access denied".to_string(),
                    is_error: true,
                })
                .await
                .unwrap();
            event_tx
                .send(AgentEvent::TextDelta("I could not read it.".to_string()))
                .await
                .unwrap();
            event_tx
                .send(AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn,
                    },
                })
                .await
                .unwrap();
        });
        let input = Cursor::new("inspect files\n/quit\n");
        let mut output = Vec::new();

        // Act
        run(agent, input, &mut output).await.unwrap();
        frontend.await.unwrap();

        // Assert
        let output = String::from_utf8(output).unwrap();
        assert_eq!(
            output,
            "> \n[tool: read_file null]\n\n[tool: read_file null]\n[tool error: access denied]\nI could not read it.\n> "
        );
        assert!(!output.contains("secret successful output"));
    }
}
