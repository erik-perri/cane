use anyhow::Context;
use cane_core::{
    AgentCommand, AgentEvent, AgentHandle, ApprovalDecision, ApprovalLifetime, ApprovalSubject,
    CapabilityKind, ShutdownReason, TurnOutcome,
};
use std::io::{BufRead, Write};
use tokio::sync::mpsc;

struct InputLines {
    receiver: mpsc::Receiver<std::io::Result<Option<String>>>,
}

#[cfg(test)]
struct TestAgent {
    cancel: tokio_util::sync::CancellationToken,
    commands: mpsc::Sender<AgentCommand>,
    events: mpsc::Receiver<AgentEvent>,
}

impl InputLines {
    #[cfg(test)]
    fn from_reader(reader: impl BufRead + Send + 'static) -> Self {
        Self::spawn(move |sender| read_lines(reader, sender))
    }

    fn stdin() -> Self {
        Self::spawn(move |sender| {
            let stdin = std::io::stdin();
            read_lines(stdin.lock(), sender);
        })
    }

    fn spawn(
        read: impl FnOnce(mpsc::Sender<std::io::Result<Option<String>>>) + Send + 'static,
    ) -> Self {
        let (sender, receiver) = mpsc::channel(1);
        std::thread::spawn(move || read(sender));
        Self { receiver }
    }

    async fn recv(&mut self) -> std::io::Result<Option<String>> {
        self.receiver.recv().await.unwrap_or(Ok(None))
    }
}

fn read_lines(mut reader: impl BufRead, sender: mpsc::Sender<std::io::Result<Option<String>>>) {
    loop {
        let mut line = String::new();
        let result = match reader.read_line(&mut line) {
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(line.trim_end().to_owned())),
            Err(error) => Err(error),
        };
        let finished = !matches!(result, Ok(Some(_)));

        if sender.blocking_send(result).is_err() || finished {
            return;
        }
    }
}

#[cfg(test)]
async fn run(
    agent: TestAgent,
    input: impl BufRead + Send + 'static,
    output: impl Write,
) -> anyhow::Result<()> {
    let TestAgent {
        cancel,
        commands,
        mut events,
    } = agent;

    run_with_input(
        &cancel,
        &commands,
        &mut events,
        InputLines::from_reader(input),
        output,
    )
    .await
}

pub(crate) async fn run_stdio(agent: &mut AgentHandle) -> anyhow::Result<()> {
    let stdout = std::io::stdout();
    run_with_input(
        &agent.cancel,
        &agent.commands,
        &mut agent.events,
        InputLines::stdin(),
        stdout.lock(),
    )
    .await
}

async fn run_with_input(
    cancel: &tokio_util::sync::CancellationToken,
    commands: &mpsc::Sender<AgentCommand>,
    events: &mut mpsc::Receiver<AgentEvent>,
    mut input: InputLines,
    mut output: impl Write,
) -> anyhow::Result<()> {
    loop {
        write_prompt(&mut output)?;
        let line = tokio::select! {
            line = input.recv() => line?,
            event = events.recv() => {
                match event {
                    None => break,
                    Some(AgentEvent::Error(error)) => {
                        writeln!(output, "\nerror: {error}")?;
                        continue;
                    }
                    Some(AgentEvent::Warning(warning)) => {
                        writeln!(output, "\nwarning: {warning}")?;
                        continue;
                    }
                    Some(event) => {
                        return Err(anyhow::anyhow!(
                            "agent emitted an unexpected event while idle: {event:?}"
                        ));
                    }
                }
            }
        };
        let Some(line) = line else {
            let _ = commands
                .send(AgentCommand::Shutdown(ShutdownReason::InputClosed))
                .await;
            break;
        };

        if line == "/quit" {
            let _ = commands
                .send(AgentCommand::Shutdown(ShutdownReason::UserQuit))
                .await;
            break;
        }

        if commands.send(AgentCommand::UserInput(line)).await.is_err() {
            // Exit cleanly if the agent task disappears.
            break;
        }

        loop {
            let event = events
                .recv()
                .await
                .context("agent stopped before completing the turn")?;

            match event {
                AgentEvent::TextDelta(text) => {
                    write!(output, "{text}")?;
                    output.flush()?;
                }

                AgentEvent::CommandOutput(chunk) => {
                    write!(output, "{}", String::from_utf8_lossy(&chunk.bytes))?;
                    output.flush()?;
                }

                AgentEvent::ToolStarted { name, input } => {
                    writeln!(output, "\n[tool: {name} {input}]")?;
                    output.flush()?;
                }

                AgentEvent::ToolFinished {
                    output: tool_output,
                    is_error: true,
                    ..
                } => {
                    writeln!(output, "[tool error: {tool_output}]")?;
                    output.flush()?;
                }

                AgentEvent::ToolFinished { .. } => {}

                AgentEvent::ToolDenied { reason, .. } => {
                    writeln!(output, "[tool denied: {reason}]")?;
                    output.flush()?;
                }

                AgentEvent::ToolRejected { error, .. } => {
                    writeln!(output, "[tool error: {error}]")?;
                    output.flush()?;
                }

                AgentEvent::TurnComplete { outcome } => match outcome {
                    TurnOutcome::Paused { reason } => {
                        writeln!(output, "\npaused: {reason}")?;
                        break;
                    }
                    TurnOutcome::Cancelled => {
                        writeln!(output)?;
                        return Ok(());
                    }
                    TurnOutcome::Completed { .. } | TurnOutcome::Failed => {
                        writeln!(output)?;
                        break;
                    }
                },

                AgentEvent::Error(e) => writeln!(output, "\nerror: {e}")?,
                AgentEvent::Warning(warning) => writeln!(output, "\nwarning: {warning}")?,

                AgentEvent::ApprovalRequest {
                    available_lifetimes,
                    input: command_input,
                    respond_to,
                    subject,
                } => {
                    let options = approval_options(&available_lifetimes);
                    if let Some(notice) = capability_notice(&subject) {
                        writeln!(output, "\n{notice}")?;
                    }
                    writeln!(
                        output,
                        "\n[approval request: {} {command_input}] {options}",
                        subject.tool_name(),
                    )?;
                    output.flush()?;

                    let decision = tokio::select! {
                        _ = cancel.cancelled() => None,
                        decision = read_decision(
                            &mut input,
                            &mut output,
                            &available_lifetimes,
                            &subject,
                        ) => Some(decision?),
                    };

                    let Some(decision) = decision else {
                        continue;
                    };

                    if respond_to.send(decision).is_err() {
                        // Exit if the agent task disappears.
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

fn capability_notice(subject: &ApprovalSubject) -> Option<String> {
    let ApprovalSubject::Capability { capability, .. } = subject else {
        return None;
    };
    match capability.kind() {
        CapabilityKind::DockerDaemon => Some(format!(
            "WARNING: Docker daemon access can mount arbitrary host paths, use external networks, \
             and create persistent host effects. The command sandbox does not make Docker safe, \
             and containers may outlive this command or Cane. Endpoint: {}",
            capability.resource()
        )),
    }
}

async fn read_decision(
    input: &mut InputLines,
    output: &mut impl Write,
    available_lifetimes: &[ApprovalLifetime],
    subject: &ApprovalSubject,
) -> anyhow::Result<ApprovalDecision> {
    loop {
        let Some(line) = read_input(input, output).await? else {
            return Err(anyhow::anyhow!("eof"));
        };

        let allow_input = line.trim().to_lowercase();

        if allow_input.is_empty() || allow_input == "n" {
            writeln!(output, "\nreason:")?;
            output.flush()?;

            let Some(reason) = read_input(input, output).await? else {
                return Err(anyhow::anyhow!("eof"));
            };

            return Ok(ApprovalDecision::Deny { reason });
        } else if allow_input == "a" && available_lifetimes.contains(&ApprovalLifetime::Run) {
            return Ok(ApprovalDecision::Grant(
                subject.grant(ApprovalLifetime::Run),
            ));
        } else if allow_input == "w" && available_lifetimes.contains(&ApprovalLifetime::Workspace) {
            return Ok(ApprovalDecision::Grant(
                subject.grant(ApprovalLifetime::Workspace),
            ));
        } else if allow_input == "y" && available_lifetimes.contains(&ApprovalLifetime::Invocation)
        {
            return Ok(ApprovalDecision::Grant(
                subject.grant(ApprovalLifetime::Invocation),
            ));
        }
    }
}

fn approval_options(available_lifetimes: &[ApprovalLifetime]) -> String {
    let mut options = Vec::new();
    if available_lifetimes.contains(&ApprovalLifetime::Invocation) {
        options.push("y");
    }
    options.push("n");
    if available_lifetimes.contains(&ApprovalLifetime::Run) {
        options.push("a");
    }
    if available_lifetimes.contains(&ApprovalLifetime::Workspace) {
        options.push("w");
    }
    format!("[{}]", options.join(","))
}

async fn read_input(
    input: &mut InputLines,
    output: &mut impl Write,
) -> anyhow::Result<Option<String>> {
    write_prompt(output)?;
    Ok(input.recv().await?)
}

fn write_prompt(output: &mut impl Write) -> anyhow::Result<()> {
    write!(output, "> ")?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cane_core::StopReason;
    use cane_core::command::CommandOutputChunk;
    use std::io::{self, Cursor, Read};
    use std::sync::mpsc as std_mpsc;
    use tokio::sync::{mpsc, oneshot};

    fn offered_tool_lifetimes() -> Vec<ApprovalLifetime> {
        vec![ApprovalLifetime::Invocation, ApprovalLifetime::Run]
    }

    fn tool_subject() -> ApprovalSubject {
        ApprovalSubject::tool_call("call_abc", "write_file")
    }

    async fn read_tool_decision(
        input: &mut InputLines,
        output: &mut impl Write,
    ) -> anyhow::Result<ApprovalDecision> {
        read_decision(input, output, &offered_tool_lifetimes(), &tool_subject()).await
    }

    struct GatedEof {
        released: bool,
        release: std_mpsc::Receiver<()>,
    }

    impl GatedEof {
        fn wait_for_release(&mut self) -> io::Result<()> {
            if !self.released {
                self.release.recv().map_err(|_| io::ErrorKind::BrokenPipe)?;
                self.released = true;
            }
            Ok(())
        }
    }

    impl Read for GatedEof {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            self.wait_for_release()?;
            Ok(0)
        }
    }

    impl BufRead for GatedEof {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.wait_for_release()?;
            Ok(&[])
        }

        fn consume(&mut self, _amount: usize) {}
    }

    #[tokio::test]
    async fn run_displays_tool_errors_but_hides_successful_tool_output() {
        // Arrange
        let (commands, mut command_rx) = mpsc::channel(1);
        let (event_tx, events) = mpsc::channel(8);
        let agent = TestAgent {
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

    #[tokio::test]
    async fn run_displays_live_command_output_in_observed_order() {
        // Arrange
        let (commands, mut command_rx) = mpsc::channel(1);
        let (event_tx, events) = mpsc::channel(8);
        let agent = TestAgent {
            cancel: Default::default(),
            commands,
            events,
        };
        let frontend = tokio::spawn(async move {
            command_rx.recv().await.unwrap();
            event_tx
                .send(AgentEvent::ToolStarted {
                    name: "shell".to_string(),
                    input: Default::default(),
                })
                .await
                .unwrap();
            event_tx
                .send(AgentEvent::CommandOutput(CommandOutputChunk::stdout(
                    b"building ".to_vec(),
                )))
                .await
                .unwrap();
            event_tx
                .send(AgentEvent::CommandOutput(CommandOutputChunk::stderr(
                    b"warning\n".to_vec(),
                )))
                .await
                .unwrap();
            event_tx
                .send(AgentEvent::ToolFinished {
                    name: "shell".to_string(),
                    output: "process exited with code 0".to_string(),
                    is_error: false,
                })
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
        let input = Cursor::new("build\n/quit\n");
        let mut output = Vec::new();

        // Act
        run(agent, input, &mut output).await.unwrap();
        frontend.await.unwrap();

        // Assert
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "> \n[tool: shell null]\nbuilding warning\n\n> "
        );
    }

    #[tokio::test]
    async fn run_displays_an_approval_request_and_sends_allow_through_its_responder() {
        // Arrange
        let (commands, mut command_rx) = mpsc::channel(1);
        let (event_tx, events) = mpsc::channel(8);
        let agent = TestAgent {
            cancel: Default::default(),
            commands,
            events,
        };
        let frontend = tokio::spawn(async move {
            command_rx.recv().await.unwrap();
            let (respond_to, decision_rx) = oneshot::channel();
            event_tx
                .send(AgentEvent::ApprovalRequest {
                    available_lifetimes: offered_tool_lifetimes(),
                    input: Default::default(),
                    respond_to,
                    subject: tool_subject(),
                })
                .await
                .unwrap();
            assert_eq!(
                decision_rx.await.unwrap(),
                ApprovalDecision::Grant(tool_subject().grant(ApprovalLifetime::Invocation))
            );
            event_tx
                .send(AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn,
                    },
                })
                .await
                .unwrap();
        });
        let input = Cursor::new("write the file\ny\n/quit\n");
        let mut output = Vec::new();

        // Act
        run(agent, input, &mut output).await.unwrap();
        frontend.await.unwrap();

        // Assert
        let output = String::from_utf8(output).unwrap();
        assert_eq!(
            output,
            "> \n[approval request: write_file null] [y,n,a]\n> \n> "
        );
    }

    #[tokio::test]
    async fn run_writes_agent_errors_to_the_output() {
        // Arrange
        let (commands, mut command_rx) = mpsc::channel(1);
        let (event_tx, events) = mpsc::channel(8);
        let agent = TestAgent {
            cancel: Default::default(),
            commands,
            events,
        };
        let frontend = tokio::spawn(async move {
            command_rx.recv().await.unwrap();
            event_tx
                .send(AgentEvent::Error("provider fell over".to_string()))
                .await
                .unwrap();
            event_tx
                .send(AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Failed,
                })
                .await
                .unwrap();
        });
        let input = Cursor::new("hello\n/quit\n");
        let mut output = Vec::new();

        // Act
        run(agent, input, &mut output).await.unwrap();
        frontend.await.unwrap();

        // Assert
        let output = String::from_utf8(output).unwrap();
        assert_eq!(output, "> \nerror: provider fell over\n\n> ");
    }

    #[tokio::test]
    async fn run_displays_a_pause_and_accepts_a_continuation() {
        // Arrange
        let (commands, mut command_rx) = mpsc::channel(2);
        let (event_tx, events) = mpsc::channel(2);
        let agent = TestAgent {
            cancel: Default::default(),
            commands,
            events,
        };
        let frontend = tokio::spawn(async move {
            assert_eq!(
                command_rx.recv().await,
                Some(AgentCommand::UserInput("start".to_string()))
            );
            event_tx
                .send(AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Paused {
                        reason: "provider round budget reached".to_string(),
                    },
                })
                .await
                .unwrap();
            assert_eq!(
                command_rx.recv().await,
                Some(AgentCommand::UserInput("Continue".to_string()))
            );
            event_tx
                .send(AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Completed {
                        stop_reason: StopReason::EndTurn,
                    },
                })
                .await
                .unwrap();
        });
        let input = Cursor::new("start\nContinue\n/quit\n");
        let mut output = Vec::new();

        // Act
        run(agent, input, &mut output).await.unwrap();
        frontend.await.unwrap();

        // Assert
        let output = String::from_utf8(output).unwrap();
        assert_eq!(output, "> \npaused: provider round budget reached\n> \n> ");
    }

    #[tokio::test]
    async fn run_returns_after_a_cancelled_turn_without_processing_more_input() {
        // Arrange
        let (commands, mut command_rx) = mpsc::channel(1);
        let (event_tx, events) = mpsc::channel(8);
        let agent = TestAgent {
            cancel: Default::default(),
            commands,
            events,
        };
        let frontend = tokio::spawn(async move {
            let first_command = command_rx.recv().await.unwrap();
            event_tx
                .send(AgentEvent::TurnComplete {
                    outcome: TurnOutcome::Cancelled,
                })
                .await
                .unwrap();
            let next_command = command_rx.recv().await;
            (first_command, next_command)
        });
        let input = Cursor::new("start something\nnever read\n");
        let mut output = Vec::new();

        // Act
        run(agent, input, &mut output).await.unwrap();
        let (first_command, next_command) = frontend.await.unwrap();

        // Assert
        let output = String::from_utf8(output).unwrap();
        assert_eq!(output, "> \n");
        assert_eq!(
            first_command,
            AgentCommand::UserInput("start something".to_string())
        );
        assert_eq!(next_command, None);
    }

    #[tokio::test]
    async fn run_treats_idle_eof_as_a_clean_exit() {
        // Arrange
        let (commands, mut command_rx) = mpsc::channel(1);
        let (_event_tx, events) = mpsc::channel(1);
        let agent = TestAgent {
            cancel: Default::default(),
            commands,
            events,
        };
        let mut output = Vec::new();

        // Act
        let result = run(agent, Cursor::new(""), &mut output).await;

        // Assert
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(String::from_utf8(output).unwrap(), "> ");
        assert_eq!(
            command_rx.recv().await,
            Some(AgentCommand::Shutdown(ShutdownReason::InputClosed))
        );
    }

    #[tokio::test]
    async fn run_reports_an_explicit_user_quit() {
        // Arrange
        let (commands, mut command_rx) = mpsc::channel(1);
        let (_event_tx, events) = mpsc::channel(1);
        let agent = TestAgent {
            cancel: Default::default(),
            commands,
            events,
        };
        let mut output = Vec::new();

        // Act
        let result = run(agent, Cursor::new("/quit\n"), &mut output).await;

        // Assert
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(String::from_utf8(output).unwrap(), "> ");
        assert_eq!(
            command_rx.recv().await,
            Some(AgentCommand::Shutdown(ShutdownReason::UserQuit))
        );
    }

    #[tokio::test]
    async fn run_exits_cleanly_when_the_agent_command_channel_is_closed() {
        // Arrange
        let (commands, command_rx) = mpsc::channel(1);
        drop(command_rx);
        let (_event_tx, events) = mpsc::channel(8);
        let agent = TestAgent {
            cancel: Default::default(),
            commands,
            events,
        };
        let input = Cursor::new("hello\n");
        let mut output = Vec::new();

        // Act
        let result = run(agent, input, &mut output).await;

        // Assert
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(String::from_utf8(output).unwrap(), "> ");
    }

    #[tokio::test]
    async fn run_exits_when_an_idle_agent_stops_while_input_is_blocked() {
        // Arrange
        let (commands, _command_rx) = mpsc::channel(1);
        let (event_tx, events) = mpsc::channel(1);
        drop(event_tx);
        let agent = TestAgent {
            cancel: Default::default(),
            commands,
            events,
        };
        let (release_tx, release_rx) = std_mpsc::channel();
        let input = GatedEof {
            released: false,
            release: release_rx,
        };
        let mut output = Vec::new();

        // Act
        let result = run(agent, input, &mut output).await;
        release_tx.send(()).unwrap();

        // Assert
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(String::from_utf8(output).unwrap(), "> ");
    }

    #[tokio::test]
    async fn read_decision_returns_deny_with_the_entered_reason() {
        // Arrange
        let mut input = InputLines::from_reader(Cursor::new("n\nit would clobber my changes\n"));
        let mut output = Vec::new();

        // Act
        let decision = read_tool_decision(&mut input, &mut output).await.unwrap();

        // Assert
        assert_eq!(
            decision,
            ApprovalDecision::Deny {
                reason: "it would clobber my changes".to_string()
            }
        );
        assert_eq!(String::from_utf8(output).unwrap(), "> \nreason:\n> ");
    }

    #[test]
    fn approval_options_only_list_offered_grant_lifetimes() {
        // Arrange
        let available_lifetimes = vec![ApprovalLifetime::Run, ApprovalLifetime::Workspace];

        // Act
        let options = approval_options(&available_lifetimes);

        // Assert
        assert_eq!(options, "[n,a,w]");
    }

    #[test]
    fn docker_capability_notice_states_the_host_authority_and_persistence_risks() {
        // Arrange
        let subject = ApprovalSubject::capability(
            cane_core::NamedCapability::docker_daemon("unix:///run/user/1000/docker.sock"),
            "shell-1",
            "shell",
        );

        // Act
        let notice = capability_notice(&subject).unwrap();

        // Assert
        assert!(notice.contains("mount arbitrary host paths"));
        assert!(notice.contains("external networks"));
        assert!(notice.contains("persistent host effects"));
        assert!(notice.contains("does not make Docker safe"));
        assert!(notice.contains("containers may outlive"));
        assert!(notice.contains("unix:///run/user/1000/docker.sock"));
    }

    #[tokio::test]
    async fn read_decision_treats_an_empty_decision_as_deny() {
        // Arrange
        let mut input = InputLines::from_reader(Cursor::new("\nnot like that\n"));
        let mut output = Vec::new();

        // Act
        let decision = read_tool_decision(&mut input, &mut output).await.unwrap();

        // Assert
        assert_eq!(
            decision,
            ApprovalDecision::Deny {
                reason: "not like that".to_string()
            }
        );
    }

    #[tokio::test]
    async fn read_decision_accepts_allow_for_run_case_insensitively_with_whitespace() {
        // Arrange
        let mut input = InputLines::from_reader(Cursor::new("  A  \n"));
        let mut output = Vec::new();

        // Act
        let decision = read_tool_decision(&mut input, &mut output).await.unwrap();

        // Assert
        assert_eq!(
            decision,
            ApprovalDecision::Grant(tool_subject().grant(ApprovalLifetime::Run))
        );
    }

    #[tokio::test]
    async fn read_decision_reprompts_after_unrecognized_input() {
        // Arrange
        let mut input = InputLines::from_reader(Cursor::new("maybe\ny\n"));
        let mut output = Vec::new();

        // Act
        let decision = read_tool_decision(&mut input, &mut output).await.unwrap();

        // Assert
        assert_eq!(
            decision,
            ApprovalDecision::Grant(tool_subject().grant(ApprovalLifetime::Invocation))
        );
        let output = String::from_utf8(output).unwrap();
        assert_eq!(
            output.matches("> ").count(),
            2,
            "junk input must reprompt, not fall through: {output:?}"
        );
    }

    #[tokio::test]
    async fn read_decision_errors_on_eof_before_a_decision() {
        // Arrange
        let mut input = InputLines::from_reader(Cursor::new(""));
        let mut output = Vec::new();

        // Act
        let result = read_tool_decision(&mut input, &mut output).await;

        // Assert
        assert_eq!(result.unwrap_err().to_string(), "eof");
    }

    #[tokio::test]
    async fn read_decision_errors_on_eof_before_a_denial_reason() {
        // Arrange
        let mut input = InputLines::from_reader(Cursor::new("n\n"));
        let mut output = Vec::new();

        // Act
        let result = read_tool_decision(&mut input, &mut output).await;

        // Assert
        assert_eq!(result.unwrap_err().to_string(), "eof");
        assert!(String::from_utf8(output).unwrap().contains("reason:"));
    }
}
