use crate::Workspace;
use crate::command::{
    CommandDeadline, CommandEnvironmentConfig, CommandExecutionError, CommandExecutor,
    PreparedShellCommand, format_command_result, prepare_shell_command,
};
use crate::journal::{
    CapturedStream, CommandTermination as JournalCommandTermination, ToolExecutionCompleted,
    ToolExecutionStarted,
};
use crate::protocol::{AgentEvent, ApprovalLifetime, ApprovalRequirement, EventSink};
use crate::tools::{
    PreparedInvocation, Tool, ToolDefinition, ToolExecutionError, ToolExecutionOutput,
};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const PREVIEW_CHANNEL_CAPACITY: usize = 16;

pub(super) struct ShellTool {
    environment: CommandEnvironmentConfig,
    events: EventSink,
    executor: Arc<dyn CommandExecutor>,
    workspace: Arc<Workspace>,
}

impl ShellTool {
    pub(super) fn new(
        workspace: Arc<Workspace>,
        environment: CommandEnvironmentConfig,
        events: EventSink,
        executor: Arc<dyn CommandExecutor>,
    ) -> Self {
        Self {
            environment,
            events,
            executor,
            workspace,
        }
    }
}

#[async_trait::async_trait]
impl Tool for ShellTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "shell".to_string(),
            description: "Run a command with fixed non-login Bash in a fresh process. The working \
                directory defaults to the workspace root and must remain inside the workspace. \
                Commands require approval and time out after 600 seconds by default."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact Bash command to execute."
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds. Defaults to 600.",
                        "minimum": 1,
                        "maximum": 1800
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Optional working directory relative to the workspace root."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    async fn prepare(&self, input: Value) -> Result<Box<dyn PreparedInvocation>, String> {
        let prepared = prepare_shell_command(&self.environment, input, &self.workspace)
            .map_err(|error| error.to_string())?;

        Ok(Box::new(PreparedShellInvocation {
            events: self.events.clone(),
            executor: Arc::clone(&self.executor),
            prepared,
        }))
    }
}

struct PreparedShellInvocation {
    events: EventSink,
    executor: Arc<dyn CommandExecutor>,
    prepared: PreparedShellCommand,
}

#[async_trait::async_trait]
impl PreparedInvocation for PreparedShellInvocation {
    fn approval_requirement(&self) -> ApprovalRequirement {
        ApprovalRequirement::Required
    }

    fn available_grant_lifetimes(&self) -> &'static [ApprovalLifetime] {
        &[ApprovalLifetime::Invocation]
    }

    fn execution_started(&self) -> Option<ToolExecutionStarted> {
        Some(ToolExecutionStarted::Shell {
            capabilities: Vec::new(),
        })
    }

    async fn execute(
        self: Box<Self>,
        cancel: CancellationToken,
    ) -> Result<ToolExecutionOutput, ToolExecutionError> {
        let Self {
            events,
            executor,
            prepared,
        } = *self;
        let deadline = CommandDeadline::after(prepared.timeout());
        let (preview, mut preview_receiver) = mpsc::channel(PREVIEW_CHANNEL_CAPACITY);
        let preview_task = tokio::spawn(async move {
            while let Some(chunk) = preview_receiver.recv().await {
                events.emit_best_effort(AgentEvent::CommandOutput(chunk));
            }
        });
        let result = executor
            .execute(cancel, deadline, preview, prepared.into_request())
            .await;
        preview_task
            .await
            .map_err(|error| ToolExecutionError::ToolError(error.to_string()))?;
        let result = result.map_err(|error| match error {
            CommandExecutionError::Cancelled => ToolExecutionError::Cancelled,
            CommandExecutionError::Failed { .. } => {
                ToolExecutionError::ToolError(error.to_string())
            }
        })?;

        let execution = shell_execution_completed(&result);

        Ok(ToolExecutionOutput {
            content: format_command_result(&result),
            execution: Some(execution),
        })
    }
}

fn shell_execution_completed(result: &crate::command::CommandResult) -> ToolExecutionCompleted {
    let termination = match result.termination {
        crate::command::CommandTermination::Exited { code } => {
            JournalCommandTermination::Exited { code }
        }
        crate::command::CommandTermination::Signaled { signal } => {
            JournalCommandTermination::Signaled { signal }
        }
        crate::command::CommandTermination::TimedOut => JournalCommandTermination::TimedOut,
    };

    ToolExecutionCompleted::Shell {
        stderr: CapturedStream {
            bytes: result.output.stderr_bytes,
            truncated: result.output.stderr_truncated(),
        },
        stdout: CapturedStream {
            bytes: result.output.stdout_bytes,
            truncated: result.output.stdout_truncated(),
        },
        termination,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{
        CapturedOutput, CommandExecutionMode, CommandExecutorDescriptor, CommandOutputChunk,
        CommandOutputSender, CommandRequest, CommandResult, CommandTermination,
    };
    use crate::tools::{ShellToolConfig, ToolSet, ToolTestExt};
    use async_trait::async_trait;
    use std::sync::Mutex;
    use tempfile::TempDir;

    #[derive(Debug)]
    struct ExecutionObservation {
        cancelled: bool,
        request: CommandRequest,
    }

    struct FakeExecutor {
        observations: Mutex<Vec<ExecutionObservation>>,
        result: Result<CommandResult, CommandExecutionError>,
    }

    #[async_trait]
    impl CommandExecutor for FakeExecutor {
        fn descriptor(&self) -> CommandExecutorDescriptor {
            CommandExecutorDescriptor {
                backend: "fake".to_string(),
                mode: CommandExecutionMode::Sandboxed,
            }
        }

        async fn execute(
            &self,
            cancel: CancellationToken,
            _deadline: CommandDeadline,
            output: CommandOutputSender,
            request: CommandRequest,
        ) -> Result<CommandResult, CommandExecutionError> {
            self.observations
                .lock()
                .unwrap()
                .push(ExecutionObservation {
                    cancelled: cancel.is_cancelled(),
                    request,
                });

            match &self.result {
                Ok(result) => {
                    for chunk in &result.output.chunks {
                        let _ = output.try_send(chunk.clone());
                    }
                    Ok(result.clone())
                }
                Err(CommandExecutionError::Cancelled) => Err(CommandExecutionError::Cancelled),
                Err(CommandExecutionError::Failed { message }) => {
                    Err(CommandExecutionError::Failed {
                        message: message.clone(),
                    })
                }
            }
        }
    }

    fn shell_tool(
        result: Result<CommandResult, CommandExecutionError>,
    ) -> (
        TempDir,
        ShellTool,
        Arc<FakeExecutor>,
        mpsc::Receiver<AgentEvent>,
    ) {
        let root = TempDir::new().unwrap();
        let workspace = Arc::new(Workspace::new(root.path().to_path_buf()).unwrap());
        let executor = Arc::new(FakeExecutor {
            observations: Mutex::new(Vec::new()),
            result,
        });
        let environment = CommandEnvironmentConfig::new(
            "/sandbox/home",
            vec!["/usr/bin".to_string()],
            "/sandbox/tmp",
        )
        .unwrap();
        let (events, event_receiver) = mpsc::channel(16);
        let tool = ShellTool::new(
            workspace,
            environment,
            EventSink::new(events),
            Arc::clone(&executor) as Arc<dyn CommandExecutor>,
        );

        (root, tool, executor, event_receiver)
    }

    #[test]
    fn shell_definition_describes_the_prepared_command_contract() {
        // Arrange
        let result = CommandResult {
            output: CapturedOutput::complete(Vec::new()),
            termination: CommandTermination::Exited { code: 0 },
        };
        let (_root, tool, _executor, _events) = shell_tool(Ok(result));

        // Act
        let definition = tool.definition();

        // Assert
        assert_eq!(definition.name, "shell");
        assert_eq!(
            definition.input_schema["required"],
            serde_json::json!(["command"])
        );
        assert_eq!(definition.input_schema["additionalProperties"], false);
        assert_eq!(
            definition.input_schema["properties"]["timeout_seconds"]["maximum"],
            1800
        );
    }

    #[test]
    fn tool_set_registers_shell_only_when_configured() {
        // Arrange
        let root = TempDir::new().unwrap();
        let workspace = Arc::new(Workspace::new(root.path().to_path_buf()).unwrap());
        let executor = Arc::new(FakeExecutor {
            observations: Mutex::new(Vec::new()),
            result: Ok(CommandResult {
                output: CapturedOutput::complete(Vec::new()),
                termination: CommandTermination::Exited { code: 0 },
            }),
        });
        let environment =
            CommandEnvironmentConfig::new("/sandbox/home", vec!["/usr/bin".to_string()], "/tmp")
                .unwrap();
        let (events, _event_receiver) = mpsc::channel(16);

        // Act
        let without_shell = ToolSet::new(Arc::clone(&workspace), None);
        let with_shell = ToolSet::new(
            workspace,
            Some(ShellToolConfig {
                environment,
                events: EventSink::new(events),
                executor,
            }),
        );

        // Assert
        assert!(without_shell.locate("shell").is_err());
        assert!(with_shell.locate("shell").is_ok());
    }

    #[tokio::test]
    async fn shell_prepares_executes_and_formats_a_nonzero_result() {
        // Arrange
        let result = CommandResult {
            output: CapturedOutput::complete(vec![CommandOutputChunk::stderr(
                b"compile failed\n".to_vec(),
            )]),
            termination: CommandTermination::Exited { code: 2 },
        };
        let (root, tool, executor, _events) = shell_tool(Ok(result));
        let input = serde_json::json!({
            "command": "cargo check",
            "timeout_seconds": 30
        });

        // Act
        let output = tool.execute(input).await.unwrap();

        // Assert
        assert_eq!(
            output,
            "process exited with code 2\n\
             output (15 bytes, complete):\n\
             compile failed\n"
        );
        let observations = executor.observations.lock().unwrap();
        assert_eq!(observations.len(), 1);
        assert!(!observations[0].cancelled);
        assert_eq!(observations[0].request.workdir, root.path());
        assert_eq!(observations[0].request.executable, "/bin/bash");
        assert_eq!(
            observations[0].request.arguments,
            ["--noprofile", "--norc", "-c", "cargo check"]
        );
    }

    #[tokio::test]
    async fn shell_requires_a_one_invocation_approval() {
        // Arrange
        let result = CommandResult {
            output: CapturedOutput::complete(Vec::new()),
            termination: CommandTermination::Exited { code: 0 },
        };
        let (_root, tool, _executor, _events) = shell_tool(Ok(result));

        // Act
        let invocation = tool
            .prepare(serde_json::json!({ "command": "true" }))
            .await
            .unwrap();

        // Assert
        assert_eq!(
            invocation.approval_requirement(),
            ApprovalRequirement::Required
        );
        assert_eq!(
            invocation.available_grant_lifetimes(),
            [ApprovalLifetime::Invocation]
        );
    }

    #[tokio::test]
    async fn shell_reports_structured_start_and_completion_diagnostics() {
        // Arrange
        let result = CommandResult {
            output: CapturedOutput {
                chunks: vec![
                    CommandOutputChunk::stderr(b"stderr tail".to_vec()),
                    CommandOutputChunk::stdout(b"stdout".to_vec()),
                ],
                stderr_bytes: 50,
                stdout_bytes: 6,
            },
            termination: CommandTermination::TimedOut,
        };
        let (_root, tool, _executor, mut events) = shell_tool(Ok(result));
        let invocation = tool
            .prepare(serde_json::json!({ "command": "cargo test" }))
            .await
            .unwrap();

        // Act
        let started = invocation.execution_started();
        let completed = invocation.execute(CancellationToken::new()).await.unwrap();

        // Assert
        assert_eq!(
            started,
            Some(ToolExecutionStarted::Shell {
                capabilities: Vec::new(),
            })
        );
        assert_eq!(
            completed.execution,
            Some(ToolExecutionCompleted::Shell {
                stderr: CapturedStream {
                    bytes: 50,
                    truncated: true,
                },
                stdout: CapturedStream {
                    bytes: 6,
                    truncated: false,
                },
                termination: JournalCommandTermination::TimedOut,
            })
        );
        assert!(completed.content.starts_with("process timed out\n"));
        let AgentEvent::CommandOutput(stderr) = events.try_recv().unwrap() else {
            panic!("expected stderr preview");
        };
        let AgentEvent::CommandOutput(stdout) = events.try_recv().unwrap() else {
            panic!("expected stdout preview");
        };
        assert_eq!(stderr, CommandOutputChunk::stderr(b"stderr tail".to_vec()));
        assert_eq!(stdout, CommandOutputChunk::stdout(b"stdout".to_vec()));
    }

    #[tokio::test]
    async fn shell_maps_executor_cancellation_to_tool_cancellation() {
        // Arrange
        let (_root, tool, _executor, _events) = shell_tool(Err(CommandExecutionError::Cancelled));
        let invocation = tool
            .prepare(serde_json::json!({ "command": "sleep 10" }))
            .await
            .unwrap();

        // Act
        let result = invocation.execute(CancellationToken::new()).await;

        // Assert
        assert_eq!(result, Err(ToolExecutionError::Cancelled));
    }
}
