use crate::Workspace;
use crate::command::{
    CommandDeadline, CommandEnvironmentConfig, CommandExecutionError, CommandExecutor,
    DockerEndpoint, DockerExecutableName, DockerIntegration, PreparedShellCommand,
    format_command_result, invokes_direct_executable, prepare_shell_command,
};
use crate::journal::{
    CapturedStream, CommandTermination as JournalCommandTermination, ExecutionCapability,
    ToolExecutionCompleted, ToolExecutionStarted,
};
use crate::protocol::{
    AgentEvent, ApprovalLifetime, ApprovalRequirement, EventSink, NamedCapability,
};
use crate::tools::{
    AuthorizedCapability, CapabilityRequest, PreparedInvocation, Tool, ToolDefinition,
    ToolExecutionError, ToolExecutionOutput,
};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const PREVIEW_CHANNEL_CAPACITY: usize = 16;
const DOCKER_CAPABILITY_LIFETIMES: &[ApprovalLifetime] =
    &[ApprovalLifetime::Run, ApprovalLifetime::Workspace];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShellIntegration {
    Docker(DockerIntegration),
}

#[derive(Clone, Default)]
pub(crate) struct ShellIntegrations {
    docker: Option<DockerIntegration>,
}

impl ShellIntegrations {
    pub(crate) fn insert(&mut self, integration: ShellIntegration) {
        match integration {
            ShellIntegration::Docker(integration) => self.docker = Some(integration),
        }
    }

    fn docker_integration_for(&self, prepared: &PreparedShellCommand) -> Option<DockerIntegration> {
        self.docker
            .as_ref()
            .filter(|integration| {
                [
                    (DockerExecutableName::Docker, "docker"),
                    (DockerExecutableName::DockerCompose, "docker-compose"),
                ]
                .into_iter()
                .any(|(name, executable)| {
                    integration.supports(name)
                        && invokes_direct_executable(prepared.command(), executable)
                })
            })
            .cloned()
    }

    pub(crate) fn docker_endpoint(&self) -> Option<&DockerEndpoint> {
        self.docker.as_ref().map(DockerIntegration::endpoint)
    }
}

pub(super) struct ShellTool {
    environment: CommandEnvironmentConfig,
    events: EventSink,
    executor: Arc<dyn CommandExecutor>,
    integrations: ShellIntegrations,
    workspace: Arc<Workspace>,
}

impl ShellTool {
    pub(super) fn new(
        workspace: Arc<Workspace>,
        integrations: ShellIntegrations,
        environment: CommandEnvironmentConfig,
        events: EventSink,
        executor: Arc<dyn CommandExecutor>,
    ) -> Self {
        Self {
            environment,
            events,
            executor,
            integrations,
            workspace,
        }
    }
}

#[async_trait::async_trait]
impl Tool for ShellTool {
    fn definition(&self) -> ToolDefinition {
        let docker_available = self.integrations.docker.is_some();
        ToolDefinition {
            name: "shell".to_string(),
            description: shell_description(docker_available),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": shell_command_description(docker_available)
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
        let docker = self.integrations.docker_integration_for(&prepared);

        Ok(Box::new(PreparedShellInvocation {
            authorized_capability: None,
            docker,
            events: self.events.clone(),
            executor: Arc::clone(&self.executor),
            prepared,
        }))
    }
}

fn shell_description(docker_available: bool) -> String {
    let mut description = "Run an exact command with `/bin/bash --noprofile --norc -c` in a \
            fresh process. The working directory defaults to the workspace root and may be changed \
            per call with `workdir`; changes such as `cd` do not persist between calls. \
            `timeout_seconds` defaults to 600 and must be between 1 and 1800."
        .to_string();
    if docker_available {
        description.push_str(
            " A shell invocation containing an actual command-position `docker` or \
             `docker-compose` command can request Docker daemon access. When authorized, the \
             entire shell invocation and all descendants receive that access. Mere mentions, \
             executable paths, and opaque wrappers do not request it.",
        );
    }

    description
}

fn shell_command_description(docker_available: bool) -> &'static str {
    if docker_available {
        "The exact Bash command to execute. An actual command-position `docker` or \
         `docker-compose` command makes the entire shell invocation and its descendants eligible \
         for Docker daemon access; mentioning Docker as data or hiding it behind a wrapper does not."
    } else {
        "The exact Bash command to execute."
    }
}

struct PreparedShellInvocation {
    authorized_capability: Option<AuthorizedCapability>,
    docker: Option<DockerIntegration>,
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

    fn capability_request(&self) -> Option<CapabilityRequest> {
        self.docker.as_ref().map(|integration| CapabilityRequest {
            available_lifetimes: DOCKER_CAPABILITY_LIFETIMES,
            capability: NamedCapability::docker_daemon(integration.endpoint().resource()),
        })
    }

    fn authorize_capability(&mut self, authorization: AuthorizedCapability) -> Result<(), String> {
        let integration = self
            .docker
            .as_ref()
            .ok_or_else(|| "shell invocation did not request Docker access".to_string())?;
        let expected = NamedCapability::docker_daemon(integration.endpoint().resource());
        if authorization.capability != expected {
            return Err("capability authorization does not match the Docker endpoint".to_string());
        }

        self.prepared.authorize_docker(integration.clone());
        self.authorized_capability = Some(authorization);
        Ok(())
    }

    fn execution_started(&self) -> Option<ToolExecutionStarted> {
        Some(ToolExecutionStarted::Shell {
            capabilities: self
                .authorized_capability
                .iter()
                .map(|authorization| ExecutionCapability {
                    capability: authorization.capability.clone(),
                    source: authorization.source.clone(),
                })
                .collect(),
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
            ..
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
    #[cfg(unix)]
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    use std::os::unix::net::UnixListener;
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
        shell_tool_with_docker(result, None)
    }

    fn shell_tool_with_docker(
        result: Result<CommandResult, CommandExecutionError>,
        docker: Option<DockerIntegration>,
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
        let mut integrations = ShellIntegrations::default();
        if let Some(integration) = docker {
            integrations.insert(ShellIntegration::Docker(integration));
        }
        let tool = ShellTool::new(
            workspace,
            integrations,
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
        assert!(definition.description.contains("`cd` do not persist"));
        assert!(definition.description.contains("defaults to 600"));
        assert!(!definition.description.contains("Docker"));
        assert!(!definition.description.contains("sandbox"));
        assert!(!definition.description.contains("approval"));
    }

    #[test]
    fn shell_definition_advertises_only_configured_docker_command_support() {
        // Arrange
        let docker_available = true;

        // Act
        let description = shell_description(docker_available);
        let command_description = shell_command_description(docker_available);

        // Assert
        assert!(description.contains("actual command-position `docker`"));
        assert!(description.contains("entire shell invocation and all descendants"));
        assert!(description.contains("opaque wrappers"));
        assert!(command_description.contains("command-position `docker`"));
        assert!(command_description.contains("entire shell invocation"));
        assert!(command_description.contains("mentioning Docker as data"));
        assert!(!description.contains("sandbox"));
        assert!(!description.contains("approval"));
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
                integrations: ShellIntegrations::default(),
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
        let (_root, tool, executor, _events) = shell_tool(Ok(result));
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
        assert_eq!(observations[0].request.workdir, tool.workspace.root());
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

    #[cfg(unix)]
    #[tokio::test]
    async fn direct_docker_command_requires_separate_run_capability_authorization() {
        // Arrange
        let socket_root = TempDir::new().unwrap();
        let socket_path = socket_root.path().join("docker.sock");
        let _socket = UnixListener::bind(&socket_path).unwrap();
        let endpoint = DockerEndpoint::validate(&socket_path).unwrap();
        let docker_binary = socket_root.path().join("docker");
        fs::write(&docker_binary, "#!/bin/true\n").unwrap();
        fs::set_permissions(&docker_binary, fs::Permissions::from_mode(0o755)).unwrap();
        let executable = crate::command::DockerExecutable::validate(
            DockerExecutableName::Docker,
            &docker_binary,
        )
        .unwrap();
        let docker =
            DockerIntegration::new(endpoint.clone()).with_executables([executable.clone()]);
        let result = CommandResult {
            output: CapturedOutput::complete(Vec::new()),
            termination: CommandTermination::Exited { code: 0 },
        };
        let (_root, tool, executor, _events) = shell_tool_with_docker(Ok(result), Some(docker));
        let mut invocation = tool
            .prepare(serde_json::json!({ "command": "docker ps" }))
            .await
            .unwrap();
        let approval_id = "appr_01ARZ3NDEKTSV4RRFFQ69G5FAY".parse().unwrap();

        // Act
        let capability_request = invocation.capability_request().unwrap();
        invocation
            .authorize_capability(AuthorizedCapability {
                capability: capability_request.capability.clone(),
                source: crate::journal::CapabilityAuthorizationSource::Approval { approval_id },
            })
            .unwrap();
        let started = invocation.execution_started();
        let output = invocation.execute(CancellationToken::new()).await.unwrap();

        // Assert
        assert_eq!(
            capability_request,
            CapabilityRequest {
                available_lifetimes: &[ApprovalLifetime::Run, ApprovalLifetime::Workspace,],
                capability: NamedCapability::docker_daemon(endpoint.resource()),
            }
        );
        assert_eq!(
            started,
            Some(ToolExecutionStarted::Shell {
                capabilities: vec![ExecutionCapability {
                    capability: capability_request.capability,
                    source: crate::journal::CapabilityAuthorizationSource::Approval { approval_id },
                }],
            })
        );
        assert_eq!(
            output.content,
            "process exited with code 0\noutput (0 bytes, complete):\n"
        );
        assert_eq!(
            executor.observations.lock().unwrap()[0]
                .request
                .docker_endpoint,
            Some(endpoint)
        );
        assert_eq!(
            executor.observations.lock().unwrap()[0]
                .request
                .docker_executables,
            vec![executable]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn compound_invocation_with_a_direct_docker_segment_requests_daemon_access_for_all() {
        // Arrange
        let socket_root = TempDir::new().unwrap();
        let socket_path = socket_root.path().join("docker.sock");
        let _socket = UnixListener::bind(&socket_path).unwrap();
        let endpoint = DockerEndpoint::validate(socket_path).unwrap();
        let result = CommandResult {
            output: CapturedOutput::complete(Vec::new()),
            termination: CommandTermination::Exited { code: 0 },
        };
        let (_root, tool, executor, _events) =
            shell_tool_with_docker(Ok(result), Some(endpoint.clone().into()));
        let mut invocation = tool
            .prepare(serde_json::json!({
                "command": "which docker 2>&1; docker ps | cat"
            }))
            .await
            .unwrap();
        let approval_id = "appr_01ARZ3NDEKTSV4RRFFQ69G5FAY".parse().unwrap();

        // Act
        let capability_request = invocation.capability_request().unwrap();
        invocation
            .authorize_capability(AuthorizedCapability {
                capability: capability_request.capability.clone(),
                source: crate::journal::CapabilityAuthorizationSource::Approval { approval_id },
            })
            .unwrap();
        let output = invocation.execute(CancellationToken::new()).await.unwrap();

        // Assert
        assert_eq!(
            capability_request.capability,
            NamedCapability::docker_daemon(endpoint.resource())
        );
        assert_eq!(
            output.content,
            "process exited with code 0\noutput (0 bytes, complete):\n"
        );
        assert_eq!(
            executor.observations.lock().unwrap()[0]
                .request
                .docker_endpoint,
            Some(endpoint)
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn docker_mentions_and_opaque_wrappers_request_no_daemon_access() {
        // Arrange
        let socket_root = TempDir::new().unwrap();
        let socket_path = socket_root.path().join("docker.sock");
        let _socket = UnixListener::bind(&socket_path).unwrap();
        let endpoint = DockerEndpoint::validate(socket_path).unwrap();
        let result = CommandResult {
            output: CapturedOutput::complete(Vec::new()),
            termination: CommandTermination::Exited { code: 0 },
        };
        let (_root, tool, _executor, _events) =
            shell_tool_with_docker(Ok(result), Some(endpoint.into()));

        for command in [
            "echo docker; true",
            "grep docker README.md",
            "/usr/bin/docker ps",
            "env docker ps",
            "make docker-test",
        ] {
            // Act
            let invocation = tool
                .prepare(serde_json::json!({ "command": command }))
                .await
                .unwrap();

            // Assert
            assert!(invocation.capability_request().is_none(), "{command}");
        }
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
