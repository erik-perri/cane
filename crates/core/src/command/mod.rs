mod classify;
mod deadline;
mod diagnostics;
mod docker;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
mod linux_executor;
mod output;
mod policy;

pub(crate) use classify::invokes_direct_executable;
pub use classify::{CommandClassification, SimpleCommand, classify_command};
pub use deadline::CommandDeadline;
pub use diagnostics::{
    DiagnosticImportance, DiagnosticStatus, SandboxDiagnosticFinding, SandboxDiagnosticInput,
    SandboxDiagnosticReport, diagnose_sandbox,
};
pub use docker::{
    DockerEndpoint, DockerEndpointError, DockerExecutable, DockerExecutableError,
    DockerExecutableName, DockerIntegration,
};
#[cfg(target_os = "linux")]
pub use linux::{
    BubblewrapInstallation, BubblewrapResolutionError, LinuxSandboxOperation, LinuxSandboxPlan,
    compile_linux_sandbox_plan, resolve_bubblewrap,
};
#[cfg(target_os = "linux")]
pub use linux_executor::{BubblewrapExecutor, UnsafeExecutor, establish_bubblewrap_executor};
pub use output::{
    CapturedOutput, CommandOutputChunk, CommandOutputStream, CommandResult, CommandTermination,
    MAX_COMMAND_RESULT_BYTES, format_command_result,
};
pub use policy::{
    CommandSandboxPolicy, CommandSandboxPolicyConfig, CommandSandboxPolicyError,
    SandboxFilesystemAccess, SandboxPathGrant, SandboxPathPurpose, build_command_sandbox_policy,
};

use crate::Workspace;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const BASH_PATH: &str = "/bin/bash";
const DEFAULT_TIMEOUT_SECONDS: u64 = 10 * 60;
const MAX_TIMEOUT_SECONDS: u64 = 30 * 60;
const MIN_TIMEOUT_SECONDS: u64 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandEnvironmentConfig {
    home: String,
    path: Vec<String>,
    temp_directory: String,
}

impl CommandEnvironmentConfig {
    pub fn new(
        home: impl Into<String>,
        path: Vec<String>,
        temp_directory: impl Into<String>,
    ) -> Result<Self, ShellPreparationError> {
        let config = Self {
            home: home.into(),
            path,
            temp_directory: temp_directory.into(),
        };

        config.validate()?;

        Ok(config)
    }

    pub fn environment(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("GIT_OPTIONAL_LOCKS".to_string(), "0".to_string()),
            ("HOME".to_string(), self.home.clone()),
            ("LANG".to_string(), "C.UTF-8".to_string()),
            ("LC_ALL".to_string(), "C.UTF-8".to_string()),
            ("NO_COLOR".to_string(), "1".to_string()),
            ("PATH".to_string(), self.path.join(":")),
            ("SHELL".to_string(), BASH_PATH.to_string()),
            ("TERM".to_string(), "dumb".to_string()),
            ("TMPDIR".to_string(), self.temp_directory.clone()),
        ])
    }

    fn validate(&self) -> Result<(), ShellPreparationError> {
        validate_environment_value("home", &self.home)?;
        validate_environment_value("temporary directory", &self.temp_directory)?;

        if self.path.is_empty() {
            return Err(ShellPreparationError::Environment {
                detail: "PATH must contain at least one directory".to_string(),
            });
        }

        for entry in &self.path {
            validate_environment_value("PATH entry", entry)?;

            if entry.contains(':') {
                return Err(ShellPreparationError::Environment {
                    detail: format!("PATH entry `{entry}` must not contain `:`"),
                });
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandRequest {
    pub arguments: Vec<String>,
    pub docker_endpoint: Option<DockerEndpoint>,
    pub docker_executables: Vec<DockerExecutable>,
    pub environment: BTreeMap<String, String>,
    pub executable: String,
    pub workdir: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedShellCommand {
    classification: CommandClassification,
    command: String,
    request: CommandRequest,
    timeout: Duration,
}

impl PreparedShellCommand {
    pub(crate) fn authorize_docker(&mut self, integration: DockerIntegration) {
        self.request.docker_endpoint = Some(integration.endpoint().clone());
        self.request.docker_executables = integration.executables().to_vec();
    }

    pub fn classification(&self) -> &CommandClassification {
        &self.classification
    }

    pub fn command(&self) -> &str {
        &self.command
    }

    pub fn into_request(self) -> CommandRequest {
        self.request
    }

    pub fn request(&self) -> &CommandRequest {
        &self.request
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellInput {
    command: String,
    #[serde(default)]
    timeout_seconds: Option<u64>,
    #[serde(default)]
    workdir: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandExecutorDescriptor {
    pub backend: String,
    pub mode: CommandExecutionMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandExecutionMode {
    Sandboxed,
    Unsafe,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CommandExecutionError {
    #[error("command execution was cancelled")]
    Cancelled,

    #[error("command executor failed: {message}")]
    Failed { message: String },
}

/// Best-effort live command output.
///
/// Executors must use nonblocking sends so a slow frontend cannot prevent child
/// output pipes from being drained. The returned `CommandResult` remains the
/// authoritative bounded output.
pub type CommandOutputSender = mpsc::Sender<CommandOutputChunk>;

#[async_trait]
pub trait CommandExecutor: Send + Sync {
    fn descriptor(&self) -> CommandExecutorDescriptor;

    async fn execute(
        &self,
        cancel: CancellationToken,
        deadline: CommandDeadline,
        output: CommandOutputSender,
        request: CommandRequest,
    ) -> Result<CommandResult, CommandExecutionError>;
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ShellPreparationError {
    #[error("invalid command environment: {detail}")]
    Environment { detail: String },

    #[error("invalid shell input: {detail}")]
    Input { detail: String },
}

pub fn prepare_shell_command(
    environment: &CommandEnvironmentConfig,
    input: Value,
    workspace: &Workspace,
) -> Result<PreparedShellCommand, ShellPreparationError> {
    let input: ShellInput =
        serde_json::from_value(input).map_err(|error| ShellPreparationError::Input {
            detail: error.to_string(),
        })?;

    if input.command.trim().is_empty() {
        return Err(ShellPreparationError::Input {
            detail: "command must not be empty".to_string(),
        });
    }

    let timeout_seconds = input.timeout_seconds.unwrap_or(DEFAULT_TIMEOUT_SECONDS);
    if !(MIN_TIMEOUT_SECONDS..=MAX_TIMEOUT_SECONDS).contains(&timeout_seconds) {
        return Err(ShellPreparationError::Input {
            detail: format!(
                "timeout_seconds must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}"
            ),
        });
    }

    let workdir = resolve_workdir(input.workdir.as_deref(), workspace)?;
    let command = input.command;
    let classification = classify_command(&command);

    let request = CommandRequest {
        arguments: vec![
            "--noprofile".to_string(),
            "--norc".to_string(),
            "-c".to_string(),
            command.clone(),
        ],
        docker_endpoint: None,
        docker_executables: Vec::new(),
        environment: environment.environment(),
        executable: BASH_PATH.to_string(),
        workdir,
    };

    Ok(PreparedShellCommand {
        classification,
        command,
        request,
        timeout: Duration::from_secs(timeout_seconds),
    })
}

fn resolve_workdir(
    workdir: Option<&str>,
    workspace: &Workspace,
) -> Result<PathBuf, ShellPreparationError> {
    let Some(workdir) = workdir else {
        return Ok(workspace.root().to_path_buf());
    };

    if Path::new(workdir).is_absolute() {
        return Err(ShellPreparationError::Input {
            detail: "workdir must be relative to the workspace".to_string(),
        });
    }

    let resolved = workspace
        .resolve(workdir)
        .map_err(|detail| ShellPreparationError::Input {
            detail: format!("invalid workdir: {detail}"),
        })?;

    if !resolved.is_dir() {
        return Err(ShellPreparationError::Input {
            detail: format!("workdir `{workdir}` is not a directory"),
        });
    }

    Ok(resolved)
}

fn validate_environment_value(name: &str, value: &str) -> Result<(), ShellPreparationError> {
    if value.is_empty() {
        return Err(ShellPreparationError::Environment {
            detail: format!("{name} must not be empty"),
        });
    }

    if value.contains('\0') {
        return Err(ShellPreparationError::Environment {
            detail: format!("{name} must not contain a NUL byte"),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::TempDir;

    fn environment() -> CommandEnvironmentConfig {
        CommandEnvironmentConfig::new(
            "/sandbox/home",
            vec!["/usr/local/bin".to_string(), "/usr/bin".to_string()],
            "/sandbox/tmp",
        )
        .unwrap()
    }

    fn workspace() -> (TempDir, Workspace) {
        let root = TempDir::new().unwrap();
        let workspace = Workspace::new(root.path().to_path_buf()).unwrap();

        (root, workspace)
    }

    #[test]
    fn prepare_shell_command_preserves_the_command_and_builds_fixed_bash_arguments() {
        // Arrange
        let (_root, workspace) = workspace();
        let command = "printf '  unchanged  ' | sed 's/x/y/'";
        let input = serde_json::json!({ "command": command });

        // Act
        let prepared = prepare_shell_command(&environment(), input, &workspace).unwrap();

        // Assert
        assert_eq!(prepared.command(), command);
        assert_eq!(prepared.request().executable, "/bin/bash");
        assert_eq!(
            prepared.request().arguments,
            ["--noprofile", "--norc", "-c", command]
        );
        assert_eq!(prepared.request().docker_endpoint, None);
        assert_eq!(prepared.request().workdir, workspace.root());
        assert_eq!(prepared.classification(), &CommandClassification::Complex);
    }

    #[test]
    fn prepare_shell_command_constructs_only_the_declared_environment() {
        // Arrange
        let (_root, workspace) = workspace();
        let input = serde_json::json!({ "command": "cargo test" });

        // Act
        let prepared = prepare_shell_command(&environment(), input, &workspace).unwrap();

        // Assert
        assert_eq!(
            prepared.request().environment,
            BTreeMap::from([
                ("GIT_OPTIONAL_LOCKS".to_string(), "0".to_string()),
                ("HOME".to_string(), "/sandbox/home".to_string()),
                ("LANG".to_string(), "C.UTF-8".to_string()),
                ("LC_ALL".to_string(), "C.UTF-8".to_string()),
                ("NO_COLOR".to_string(), "1".to_string()),
                ("PATH".to_string(), "/usr/local/bin:/usr/bin".to_string()),
                ("SHELL".to_string(), "/bin/bash".to_string()),
                ("TERM".to_string(), "dumb".to_string()),
                ("TMPDIR".to_string(), "/sandbox/tmp".to_string()),
            ])
        );
        assert!(!prepared.request().environment.contains_key("BASH_ENV"));
        assert!(
            !prepared
                .request()
                .environment
                .contains_key("OPENAI_API_KEY")
        );
        assert!(!prepared.request().environment.contains_key("SSH_AUTH_SOCK"));
    }

    #[test]
    fn command_environment_rejects_values_that_cannot_form_the_baseline() {
        // Arrange
        let configurations = [
            CommandEnvironmentConfig::new("", vec!["/usr/bin".to_string()], "/tmp"),
            CommandEnvironmentConfig::new("/home", Vec::new(), "/tmp"),
            CommandEnvironmentConfig::new(
                "/home",
                vec!["/usr/bin:/workspace/bin".to_string()],
                "/tmp",
            ),
            CommandEnvironmentConfig::new("/home", vec!["/usr/bin\0hidden".to_string()], "/tmp"),
            CommandEnvironmentConfig::new("/home", vec!["/usr/bin".to_string()], "/tmp\0hidden"),
        ];

        // Act
        let errors = configurations.map(Result::unwrap_err);

        // Assert
        assert_eq!(
            errors.map(|error| matches!(error, ShellPreparationError::Environment { .. })),
            [true; 5]
        );
    }

    #[test]
    fn prepare_shell_command_uses_the_documented_default_and_timeout_bounds() {
        // Arrange
        let (_root, workspace) = workspace();
        let inputs = [
            serde_json::json!({ "command": "true" }),
            serde_json::json!({ "command": "true", "timeout_seconds": 1 }),
            serde_json::json!({ "command": "true", "timeout_seconds": 1_800 }),
        ];

        // Act
        let timeouts = inputs.map(|input| {
            prepare_shell_command(&environment(), input, &workspace)
                .unwrap()
                .timeout()
        });

        // Assert
        assert_eq!(
            timeouts,
            [
                Duration::from_secs(600),
                Duration::from_secs(1),
                Duration::from_secs(1_800),
            ]
        );
    }

    #[test]
    fn prepare_shell_command_rejects_timeouts_outside_the_documented_bounds() {
        // Arrange
        let (_root, workspace) = workspace();
        let inputs = [
            serde_json::json!({ "command": "true", "timeout_seconds": 0 }),
            serde_json::json!({ "command": "true", "timeout_seconds": 1_801 }),
        ];

        // Act
        let errors = inputs
            .map(|input| prepare_shell_command(&environment(), input, &workspace).unwrap_err());

        // Assert
        assert!(errors.iter().all(|error| matches!(
            error,
            ShellPreparationError::Input { detail }
                if detail == "timeout_seconds must be between 1 and 1800"
        )));
    }

    #[test]
    fn prepare_shell_command_resolves_a_workspace_relative_workdir() {
        // Arrange
        let (root, workspace) = workspace();
        let child = root.path().join("crates").join("core");
        fs::create_dir_all(&child).unwrap();
        let input = serde_json::json!({
            "command": "cargo test",
            "workdir": "crates/core",
        });

        // Act
        let prepared = prepare_shell_command(&environment(), input, &workspace).unwrap();

        // Assert
        assert_eq!(
            prepared.request().workdir,
            dunce::canonicalize(child).unwrap()
        );
    }

    #[test]
    fn prepare_shell_command_rejects_invalid_workdirs() {
        // Arrange
        let (root, workspace) = workspace();
        let file = root.path().join("Cargo.toml");
        fs::write(&file, "[package]").unwrap();
        let inputs = [
            serde_json::json!({ "command": "pwd", "workdir": file }),
            serde_json::json!({ "command": "pwd", "workdir": "../outside" }),
            serde_json::json!({ "command": "pwd", "workdir": "missing" }),
            serde_json::json!({ "command": "pwd", "workdir": "Cargo.toml" }),
            serde_json::json!({ "command": "pwd", "workdir": "" }),
        ];

        // Act
        let errors = inputs
            .map(|input| prepare_shell_command(&environment(), input, &workspace).unwrap_err());

        // Assert
        assert!(
            errors
                .iter()
                .all(|error| matches!(error, ShellPreparationError::Input { .. }))
        );
    }

    #[test]
    fn prepare_shell_command_rejects_invalid_input_shapes_and_empty_commands() {
        // Arrange
        let inputs = [
            serde_json::json!({}),
            serde_json::json!({ "command": 42 }),
            serde_json::json!({ "command": "true", "unknown": true }),
            serde_json::json!({ "command": " \n\t " }),
        ];
        let (_root, workspace) = workspace();

        // Act
        let errors = inputs
            .map(|input| prepare_shell_command(&environment(), input, &workspace).unwrap_err());

        // Assert
        assert!(
            errors
                .iter()
                .all(|error| matches!(error, ShellPreparationError::Input { .. }))
        );
    }

    #[derive(Debug, Eq, PartialEq)]
    struct ExecutionObservation {
        cancelled: bool,
        deadline: tokio::time::Instant,
        request: CommandRequest,
    }

    struct FakeExecutor {
        observations: Mutex<Vec<ExecutionObservation>>,
        result: CommandResult,
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
            deadline: CommandDeadline,
            output: CommandOutputSender,
            request: CommandRequest,
        ) -> Result<CommandResult, CommandExecutionError> {
            self.observations
                .lock()
                .unwrap()
                .push(ExecutionObservation {
                    cancelled: cancel.is_cancelled(),
                    deadline: deadline.at(),
                    request,
                });

            for chunk in &self.result.output.chunks {
                let _ = output.try_send(chunk.clone());
            }

            Ok(self.result.clone())
        }
    }

    #[tokio::test]
    async fn command_executor_contract_streams_and_returns_ordered_output_without_a_process_backend()
     {
        // Arrange
        let (_root, workspace) = workspace();
        let prepared = prepare_shell_command(
            &environment(),
            serde_json::json!({ "command": "cargo test" }),
            &workspace,
        )
        .unwrap();
        let expected_request = prepared.request().clone();
        let cancel = CancellationToken::new();
        let deadline = CommandDeadline::after(prepared.timeout());
        let expected_deadline = deadline.at();
        let (output_tx, mut output_rx) = mpsc::channel(4);
        let expected_output = vec![
            CommandOutputChunk::stdout(b"fake stdout\n".to_vec()),
            CommandOutputChunk::stderr(b"fake stderr\n".to_vec()),
        ];
        let executor = FakeExecutor {
            observations: Mutex::new(Vec::new()),
            result: CommandResult {
                output: CapturedOutput::complete(expected_output.clone()),
                termination: CommandTermination::Exited { code: 0 },
            },
        };

        // Act
        let descriptor = executor.descriptor();
        let result = executor
            .execute(cancel, deadline, output_tx, prepared.into_request())
            .await
            .unwrap();
        let mut observed_output = Vec::new();
        while let Some(chunk) = output_rx.recv().await {
            observed_output.push(chunk);
        }

        // Assert
        assert_eq!(
            descriptor,
            CommandExecutorDescriptor {
                backend: "fake".to_string(),
                mode: CommandExecutionMode::Sandboxed,
            }
        );
        assert_eq!(observed_output, expected_output);
        assert_eq!(result.output.chunks, expected_output);
        assert_eq!(
            *executor.observations.lock().unwrap(),
            [ExecutionObservation {
                cancelled: false,
                deadline: expected_deadline,
                request: expected_request,
            }]
        );
    }
}
