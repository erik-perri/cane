use super::{
    CapturedOutput, CommandDeadline, CommandExecutionError, CommandExecutionMode, CommandExecutor,
    CommandExecutorDescriptor, CommandOutputChunk, CommandOutputSender, CommandOutputStream,
    CommandRequest, CommandResult, CommandTermination, DockerExecutableName, LinuxSandboxPlan,
    MAX_COMMAND_RESULT_BYTES, SandboxDiagnosticInput, SandboxDiagnosticReport,
    compile_linux_sandbox_plan, diagnose_sandbox,
};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::io;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio::time::Sleep;
use tokio_util::sync::CancellationToken;

const OUTPUT_READ_BYTES: usize = 8 * 1024;
const PIPE_EVENT_CAPACITY: usize = 16;
const PRIVATE_DOCKER_BIN: &str = "/tmp/cane-docker-bin";
const PRIVATE_DOCKER_CONFIG: &str = "/tmp/cane-docker-config";
const PRIVATE_DOCKER_PLUGIN_DIR: &str = "/tmp/cane-docker-config/cli-plugins";
const PRIVATE_DOCKER_HOST: &str = "unix:///tmp/cane-docker.sock";
const PRIVATE_DOCKER_SOCKET: &str = "/tmp/cane-docker.sock";
const TERMINATION_GRACE: Duration = Duration::from_secs(2);

pub struct BubblewrapExecutor {
    launcher: PathBuf,
    plan: LinuxSandboxPlan,
    workspace: PathBuf,
}

pub struct UnsafeExecutor {
    workspace: PathBuf,
}

impl UnsafeExecutor {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }

    fn prepare_process(
        &self,
        mut request: CommandRequest,
    ) -> Result<Command, CommandExecutionError> {
        validate_request(&request, &self.workspace)?;
        expose_host_docker_endpoint(&mut request);

        let mut process = Command::new(request.executable);
        process
            .args(request.arguments)
            .env_clear()
            .envs(request.environment)
            .current_dir(request.workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .process_group(0);

        Ok(process)
    }
}

impl BubblewrapExecutor {
    fn from_diagnostics(report: &SandboxDiagnosticReport) -> Result<Self, CommandExecutionError> {
        if report.platform != "linux" || report.backend != "bubblewrap" || !report.is_supported() {
            return Err(execution_failure_message(
                "Bubblewrap executor requires successful Linux sandbox diagnostics",
            ));
        }
        let launcher = report.bubblewrap_path.clone().ok_or_else(|| {
            execution_failure_message("sandbox diagnostics did not resolve Bubblewrap")
        })?;
        let policy = report.policy.as_ref().ok_or_else(|| {
            execution_failure_message("sandbox diagnostics did not produce a command policy")
        })?;

        Ok(Self {
            launcher,
            plan: compile_linux_sandbox_plan(policy),
            workspace: policy.workspace().to_path_buf(),
        })
    }

    #[cfg(test)]
    fn with_launcher(launcher: PathBuf, policy: &super::CommandSandboxPolicy) -> Self {
        Self {
            launcher,
            plan: compile_linux_sandbox_plan(policy),
            workspace: policy.workspace().to_path_buf(),
        }
    }

    fn prepare_process(&self, request: CommandRequest) -> Result<Command, CommandExecutionError> {
        validate_request(&request, &self.workspace)?;

        let mut process = Command::new(&self.launcher);
        process.args(self.plan.arguments());
        if let Some(endpoint) = &request.docker_endpoint {
            process
                .arg("--bind")
                .arg(endpoint.path())
                .arg(PRIVATE_DOCKER_SOCKET);
        }
        if !request.docker_executables.is_empty() {
            process
                .arg("--dir")
                .arg(PRIVATE_DOCKER_BIN)
                .arg("--dir")
                .arg(PRIVATE_DOCKER_CONFIG);
            for executable in &request.docker_executables {
                process
                    .arg("--ro-bind")
                    .arg(executable.path())
                    .arg(Path::new(PRIVATE_DOCKER_BIN).join(executable.name().as_str()));
            }
            if let Some(compose) = request
                .docker_executables
                .iter()
                .find(|executable| executable.name() == DockerExecutableName::DockerCompose)
            {
                process
                    .arg("--dir")
                    .arg(PRIVATE_DOCKER_PLUGIN_DIR)
                    .arg("--ro-bind")
                    .arg(compose.path())
                    .arg(
                        Path::new(PRIVATE_DOCKER_PLUGIN_DIR)
                            .join(DockerExecutableName::DockerCompose.as_str()),
                    );
            }
        }
        process.arg("--chdir").arg(&request.workdir);

        let private_docker_path = (!request.docker_executables.is_empty()).then(|| {
            request.environment.get("PATH").map_or_else(
                || PRIVATE_DOCKER_BIN.to_string(),
                |path| format!("{PRIVATE_DOCKER_BIN}:{path}"),
            )
        });
        for (name, value) in request.environment {
            process.arg("--setenv").arg(name).arg(value);
        }
        if request.docker_endpoint.is_some() {
            process
                .arg("--setenv")
                .arg("DOCKER_HOST")
                .arg(PRIVATE_DOCKER_HOST);
        }
        if let Some(path) = private_docker_path {
            process
                .arg("--setenv")
                .arg("PATH")
                .arg(path)
                .arg("--setenv")
                .arg("DOCKER_CONFIG")
                .arg(PRIVATE_DOCKER_CONFIG);
        }
        process
            .arg("--")
            .arg(request.executable)
            .args(request.arguments)
            .env_clear()
            .current_dir("/")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .process_group(0);

        Ok(process)
    }
}

fn expose_host_docker_endpoint(request: &mut CommandRequest) {
    if let Some(endpoint) = &request.docker_endpoint {
        request
            .environment
            .insert("DOCKER_HOST".to_string(), endpoint.resource().to_string());
    }
}

pub fn establish_bubblewrap_executor(
    input: SandboxDiagnosticInput,
) -> Result<(BubblewrapExecutor, SandboxDiagnosticReport), Box<SandboxDiagnosticReport>> {
    let report = diagnose_sandbox(input);
    match BubblewrapExecutor::from_diagnostics(&report) {
        Ok(executor) => Ok((executor, report)),
        Err(_) => Err(Box::new(report)),
    }
}

#[async_trait]
impl CommandExecutor for BubblewrapExecutor {
    fn descriptor(&self) -> CommandExecutorDescriptor {
        CommandExecutorDescriptor {
            backend: "bubblewrap".to_string(),
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
        let mut process = self.prepare_process(request)?;
        let child = process
            .spawn()
            .map_err(|error| execution_failure("could not spawn Bubblewrap", error))?;

        supervise_process(child, cancel, deadline, output).await
    }
}

#[async_trait]
impl CommandExecutor for UnsafeExecutor {
    fn descriptor(&self) -> CommandExecutorDescriptor {
        CommandExecutorDescriptor {
            backend: "host".to_string(),
            mode: CommandExecutionMode::Unsafe,
        }
    }

    async fn execute(
        &self,
        cancel: CancellationToken,
        deadline: CommandDeadline,
        output: CommandOutputSender,
        request: CommandRequest,
    ) -> Result<CommandResult, CommandExecutionError> {
        let mut process = self.prepare_process(request)?;
        let child = process
            .spawn()
            .map_err(|error| execution_failure("could not spawn host command", error))?;

        supervise_process(child, cancel, deadline, output).await
    }
}

fn validate_request(
    request: &CommandRequest,
    workspace: &Path,
) -> Result<(), CommandExecutionError> {
    if !Path::new(&request.executable).is_absolute() {
        return Err(execution_failure_message(
            "prepared executable must be an absolute path",
        ));
    }
    validate_os_argument("executable", &request.executable)?;
    for argument in &request.arguments {
        validate_os_argument("argument", argument)?;
    }
    for (name, value) in &request.environment {
        if name.is_empty() || name.contains('=') || name.contains('\0') {
            return Err(execution_failure_message(
                "environment variable names must be nonempty and contain neither `=` nor NUL",
            ));
        }
        validate_os_argument("environment value", value)?;
    }
    if request.docker_endpoint.is_none() && !request.docker_executables.is_empty() {
        return Err(execution_failure_message(
            "Docker executables require an authorized Docker endpoint",
        ));
    }

    let workdir = std::fs::canonicalize(&request.workdir)
        .map_err(|error| execution_failure("could not resolve command workdir", error))?;
    if !workdir.is_dir() || !workdir.starts_with(workspace) {
        return Err(execution_failure_message(
            "prepared workdir must be a directory inside the workspace",
        ));
    }

    Ok(())
}

fn validate_os_argument(name: &str, value: &str) -> Result<(), CommandExecutionError> {
    if value.contains('\0') {
        return Err(execution_failure_message(format!(
            "{name} must not contain NUL"
        )));
    }
    Ok(())
}

async fn supervise_process(
    mut child: Child,
    cancel: CancellationToken,
    deadline: CommandDeadline,
    preview: CommandOutputSender,
) -> Result<CommandResult, CommandExecutionError> {
    let process_id = child
        .id()
        .and_then(|id| i32::try_from(id).ok())
        .ok_or_else(|| execution_failure_message("command did not report a valid process ID"))?;
    let mut process_guard = ProcessGroupGuard::new(process_id);
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| execution_failure_message("command stdout was not piped"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| execution_failure_message("command stderr was not piped"))?;
    let (events_tx, mut events_rx) = mpsc::channel(PIPE_EVENT_CAPACITY);
    tokio::spawn(read_pipe(
        stdout,
        CommandOutputStream::Stdout,
        events_tx.clone(),
    ));
    tokio::spawn(read_pipe(stderr, CommandOutputStream::Stderr, events_tx));

    let mut capture = TailCapture::default();
    let mut status = None;
    let mut open_pipes = 2_u8;
    let mut requested_outcome = None;
    let mut force_kill = None;
    let mut failure = None;

    while status.is_none() || open_pipes != 0 {
        tokio::select! {
            biased;

            _ = cancel.cancelled(), if requested_outcome.is_none() => {
                requested_outcome = Some(RequestedOutcome::Cancelled);
                begin_termination(process_id, &mut force_kill, &mut failure);
            }

            _ = deadline.elapsed(), if requested_outcome.is_none() => {
                requested_outcome = Some(RequestedOutcome::TimedOut);
                begin_termination(process_id, &mut force_kill, &mut failure);
            }

            _ = wait_for_force_kill(&mut force_kill), if force_kill.is_some() => {
                if let Err(error) = signal_process_group(process_id, libc::SIGKILL) {
                    record_failure(&mut failure, "could not kill command process group", error);
                }
                force_kill = None;
            }

            wait_result = child.wait(), if status.is_none() => {
                match wait_result {
                    Ok(exit_status) => status = Some(exit_status),
                    Err(error) => {
                        record_failure(&mut failure, "could not wait for command", error);
                        begin_termination(process_id, &mut force_kill, &mut failure);
                    }
                }
            }

            event = events_rx.recv(), if open_pipes != 0 => {
                match event {
                    Some(PipeEvent::Chunk(chunk)) => {
                        let _ = preview.try_send(chunk.clone());
                        capture.push(chunk);
                    }
                    Some(PipeEvent::Closed) => open_pipes = open_pipes.saturating_sub(1),
                    Some(PipeEvent::Failed(error)) => {
                        open_pipes = open_pipes.saturating_sub(1);
                        record_failure(&mut failure, "could not read command output", error);
                        begin_termination(process_id, &mut force_kill, &mut failure);
                    }
                    None => open_pipes = 0,
                }
            }
        }
    }

    process_guard.disarm();

    if requested_outcome == Some(RequestedOutcome::Cancelled) {
        return Err(CommandExecutionError::Cancelled);
    }
    if let Some(message) = failure {
        return Err(CommandExecutionError::Failed { message });
    }

    let output = capture.finish();
    if requested_outcome == Some(RequestedOutcome::TimedOut) {
        return Ok(CommandResult {
            output,
            termination: CommandTermination::TimedOut,
        });
    }

    let status = status.expect("supervision loop waits for child status");
    let termination = if let Some(code) = status.code() {
        CommandTermination::Exited { code }
    } else {
        CommandTermination::Signaled {
            signal: status.signal().unwrap_or_default(),
        }
    };

    Ok(CommandResult {
        output,
        termination,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestedOutcome {
    Cancelled,
    TimedOut,
}

fn begin_termination(
    process_id: i32,
    force_kill: &mut Option<Pin<Box<Sleep>>>,
    failure: &mut Option<String>,
) {
    if force_kill.is_some() {
        return;
    }
    if let Err(error) = signal_process_group(process_id, libc::SIGTERM) {
        record_failure(failure, "could not terminate command process group", error);
    }
    *force_kill = Some(Box::pin(tokio::time::sleep(TERMINATION_GRACE)));
}

async fn wait_for_force_kill(force_kill: &mut Option<Pin<Box<Sleep>>>) {
    match force_kill {
        Some(sleep) => sleep.as_mut().await,
        None => std::future::pending().await,
    }
}

fn signal_process_group(process_id: i32, signal: i32) -> io::Result<()> {
    // SAFETY: `kill` is called with a negated, validated child PID to target only
    // the private process group created for this invocation.
    let result = unsafe { libc::kill(-process_id, signal) };
    if result == 0 {
        return Ok(());
    }

    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error)
    }
}

struct ProcessGroupGuard {
    armed: bool,
    process_id: i32,
}

impl ProcessGroupGuard {
    fn new(process_id: i32) -> Self {
        Self {
            armed: true,
            process_id,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = signal_process_group(self.process_id, libc::SIGKILL);
        }
    }
}

enum PipeEvent {
    Chunk(CommandOutputChunk),
    Closed,
    Failed(io::Error),
}

async fn read_pipe(
    mut pipe: impl AsyncRead + Unpin,
    stream: CommandOutputStream,
    events: mpsc::Sender<PipeEvent>,
) {
    let mut buffer = vec![0_u8; OUTPUT_READ_BYTES];
    loop {
        match pipe.read(&mut buffer).await {
            Ok(0) => {
                let _ = events.send(PipeEvent::Closed).await;
                return;
            }
            Ok(bytes) => {
                let chunk = CommandOutputChunk {
                    bytes: buffer[..bytes].to_vec(),
                    stream,
                };
                if events.send(PipeEvent::Chunk(chunk)).await.is_err() {
                    return;
                }
            }
            Err(error) => {
                let _ = events.send(PipeEvent::Failed(error)).await;
                return;
            }
        }
    }
}

#[derive(Default)]
struct TailCapture {
    chunks: VecDeque<CommandOutputChunk>,
    retained_bytes: usize,
    stderr_bytes: u64,
    stdout_bytes: u64,
}

impl TailCapture {
    fn push(&mut self, chunk: CommandOutputChunk) {
        let observed_bytes = u64::try_from(chunk.bytes.len()).unwrap_or(u64::MAX);
        match chunk.stream {
            CommandOutputStream::Stderr => {
                self.stderr_bytes = self.stderr_bytes.saturating_add(observed_bytes);
            }
            CommandOutputStream::Stdout => {
                self.stdout_bytes = self.stdout_bytes.saturating_add(observed_bytes);
            }
        }

        self.retained_bytes = self.retained_bytes.saturating_add(chunk.bytes.len());
        self.chunks.push_back(chunk);
        self.truncate_front();
    }

    fn truncate_front(&mut self) {
        while self.retained_bytes > MAX_COMMAND_RESULT_BYTES {
            let remove = self.retained_bytes - MAX_COMMAND_RESULT_BYTES;
            let Some(front) = self.chunks.front_mut() else {
                self.retained_bytes = 0;
                return;
            };
            if remove >= front.bytes.len() {
                self.retained_bytes -= front.bytes.len();
                self.chunks.pop_front();
            } else {
                front.bytes.drain(..remove);
                self.retained_bytes -= remove;
            }
        }
    }

    fn finish(self) -> CapturedOutput {
        CapturedOutput {
            chunks: self.chunks.into(),
            stderr_bytes: self.stderr_bytes,
            stdout_bytes: self.stdout_bytes,
        }
    }
}

fn record_failure(failure: &mut Option<String>, context: &str, error: io::Error) {
    if failure.is_none() {
        *failure = Some(format!("{context}: {error}"));
    }
}

fn execution_failure(context: &str, error: impl std::fmt::Display) -> CommandExecutionError {
    execution_failure_message(format!("{context}: {error}"))
}

fn execution_failure_message(message: impl Into<String>) -> CommandExecutionError {
    CommandExecutionError::Failed {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{
        CommandSandboxPolicyConfig, DockerEndpoint, DockerExecutable, DockerExecutableName,
        build_command_sandbox_policy,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::net::TcpListener;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::fs::symlink;
    use std::os::unix::net::UnixListener;
    use std::str::FromStr;
    use tempfile::TempDir;

    struct ExecutorFixture {
        _root: TempDir,
        executor: BubblewrapExecutor,
        workspace: PathBuf,
    }

    struct SharedMemorySegment(i32);

    impl SharedMemorySegment {
        fn create() -> Self {
            // SAFETY: `shmget` creates a new, private one-byte test segment. The
            // returned identifier is owned by this guard and removed on drop.
            let identifier = unsafe { libc::shmget(libc::IPC_PRIVATE, 1, libc::IPC_CREAT | 0o600) };
            assert_ne!(
                identifier,
                -1,
                "could not create host shared-memory fixture: {}",
                io::Error::last_os_error()
            );
            Self(identifier)
        }
    }

    impl Drop for SharedMemorySegment {
        fn drop(&mut self) {
            // SAFETY: this removes the test segment owned by the guard. A null
            // buffer is valid for `IPC_RMID`.
            let _ = unsafe { libc::shmctl(self.0, libc::IPC_RMID, std::ptr::null_mut()) };
        }
    }

    fn fixture() -> ExecutorFixture {
        let root = TempDir::new().unwrap();
        let workspace = root.path().join("workspace");
        let runtime = root.path().join("runtime");
        fs::create_dir_all(&workspace).unwrap();
        fs::create_dir_all(&runtime).unwrap();
        let launcher = root.path().join("fake-bwrap");
        fs::write(
            &launcher,
            "#!/bin/sh\n\
             while [ \"$#\" -gt 0 ]; do\n\
               if [ \"$1\" = \"--chdir\" ]; then cd \"$2\" || exit 125; shift 2\n\
               elif [ \"$1\" = \"--setenv\" ]; then export \"$2=$3\"; shift 3\n\
               elif [ \"$1\" = \"--\" ]; then shift; exec \"$@\"\n\
               else shift\n\
               fi\n\
             done\n\
             exit 125\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&launcher).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&launcher, permissions).unwrap();
        let policy = build_command_sandbox_policy(CommandSandboxPolicyConfig {
            git_metadata_roots: Vec::new(),
            inherited_path: vec![runtime.clone()],
            pass_through_roots: Vec::new(),
            private_home: PathBuf::from("/home/cane"),
            private_temp: PathBuf::from("/tmp"),
            runtime_roots: vec![runtime],
            toolchain_roots: Vec::new(),
            workspace: workspace.clone(),
        })
        .unwrap();
        let executor = BubblewrapExecutor::with_launcher(launcher, &policy);

        ExecutorFixture {
            _root: root,
            executor,
            workspace,
        }
    }

    fn request(workspace: &Path, command: &str) -> CommandRequest {
        CommandRequest {
            arguments: vec![
                "--noprofile".to_string(),
                "--norc".to_string(),
                "-c".to_string(),
                command.to_string(),
            ],
            docker_endpoint: None,
            docker_executables: Vec::new(),
            environment: BTreeMap::from([
                ("PATH".to_string(), "/usr/bin".to_string()),
                ("TEST_VALUE".to_string(), "explicit".to_string()),
            ]),
            executable: "/bin/bash".to_string(),
            workdir: workspace.to_path_buf(),
        }
    }

    fn process_arguments(process: &Command) -> Vec<String> {
        process
            .as_std()
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn bubblewrap_exposes_docker_only_for_a_capability_bearing_request() {
        // Arrange
        let fixture = fixture();
        let socket_path = fixture._root.path().join("docker.sock");
        let _socket = UnixListener::bind(&socket_path).unwrap();
        let endpoint = DockerEndpoint::validate(&socket_path).unwrap();
        let docker_binary = fixture._root.path().join("docker");
        fs::write(&docker_binary, "#!/bin/true\n").unwrap();
        fs::set_permissions(&docker_binary, fs::Permissions::from_mode(0o755)).unwrap();
        let executable =
            DockerExecutable::validate(DockerExecutableName::Docker, &docker_binary).unwrap();
        let compose_binary = fixture._root.path().join("docker-compose");
        fs::write(&compose_binary, "#!/bin/true\n").unwrap();
        fs::set_permissions(&compose_binary, fs::Permissions::from_mode(0o755)).unwrap();
        let compose =
            DockerExecutable::validate(DockerExecutableName::DockerCompose, &compose_binary)
                .unwrap();
        let mut docker_request = request(&fixture.workspace, "docker ps");
        docker_request.environment.insert(
            "DOCKER_HOST".to_string(),
            "unix:///host/guess.sock".to_string(),
        );
        docker_request.docker_endpoint = Some(endpoint.clone());
        docker_request.docker_executables = vec![executable.clone(), compose.clone()];
        let ordinary_request = request(&fixture.workspace, "cargo test");

        // Act
        let docker_process = fixture.executor.prepare_process(docker_request).unwrap();
        let ordinary_process = fixture.executor.prepare_process(ordinary_request).unwrap();
        let docker_arguments = process_arguments(&docker_process);
        let ordinary_arguments = process_arguments(&ordinary_process);
        let docker_host_values = docker_arguments
            .windows(3)
            .filter(|arguments| arguments[0] == "--setenv" && arguments[1] == "DOCKER_HOST")
            .map(|arguments| arguments[2].as_str())
            .collect::<Vec<_>>();
        let docker_path_values = docker_arguments
            .windows(3)
            .filter(|arguments| arguments[0] == "--setenv" && arguments[1] == "PATH")
            .map(|arguments| arguments[2].as_str())
            .collect::<Vec<_>>();
        let docker_config_values = docker_arguments
            .windows(3)
            .filter(|arguments| arguments[0] == "--setenv" && arguments[1] == "DOCKER_CONFIG")
            .map(|arguments| arguments[2].as_str())
            .collect::<Vec<_>>();

        // Assert
        assert!(docker_arguments.windows(3).any(|arguments| {
            arguments
                == [
                    "--bind",
                    endpoint.path().to_str().unwrap(),
                    PRIVATE_DOCKER_SOCKET,
                ]
        }));
        assert_eq!(
            docker_host_values,
            ["unix:///host/guess.sock", PRIVATE_DOCKER_HOST]
        );
        assert!(docker_arguments.windows(3).any(|arguments| {
            arguments
                == [
                    "--ro-bind",
                    executable.path().to_str().unwrap(),
                    "/tmp/cane-docker-bin/docker",
                ]
        }));
        assert!(docker_arguments.windows(3).any(|arguments| {
            arguments
                == [
                    "--ro-bind",
                    compose.path().to_str().unwrap(),
                    "/tmp/cane-docker-config/cli-plugins/docker-compose",
                ]
        }));
        assert_eq!(
            docker_path_values,
            ["/usr/bin", "/tmp/cane-docker-bin:/usr/bin"]
        );
        assert_eq!(docker_config_values, [PRIVATE_DOCKER_CONFIG]);
        assert!(
            !ordinary_arguments
                .iter()
                .any(|argument| argument == endpoint.path().to_str().unwrap())
        );
        assert!(
            !ordinary_arguments
                .iter()
                .any(|argument| argument == PRIVATE_DOCKER_HOST)
        );
        assert!(
            !ordinary_arguments
                .iter()
                .any(|argument| argument == PRIVATE_DOCKER_BIN)
        );
    }

    #[test]
    fn unsafe_executor_adds_docker_host_only_to_a_capability_bearing_request() {
        // Arrange
        let fixture = fixture();
        let executor = UnsafeExecutor::new(fixture.workspace.clone());
        let socket_path = fixture._root.path().join("docker.sock");
        let _socket = UnixListener::bind(&socket_path).unwrap();
        let endpoint = DockerEndpoint::validate(&socket_path).unwrap();
        let mut docker_request = request(&fixture.workspace, "docker ps");
        docker_request.docker_endpoint = Some(endpoint.clone());
        let ordinary_request = request(&fixture.workspace, "cargo test");

        // Act
        let docker_process = executor.prepare_process(docker_request).unwrap();
        let ordinary_process = executor.prepare_process(ordinary_request).unwrap();
        let docker_environment = docker_process
            .as_std()
            .get_envs()
            .collect::<BTreeMap<_, _>>();
        let ordinary_environment = ordinary_process
            .as_std()
            .get_envs()
            .collect::<BTreeMap<_, _>>();

        // Assert
        assert_eq!(
            docker_environment.get(std::ffi::OsStr::new("DOCKER_HOST")),
            Some(&Some(std::ffi::OsStr::new(endpoint.resource())))
        );
        assert!(!ordinary_environment.contains_key(std::ffi::OsStr::new("DOCKER_HOST")));
    }

    #[test]
    fn tail_capture_bounds_retained_bytes_but_counts_every_observed_stream_byte() {
        // Arrange
        let mut capture = TailCapture::default();
        let stdout = vec![b'a'; MAX_COMMAND_RESULT_BYTES];
        let stderr = vec![b'b'; 17];

        // Act
        capture.push(CommandOutputChunk::stdout(stdout));
        capture.push(CommandOutputChunk::stderr(stderr.clone()));
        let output = capture.finish();
        let retained = output
            .chunks
            .iter()
            .map(|chunk| chunk.bytes.len())
            .sum::<usize>();

        // Assert
        assert_eq!(retained, MAX_COMMAND_RESULT_BYTES);
        assert_eq!(output.stdout_bytes, MAX_COMMAND_RESULT_BYTES as u64);
        assert_eq!(output.stderr_bytes, 17);
        assert_eq!(output.chunks.last().unwrap().bytes, stderr);
    }

    #[tokio::test]
    async fn executor_drains_both_streams_and_returns_a_nonzero_exit_normally() {
        // Arrange
        let fixture = fixture();
        let request = request(
            &fixture.workspace,
            "printf 'stdout:%s' \"$TEST_VALUE\"; printf 'stderr' >&2; exit 7",
        );
        let (preview_tx, mut preview_rx) = mpsc::channel(4);

        // Act
        let result = fixture
            .executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                preview_tx,
                request,
            )
            .await
            .unwrap();
        let mut preview = Vec::new();
        while let Some(chunk) = preview_rx.recv().await {
            preview.push(chunk);
        }

        // Assert
        assert_eq!(result.termination, CommandTermination::Exited { code: 7 });
        assert_eq!(result.output.stdout_bytes, 15);
        assert_eq!(result.output.stderr_bytes, 6);
        assert!(result.output.chunks.iter().any(|chunk| {
            chunk.stream == CommandOutputStream::Stdout && chunk.bytes == b"stdout:explicit"
        }));
        assert!(result.output.chunks.iter().any(|chunk| {
            chunk.stream == CommandOutputStream::Stderr && chunk.bytes == b"stderr"
        }));
        assert!(!preview.is_empty());
    }

    #[tokio::test]
    async fn executor_reports_signal_termination_without_a_synthetic_exit_code() {
        // Arrange
        let fixture = fixture();
        let request = request(&fixture.workspace, "kill -TERM $$");
        let (preview_tx, _preview_rx) = mpsc::channel(1);

        // Act
        let result = fixture
            .executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                preview_tx,
                request,
            )
            .await
            .unwrap();

        // Assert
        assert_eq!(
            result.termination,
            CommandTermination::Signaled {
                signal: libc::SIGTERM,
            }
        );
    }

    #[tokio::test]
    async fn unsafe_executor_clears_the_ambient_environment() {
        // Arrange
        let fixture = fixture();
        let executor = UnsafeExecutor::new(fixture.workspace.clone());
        let request = request(
            &fixture.workspace,
            "printf '%s:%s' \"${AMBIENT_SECRET-unset}\" \"$TEST_VALUE\"",
        );
        let (preview_tx, _preview_rx) = mpsc::channel(1);

        // Act
        let descriptor = executor.descriptor();
        let result = executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                preview_tx,
                request,
            )
            .await
            .unwrap();
        let output = result
            .output
            .chunks
            .iter()
            .flat_map(|chunk| chunk.bytes.iter().copied())
            .collect::<Vec<_>>();

        // Assert
        assert_eq!(
            descriptor,
            CommandExecutorDescriptor {
                backend: "host".to_string(),
                mode: CommandExecutionMode::Unsafe,
            }
        );
        assert_eq!(output, b"unset:explicit");
    }

    #[tokio::test]
    async fn large_concurrent_streams_are_drained_without_preview_backpressure() {
        // Arrange
        let fixture = fixture();
        let request = request(
            &fixture.workspace,
            "/usr/bin/head -c 65536 /dev/zero & \
             /usr/bin/head -c 65536 /dev/zero >&2 & wait",
        );
        let (preview_tx, _preview_rx) = mpsc::channel(1);

        // Act
        let result = fixture
            .executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                preview_tx,
                request,
            )
            .await
            .unwrap();
        let retained = result
            .output
            .chunks
            .iter()
            .map(|chunk| chunk.bytes.len())
            .sum::<usize>();

        // Assert
        assert_eq!(result.termination, CommandTermination::Exited { code: 0 });
        assert_eq!(result.output.stdout_bytes, 65_536);
        assert_eq!(result.output.stderr_bytes, 65_536);
        assert_eq!(retained, MAX_COMMAND_RESULT_BYTES);
    }

    #[tokio::test]
    async fn executor_continues_draining_after_the_preview_receiver_is_dropped() {
        // Arrange
        let fixture = fixture();
        let request = request(
            &fixture.workspace,
            "/usr/bin/head -c 65536 /dev/zero; \
             /usr/bin/head -c 65536 /dev/zero >&2",
        );
        let (preview_tx, preview_rx) = mpsc::channel(1);
        drop(preview_rx);

        // Act
        let result = fixture
            .executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                preview_tx,
                request,
            )
            .await
            .unwrap();

        // Assert
        assert_eq!(result.termination, CommandTermination::Exited { code: 0 });
        assert_eq!(result.output.stdout_bytes, 65_536);
        assert_eq!(result.output.stderr_bytes, 65_536);
    }

    #[tokio::test]
    async fn resetting_the_deadline_extends_a_running_execution() {
        // Arrange
        let fixture = fixture();
        let request = request(
            &fixture.workspace,
            "printf 'started'; /bin/sleep 1.2; printf 'completed'",
        );
        let deadline = CommandDeadline::after(Duration::from_secs(5));
        let extension = deadline.clone();
        let (preview_tx, mut preview_rx) = mpsc::channel(1);

        // Act
        let execution =
            fixture
                .executor
                .execute(CancellationToken::new(), deadline, preview_tx, request);
        let extend = async move {
            let started = tokio::time::timeout(Duration::from_secs(5), preview_rx.recv())
                .await
                .expect("command did not start before the test timeout")
                .expect("command output closed before the start marker");
            assert_eq!(started, CommandOutputChunk::stdout(b"started".to_vec()));
            extension.reset_after(Duration::from_secs(1));
            tokio::time::sleep(Duration::from_millis(50)).await;
            extension.reset_after(Duration::from_secs(2));
        };
        let (result, ()) = tokio::join!(execution, extend);

        // Assert
        assert_eq!(
            result.unwrap().termination,
            CommandTermination::Exited { code: 0 }
        );
    }

    #[tokio::test]
    async fn executor_returns_captured_output_with_a_timeout() {
        // Arrange
        let fixture = fixture();
        let request = request(&fixture.workspace, "printf 'started'; sleep 60");
        let (preview_tx, _preview_rx) = mpsc::channel(1);

        // Act
        let result = fixture
            .executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_millis(200)),
                preview_tx,
                request,
            )
            .await
            .unwrap();

        // Assert
        assert_eq!(result.termination, CommandTermination::TimedOut);
        assert_eq!(result.output.stdout_bytes, 7);
    }

    #[tokio::test]
    async fn executor_cancellation_cleans_up_before_returning_cancelled() {
        // Arrange
        let fixture = fixture();
        let request = request(&fixture.workspace, "printf 'started'; sleep 60");
        let cancel = CancellationToken::new();
        let cancel_after_start = cancel.clone();
        let (preview_tx, _preview_rx) = mpsc::channel(1);

        // Act
        let execution = fixture.executor.execute(
            cancel,
            CommandDeadline::after(Duration::from_secs(5)),
            preview_tx,
            request,
        );
        let cancellation = async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            cancel_after_start.cancel();
        };
        let (result, ()) = tokio::join!(execution, cancellation);

        // Assert
        assert_eq!(result, Err(CommandExecutionError::Cancelled));
    }

    #[tokio::test]
    async fn dropping_execution_kills_a_background_descendant() {
        // Arrange
        let fixture = fixture();
        let pid_file = fixture.workspace.join("descendant.pid");
        let request = request(
            &fixture.workspace,
            "sleep 60 & child=$!; printf '%s' \"$child\" > descendant.pid; wait",
        );
        let (preview_tx, _preview_rx) = mpsc::channel(1);
        let mut execution = fixture.executor.execute(
            CancellationToken::new(),
            CommandDeadline::after(Duration::from_secs(60)),
            preview_tx,
            request,
        );
        let wait_for_pid = async {
            loop {
                if let Ok(pid) = fs::read_to_string(&pid_file) {
                    break i32::from_str(&pid).unwrap();
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        };
        let descendant_pid = tokio::select! {
            result = &mut execution => panic!("execution finished unexpectedly: {result:?}"),
            pid = wait_for_pid => pid,
        };

        // Act
        drop(execution);
        let cleanup_completed = tokio::time::timeout(Duration::from_secs(2), async {
            while process_exists(descendant_pid) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;

        // Assert
        assert!(
            cleanup_completed.is_ok(),
            "descendant process {descendant_pid} survived the dropped execution"
        );
    }

    #[tokio::test]
    async fn executor_rejects_a_workdir_outside_the_policy_before_spawning() {
        // Arrange
        let fixture = fixture();
        let outside = TempDir::new().unwrap();
        let request = request(outside.path(), "true");
        let (preview_tx, _preview_rx) = mpsc::channel(1);

        // Act
        let error = fixture
            .executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                preview_tx,
                request,
            )
            .await
            .unwrap_err();

        // Assert
        assert!(matches!(
            error,
            CommandExecutionError::Failed { message }
                if message.contains("inside the workspace")
        ));
    }

    #[tokio::test]
    #[ignore = "requires installed Bubblewrap and enabled unprivileged namespaces"]
    async fn installed_bubblewrap_enforces_base_filesystem_and_environment_policy() {
        // Arrange
        let workspace = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let outside_file = outside.path().join("host-secret");
        let host_socket_path = outside.path().join("host.sock");
        fs::write(&outside_file, "secret").unwrap();
        symlink(&outside_file, workspace.path().join("outside-link")).unwrap();
        fs::write(
            workspace.path().join("network.py"),
            r#"import ctypes
import os
import socket

libc = ctypes.CDLL(None, use_errno=True)
libc.shmat.restype = ctypes.c_void_p
assert libc.shmat(int(os.environ["HOST_SHMID"]), None, 0) == ctypes.c_void_p(-1).value

listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
pid = os.fork()
if pid == 0:
    client = socket.socket()
    client.connect(listener.getsockname())
    client.sendall(b"ok")
    os._exit(0)
connection, _ = listener.accept()
assert connection.recv(2) == b"ok"
_, status = os.waitpid(pid, 0)
assert status == 0
try:
    socket.create_connection(("1.1.1.1", 53), timeout=0.1)
except OSError:
    pass
else:
    raise AssertionError("external network was reachable")
"#,
        )
        .unwrap();
        initialize_git_repository(workspace.path());
        let host_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let host_socket = UnixListener::bind(&host_socket_path).unwrap();
        let host_shared_memory = SharedMemorySegment::create();
        let host_port = host_listener.local_addr().unwrap().port();
        let (executor, report) =
            establish_bubblewrap_executor(crate::command::SandboxDiagnosticInput {
                docker_endpoint: None,
                inherited_path: vec![PathBuf::from("/usr/bin"), PathBuf::from("/bin")],
                pass_through_roots: Vec::new(),
                toolchain_roots: Vec::new(),
                workspace: workspace.path().to_path_buf(),
            })
            .unwrap();
        assert!(report.is_supported());
        let mut command_request = request(
            workspace.path(),
            "test ! -e \"$OUTSIDE_PATH\" && \
             test ! -e outside-link && \
             test ! -e \"$HOST_SOCKET\" && \
             test ! -e /etc/passwd && \
             test ! -e /proc/1/root/etc/passwd && \
             test ! -e /proc/self/fd/3 && \
             test \"$(git log -1 --format=%s)\" = 'sandbox-history' && \
             touch workspace-write && \
             ! git add workspace-write 2>/dev/null && \
             ! git branch sandbox-mutation 2>/dev/null && \
             ! git config cane.mutation blocked 2>/dev/null && \
             ! touch .git/hooks/cane 2>/dev/null && \
             ! : > \"/dev/tcp/127.0.0.1/$HOST_PORT\" 2>/dev/null && \
             python3 network.py && \
             test \"$HOME\" = /home/cane && \
             test \"$TMPDIR\" = /tmp && \
             touch \"$HOME/private\" \"$TMPDIR/private\"",
        );
        command_request.environment.extend([
            ("HOME".to_string(), "/home/cane".to_string()),
            ("GIT_OPTIONAL_LOCKS".to_string(), "0".to_string()),
            ("HOST_PORT".to_string(), host_port.to_string()),
            ("HOST_SHMID".to_string(), host_shared_memory.0.to_string()),
            (
                "HOST_SOCKET".to_string(),
                host_socket_path.to_string_lossy().into_owned(),
            ),
            (
                "OUTSIDE_PATH".to_string(),
                outside_file.to_string_lossy().into_owned(),
            ),
            ("PATH".to_string(), "/usr/bin:/bin".to_string()),
            ("TMPDIR".to_string(), "/tmp".to_string()),
        ]);
        let (preview_tx, _preview_rx) = mpsc::channel(1);
        let private_state_request = request(
            workspace.path(),
            "test ! -e \"$HOME/private\" && test ! -e \"$TMPDIR/private\"",
        );
        let (private_preview_tx, _private_preview_rx) = mpsc::channel(1);

        // Act
        let result = executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                preview_tx,
                command_request,
            )
            .await
            .unwrap();
        let private_state_result = executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                private_preview_tx,
                private_state_request,
            )
            .await
            .unwrap();

        // Assert
        assert_eq!(
            result.termination,
            CommandTermination::Exited { code: 0 },
            "{}",
            crate::command::format_command_result(&result)
        );
        assert_eq!(
            private_state_result.termination,
            CommandTermination::Exited { code: 0 }
        );
        assert!(workspace.path().join("workspace-write").is_file());
        assert!(!workspace.path().join(".git/hooks/cane").exists());
        drop(host_listener);
        drop(host_socket);
    }

    #[tokio::test]
    #[ignore = "requires installed Bubblewrap and enabled unprivileged namespaces"]
    async fn installed_bubblewrap_exposes_only_prepared_and_executor_environment() {
        // Arrange
        let workspace = TempDir::new().unwrap();
        let (executor, _report) =
            establish_bubblewrap_executor(crate::command::SandboxDiagnosticInput {
                docker_endpoint: None,
                inherited_path: vec![PathBuf::from("/usr/bin"), PathBuf::from("/bin")],
                pass_through_roots: Vec::new(),
                toolchain_roots: Vec::new(),
                workspace: workspace.path().to_path_buf(),
            })
            .unwrap();
        let request = CommandRequest {
            arguments: Vec::new(),
            docker_endpoint: None,
            docker_executables: Vec::new(),
            environment: BTreeMap::from([("ONLY_VALUE".to_string(), "prepared".to_string())]),
            executable: "/usr/bin/env".to_string(),
            workdir: workspace.path().to_path_buf(),
        };
        let (preview_tx, _preview_rx) = mpsc::channel(1);

        // Act
        let result = executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                preview_tx,
                request,
            )
            .await
            .unwrap();
        let output = result
            .output
            .chunks
            .iter()
            .flat_map(|chunk| chunk.bytes.iter().copied())
            .collect::<Vec<_>>();
        let output = String::from_utf8(output).unwrap();
        let environment = output.lines().map(str::to_string).collect::<BTreeSet<_>>();

        // Assert
        assert_eq!(result.termination, CommandTermination::Exited { code: 0 });
        assert_eq!(
            environment,
            BTreeSet::from([
                "ONLY_VALUE=prepared".to_string(),
                format!("PWD={}", workspace.path().display()),
            ])
        );
    }

    #[tokio::test]
    #[ignore = "requires installed Bubblewrap and enabled unprivileged namespaces"]
    async fn installed_bubblewrap_keeps_linked_worktree_metadata_read_only() {
        // Arrange
        let root = TempDir::new().unwrap();
        let repository = root.path().join("repository");
        let workspace = root.path().join("workspace");
        fs::create_dir(&repository).unwrap();
        initialize_git_repository(&repository);
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(&repository)
            .args(["worktree", "add", "--quiet", "-b", "linked-test"])
            .arg(&workspace)
            .status()
            .unwrap();
        assert!(status.success());
        let (executor, _report) =
            establish_bubblewrap_executor(crate::command::SandboxDiagnosticInput {
                docker_endpoint: None,
                inherited_path: vec![PathBuf::from("/usr/bin"), PathBuf::from("/bin")],
                pass_through_roots: Vec::new(),
                toolchain_roots: Vec::new(),
                workspace: workspace.clone(),
            })
            .unwrap();
        let request = request(
            &workspace,
            "test \"$(git log -1 --format=%s)\" = 'sandbox-history' && \
             touch workspace-write && \
             ! git add workspace-write 2>/dev/null && \
             ! git update-ref refs/heads/sandbox-mutation HEAD 2>/dev/null && \
             ! git config cane.mutation blocked 2>/dev/null && \
             ! touch \"$(git rev-parse --git-path hooks)/cane\" 2>/dev/null",
        );
        let (preview_tx, _preview_rx) = mpsc::channel(1);

        // Act
        let result = executor
            .execute(
                CancellationToken::new(),
                CommandDeadline::after(Duration::from_secs(5)),
                preview_tx,
                request,
            )
            .await
            .unwrap();

        // Assert
        assert_eq!(result.termination, CommandTermination::Exited { code: 0 });
        assert!(workspace.join("workspace-write").is_file());
        assert!(!repository.join(".git/hooks/cane").exists());
        assert!(!repository.join(".git/refs/heads/sandbox-mutation").exists());
    }

    #[tokio::test]
    #[ignore = "requires installed Bubblewrap and enabled unprivileged namespaces"]
    async fn installed_bubblewrap_drop_prevents_background_process_survival() {
        // Arrange
        let workspace = TempDir::new().unwrap();
        let (executor, _report) =
            establish_bubblewrap_executor(crate::command::SandboxDiagnosticInput {
                docker_endpoint: None,
                inherited_path: vec![PathBuf::from("/usr/bin"), PathBuf::from("/bin")],
                pass_through_roots: Vec::new(),
                toolchain_roots: Vec::new(),
                workspace: workspace.path().to_path_buf(),
            })
            .unwrap();
        let request = request(
            workspace.path(),
            "(/bin/sleep 1; touch orphaned) & touch started; wait",
        );
        let started = workspace.path().join("started");
        let orphaned = workspace.path().join("orphaned");
        let (preview_tx, _preview_rx) = mpsc::channel(1);
        let mut execution = executor.execute(
            CancellationToken::new(),
            CommandDeadline::after(Duration::from_secs(10)),
            preview_tx,
            request,
        );
        let wait_for_start = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if started.exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
        tokio::select! {
            result = &mut execution => panic!("execution finished unexpectedly: {result:?}"),
            ready = wait_for_start => ready.unwrap(),
        }

        // Act
        drop(execution);
        tokio::time::sleep(Duration::from_millis(1_200)).await;

        // Assert
        assert!(
            !orphaned.exists(),
            "background process survived after the execution future was dropped"
        );
    }

    fn process_exists(process_id: i32) -> bool {
        // SAFETY: signal zero performs existence checking only and the PID was
        // parsed from the test child itself.
        let result = unsafe { libc::kill(process_id, 0) };
        result == 0 || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }

    fn initialize_git_repository(workspace: &Path) {
        let git = |arguments: &[&str]| {
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(workspace)
                .args(arguments)
                .status()
                .unwrap();
            assert!(status.success(), "git command failed: {arguments:?}");
        };

        git(&["init", "--quiet"]);
        fs::write(workspace.join("tracked.txt"), "history").unwrap();
        git(&["add", "tracked.txt"]);
        git(&[
            "-c",
            "user.name=Cane Test",
            "-c",
            "user.email=cane@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "sandbox-history",
        ]);
    }
}
