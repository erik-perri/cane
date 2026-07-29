use super::{
    CommandSandboxPolicy, CommandSandboxPolicyConfig, DiagnosticStatus, SandboxDiagnosticFinding,
    SandboxDiagnosticInput, SandboxDiagnosticReport, SandboxFilesystemAccess,
    build_command_sandbox_policy,
};
use std::collections::HashSet;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use thiserror::Error;

const PRIVATE_HOME: &str = "/home/cane";
const PRIVATE_TEMP: &str = "/tmp";
const MAX_GIT_POINTER_BYTES: u64 = 16 * 1024;
const TRUSTED_BUBBLEWRAP_CANDIDATES: &[&str] = &[
    "/usr/bin/bwrap",
    "/bin/bwrap",
    "/run/current-system/sw/bin/bwrap",
];
const TRUSTED_BUBBLEWRAP_TARGET_ROOTS: &[&str] = &["/usr/bin", "/bin", "/nix/store"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BubblewrapInstallation {
    path: PathBuf,
}

impl BubblewrapInstallation {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum BubblewrapResolutionError {
    #[error(
        "Bubblewrap was not found at a trusted system location; checked: {}",
        .searched.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(", ")
    )]
    NotFound { searched: Vec<PathBuf> },

    #[error("trusted Bubblewrap candidates were rejected: {}", .rejections.join("; "))]
    Rejected { rejections: Vec<String> },
}

pub fn resolve_bubblewrap() -> Result<BubblewrapInstallation, BubblewrapResolutionError> {
    let candidates = TRUSTED_BUBBLEWRAP_CANDIDATES
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let trusted_roots = TRUSTED_BUBBLEWRAP_TARGET_ROOTS
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();

    resolve_bubblewrap_from(&candidates, &trusted_roots)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinuxSandboxOperation {
    CreateDirectory { path: PathBuf },
    DeviceFilesystem { path: PathBuf },
    PrivateTmpfs { path: PathBuf },
    ProcFilesystem { path: PathBuf },
    ReadOnlyBind { path: PathBuf },
    ReadWriteBind { path: PathBuf },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinuxSandboxPlan {
    arguments: Vec<OsString>,
    operations: Vec<LinuxSandboxOperation>,
}

impl LinuxSandboxPlan {
    pub fn arguments(&self) -> &[OsString] {
        &self.arguments
    }

    pub fn operations(&self) -> &[LinuxSandboxOperation] {
        &self.operations
    }
}

pub fn compile_linux_sandbox_plan(policy: &CommandSandboxPolicy) -> LinuxSandboxPlan {
    let mut operations = Vec::new();
    let mut created_directories = HashSet::new();

    for private_path in [policy.private_home(), policy.private_temp()] {
        add_parent_directories(private_path, &mut created_directories, &mut operations);
        add_directory(
            private_path.to_path_buf(),
            &mut created_directories,
            &mut operations,
        );
        operations.push(LinuxSandboxOperation::PrivateTmpfs {
            path: private_path.to_path_buf(),
        });
    }

    for grant in policy.grants() {
        add_parent_directories(&grant.path, &mut created_directories, &mut operations);
        operations.push(match grant.access {
            SandboxFilesystemAccess::ReadOnly => LinuxSandboxOperation::ReadOnlyBind {
                path: grant.path.clone(),
            },
            SandboxFilesystemAccess::ReadWrite => LinuxSandboxOperation::ReadWriteBind {
                path: grant.path.clone(),
            },
        });
    }

    add_directory(
        PathBuf::from("/proc"),
        &mut created_directories,
        &mut operations,
    );
    operations.push(LinuxSandboxOperation::ProcFilesystem {
        path: PathBuf::from("/proc"),
    });
    add_directory(
        PathBuf::from("/dev"),
        &mut created_directories,
        &mut operations,
    );
    operations.push(LinuxSandboxOperation::DeviceFilesystem {
        path: PathBuf::from("/dev"),
    });

    let mut arguments = namespace_arguments();
    for operation in &operations {
        append_operation_arguments(operation, &mut arguments);
    }

    LinuxSandboxPlan {
        arguments,
        operations,
    }
}

pub(crate) fn diagnose_linux_sandbox(input: SandboxDiagnosticInput) -> SandboxDiagnosticReport {
    diagnose_linux_sandbox_with(
        input,
        &SystemProbeRunner,
        TRUSTED_BUBBLEWRAP_CANDIDATES
            .iter()
            .map(PathBuf::from)
            .collect(),
        TRUSTED_BUBBLEWRAP_TARGET_ROOTS
            .iter()
            .map(PathBuf::from)
            .collect(),
    )
}

trait ProbeRunner {
    fn run(&self, executable: &Path, arguments: &[OsString]) -> io::Result<ProbeOutput>;
}

struct SystemProbeRunner;

impl ProbeRunner for SystemProbeRunner {
    fn run(&self, executable: &Path, arguments: &[OsString]) -> io::Result<ProbeOutput> {
        let output = Command::new(executable)
            .args(arguments)
            .env_clear()
            .stdin(Stdio::null())
            .output()?;

        Ok(ProbeOutput {
            status: output.status.success(),
            stderr: output.stderr,
            stdout: output.stdout,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProbeOutput {
    status: bool,
    stderr: Vec<u8>,
    stdout: Vec<u8>,
}

fn diagnose_linux_sandbox_with(
    input: SandboxDiagnosticInput,
    runner: &impl ProbeRunner,
    candidates: Vec<PathBuf>,
    trusted_roots: Vec<PathBuf>,
) -> SandboxDiagnosticReport {
    let mut findings = vec![SandboxDiagnosticFinding::mandatory(
        "platform",
        DiagnosticStatus::Passed,
        "Linux selected",
    )];
    let runtime_roots = linux_runtime_roots();

    for path in &runtime_roots {
        findings.push(SandboxDiagnosticFinding::mandatory(
            format!("runtime path {}", path.display()),
            DiagnosticStatus::Passed,
            "available for read-only exposure",
        ));
    }

    for required in [Path::new("/bin/bash"), Path::new("/bin/true")] {
        let (status, detail) = if required.is_file() {
            (
                DiagnosticStatus::Passed,
                "required executable is available".to_string(),
            )
        } else {
            (
                DiagnosticStatus::Failed,
                "required executable is missing".to_string(),
            )
        };
        findings.push(SandboxDiagnosticFinding::mandatory(
            format!("runtime executable {}", required.display()),
            status,
            detail,
        ));
    }

    let (workspace_status, workspace_detail) = match fs::canonicalize(&input.workspace) {
        Ok(workspace) if workspace.is_dir() => (
            DiagnosticStatus::Passed,
            format!(
                "read/write root is representable at {}",
                workspace.display()
            ),
        ),
        Ok(workspace) => (
            DiagnosticStatus::Failed,
            format!("{} is not a directory", workspace.display()),
        ),
        Err(error) => (DiagnosticStatus::Failed, error.to_string()),
    };
    findings.push(SandboxDiagnosticFinding::mandatory(
        "Workspace",
        workspace_status,
        workspace_detail,
    ));

    let git_metadata = match discover_git_metadata(&input.workspace) {
        Ok(paths) => {
            let detail = if paths.is_empty() {
                "workspace is not inside a Git repository".to_string()
            } else {
                format!(
                    "read-only metadata: {}",
                    paths
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Git metadata",
                DiagnosticStatus::Passed,
                detail,
            ));
            paths
        }
        Err(error) => {
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Git metadata",
                DiagnosticStatus::Failed,
                error.to_string(),
            ));
            Vec::new()
        }
    };

    let policy_config = CommandSandboxPolicyConfig {
        git_metadata_roots: git_metadata,
        inherited_path: input.inherited_path,
        pass_through_roots: input.pass_through_roots,
        private_home: PathBuf::from(PRIVATE_HOME),
        private_temp: PathBuf::from(PRIVATE_TEMP),
        runtime_roots: runtime_roots.clone(),
        toolchain_roots: input.toolchain_roots,
        workspace: input.workspace,
    };
    let policy = match build_command_sandbox_policy(policy_config) {
        Ok(policy) => {
            let plan = compile_linux_sandbox_plan(&policy);
            let exposes_only_declared_roots = plan.operations().iter().all(|operation| {
                !matches!(
                    operation,
                    LinuxSandboxOperation::ReadOnlyBind { path }
                        | LinuxSandboxOperation::ReadWriteBind { path }
                        if path == Path::new("/")
                )
            });
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Linux policy plan",
                if exposes_only_declared_roots {
                    DiagnosticStatus::Passed
                } else {
                    DiagnosticStatus::Failed
                },
                format!(
                    "{} positive mount operations; host root is not mounted",
                    plan.operations().len()
                ),
            ));
            Some(policy)
        }
        Err(error) => {
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Linux policy plan",
                DiagnosticStatus::Failed,
                error.to_string(),
            ));
            None
        }
    };

    add_docker_finding(input.docker_endpoint.as_deref(), &mut findings);

    let installation = resolve_bubblewrap_from(&candidates, &trusted_roots);
    let (bubblewrap_path, bubblewrap_version) = match installation {
        Ok(installation) => {
            let path = installation.path;
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Bubblewrap resolution",
                DiagnosticStatus::Passed,
                format!("resolved trusted launcher {}", path.display()),
            ));
            let version = probe_version(&path, runner, &mut findings);
            run_capability_probes(&path, &runtime_roots, runner, &mut findings);
            (Some(path), version)
        }
        Err(error) => {
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Bubblewrap resolution",
                DiagnosticStatus::Failed,
                error.to_string(),
            ));
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Bubblewrap version",
                DiagnosticStatus::Failed,
                "not probed because no trusted launcher was resolved",
            ));
            for capability in PROBE_CAPABILITIES {
                findings.push(SandboxDiagnosticFinding::mandatory(
                    capability.name,
                    DiagnosticStatus::Failed,
                    "not probed because no trusted launcher was resolved",
                ));
            }
            (None, None)
        }
    };

    SandboxDiagnosticReport {
        backend: "bubblewrap".to_string(),
        bubblewrap_path,
        bubblewrap_version,
        findings,
        platform: "linux".to_string(),
        policy,
    }
}

fn probe_version(
    path: &Path,
    runner: &impl ProbeRunner,
    findings: &mut Vec<SandboxDiagnosticFinding>,
) -> Option<String> {
    match runner.run(path, &[OsString::from("--version")]) {
        Ok(output) if output.status => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let valid = !version.is_empty();
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Bubblewrap version",
                if valid {
                    DiagnosticStatus::Passed
                } else {
                    DiagnosticStatus::Failed
                },
                if valid {
                    version.clone()
                } else {
                    "version output was empty".to_string()
                },
            ));
            valid.then_some(version)
        }
        Ok(output) => {
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Bubblewrap version",
                DiagnosticStatus::Failed,
                probe_failure_detail(&output),
            ));
            None
        }
        Err(error) => {
            findings.push(SandboxDiagnosticFinding::mandatory(
                "Bubblewrap version",
                DiagnosticStatus::Failed,
                error.to_string(),
            ));
            None
        }
    }
}

#[derive(Clone, Copy)]
struct ProbeCapability {
    arguments: &'static [&'static str],
    name: &'static str,
}

const PROBE_CAPABILITIES: &[ProbeCapability] = &[
    ProbeCapability {
        arguments: &[],
        name: "mount namespace",
    },
    ProbeCapability {
        arguments: &[],
        name: "user namespace",
    },
    ProbeCapability {
        arguments: &["--unshare-pid"],
        name: "PID namespace",
    },
    ProbeCapability {
        arguments: &["--unshare-net"],
        name: "network namespace",
    },
    ProbeCapability {
        arguments: &["--unshare-ipc"],
        name: "IPC namespace",
    },
    ProbeCapability {
        arguments: &["--unshare-uts"],
        name: "UTS namespace",
    },
    ProbeCapability {
        arguments: &["--new-session"],
        name: "session isolation",
    },
    ProbeCapability {
        arguments: &["--die-with-parent"],
        name: "parent-death supervision",
    },
    ProbeCapability {
        arguments: &[
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--new-session",
            "--die-with-parent",
        ],
        name: "combined production capabilities",
    },
];

fn run_capability_probes(
    path: &Path,
    runtime_roots: &[PathBuf],
    runner: &impl ProbeRunner,
    findings: &mut Vec<SandboxDiagnosticFinding>,
) {
    for capability in PROBE_CAPABILITIES {
        let arguments = probe_arguments(runtime_roots, capability.arguments);
        let (status, detail) = match runner.run(path, &arguments) {
            Ok(output) if output.status => (
                DiagnosticStatus::Passed,
                "functional sandbox invocation succeeded".to_string(),
            ),
            Ok(output) => (DiagnosticStatus::Failed, probe_failure_detail(&output)),
            Err(error) => (DiagnosticStatus::Failed, error.to_string()),
        };
        findings.push(SandboxDiagnosticFinding::mandatory(
            capability.name,
            status,
            detail,
        ));
    }
}

fn probe_arguments(runtime_roots: &[PathBuf], additions: &[&str]) -> Vec<OsString> {
    let mut arguments = vec![
        OsString::from("--unshare-user"),
        OsString::from("--clearenv"),
    ];
    arguments.extend(additions.iter().map(OsString::from));

    for root in runtime_roots {
        arguments.push(OsString::from("--ro-bind"));
        arguments.push(root.as_os_str().to_owned());
        arguments.push(root.as_os_str().to_owned());
    }

    arguments.extend([
        OsString::from("--proc"),
        OsString::from("/proc"),
        OsString::from("--dev"),
        OsString::from("/dev"),
        OsString::from("--chdir"),
        OsString::from("/"),
        OsString::from("--"),
        OsString::from("/bin/true"),
    ]);
    arguments
}

fn probe_failure_detail(output: &ProbeOutput) -> String {
    let bytes = if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    };
    let detail = String::from_utf8_lossy(bytes);
    let detail = detail.trim();

    if detail.is_empty() {
        "probe exited unsuccessfully without output".to_string()
    } else {
        detail.chars().take(400).collect()
    }
}

fn resolve_bubblewrap_from(
    candidates: &[PathBuf],
    trusted_roots: &[PathBuf],
) -> Result<BubblewrapInstallation, BubblewrapResolutionError> {
    use std::os::unix::fs::PermissionsExt;

    let mut found_candidate = false;
    let mut rejections = Vec::new();
    for candidate in candidates {
        let metadata = match fs::metadata(candidate) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => {
                found_candidate = true;
                rejections.push(format!("{}: {error}", candidate.display()));
                continue;
            }
        };
        found_candidate = true;

        if !metadata.is_file() {
            rejections.push(format!("{} is not a regular file", candidate.display()));
            continue;
        }
        if metadata.permissions().mode() & 0o111 == 0 {
            rejections.push(format!("{} is not executable", candidate.display()));
            continue;
        }

        let canonical = match fs::canonicalize(candidate) {
            Ok(canonical) => canonical,
            Err(error) => {
                rejections.push(format!("{}: {error}", candidate.display()));
                continue;
            }
        };
        let trusted = trusted_roots.iter().any(|root| {
            fs::canonicalize(root).is_ok_and(|canonical_root| canonical.starts_with(canonical_root))
        });
        if !trusted {
            rejections.push(format!(
                "{} resolves outside trusted system roots to {}",
                candidate.display(),
                canonical.display()
            ));
            continue;
        }

        return Ok(BubblewrapInstallation { path: canonical });
    }

    if found_candidate {
        Err(BubblewrapResolutionError::Rejected { rejections })
    } else {
        Err(BubblewrapResolutionError::NotFound {
            searched: candidates.to_vec(),
        })
    }
}

fn linux_runtime_roots() -> Vec<PathBuf> {
    [
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/nix/store",
        "/run/current-system/sw",
    ]
    .into_iter()
    .map(PathBuf::from)
    .filter(|path| path.exists())
    .collect()
}

fn discover_git_metadata(workspace: &Path) -> io::Result<Vec<PathBuf>> {
    let mut directory = Some(workspace);
    while let Some(candidate) = directory {
        let dot_git = candidate.join(".git");
        match fs::symlink_metadata(&dot_git) {
            Ok(metadata) if metadata.is_dir() => {
                return Ok(vec![fs::canonicalize(dot_git)?]);
            }
            Ok(metadata) if metadata.is_file() => {
                return linked_worktree_metadata(&dot_git);
            }
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} is neither a file nor directory", dot_git.display()),
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        directory = candidate.parent();
    }

    Ok(Vec::new())
}

fn linked_worktree_metadata(dot_git: &Path) -> io::Result<Vec<PathBuf>> {
    let contents = read_git_pointer(dot_git)?;
    let git_dir = contents
        .trim()
        .strip_prefix("gitdir: ")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid .git file"))?;
    let git_dir = Path::new(git_dir);
    let git_dir = if git_dir.is_absolute() {
        git_dir.to_path_buf()
    } else {
        dot_git
            .parent()
            .expect(".git file has a parent")
            .join(git_dir)
    };
    let git_dir = fs::canonicalize(git_dir)?;
    let canonical_dot_git = fs::canonicalize(dot_git)?;
    let backlink_file = git_dir.join("gitdir");
    let common_dir_file = git_dir.join("commondir");
    if !backlink_file.is_file() || !common_dir_file.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} does not identify a supported linked worktree",
                dot_git.display()
            ),
        ));
    }

    let backlink = resolve_git_pointer_path(&git_dir, read_git_pointer(&backlink_file)?.trim());
    if fs::canonicalize(backlink)? != canonical_dot_git {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} does not point back to {}",
                backlink_file.display(),
                dot_git.display()
            ),
        ));
    }

    let common_dir = resolve_git_pointer_path(&git_dir, read_git_pointer(&common_dir_file)?.trim());
    let common_dir = fs::canonicalize(common_dir)?;
    let expected_worktrees = common_dir.join("worktrees");
    if git_dir.parent() != Some(expected_worktrees.as_path()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} is outside the common Git worktree administration directory",
                git_dir.display()
            ),
        ));
    }

    Ok(vec![canonical_dot_git, git_dir, common_dir])
}

fn read_git_pointer(path: &Path) -> io::Result<String> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > MAX_GIT_POINTER_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds the {MAX_GIT_POINTER_BYTES}-byte Git pointer limit",
                path.display()
            ),
        ));
    }

    fs::read_to_string(path)
}

fn resolve_git_pointer_path(base: &Path, value: &str) -> PathBuf {
    let path = Path::new(value);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn namespace_arguments() -> Vec<OsString> {
    [
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--clearenv",
    ]
    .into_iter()
    .map(OsString::from)
    .collect()
}

fn add_parent_directories(
    path: &Path,
    created: &mut HashSet<PathBuf>,
    operations: &mut Vec<LinuxSandboxOperation>,
) {
    let mut parents = path
        .ancestors()
        .skip(1)
        .filter(|parent| parent.parent().is_some())
        .map(Path::to_path_buf)
        .collect::<Vec<_>>();
    parents.reverse();

    for parent in parents {
        add_directory(parent, created, operations);
    }
}

fn add_directory(
    path: PathBuf,
    created: &mut HashSet<PathBuf>,
    operations: &mut Vec<LinuxSandboxOperation>,
) {
    if created.insert(path.clone()) {
        operations.push(LinuxSandboxOperation::CreateDirectory { path });
    }
}

fn append_operation_arguments(operation: &LinuxSandboxOperation, arguments: &mut Vec<OsString>) {
    let (flag, path) = match operation {
        LinuxSandboxOperation::CreateDirectory { path } => ("--dir", path),
        LinuxSandboxOperation::DeviceFilesystem { path } => ("--dev", path),
        LinuxSandboxOperation::PrivateTmpfs { path } => ("--tmpfs", path),
        LinuxSandboxOperation::ProcFilesystem { path } => ("--proc", path),
        LinuxSandboxOperation::ReadOnlyBind { path } => {
            arguments.push(OsString::from("--ro-bind"));
            arguments.push(path.as_os_str().to_owned());
            arguments.push(path.as_os_str().to_owned());
            return;
        }
        LinuxSandboxOperation::ReadWriteBind { path } => {
            arguments.push(OsString::from("--bind"));
            arguments.push(path.as_os_str().to_owned());
            arguments.push(path.as_os_str().to_owned());
            return;
        }
    };
    arguments.push(OsString::from(flag));
    arguments.push(path.as_os_str().to_owned());
}

fn add_docker_finding(endpoint: Option<&Path>, findings: &mut Vec<SandboxDiagnosticFinding>) {
    use std::os::unix::fs::FileTypeExt;

    let Some(endpoint) = endpoint else {
        findings.push(SandboxDiagnosticFinding::optional(
            "Docker endpoint",
            DiagnosticStatus::Failed,
            "no eligible local Unix socket was discovered",
        ));
        return;
    };

    let (status, detail) = match fs::metadata(endpoint) {
        Ok(metadata) if metadata.file_type().is_socket() => (
            DiagnosticStatus::Passed,
            format!("local Unix socket available at {}", endpoint.display()),
        ),
        Ok(_) => (
            DiagnosticStatus::Failed,
            format!("{} is not a Unix socket", endpoint.display()),
        ),
        Err(error) => (
            DiagnosticStatus::Failed,
            format!("{} is unavailable: {error}", endpoint.display()),
        ),
    };
    findings.push(SandboxDiagnosticFinding::optional(
        "Docker endpoint",
        status,
        detail,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::ffi::OsStr;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use tempfile::TempDir;

    #[derive(Default)]
    struct FakeProbeRunner {
        calls: RefCell<Vec<(PathBuf, Vec<OsString>)>>,
        fail_argument: Option<&'static str>,
    }

    impl ProbeRunner for FakeProbeRunner {
        fn run(&self, executable: &Path, arguments: &[OsString]) -> io::Result<ProbeOutput> {
            self.calls
                .borrow_mut()
                .push((executable.to_path_buf(), arguments.to_vec()));
            let failed = self.fail_argument.is_some_and(|argument| {
                arguments
                    .iter()
                    .any(|candidate| candidate == OsStr::new(argument))
            });

            Ok(if arguments == [OsString::from("--version")] {
                ProbeOutput {
                    status: true,
                    stderr: Vec::new(),
                    stdout: b"bubblewrap 1.2.3\n".to_vec(),
                }
            } else if failed {
                ProbeOutput {
                    status: false,
                    stderr: b"capability unavailable".to_vec(),
                    stdout: Vec::new(),
                }
            } else {
                ProbeOutput {
                    status: true,
                    stderr: Vec::new(),
                    stdout: Vec::new(),
                }
            })
        }
    }

    fn executable(path: &Path) {
        fs::write(path, "#!/bin/true\n").unwrap();
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).unwrap();
    }

    fn diagnostic_fixture() -> (TempDir, SandboxDiagnosticInput, PathBuf, PathBuf) {
        let root = TempDir::new().unwrap();
        let trusted_root = root.path().join("trusted");
        let workspace = root.path().join("workspace");
        fs::create_dir_all(&trusted_root).unwrap();
        fs::create_dir_all(&workspace).unwrap();
        let launcher = trusted_root.join("bwrap");
        executable(&launcher);
        let input = SandboxDiagnosticInput {
            docker_endpoint: None,
            inherited_path: vec![PathBuf::from("/usr/bin"), PathBuf::from("/bin")],
            pass_through_roots: Vec::new(),
            toolchain_roots: Vec::new(),
            workspace,
        };

        (root, input, launcher, trusted_root)
    }

    #[test]
    fn trusted_resolution_ignores_path_and_accepts_only_fixed_candidate_targets() {
        // Arrange
        let root = TempDir::new().unwrap();
        let trusted_root = root.path().join("trusted");
        let workspace_root = root.path().join("workspace");
        fs::create_dir_all(&trusted_root).unwrap();
        fs::create_dir_all(&workspace_root).unwrap();
        let trusted_launcher = trusted_root.join("bwrap");
        let workspace_launcher = workspace_root.join("bwrap");
        executable(&trusted_launcher);
        executable(&workspace_launcher);
        let candidates = vec![workspace_launcher, trusted_launcher.clone()];

        // Act
        let installation = resolve_bubblewrap_from(&candidates, &[trusted_root]).unwrap();

        // Assert
        assert_eq!(
            installation.path(),
            fs::canonicalize(trusted_launcher).unwrap()
        );
    }

    #[test]
    fn trusted_resolution_rejects_a_candidate_symlinked_outside_system_roots() {
        // Arrange
        let root = TempDir::new().unwrap();
        let trusted_root = root.path().join("trusted");
        let outside_root = root.path().join("workspace");
        fs::create_dir_all(&trusted_root).unwrap();
        fs::create_dir_all(&outside_root).unwrap();
        let outside_launcher = outside_root.join("bwrap");
        let candidate = trusted_root.join("bwrap");
        executable(&outside_launcher);
        symlink(outside_launcher, &candidate).unwrap();

        // Act
        let error = resolve_bubblewrap_from(&[candidate], &[trusted_root]).unwrap_err();

        // Assert
        assert!(matches!(
            error,
            BubblewrapResolutionError::Rejected { rejections }
                if rejections[0].contains("outside trusted system roots")
        ));
    }

    #[test]
    fn linux_plan_uses_positive_mounts_and_overlays_git_metadata_read_only() {
        // Arrange
        let root = TempDir::new().unwrap();
        let workspace = root.path().join("workspace");
        let runtime = root.path().join("runtime");
        let git_metadata = workspace.join(".git");
        fs::create_dir_all(&runtime).unwrap();
        fs::create_dir_all(&git_metadata).unwrap();
        let policy = build_command_sandbox_policy(CommandSandboxPolicyConfig {
            git_metadata_roots: vec![git_metadata.clone()],
            inherited_path: vec![runtime.clone()],
            pass_through_roots: Vec::new(),
            private_home: PathBuf::from("/home/cane"),
            private_temp: PathBuf::from("/tmp"),
            runtime_roots: vec![runtime.clone()],
            toolchain_roots: Vec::new(),
            workspace: workspace.clone(),
        })
        .unwrap();

        // Act
        let plan = compile_linux_sandbox_plan(&policy);

        // Assert
        assert!(plan.arguments().contains(&OsString::from("--unshare-net")));
        assert!(!plan.arguments().iter().any(|argument| argument == "/"));
        let workspace_index = plan
            .operations()
            .iter()
            .position(|operation| {
                operation
                    == &LinuxSandboxOperation::ReadWriteBind {
                        path: workspace.clone(),
                    }
            })
            .unwrap();
        let git_index = plan
            .operations()
            .iter()
            .position(|operation| {
                operation
                    == &LinuxSandboxOperation::ReadOnlyBind {
                        path: git_metadata.clone(),
                    }
            })
            .unwrap();
        assert!(workspace_index < git_index);
        assert!(
            plan.operations()
                .contains(&LinuxSandboxOperation::ReadOnlyBind { path: runtime })
        );
    }

    #[test]
    fn diagnostics_use_one_probe_path_and_keep_capability_failures_distinct() {
        // Arrange
        let (_root, input, launcher, trusted_root) = diagnostic_fixture();
        let runner = FakeProbeRunner {
            calls: RefCell::new(Vec::new()),
            fail_argument: Some("--unshare-net"),
        };

        // Act
        let report =
            diagnose_linux_sandbox_with(input, &runner, vec![launcher.clone()], vec![trusted_root]);

        // Assert
        assert_eq!(
            report.bubblewrap_path,
            Some(fs::canonicalize(launcher).unwrap())
        );
        assert_eq!(
            report.bubblewrap_version.as_deref(),
            Some("bubblewrap 1.2.3")
        );
        assert!(!report.is_supported());
        assert!(report.findings.iter().any(|finding| {
            finding.name == "network namespace" && finding.status == DiagnosticStatus::Failed
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.name == "PID namespace" && finding.status == DiagnosticStatus::Passed
        }));
        assert_eq!(runner.calls.borrow().len(), 1 + PROBE_CAPABILITIES.len());
        assert!(
            runner
                .calls
                .borrow()
                .iter()
                .all(|(path, _)| { path == report.bubblewrap_path.as_deref().unwrap() })
        );
    }

    #[test]
    fn diagnostics_fail_closed_without_running_probes_when_bubblewrap_is_missing() {
        // Arrange
        let (root, input, _launcher, trusted_root) = diagnostic_fixture();
        let missing = root.path().join("trusted").join("missing-bwrap");
        let runner = FakeProbeRunner::default();

        // Act
        let report = diagnose_linux_sandbox_with(input, &runner, vec![missing], vec![trusted_root]);

        // Assert
        assert!(!report.is_supported());
        assert!(report.bubblewrap_path.is_none());
        assert!(runner.calls.borrow().is_empty());
        assert!(report.findings.iter().any(|finding| {
            finding.name == "Bubblewrap resolution" && finding.status == DiagnosticStatus::Failed
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.name == "combined production capabilities"
                && finding.status == DiagnosticStatus::Failed
        }));
    }

    #[test]
    fn linked_worktree_metadata_includes_the_git_file_private_dir_and_common_dir() {
        // Arrange
        let root = TempDir::new().unwrap();
        let workspace = root.path().join("worktree");
        let common = root.path().join("repository.git");
        let git_dir = common.join("worktrees").join("worktree");
        fs::create_dir_all(&workspace).unwrap();
        fs::create_dir_all(&git_dir).unwrap();
        fs::write(
            workspace.join(".git"),
            format!("gitdir: {}\n", git_dir.display()),
        )
        .unwrap();
        fs::write(git_dir.join("commondir"), "../..\n").unwrap();
        fs::write(
            git_dir.join("gitdir"),
            format!("{}\n", workspace.join(".git").display()),
        )
        .unwrap();

        // Act
        let metadata = discover_git_metadata(&workspace).unwrap();

        // Assert
        assert_eq!(
            metadata,
            [
                fs::canonicalize(workspace.join(".git")).unwrap(),
                fs::canonicalize(git_dir).unwrap(),
                fs::canonicalize(common).unwrap(),
            ]
        );
    }

    #[test]
    fn workspace_git_file_cannot_grant_an_arbitrary_external_directory() {
        // Arrange
        let root = TempDir::new().unwrap();
        let workspace = root.path().join("workspace");
        let outside = root.path().join("outside");
        fs::create_dir_all(&workspace).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(
            workspace.join(".git"),
            format!("gitdir: {}\n", outside.display()),
        )
        .unwrap();

        // Act
        let error = discover_git_metadata(&workspace).unwrap_err();

        // Assert
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("supported linked worktree"));
    }

    #[test]
    #[ignore = "requires installed Bubblewrap and enabled unprivileged namespaces"]
    fn installed_bubblewrap_passes_the_functional_probe() {
        // Arrange
        let workspace = tempfile::tempdir().unwrap();
        let input = SandboxDiagnosticInput {
            docker_endpoint: None,
            inherited_path: vec![PathBuf::from("/usr/bin"), PathBuf::from("/bin")],
            pass_through_roots: Vec::new(),
            toolchain_roots: Vec::new(),
            workspace: workspace.path().to_path_buf(),
        };

        // Act
        let report = diagnose_linux_sandbox(input);

        // Assert
        assert!(report.is_supported(), "{}", report.render());
    }
}
