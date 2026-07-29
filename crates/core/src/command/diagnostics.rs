use super::CommandSandboxPolicy;
use std::fmt::Write;
use std::path::PathBuf;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SandboxDiagnosticInput {
    pub docker_endpoint: Option<PathBuf>,
    pub inherited_path: Vec<PathBuf>,
    pub pass_through_roots: Vec<PathBuf>,
    pub toolchain_roots: Vec<PathBuf>,
    pub workspace: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticImportance {
    Mandatory,
    Optional,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticStatus {
    Failed,
    Passed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SandboxDiagnosticFinding {
    pub detail: String,
    pub importance: DiagnosticImportance,
    pub name: String,
    pub status: DiagnosticStatus,
}

impl SandboxDiagnosticFinding {
    pub(crate) fn mandatory(
        name: impl Into<String>,
        status: DiagnosticStatus,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            detail: detail.into(),
            importance: DiagnosticImportance::Mandatory,
            name: name.into(),
            status,
        }
    }

    pub(crate) fn optional(
        name: impl Into<String>,
        status: DiagnosticStatus,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            detail: detail.into(),
            importance: DiagnosticImportance::Optional,
            name: name.into(),
            status,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SandboxDiagnosticReport {
    pub backend: String,
    pub bubblewrap_path: Option<PathBuf>,
    pub bubblewrap_version: Option<String>,
    pub findings: Vec<SandboxDiagnosticFinding>,
    pub platform: String,
    pub policy: Option<CommandSandboxPolicy>,
}

impl SandboxDiagnosticReport {
    pub fn is_supported(&self) -> bool {
        self.findings.iter().all(|finding| {
            finding.importance == DiagnosticImportance::Optional
                || finding.status == DiagnosticStatus::Passed
        })
    }

    pub fn render(&self) -> String {
        let mut rendered = String::new();
        writeln!(rendered, "Sandbox").unwrap();
        writeln!(rendered, "  platform: {}", self.platform).unwrap();
        writeln!(rendered, "  backend: {}", self.backend).unwrap();
        writeln!(
            rendered,
            "  bubblewrap: {}",
            self.bubblewrap_path.as_deref().map_or_else(
                || "not resolved".to_string(),
                |path| path.display().to_string()
            )
        )
        .unwrap();
        writeln!(
            rendered,
            "  version: {}",
            self.bubblewrap_version.as_deref().unwrap_or("unavailable")
        )
        .unwrap();

        for finding in &self.findings {
            let status = match finding.status {
                DiagnosticStatus::Failed => "FAIL",
                DiagnosticStatus::Passed => "ok",
            };
            let importance = match finding.importance {
                DiagnosticImportance::Mandatory => "required",
                DiagnosticImportance::Optional => "optional",
            };
            writeln!(
                rendered,
                "  [{status}] {} ({importance}): {}",
                finding.name, finding.detail
            )
            .unwrap();
        }

        if let Some(policy) = &self.policy {
            writeln!(rendered, "  exposed roots:").unwrap();
            for grant in policy.grants() {
                writeln!(
                    rendered,
                    "    {:?} {:?}: {}",
                    grant.access,
                    grant.purpose,
                    grant.path.display()
                )
                .unwrap();
            }
            writeln!(
                rendered,
                "  command PATH: {}",
                std::env::join_paths(policy.executable_path()).map_or_else(
                    |_| "<not representable>".to_string(),
                    |path| path.to_string_lossy().into_owned()
                )
            )
            .unwrap();
        }

        rendered
    }
}

pub fn diagnose_sandbox(input: SandboxDiagnosticInput) -> SandboxDiagnosticReport {
    #[cfg(target_os = "linux")]
    {
        super::linux::diagnose_linux_sandbox(input)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = input;
        SandboxDiagnosticReport {
            backend: "unavailable".to_string(),
            bubblewrap_path: None,
            bubblewrap_version: None,
            findings: vec![SandboxDiagnosticFinding::mandatory(
                "platform",
                DiagnosticStatus::Failed,
                "sandboxed shell enforcement is currently available only on Linux",
            )],
            platform: std::env::consts::OS.to_string(),
            policy: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(
        importance: DiagnosticImportance,
        status: DiagnosticStatus,
    ) -> SandboxDiagnosticFinding {
        SandboxDiagnosticFinding {
            detail: "detail".to_string(),
            importance,
            name: "finding".to_string(),
            status,
        }
    }

    #[test]
    fn optional_failures_do_not_make_the_sandbox_unsupported() {
        // Arrange
        let report = SandboxDiagnosticReport {
            backend: "bubblewrap".to_string(),
            bubblewrap_path: None,
            bubblewrap_version: None,
            findings: vec![
                finding(DiagnosticImportance::Mandatory, DiagnosticStatus::Passed),
                finding(DiagnosticImportance::Optional, DiagnosticStatus::Failed),
            ],
            platform: "linux".to_string(),
            policy: None,
        };

        // Act
        let supported = report.is_supported();

        // Assert
        assert!(supported);
    }

    #[test]
    fn mandatory_failures_make_the_sandbox_unsupported_and_render_distinctly() {
        // Arrange
        let report = SandboxDiagnosticReport {
            backend: "bubblewrap".to_string(),
            bubblewrap_path: None,
            bubblewrap_version: None,
            findings: vec![finding(
                DiagnosticImportance::Mandatory,
                DiagnosticStatus::Failed,
            )],
            platform: "linux".to_string(),
            policy: None,
        };

        // Act
        let supported = report.is_supported();
        let rendered = report.render();

        // Assert
        assert!(!supported);
        assert!(rendered.contains("[FAIL] finding (required): detail"));
        assert!(rendered.contains("bubblewrap: not resolved"));
    }
}
