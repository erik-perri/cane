use std::ffi::OsStr;
use std::process::{Command, Output, Stdio};
use tempfile::tempdir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn cane_command(cane_home: &OsStr) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_cane"));
    command
        .env_clear()
        .env("CANE_HOME", cane_home)
        .env("RUST_LOG", "off")
        .stdin(Stdio::null());
    if let Some(profile) = std::env::var_os("LLVM_PROFILE_FILE") {
        command.env("LLVM_PROFILE_FILE", profile);
    }
    command
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn chat_startup_reports_each_missing_required_provider_variable() {
    let root = tempdir().unwrap();
    let cases: &[(&[(&str, &str)], &str)] = &[
        (&[], "CANE_API_KEY not set"),
        (&[("CANE_API_KEY", "test-key")], "CANE_BASE_URL not set"),
        (
            &[
                ("CANE_API_KEY", "test-key"),
                ("CANE_BASE_URL", "http://127.0.0.1:1"),
            ],
            "CANE_MODEL not set",
        ),
    ];

    for &(environment, expected) in cases {
        let output = cane_command(root.path().as_os_str())
            .arg("--no-shell")
            .envs(environment.iter().copied())
            .output()
            .unwrap();

        assert!(!output.status.success());
        assert!(
            stderr(&output).contains(expected),
            "stderr: {}",
            stderr(&output)
        );
    }
}

#[test]
fn chat_startup_rejects_an_invalid_max_tokens_value() {
    let root = tempdir().unwrap();
    let output = cane_command(root.path().as_os_str())
        .arg("--no-shell")
        .env("CANE_API_KEY", "test-key")
        .env("CANE_BASE_URL", "http://127.0.0.1:1")
        .env("CANE_MODEL", "test-model")
        .env("CANE_MAX_TOKENS", "many")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("CANE_MAX_TOKENS must be an integer"),
        "stderr: {}",
        stderr(&output)
    );
}

#[test]
fn invalid_cli_arguments_fail_before_provider_configuration_is_read() {
    let root = tempdir().unwrap();
    let output = cane_command(root.path().as_os_str())
        .args(["--no-shell", "--unsafe-shell"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("--no-shell and --unsafe-shell are mutually exclusive"),
        "stderr: {}",
        stderr(&output)
    );
    assert!(!stderr(&output).contains("CANE_API_KEY"));
}

#[test]
fn doctor_prints_a_report_and_sets_status_from_required_findings() {
    let root = tempdir().unwrap();
    let output = cane_command(root.path().as_os_str())
        .arg("--doctor")
        .current_dir(root.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let has_required_failure = stdout
        .lines()
        .any(|line| line.contains("[FAIL]") && line.contains("(required):"));

    assert!(stdout.starts_with("Sandbox\n"), "stdout: {stdout}");
    assert!(stdout.contains("  platform:"), "stdout: {stdout}");
    assert!(stdout.contains("  backend:"), "stdout: {stdout}");
    assert_eq!(
        output.status.success(),
        !has_required_failure,
        "stdout: {stdout}"
    );
}

#[tokio::test]
async fn no_shell_chat_completes_a_turn_against_the_configured_provider() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"goodbye\"},\"finish_reason\":\"stop\",\"index\":0}]}\n\n",
                    "data: [DONE]\n\n"
                )),
        )
        .expect(1)
        .mount(&server)
        .await;

    let root = tempdir().unwrap();
    let mut child = cane_command(root.path().as_os_str())
        .arg("--no-shell")
        .env("CANE_API_KEY", "test-key")
        .env("CANE_BASE_URL", server.uri())
        .env("CANE_MODEL", "test-model")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write;
    child.stdin.take().unwrap().write_all(b"hello\n").unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(output.status.success(), "stderr: {}", stderr(&output));
    assert!(String::from_utf8_lossy(&output.stdout).contains("goodbye"));
}
