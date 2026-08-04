use std::convert::Infallible;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};

use crate::{
    ArchiveError, ArchiveRefusal, ConfigFile, ConfigFileError, DEFAULT_LOCK_TIMEOUT, DocumentCodec,
    InvalidDocument, LoadFailure, LoadOutcome, PersistenceError, RefreshOutcome, UpdateError,
    generate_schema, pretty_json,
};

const CURRENT_VERSION: u64 = 1;
const PINNED_SCHEMA: &str = r##"{
  "$defs": {
    "SchemaV0": {
      "description": "An older test configuration document.",
      "properties": {
        "schema_version": {
          "const": 0,
          "type": "integer"
        },
        "value": {
          "type": "string"
        }
      },
      "required": [
        "schema_version",
        "value"
      ],
      "type": "object"
    },
    "SchemaV1": {
      "description": "The current test configuration document.",
      "properties": {
        "schema_version": {
          "const": 1,
          "type": "integer"
        },
        "values": {
          "items": {
            "type": "string"
          },
          "type": "array"
        }
      },
      "required": [
        "schema_version",
        "values"
      ],
      "type": "object"
    }
  },
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "anyOf": [
    {
      "$ref": "#/$defs/SchemaV0"
    },
    {
      "$ref": "#/$defs/SchemaV1"
    }
  ],
  "title": "SupportedSchema"
}
"##;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CurrentDocument {
    schema_version: u64,
    values: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OldDocument {
    schema_version: u64,
    value: String,
}

#[derive(Debug, Clone, Copy)]
struct TestCodec;

impl DocumentCodec for TestCodec {
    type Current = CurrentDocument;
    type Error = String;

    fn current_version(&self) -> u64 {
        CURRENT_VERSION
    }

    fn supports_version(&self, version: u64) -> bool {
        matches!(version, 0 | CURRENT_VERSION)
    }

    fn decode(&self, version: u64, document: Value) -> Result<Self::Current, Self::Error> {
        match version {
            0 => {
                let old = serde_json::from_value::<OldDocument>(document)
                    .map_err(|error| error.to_string())?;
                if old.schema_version != version {
                    return Err("decoded version did not match discriminator".to_owned());
                }
                Ok(CurrentDocument {
                    schema_version: CURRENT_VERSION,
                    values: vec![old.value],
                })
            }
            CURRENT_VERSION => serde_json::from_value(document).map_err(|error| error.to_string()),
            _ => unreachable!("unsupported versions are rejected before decode"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LimitedCodec;

impl DocumentCodec for LimitedCodec {
    type Current = CurrentDocument;
    type Error = String;

    fn current_version(&self) -> u64 {
        TestCodec.current_version()
    }

    fn supports_version(&self, version: u64) -> bool {
        TestCodec.supports_version(version)
    }

    fn decode(&self, version: u64, document: Value) -> Result<Self::Current, Self::Error> {
        let document = TestCodec.decode(version, document)?;
        if document.values.len() > 1 {
            return Err("too many values".to_owned());
        }
        Ok(document)
    }
}

#[derive(Debug, Clone, Copy)]
struct UnsupportedCurrentCodec;

impl DocumentCodec for UnsupportedCurrentCodec {
    type Current = CurrentDocument;
    type Error = String;

    fn current_version(&self) -> u64 {
        2
    }

    fn supports_version(&self, version: u64) -> bool {
        version == CURRENT_VERSION
    }

    fn decode(&self, _version: u64, _document: Value) -> Result<Self::Current, Self::Error> {
        panic!("an unsupported current version must be rejected before decoding")
    }
}

#[derive(Debug, JsonSchema)]
#[serde(untagged)]
enum SupportedSchema {
    Old(SchemaV0),
    Current(SchemaV1),
}

#[derive(Debug, JsonSchema)]
#[schemars(description = "An older test configuration document.")]
struct SchemaV0 {
    #[schemars(schema_with = "version_zero_schema")]
    schema_version: u64,
    value: String,
}

#[derive(Debug, JsonSchema)]
#[schemars(description = "The current test configuration document.")]
struct SchemaV1 {
    #[schemars(schema_with = "version_one_schema")]
    schema_version: u64,
    values: Vec<String>,
}

fn version_zero_schema(_generator: &mut SchemaGenerator) -> Schema {
    json_schema!({"type": "integer", "const": 0})
}

fn version_one_schema(_generator: &mut SchemaGenerator) -> Schema {
    json_schema!({"type": "integer", "const": 1})
}

fn fixture(root: &TempDir) -> ConfigFile {
    let directory = root.path().join("config");
    ConfigFile::new(directory.join("settings.json")).unwrap()
}

fn current(values: &[&str]) -> CurrentDocument {
    CurrentDocument {
        schema_version: CURRENT_VERSION,
        values: values.iter().map(|value| (*value).to_owned()).collect(),
    }
}

fn write(path: &Path, contents: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

fn spawn_lock_holder(config: &ConfigFile, hold_for: Duration) -> Child {
    let marker = config.document_path().with_extension("lock-holder-ready");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "tests::child_process_holds_writer_lock",
        ])
        .env("CANE_CONFIG_TEST_DOCUMENT", config.document_path())
        .env("CANE_CONFIG_TEST_LOCK_MARKER", &marker)
        .env(
            "CANE_CONFIG_TEST_LOCK_HOLD_MILLIS",
            hold_for.as_millis().to_string(),
        )
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    let started = Instant::now();
    while !marker.exists() {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("lock-holder subprocess exited before acquiring the lock: {status}");
        }
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "timed out waiting for lock-holder subprocess"
        );
        thread::sleep(Duration::from_millis(5));
    }
    child
}

#[test]
fn sidecar_paths_are_derived_and_distinct() {
    // Arrange
    let root = tempdir().unwrap();
    let document = root.path().join("config/settings.json");

    // Act
    let config = ConfigFile::new(&document).unwrap();

    // Assert
    assert_eq!(config.document_path(), document);
    assert_eq!(config.lock_path(), root.path().join("config/settings.lock"));
    assert_eq!(
        config.schema_path(),
        root.path().join("config/settings.schema.json")
    );
    assert_ne!(config.document_path(), config.lock_path());
    assert_ne!(config.document_path(), config.schema_path());
    assert_ne!(config.lock_path(), config.schema_path());
    assert_eq!(config.lock_timeout(), DEFAULT_LOCK_TIMEOUT);
}

#[test]
fn constructor_rejects_a_document_that_would_alias_its_lock_sidecar() {
    for file_name in ["settings.lock", "settings.LOCK"] {
        // Arrange
        let root = tempdir().unwrap();
        let document = root.path().join("config").join(file_name);

        // Act
        let result = ConfigFile::new(&document);

        // Assert
        assert!(matches!(
            result,
            Err(ConfigFileError::DerivedPathCollision {
                document: rejected,
                sidecar_kind: "lock"
            }) if rejected == document
        ));
    }
}

#[test]
fn missing_document_has_a_distinct_outcome() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);

    // Act
    let outcome = config.load(&TestCodec);

    // Assert
    assert!(matches!(outcome, LoadOutcome::Missing));
}

#[test]
fn malformed_and_duplicate_key_documents_are_invalid() {
    for contents in [
        br#"{"schema_version":1,"values":["unfinished"]"#.as_slice(),
        br#"{"schema_version":1,"schema_version":1,"values":[]}"#.as_slice(),
    ] {
        // Arrange
        let root = tempdir().unwrap();
        let config = fixture(&root);
        write(config.document_path(), contents);

        // Act
        let outcome = config.load(&TestCodec);

        // Assert
        assert!(matches!(
            outcome,
            LoadOutcome::Invalid(InvalidDocument::MalformedJson(_))
        ));
    }
}

#[test]
fn current_document_loads_into_the_current_model() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    write(
        config.document_path(),
        br#"{"schema_version":1,"values":["current"]}"#,
    );

    // Act
    let outcome = config.load(&TestCodec);

    // Assert
    let LoadOutcome::Loaded(loaded) = outcome else {
        panic!("expected a loaded document");
    };
    assert_eq!(loaded.source_version, CURRENT_VERSION);
    assert_eq!(loaded.document, current(&["current"]));
    assert!(!loaded.was_migrated(&TestCodec));
}

#[test]
fn supported_old_document_migrates_without_rewriting_during_load() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    let persisted = br#"{"schema_version":0,"value":"old formatting"}"#;
    write(config.document_path(), persisted);

    // Act
    let outcome = config.load(&TestCodec);

    // Assert
    let LoadOutcome::Loaded(loaded) = outcome else {
        panic!("expected a loaded document");
    };
    assert_eq!(loaded.document, current(&["old formatting"]));
    assert!(loaded.was_migrated(&TestCodec));
    assert_eq!(fs::read(config.document_path()).unwrap(), persisted);
}

#[test]
fn next_successful_update_after_migration_writes_the_current_version() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    write(
        config.document_path(),
        br#"{"schema_version":0,"value":"old"}"#,
    );

    // Act
    config
        .update(&TestCodec, |existing| {
            let mut document = existing.unwrap();
            document.values.push("new".to_owned());
            Ok::<_, Infallible>(document)
        })
        .unwrap();

    // Assert
    let persisted: Value =
        serde_json::from_slice(&fs::read(config.document_path()).unwrap()).unwrap();
    assert_eq!(persisted["schema_version"], CURRENT_VERSION);
    assert_eq!(persisted["values"], json!(["old", "new"]));
}

#[test]
fn unsupported_document_has_a_distinct_outcome() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    write(
        config.document_path(),
        br#"{"schema_version":99,"future":true}"#,
    );

    // Act
    let outcome = config.load(&TestCodec);

    // Assert
    assert!(matches!(
        outcome,
        LoadOutcome::UnsupportedVersion { version: 99 }
    ));
}

#[test]
fn unreadable_shape_has_a_distinct_io_outcome() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    fs::create_dir_all(config.document_path()).unwrap();

    // Act
    let outcome = config.load(&TestCodec);

    // Assert
    let LoadOutcome::Io(error) = outcome else {
        panic!("expected an I/O outcome");
    };
    assert_eq!(error.operation(), "read");
    assert_eq!(error.path(), config.document_path());
}

#[test]
fn pretty_json_is_canonical_and_has_one_trailing_newline() {
    // Arrange
    let first = json!({"z": {"second": 2, "first": 1}, "a": true});
    let second = json!({"a": true, "z": {"first": 1, "second": 2}});

    // Act
    let first_bytes = pretty_json(&first).unwrap();
    let second_bytes = pretty_json(&second).unwrap();

    // Assert
    assert_eq!(first_bytes, second_bytes);
    assert_eq!(
        String::from_utf8(first_bytes).unwrap(),
        "{\n  \"a\": true,\n  \"z\": {\n    \"first\": 1,\n    \"second\": 2\n  }\n}\n"
    );
}

#[test]
fn generated_schema_is_pinned_to_draft_2020_12() {
    // Arrange
    let representative_values = [
        SupportedSchema::Old(SchemaV0 {
            schema_version: 0,
            value: String::new(),
        }),
        SupportedSchema::Current(SchemaV1 {
            schema_version: 1,
            values: Vec::new(),
        }),
    ];
    for value in representative_values {
        match value {
            SupportedSchema::Old(document) => {
                assert_eq!(document.schema_version, 0);
                assert!(document.value.is_empty());
            }
            SupportedSchema::Current(document) => {
                assert_eq!(document.schema_version, 1);
                assert!(document.values.is_empty());
            }
        }
    }

    // Act
    let bytes = generate_schema::<SupportedSchema>().unwrap();
    let schema: Value = serde_json::from_slice(&bytes).unwrap();

    // Assert
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert_eq!(bytes.last(), Some(&b'\n'));
    assert_eq!(bytes, PINNED_SCHEMA.as_bytes());
}

#[test]
fn schema_refresh_writes_only_when_content_differs() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);

    // Act
    let created = config.refresh_schema::<SupportedSchema>().unwrap();
    #[cfg(unix)]
    let original_inode = inode(config.schema_path());
    let unchanged = config.refresh_schema::<SupportedSchema>().unwrap();
    write(config.schema_path(), b"stale schema\n");
    let refreshed = config.refresh_schema::<SupportedSchema>().unwrap();

    // Assert
    assert_eq!(created, RefreshOutcome::Written);
    assert_eq!(unchanged, RefreshOutcome::Unchanged);
    assert_eq!(refreshed, RefreshOutcome::Written);
    #[cfg(unix)]
    assert_ne!(original_inode, inode(config.schema_path()));
    assert_eq!(
        fs::read(config.schema_path()).unwrap(),
        generate_schema::<SupportedSchema>().unwrap()
    );
}

#[test]
fn schema_refresh_reports_an_unreadable_existing_schema() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    fs::create_dir_all(config.schema_path()).unwrap();

    // Act
    let result = config.refresh_schema::<SupportedSchema>();

    // Assert
    assert!(matches!(
        result,
        Err(PersistenceError::Io {
            operation: "read schema",
            path,
            ..
        }) if path == config.schema_path()
    ));
}

#[test]
fn concurrent_updates_reload_after_taking_the_writer_lock() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    config
        .update(&TestCodec, |_| Ok::<_, Infallible>(current(&[])))
        .unwrap();
    let barrier = Arc::new(Barrier::new(9));
    let mut threads = Vec::new();
    for index in 0..8 {
        let config = config.clone();
        let barrier = Arc::clone(&barrier);
        threads.push(thread::spawn(move || {
            barrier.wait();
            config
                .update(&TestCodec, |existing| {
                    let mut document = existing.unwrap();
                    document.values.push(index.to_string());
                    Ok::<_, Infallible>(document)
                })
                .unwrap();
        }));
    }

    // Act
    barrier.wait();
    for thread in threads {
        thread.join().unwrap();
    }

    // Assert
    let LoadOutcome::Loaded(loaded) = config.load(&TestCodec) else {
        panic!("expected a loaded document");
    };
    let mut actual = loaded.document.values;
    actual.sort();
    assert_eq!(actual, ["0", "1", "2", "3", "4", "5", "6", "7"]);
}

#[test]
fn writer_lock_is_released_when_a_process_exits_without_dropping_it() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "tests::child_process_exits_while_holding_lock",
        ])
        .env("CANE_CONFIG_TEST_DOCUMENT", config.document_path())
        .status()
        .unwrap();
    assert!(status.success());

    // Act
    let result = config.update(&TestCodec, |_| {
        Ok::<_, Infallible>(current(&["after process exit"]))
    });

    // Assert
    assert!(result.is_ok());
}

#[test]
fn writer_lock_timeout_preserves_the_document_and_skips_mutation() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    config
        .update(&TestCodec, |_| Ok::<_, Infallible>(current(&["unchanged"])))
        .unwrap();
    let original = fs::read(config.document_path()).unwrap();
    let mut holder = spawn_lock_holder(&config, Duration::from_millis(300));
    let timeout = Duration::from_millis(50);
    let bounded = config.clone().with_lock_timeout(timeout);
    let mutated = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mutation_observed = Arc::clone(&mutated);

    // Act
    let result = bounded.update(&TestCodec, move |_| {
        mutation_observed.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok::<_, Infallible>(current(&["changed"]))
    });

    // Assert
    let Err(UpdateError::Persistence(PersistenceError::LockTimeout { path, waited })) = result
    else {
        panic!("expected a writer-lock timeout");
    };
    assert_eq!(path, config.lock_path());
    assert!(waited >= timeout);
    assert!(!mutated.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(fs::read(config.document_path()).unwrap(), original);
    assert!(config.lock_path().exists());
    assert!(holder.wait().unwrap().success());
}

#[test]
fn writer_lock_acquisition_succeeds_when_holder_releases_before_timeout() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    config
        .update(&TestCodec, |_| Ok::<_, Infallible>(current(&["before"])))
        .unwrap();
    let mut holder = spawn_lock_holder(&config, Duration::from_millis(75));
    let bounded = config.clone().with_lock_timeout(Duration::from_millis(500));

    // Act
    let result = bounded.update(&TestCodec, |_| {
        Ok::<_, Infallible>(current(&["after release"]))
    });

    // Assert
    assert_eq!(result.unwrap(), current(&["after release"]));
    assert!(holder.wait().unwrap().success());
    let LoadOutcome::Loaded(loaded) = config.load(&TestCodec) else {
        panic!("expected a loaded document");
    };
    assert_eq!(loaded.document, current(&["after release"]));
}

#[test]
#[ignore = "helper subprocess selected explicitly by the process-lock test"]
fn child_process_exits_while_holding_lock() {
    let Some(document) = std::env::var_os("CANE_CONFIG_TEST_DOCUMENT") else {
        return;
    };
    let config = ConfigFile::new(PathBuf::from(document)).unwrap();
    let lock = config.acquire_writer_lock().unwrap();
    std::mem::forget(lock);
    std::process::exit(0);
}

#[test]
#[ignore = "helper subprocess selected explicitly by writer-lock contention tests"]
fn child_process_holds_writer_lock() {
    let Some(document) = std::env::var_os("CANE_CONFIG_TEST_DOCUMENT") else {
        return;
    };
    let config = ConfigFile::new(PathBuf::from(document)).unwrap();
    let lock = config.acquire_writer_lock().unwrap();
    fs::write(
        std::env::var_os("CANE_CONFIG_TEST_LOCK_MARKER").unwrap(),
        b"ready",
    )
    .unwrap();
    let hold_millis = std::env::var("CANE_CONFIG_TEST_LOCK_HOLD_MILLIS")
        .unwrap()
        .parse::<u64>()
        .unwrap();
    thread::sleep(Duration::from_millis(hold_millis));
    drop(lock);
}

#[cfg(unix)]
#[test]
fn unix_permissions_are_created_and_normalized_restrictively() {
    use std::os::unix::fs::PermissionsExt;

    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    fs::create_dir_all(config.document_path().parent().unwrap()).unwrap();
    fs::set_permissions(
        config.document_path().parent().unwrap(),
        fs::Permissions::from_mode(0o777),
    )
    .unwrap();
    write(
        config.document_path(),
        br#"{"schema_version":1,"values":[]}"#,
    );
    fs::set_permissions(config.document_path(), fs::Permissions::from_mode(0o666)).unwrap();

    // Act
    assert!(matches!(config.load(&TestCodec), LoadOutcome::Loaded(_)));
    config.refresh_schema::<SupportedSchema>().unwrap();
    config
        .update(&TestCodec, |existing| {
            Ok::<_, Infallible>(existing.unwrap())
        })
        .unwrap();

    // Assert
    assert_eq!(
        fs::metadata(config.document_path().parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    for path in [
        config.document_path(),
        config.lock_path(),
        config.schema_path(),
    ] {
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600,
            "unexpected mode for {}",
            path.display()
        );
    }
}

#[cfg(unix)]
#[test]
fn atomic_update_replaces_the_name_but_preserves_an_open_old_file() {
    use std::io::Read;

    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    config
        .update(&TestCodec, |_| Ok::<_, Infallible>(current(&["old"])))
        .unwrap();
    let old_inode = inode(config.document_path());
    let mut old_file = fs::File::open(config.document_path()).unwrap();

    // Act
    config
        .update(&TestCodec, |_| Ok::<_, Infallible>(current(&["new"])))
        .unwrap();

    // Assert
    let mut old_bytes = Vec::new();
    old_file.read_to_end(&mut old_bytes).unwrap();
    assert!(String::from_utf8(old_bytes).unwrap().contains("\"old\""));
    assert_ne!(old_inode, inode(config.document_path()));
    let LoadOutcome::Loaded(loaded) = config.load(&TestCodec) else {
        panic!("expected a loaded document");
    };
    assert_eq!(loaded.document, current(&["new"]));
}

#[test]
fn atomic_replacement_failure_is_reported_with_its_operation() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    fs::create_dir_all(config.document_path().parent().unwrap()).unwrap();

    // Act
    let result = config.update(&TestCodec, |_| {
        fs::create_dir(config.document_path()).unwrap();
        Ok::<_, Infallible>(current(&["cannot replace a directory"]))
    });

    // Assert
    let Err(UpdateError::Persistence(PersistenceError::Io {
        operation, path, ..
    })) = result
    else {
        panic!("expected an atomic replacement failure");
    };
    assert_eq!(operation, "atomically replace");
    assert_eq!(path, config.document_path());
}

#[test]
fn locked_update_refuses_an_invalid_document_before_mutation() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    write(config.document_path(), b"not json");
    let mutated = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mutation_observed = Arc::clone(&mutated);

    // Act
    let result = config.update(&TestCodec, move |_| {
        mutation_observed.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok::<_, Infallible>(current(&[]))
    });

    // Assert
    assert!(matches!(
        result,
        Err(UpdateError::Load(LoadFailure::Invalid(_)))
    ));
    assert!(!mutated.load(std::sync::atomic::Ordering::SeqCst));
}

#[test]
fn update_rejects_a_model_that_does_not_emit_the_current_version() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);

    // Act
    let result = config.update(&TestCodec, |_| {
        Ok::<_, Infallible>(CurrentDocument {
            schema_version: 0,
            values: Vec::new(),
        })
    });

    // Assert
    assert!(matches!(
        result,
        Err(UpdateError::Persistence(PersistenceError::CurrentVersion {
            expected: CURRENT_VERSION,
            found: Some(0)
        }))
    ));
    assert!(!config.document_path().exists());
}

#[test]
fn update_rejects_a_codec_that_does_not_support_its_current_version() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);

    // Act
    let result = config.update(&UnsupportedCurrentCodec, |_| {
        Ok::<_, Infallible>(CurrentDocument {
            schema_version: 2,
            values: Vec::new(),
        })
    });

    // Assert
    assert!(matches!(
        result,
        Err(UpdateError::UnsupportedCurrentVersion { version: 2 })
    ));
    assert!(!config.document_path().exists());
}

#[test]
fn update_rejects_a_current_document_that_the_codec_cannot_load() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    config
        .update(&LimitedCodec, |_| {
            Ok::<_, Infallible>(current(&["existing"]))
        })
        .unwrap();
    let original = fs::read(config.document_path()).unwrap();

    // Act
    let result = config.update(&LimitedCodec, |existing| {
        assert_eq!(existing.unwrap(), current(&["existing"]));
        Ok::<_, Infallible>(current(&["one", "two"]))
    });

    // Assert
    assert!(matches!(
        result,
        Err(UpdateError::InvalidCurrent(error)) if error == "too many values"
    ));
    assert_eq!(fs::read(config.document_path()).unwrap(), original);
}

#[test]
fn invalid_document_is_archived_only_by_explicit_operation() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    let bytes = b"{ definitely invalid";
    write(config.document_path(), bytes);
    let archive = config.document_path().with_extension("invalid.json");

    // Act
    config.archive_invalid(&TestCodec, &archive).unwrap();

    // Assert
    assert!(!config.document_path().exists());
    assert_eq!(fs::read(archive).unwrap(), bytes);
}

#[test]
fn archiving_refuses_valid_and_unsupported_documents() {
    for (contents, expected) in [
        (
            br#"{"schema_version":1,"values":[]}"#.as_slice(),
            ArchiveRefusal::Valid,
        ),
        (
            br#"{"schema_version":42}"#.as_slice(),
            ArchiveRefusal::UnsupportedVersion { version: 42 },
        ),
    ] {
        // Arrange
        let root = tempdir().unwrap();
        let config = fixture(&root);
        write(config.document_path(), contents);
        let archive = config.document_path().with_extension("invalid.json");

        // Act
        let result = config.archive_invalid(&TestCodec, &archive);

        // Assert
        assert!(matches!(
            result,
            Err(ArchiveError::Refused(actual)) if actual == expected
        ));
        assert!(config.document_path().exists());
        assert!(!archive.exists());
    }
}

#[test]
fn archiving_rejects_a_nonadjacent_destination_without_moving_the_document() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    let bytes = b"{ invalid";
    write(config.document_path(), bytes);
    let archive = root.path().join("elsewhere/invalid.json");

    // Act
    let result = config.archive_invalid(&TestCodec, &archive);

    // Assert
    assert!(matches!(
        result,
        Err(ArchiveError::Persistence(
            PersistenceError::ArchiveNotAdjacent { .. }
        ))
    ));
    assert_eq!(fs::read(config.document_path()).unwrap(), bytes);
    assert!(!archive.exists());
}

#[test]
fn archiving_rejects_an_existing_destination_without_moving_the_document() {
    // Arrange
    let root = tempdir().unwrap();
    let config = fixture(&root);
    let bytes = b"{ invalid";
    write(config.document_path(), bytes);
    let archive = config.document_path().with_extension("invalid.json");
    write(&archive, b"keep me");

    // Act
    let result = config.archive_invalid(&TestCodec, &archive);

    // Assert
    assert!(matches!(
        result,
        Err(ArchiveError::Persistence(PersistenceError::Io {
            operation: "create archive",
            path,
            ..
        })) if path == archive
    ));
    assert_eq!(fs::read(config.document_path()).unwrap(), bytes);
    assert_eq!(fs::read(archive).unwrap(), b"keep me");
}

#[cfg(unix)]
fn inode(path: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;

    fs::metadata(path).unwrap().ino()
}
