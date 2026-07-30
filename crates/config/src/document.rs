use std::collections::HashSet;
use std::fmt;
use std::io;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Serialize, Serializer};
use serde_json::{Number, Value};
use thiserror::Error;

/// Converts supported persisted document versions into one current in-memory model.
///
/// Structural and semantic validation beyond JSON syntax and `schema_version` belongs in
/// [`DocumentCodec::decode`]. Loading calls this method without writing the document back.
pub trait DocumentCodec {
    /// The current in-memory document type.
    type Current: Serialize;

    /// A document-specific structural, migration, or semantic error.
    type Error;

    /// The version emitted when the current model is saved.
    fn current_version(&self) -> u64;

    /// Whether `version` can be decoded and migrated by this codec.
    fn supports_version(&self, version: u64) -> bool;

    /// Decodes and validates `document`, migrating it in memory when `version` is older.
    ///
    /// Locked updates call this for the current version before writing, so it must enforce the
    /// same structural and semantic rules for current documents as it does during loading.
    fn decode(&self, version: u64, document: Value) -> Result<Self::Current, Self::Error>;
}

/// A successfully loaded current in-memory document.
#[derive(Debug, PartialEq, Eq)]
pub struct LoadedDocument<T> {
    /// The version found in the persisted document.
    pub source_version: u64,
    /// The decoded current model.
    pub document: T,
}

impl<T> LoadedDocument<T> {
    /// Reports whether loading migrated an older supported version in memory.
    pub fn was_migrated<C>(&self, codec: &C) -> bool
    where
        C: DocumentCodec<Current = T>,
    {
        self.source_version != codec.current_version()
    }
}

/// A malformed or absent `schema_version` member.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum InvalidVersion {
    /// The root JSON value was not an object.
    #[error("the document root must be a JSON object")]
    RootNotObject,
    /// The root object did not contain `schema_version`.
    #[error("the document is missing schema_version")]
    Missing,
    /// `schema_version` was not an unsigned integer representable as `u64`.
    #[error("schema_version must be an unsigned integer")]
    NotUnsignedInteger,
}

/// Why an existing document could not be treated as valid.
#[derive(Debug)]
pub enum InvalidDocument<E> {
    /// The bytes were not valid duplicate-free JSON.
    MalformedJson(serde_json::Error),
    /// The version discriminator was absent or malformed.
    InvalidVersion(InvalidVersion),
    /// The document-specific codec rejected a supported version.
    Rejected(E),
}

/// An I/O failure encountered while loading a document.
#[derive(Debug, Error)]
#[error("failed to {operation} {path}: {source}")]
pub struct LoadIoError {
    pub(crate) operation: &'static str,
    pub(crate) path: std::path::PathBuf,
    #[source]
    pub(crate) source: io::Error,
}

impl LoadIoError {
    /// The filesystem operation that failed.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// The path involved in the failed operation.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// The underlying operating-system error.
    pub fn source_error(&self) -> &io::Error {
        &self.source
    }
}

/// Every meaningful outcome from loading a typed versioned document.
#[derive(Debug)]
pub enum LoadOutcome<T, E> {
    /// The document does not exist.
    Missing,
    /// A supported version decoded into the current in-memory model.
    Loaded(LoadedDocument<T>),
    /// Existing bytes were malformed or rejected.
    Invalid(InvalidDocument<E>),
    /// The version discriminator is well-formed but not supported.
    UnsupportedVersion { version: u64 },
    /// The document could not be read or its required permissions could not be established.
    Io(LoadIoError),
}

/// A non-successful load used by locked update operations.
#[derive(Debug)]
pub enum LoadFailure<E> {
    /// Existing bytes were malformed or rejected.
    Invalid(InvalidDocument<E>),
    /// The version discriminator is well-formed but unsupported.
    UnsupportedVersion { version: u64 },
    /// A filesystem operation failed.
    Io(LoadIoError),
}

pub(crate) fn decode_document<C>(
    bytes: &[u8],
    codec: &C,
) -> Result<LoadedDocument<C::Current>, DecodeFailure<C::Error>>
where
    C: DocumentCodec,
{
    let unique = serde_json::from_slice::<UniqueValue>(bytes)
        .map_err(|error| DecodeFailure::Invalid(InvalidDocument::MalformedJson(error)))?;
    let value = unique.0;
    let version = extract_version(&value)
        .map_err(|error| DecodeFailure::Invalid(InvalidDocument::InvalidVersion(error)))?;

    if !codec.supports_version(version) {
        return Err(DecodeFailure::UnsupportedVersion(version));
    }

    let document = codec
        .decode(version, value)
        .map_err(|error| DecodeFailure::Invalid(InvalidDocument::Rejected(error)))?;

    Ok(LoadedDocument {
        source_version: version,
        document,
    })
}

pub(crate) enum DecodeFailure<E> {
    Invalid(InvalidDocument<E>),
    UnsupportedVersion(u64),
}

fn extract_version(value: &Value) -> Result<u64, InvalidVersion> {
    let object = value.as_object().ok_or(InvalidVersion::RootNotObject)?;
    let version = object
        .get("schema_version")
        .ok_or(InvalidVersion::Missing)?;
    version.as_u64().ok_or(InvalidVersion::NotUnsignedInteger)
}

/// Serializes JSON with stable object-key ordering, two-space indentation, and one trailing
/// newline.
pub fn pretty_json<T>(value: &T) -> Result<Vec<u8>, serde_json::Error>
where
    T: Serialize,
{
    let mut value = serde_json::to_value(value)?;
    sort_object_keys(&mut value);
    pretty_json_value(&value)
}

pub(crate) fn pretty_json_value(value: &Value) -> Result<Vec<u8>, serde_json::Error> {
    let mut value = value.clone();
    sort_object_keys(&mut value);
    let mut bytes = serde_json::to_vec_pretty(&value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn sort_object_keys(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for value in values {
                sort_object_keys(value);
            }
        }
        Value::Object(object) => {
            let old = std::mem::take(object);
            let mut entries = old.into_iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            for (key, mut value) in entries {
                sort_object_keys(&mut value);
                object.insert(key, value);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

struct UniqueValue(Value);

impl Serialize for UniqueValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueValueVisitor)
    }
}

struct UniqueValueVisitor;

impl<'de> Visitor<'de> for UniqueValueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .map(UniqueValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<UniqueValue>()? {
            values.push(value.0);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = HashSet::new();
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate object key {key:?}")));
            }
            let value = map.next_value::<UniqueValue>()?;
            values.insert(key, value.0);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}
