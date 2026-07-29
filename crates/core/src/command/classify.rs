#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandClassification {
    Complex,
    Simple(SimpleCommand),
}

impl CommandClassification {
    pub fn is_direct_docker_invocation(&self) -> bool {
        let Self::Simple(command) = self else {
            return false;
        };

        matches!(command.executable(), "docker" | "docker-compose")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimpleCommand {
    executable_index: usize,
    words: Vec<String>,
}

impl SimpleCommand {
    pub fn executable(&self) -> &str {
        &self.words[self.executable_index]
    }

    pub fn words(&self) -> &[String] {
        &self.words
    }
}

pub fn classify_command(command: &str) -> CommandClassification {
    parse_simple_command(command).map_or(
        CommandClassification::Complex,
        CommandClassification::Simple,
    )
}

#[derive(Clone, Copy)]
enum Quote {
    Double,
    Single,
    Unquoted,
}

fn parse_simple_command(command: &str) -> Option<SimpleCommand> {
    let mut characters = command.chars().peekable();
    let mut current = String::new();
    let mut quote = Quote::Unquoted;
    let mut word_started = false;
    let mut words = Vec::new();

    while let Some(character) = characters.next() {
        if matches!(character, '\n' | '\r') {
            return None;
        }

        match quote {
            Quote::Single => {
                if character == '\'' {
                    quote = Quote::Unquoted;
                } else {
                    current.push(character);
                }
            }
            Quote::Double => match character {
                '"' => quote = Quote::Unquoted,
                '$' | '`' => return None,
                '\\' => {
                    let escaped = characters.next()?;

                    if matches!(escaped, '\n' | '\r') {
                        return None;
                    }

                    if matches!(escaped, '$' | '`' | '"' | '\\') {
                        current.push(escaped);
                    } else {
                        current.push('\\');
                        current.push(escaped);
                    }
                }
                _ => current.push(character),
            },
            Quote::Unquoted => match character {
                '\'' => {
                    quote = Quote::Single;
                    word_started = true;
                }
                '"' => {
                    quote = Quote::Double;
                    word_started = true;
                }
                '\\' => {
                    let escaped = characters.next()?;

                    if matches!(escaped, '\n' | '\r') {
                        return None;
                    }

                    current.push(escaped);
                    word_started = true;
                }
                '$' | '`' | '|' | '&' | ';' | '<' | '>' | '(' | ')' | '{' | '}' | '*' | '?'
                | '[' | ']' | '~' => {
                    return None;
                }
                '#' if !word_started => return None,
                character if character.is_whitespace() => {
                    finish_word(&mut current, &mut word_started, &mut words);
                }
                _ => {
                    current.push(character);
                    word_started = true;
                }
            },
        }
    }

    if !matches!(quote, Quote::Unquoted) {
        return None;
    }

    finish_word(&mut current, &mut word_started, &mut words);

    let executable_index = words.iter().position(|word| !is_assignment(word))?;

    if is_reserved_word(&words[executable_index]) {
        return None;
    }

    Some(SimpleCommand {
        executable_index,
        words,
    })
}

fn finish_word(current: &mut String, word_started: &mut bool, words: &mut Vec<String>) {
    if *word_started {
        words.push(std::mem::take(current));
        *word_started = false;
    }
}

fn is_assignment(word: &str) -> bool {
    let Some((name, _)) = word.split_once('=') else {
        return false;
    };

    let mut characters = name.chars();

    matches!(characters.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn is_reserved_word(word: &str) -> bool {
    matches!(
        word,
        "!" | "case"
            | "coproc"
            | "do"
            | "done"
            | "elif"
            | "else"
            | "esac"
            | "fi"
            | "for"
            | "function"
            | "if"
            | "in"
            | "select"
            | "then"
            | "time"
            | "until"
            | "while"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_command_normalizes_a_simple_command_into_words() {
        // Arrange
        let command = r#"RUST_LOG=debug cargo test "package name" escaped\ value ''"#;

        // Act
        let classification = classify_command(command);

        // Assert
        assert_eq!(
            classification,
            CommandClassification::Simple(SimpleCommand {
                executable_index: 1,
                words: vec![
                    "RUST_LOG=debug".to_string(),
                    "cargo".to_string(),
                    "test".to_string(),
                    "package name".to_string(),
                    "escaped value".to_string(),
                    String::new(),
                ],
            })
        );
    }

    #[test]
    fn simple_command_identifies_the_executable_after_environment_assignments() {
        // Arrange
        let command = "RUST_LOG=debug CARGO_TERM_COLOR=never cargo test";

        // Act
        let CommandClassification::Simple(simple) = classify_command(command) else {
            panic!("expected a simple command");
        };

        // Assert
        assert_eq!(simple.executable(), "cargo");
        assert_eq!(
            simple.words(),
            ["RUST_LOG=debug", "CARGO_TERM_COLOR=never", "cargo", "test"]
        );
    }

    #[test]
    fn direct_docker_recognition_accepts_only_supported_simple_executables() {
        // Arrange
        let commands = [
            "docker",
            "docker ps",
            "DOCKER_CLI_HINTS=false docker compose ps",
            "'docker' inspect example",
            "docker-compose up",
        ];

        // Act
        let recognized =
            commands.map(|command| classify_command(command).is_direct_docker_invocation());

        // Assert
        assert_eq!(recognized, [true; 5]);
    }

    #[test]
    fn direct_docker_recognition_rejects_wrappers_paths_and_compound_syntax() {
        // Arrange
        let commands = [
            "/usr/bin/docker ps",
            "env docker ps",
            "sudo docker ps",
            "make docker-test",
            "docker ps | cat",
            "docker ps && echo done",
            "docker ps > containers.txt",
            "docker $(cat command)",
            "docker ps\nwhoami",
        ];

        // Act
        let recognized =
            commands.map(|command| classify_command(command).is_direct_docker_invocation());

        // Assert
        assert_eq!(recognized, [false; 9]);
    }

    #[test]
    fn classify_command_allows_quoted_shell_metacharacters() {
        // Arrange
        let command = r#"printf '%s' 'a | b; $(still text)' "#;

        // Act
        let classification = classify_command(command);

        // Assert
        assert_eq!(
            classification,
            CommandClassification::Simple(SimpleCommand {
                executable_index: 0,
                words: vec![
                    "printf".to_string(),
                    "%s".to_string(),
                    "a | b; $(still text)".to_string(),
                ],
            })
        );
    }

    #[test]
    fn classify_command_rejects_compound_or_dynamic_shell_syntax() {
        // Arrange
        let commands = [
            "cargo test | tee output.txt",
            "cargo test && cargo clippy",
            "cargo test; cargo clippy",
            "cargo test > output.txt",
            "echo $(whoami)",
            "echo `whoami`",
            "echo $HOME",
            "echo ${HOME}",
            "echo *",
            "echo ?",
            "echo ~",
            "echo one\necho two",
            "echo one\r\necho two",
            "echo one\\\necho two",
            "echo one # comment",
            "(cargo test)",
            "{ cargo test; }",
            "if true; then echo yes; fi",
        ];

        // Act
        let classifications = commands.map(classify_command);

        // Assert
        assert!(
            classifications
                .iter()
                .all(|classification| *classification == CommandClassification::Complex)
        );
    }

    #[test]
    fn classify_command_rejects_incomplete_or_executable_free_input() {
        // Arrange
        let commands = [
            "",
            "   ",
            "RUST_LOG=debug",
            "echo 'unterminated",
            "echo \"unterminated",
            "echo trailing\\",
            "# comment",
        ];

        // Act
        let classifications = commands.map(classify_command);

        // Assert
        assert!(
            classifications
                .iter()
                .all(|classification| *classification == CommandClassification::Complex)
        );
    }
}
