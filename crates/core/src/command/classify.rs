#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandClassification {
    Complex,
    Simple(SimpleCommand),
}

impl CommandClassification {
    pub fn is_simple_docker_invocation(&self) -> bool {
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

pub(crate) fn invokes_direct_executable(command: &str, executable: &str) -> bool {
    let Some(tokens) = tokenize_command_positions(command) else {
        return false;
    };
    let mut command_position = true;
    let mut skip_redirection_target = false;
    let mut index = 0;

    while index < tokens.len() {
        match &tokens[index] {
            CommandToken::Boundary => {
                command_position = true;
                skip_redirection_target = false;
            }
            CommandToken::Redirection => {
                skip_redirection_target = true;
            }
            CommandToken::Word(word) if skip_redirection_target => {
                skip_redirection_target = false;
            }
            CommandToken::Word(word)
                if command_position
                    && word.bytes().all(|byte| byte.is_ascii_digit())
                    && matches!(tokens.get(index + 1), Some(CommandToken::Redirection)) => {}
            CommandToken::Word(word) if command_position && is_assignment(word) => {}
            CommandToken::Word(word) if command_position && word == "!" => {}
            CommandToken::Word(word) if command_position && is_reserved_word(word) => {
                return false;
            }
            CommandToken::Word(word) if command_position => {
                if word == executable {
                    return true;
                }
                command_position = false;
            }
            CommandToken::Word(_) => {}
        }
        index += 1;
    }

    false
}

#[derive(Debug, Eq, PartialEq)]
enum CommandToken {
    Boundary,
    Redirection,
    Word(String),
}

fn tokenize_command_positions(command: &str) -> Option<Vec<CommandToken>> {
    let mut characters = command.chars().peekable();
    let mut current = String::new();
    let mut quote = Quote::Unquoted;
    let mut tokens = Vec::new();
    let mut word_started = false;

    while let Some(character) = characters.next() {
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
                '`' => return None,
                '$' if characters.peek() == Some(&'(') => return None,
                '\\' => {
                    let escaped = characters.next()?;
                    if matches!(escaped, '\n' | '\r') {
                        return None;
                    }
                    current.push(escaped);
                    word_started = true;
                }
                _ => {
                    current.push(character);
                    word_started = true;
                }
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
                '`' => return None,
                '$' if characters.peek() == Some(&'(') => return None,
                '<' | '>' => {
                    finish_token(&mut current, &mut word_started, &mut tokens);
                    if character == '<' && characters.peek() == Some(&'<') {
                        return None;
                    }
                    if characters.peek() == Some(&character) {
                        characters.next();
                    }
                    if characters.peek() == Some(&'&') {
                        characters.next();
                    }
                    tokens.push(CommandToken::Redirection);
                }
                '&' if characters.peek() == Some(&'>') => {
                    finish_token(&mut current, &mut word_started, &mut tokens);
                    characters.next();
                    if characters.peek() == Some(&'>') {
                        characters.next();
                    }
                    tokens.push(CommandToken::Redirection);
                }
                '\n' | '\r' | ';' | '|' | '&' => {
                    finish_token(&mut current, &mut word_started, &mut tokens);
                    if characters.peek() == Some(&character) {
                        characters.next();
                    }
                    tokens.push(CommandToken::Boundary);
                }
                '(' | ')' | '{' | '}' => return None,
                '#' if !word_started => {
                    finish_token(&mut current, &mut word_started, &mut tokens);
                    for comment_character in characters.by_ref() {
                        if matches!(comment_character, '\n' | '\r') {
                            tokens.push(CommandToken::Boundary);
                            break;
                        }
                    }
                }
                character if character.is_whitespace() => {
                    finish_token(&mut current, &mut word_started, &mut tokens);
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
    finish_token(&mut current, &mut word_started, &mut tokens);
    Some(tokens)
}

fn finish_token(current: &mut String, word_started: &mut bool, tokens: &mut Vec<CommandToken>) {
    if *word_started {
        tokens.push(CommandToken::Word(std::mem::take(current)));
        *word_started = false;
    }
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
    fn simple_docker_classification_accepts_only_supported_executables() {
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
            commands.map(|command| classify_command(command).is_simple_docker_invocation());

        // Assert
        assert_eq!(recognized, [true; 5]);
    }

    #[test]
    fn simple_docker_classification_rejects_wrappers_paths_and_compound_syntax() {
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
            commands.map(|command| classify_command(command).is_simple_docker_invocation());

        // Assert
        assert_eq!(recognized, [false; 9]);
    }

    #[test]
    fn command_position_recognition_finds_docker_in_common_compound_syntax() {
        // Arrange
        let commands = [
            "docker ps 2>&1; echo done",
            "which docker; docker --version 2>&1",
            "echo ready && docker compose version",
            "printf ready | docker ps",
            "RUST_LOG=debug docker-compose up > compose.log",
            "echo first\ndocker ps",
            "> probe.log docker version",
        ];

        // Act
        let recognized = commands.map(|command| invokes_direct_executable(command, "docker"));
        let compose = invokes_direct_executable(commands[4], "docker-compose");

        // Assert
        assert_eq!(recognized, [true, true, true, true, false, true, true]);
        assert!(compose);
    }

    #[test]
    fn command_position_recognition_ignores_mentions_wrappers_and_ambiguous_syntax() {
        // Arrange
        let commands = [
            "echo docker",
            "rg docker crates",
            "printf '%s' 'docker ps; docker version'",
            "echo > docker",
            "/usr/bin/docker ps",
            "env docker ps",
            "sudo docker ps",
            "docker() { malicious; }; docker",
            "cat <<EOF\ndocker ps\nEOF",
            "echo $(docker ps)",
            "if docker ps; then echo yes; fi",
            "# docker ps\necho no",
        ];

        // Act
        let recognized = commands.map(|command| invokes_direct_executable(command, "docker"));

        // Assert
        assert_eq!(recognized, [false; 12]);
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
