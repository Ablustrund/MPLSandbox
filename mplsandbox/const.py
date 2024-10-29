from dataclasses import dataclass

@dataclass
class Language:
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    GO = "go"
    RUBY = "ruby"
    RUST = "rust"
    BASH = "bash"
    TYPESCRIPT = "typescript"

@dataclass
class FILE_EXTENSION_MAPPING:
    MAPPING = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".js": Language.JAVASCRIPT,
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".sh": Language.BASH,
    ".ts": Language.TYPESCRIPT
    }
@dataclass
class CONTAINER_LANGUAGE_MAPPING:
    MAPPING = {
    Language.PYTHON: "7365b5c6ffaa",
    Language.JAVA: "6063cee04450",
    Language.JAVASCRIPT: "ce41b82700ca",
    Language.CPP: "2094ef9598af",
    Language.GO: "928e44a0b293",
    Language.RUBY: "e65e98d3a186",
    Language.RUST: "6c4c831e80d5",
    }
@dataclass
class DefaultImage:
    PYTHON = "python:3.9.19-bullseye"
    JAVA = "openjdk:11.0.12-jdk-bullseye"
    JAVASCRIPT = "node:22-bullseye"
    CPP = "gcc:11.2.0-bullseye"
    GO = "golang:1.17.0-bullseye"
    RUBY = "ruby:3.0.2-bullseye"
    RUST = "rust:latest"
    TYPESCRIPT = "node:22-bullseye"  
    BASH = "bash:latest" 


class CodeType:
    STDIN = 'stdin'
    CALL = 'call'


NotSupportedLibraryInstallation = ["JAVA"]
LanguageValues = [
    v for k, v in Language.__dict__.items() if not k.startswith("__")
]
CodeTypeValues = [ v for k, v in CodeType.__dict__.items() if not k.startswith("__")]
