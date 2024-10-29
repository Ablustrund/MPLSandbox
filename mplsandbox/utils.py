import docker
import docker.errors
from typing import Optional
import json
from docker import DockerClient
import docker
import re
from openai import OpenAI
from flask import jsonify
from guesslang import Guess
from mplsandbox.const import Language, FILE_EXTENSION_MAPPING

class ConsoleOutput:
    def __init__(self, text: str):
        self._text = text

    @property
    def text(self):
        return self._text

    def __str__(self):
        return f"ConsoleOutput(text={self.text})"

def image_exists(client: DockerClient, image: str) -> bool:
    """
    Check if a Docker image exists
    :param client: Docker client
    :param image: Docker image
    :return: True if the image exists, False otherwise
    """
    try:
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        return False
    except Exception as e:
        raise e
def get_libraries_installation_command(lang: str, library: str) -> Optional[str]:
    """
    Get the command to install libraries for the given language
    :param lang: Programming language
    :param library: List of libraries
    :return: Installation command
    """
    supported_languages = {
        Language.PYTHON: f"pip install {library}",
        Language.JAVA: f"mvn install:install-file -Dfile={library}",
        Language.JAVASCRIPT: f"yarn add {library}",
        Language.CPP: f"apt-get install {library}",
        Language.GO: f"go get -u {library}",
        Language.RUBY: f"gem install {library}"
    }
    
    if lang not in supported_languages:
        raise ValueError(f"Language {lang} is not supported")
    
    return supported_languages[lang]

def get_code_file_extension(lang: str) -> str:
    """
    Get the file extension for the given language
    :param lang: Programming language
    :return: File extension
    """
    extensions = {
        Language.PYTHON: "py",
        Language.JAVA: "java",
        Language.JAVASCRIPT: "js",
        Language.CPP: "cpp",
        Language.GO: "go",
        Language.RUBY: "rb",
        Language.RUST: "rs"
    }
    
    if lang not in extensions:
        raise ValueError(f"Language {lang} is not supported")
    
    return extensions[lang]

def get_code_execution_command(lang: str, code_file: str) -> list:
    """
    Return the execution command for the given language and code file.
    :param lang: Language of the code
    :param code_file: Path to the code file
    :return: List of execution commands
    """
    commands = {
        Language.PYTHON: [f"python {code_file}"],
        Language.JAVA: [f"java {code_file}"],
        Language.JAVASCRIPT: [f"node {code_file}"],
        Language.CPP: [f"g++ -o a.out {code_file}", "./a.out"],
        Language.GO: [f"go run {code_file}"],
        Language.RUBY: [f"ruby {code_file}"],
        Language.RUST: [f"rustc {code_file}", f"chmod +x {code_file.split('.')[0]}", f"{code_file.split('.')[0]}"],
        Language.TYPESCRIPT: [f"ts-node \"{code_file}\""],
        Language.BASH: [f"chmod +x {code_file}" , f"\"{code_file}\""]
    }

    if lang not in commands:
        raise ValueError(f"Language {lang} is not supported")
    
    return commands[lang]

def raise_error_templete(error_message: str, number: int, app=False):
    if app:
        return jsonify({"error": f"{error_message}"}), number
    else:
        raise ValueError(f"{error_message}")
    
def extract_libraries(code: str, language: str) -> list:
    libraries = []
    if language == "python":
        libraries = re.findall(r'import (\w+)|from (\w+)', code)
        libraries = [lib for pair in libraries for lib in pair if lib]
    elif language == "go":
        libraries = re.findall(r'import "(.*?)"', code)
    elif language == "cpp":
        libraries = re.findall(r'#include <(.*?)>', code)
    elif language == "javascript":
        libraries = re.findall(r'require\("(.*?)"\)|import .* from "(.*?)"', code)
        libraries = [lib for pair in libraries for lib in pair if lib]
    elif language == "java":
        libraries = re.findall(r'import (.*?);', code)
    elif language == "ruby":
        libraries = re.findall(r'require "(.*?)"', code)
    # elif language == "php":
    #     libraries = re.findall(r'use (\w+);', code)
    return libraries

def generate_install_commands(language: str, libraries: list) -> str:
    if language == "python":
        return "pip install " + " ".join(libraries) 
    elif language == "go":
        return "go get " + " ".join(libraries) 
    # elif language == "cpp":
        # C++ doesn't have a single package manager, but here is an example for apt
        # return "sudo apt-get install " + " ".join(libraries)
    elif language == "javascript":
        return "npm install " + " ".join(libraries)
    elif language == "java":
        # Assuming Maven is used
        return "\n".join([f'<dependency>\n  <groupId>groupId</groupId>\n  <artifactId>{lib}</artifactId>\n  <version>version</version>\n</dependency>' for lib in libraries])
    elif language == "ruby":
        return "gem install " + " ".join(libraries)
    # elif language == "php":
        # return "composer require " + " ".join(libraries)
    return ""

def detect_language_via_file_extension(file_extension: str) -> str:
    return FILE_EXTENSION_MAPPING.MAPPING.get(file_extension, None)

def detect_language(code: str) -> str:
    guess = Guess()
    language = guess.language_name(code)
    if isinstance(language,str):
        language = language.lower()
    return language

def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def output_answer_check(answer, output):
    def remove_newlines_and_spaces(s):
        return s.replace("\n", "").replace("\\n", "").replace(" ", "")
    answer = remove_newlines_and_spaces(answer)
    compile_feedback = remove_newlines_and_spaces(output)
    return answer == compile_feedback

def read_code_file(code_file_path):
    with open(code_file_path, "r") as file:
        return file.read()

def read_unit_file(unit_file_path):
    with open(unit_file_path, "r") as file:
        return json.load(file)

def read_libraries_file(library_file_path):
    if library_file_path:
        with open(library_file_path, "r") as file:
            return [line.strip() for line in file.readlines()]
    return []

def read_question_file(question_file_path):
    with open(question_file_path, "r") as file:
        return file.read()

def get_reward(output, lang, if_correct):
    if if_correct:
        return 1

    reward_mapping = {
        "python": lambda output: -1 if "SyntaxError" in output else -0.6 if "Error" in output else -0.3,
        "java": lambda output: -1 if "error: compilation failed" in output and f"/tmp/code" in output else -0.6 if "error" in output else -0.3,
        "cpp": lambda output: -1 if f"/tmp/code" in output else -0.6 if "error" in output else -0.3,
        "javascript": lambda output: -1 if "ReferenceError" in output and f"/tmp/code" in output else -0.6 if "Error" in output else -0.3,
        "typescript": lambda output: -1 if "TypeScript" in output and "error" in output else -0.6 if "error" in output else -0.3,
        "bash": lambda output: -1 if "bash:" in output and "command not found" in output else -0.6 if "bash:" in output and "line" in output else -0.3,
        "go": lambda output: -1 if "go:" in output and "build" in output else -0.6 if "error" in output else -0.3,
    }

    language_reward_func = reward_mapping.get(lang)
    if language_reward_func:
        return language_reward_func(output)
    else:
        return -0.3 
    
def get_completion_from_messages(api_key, messages, model, temperature=0):
    client = OpenAI(
    api_key=api_key,
    base_url="https://api3.apifans.com/v1"
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content
        
                



