import io
import os
import docker
import tarfile
from typing import List, Optional
from docker.models.images import Image
from docker.models.containers import Container
from mplsandbox.utils import (
    image_exists,
    get_libraries_installation_command,
    get_code_file_extension,
    get_code_execution_command,
)
from mplsandbox.utils import ConsoleOutput
from mplsandbox.const import (
    Language,
    LanguageValues,
    DefaultImage,
    NotSupportedLibraryInstallation,
    CONTAINER_LANGUAGE_MAPPING,
)
import ast
import astpretty
from pyflowchart import Flowchart
import javalang
import sys
import re

class AnalyzeTools:
    def __init__(
        self,
        client: Optional[docker.DockerClient] = None,
        image: Optional[str] = None,
        dockerfile: Optional[str] = None,
        lang: str = Language.PYTHON,
        keep_template: bool = False,
        verbose: bool = False,
    ):
        self._validate_inputs(image, dockerfile, lang)
        self.verbose = verbose
        self.lang = lang
        self.client = client or self._create_docker_client()
        self.image = image or DefaultImage.__dict__[lang.upper()]
        self.dockerfile = dockerfile
        self.container = None
        self.path = None
        self.keep_template = keep_template
        self.is_create_template = False
        self.is_create_container = False
        self.memory_limits = ['2G', '4G', '8G', '16G']

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def _validate_inputs(self, image, dockerfile, lang):
        if image and dockerfile:
            raise ValueError("Only one of image or dockerfile should be provided")
        if lang not in LanguageValues:
            raise ValueError(
                f"Language {lang} is not supported. Must be one of {LanguageValues}"
            )

    def _create_docker_client(self):
        if self.verbose:
            print("Using local Docker context since client is not provided..")
        return docker.from_env()

    def open(self):
        if self.dockerfile:
            self._build_image_from_dockerfile()
        elif isinstance(self.image, str):
            self._pull_image_if_needed()
        self._run_container()
        assert self.container != None

    def _build_image_from_dockerfile(self):
        self.path = os.path.dirname(self.dockerfile)
        if self.verbose:
            print(f"Building docker image from {self.dockerfile}")
            if self.keep_template:
                print(
                    "Since the `keep_template` flag is set to True, the docker image will not be removed after the session ends and remains for future use."
                )

        self.image, _ = self.client.images.build(
            path=self.path,
            dockerfile=os.path.basename(self.dockerfile),
            tag=f"sandbox-{self.lang.lower()}-{os.path.basename(self.path)}",
        )
        self.is_create_template = True

    def _pull_image_if_needed(self):
        if not image_exists(self.client, self.image):
            if self.verbose:
                print(f"Pulling image {self.image}..")
                if self.keep_template:
                    print(
                        "Since the `keep_template` flag is set to True, the docker image will not be removed after the session ends and remains for future use."
                    )
            self.image = self.client.images.pull(self.image)
            self.is_create_template = True
        else:
            self.image = self.client.images.get(self.image)
            if self.verbose:
                print(f"Using image {self.image.tags[-1]}")

    def _get_existing_container(self):
        containers = self.client.containers.list(filters={"ancestor": self.image, "status":"running"})
        if containers:
            return containers[0]
        return None

    def _check_container_exists(self, container_id):
        try:
            container = self.client.api.inspect_container(container_id)
            return True 
        except docker.errors.NotFound:
            return False

    def _run_container(self):
        for memory_limit in self.memory_limits:
            try:
                self.container = self.client.containers.run(
                    self.image,
                    detach=True,
                    tty=True,
                    mem_limit=memory_limit
                )
                self.is_create_container = True
                return  # If container is created successfully, return
            except docker.errors.ContainerError as e:
                if 'memory' in str(e):  # Check if the error is related to memory
                    if self.verbose:
                        print(f"Memory error occurred. Trying with {memory_limit}...")
                    continue  
                else:
                    raise  # If it's not a memory error, raise the exception
        raise RuntimeError("All memory limits have been tried. Failed to create container.")

    def close(self):
        if self.is_create_container:
            self._remove_container()

    def _commit_container(self):
        if isinstance(self.image, Image):
            self.container.commit(self.image.tags[-1])

    def _remove_container(self):
        self.container.remove(force=True)
        self.container = None

    def _remove_image_if_needed(self):
        if self.is_create_template and not self.keep_template:
            if not self._is_image_in_use():
                self._remove_image()
            elif self.verbose:
                print(
                    f"Image {self.image.tags[-1]} is in use by other containers. Skipping removal.."
                )


    def _remove_image(self):
        if isinstance(self.image, str):
            self.client.images.remove(self.image)
        elif isinstance(self.image, Image):
            self.image.remove(force=True)
        else:
            raise ValueError("Invalid image type")

    def _build_sh(self, code_dest_file, unit_input):
        commands = get_code_execution_command(self.lang, code_dest_file)
        sh_commands = ""
        for command in commands:
            unit_input = unit_input.replace("\n", "\\n")
            sh_commands += (
                f'echo -e "{unit_input}" | ' + command if unit_input else command
            )
            sh_commands += "\n"
        return sh_commands

        
    def run(self, code: str, unit_input: str = None, libraries: Optional[List] = None) -> ConsoleOutput:
        self._ensure_session_is_open()
        self._install_libraries_if_needed(libraries)
        code_file, code_dest_file = self._prepare_code_file(code)
        self._copy_code_to_container(code_file, code_dest_file)
        try:
            if unit_input:
                sh_text = self._build_sh(code_dest_file, unit_input)
                sh_file, sh_dest_file = self._prepare_sh_file(sh_text)
                self._copy_code_to_container(sh_file, sh_dest_file)
                return self._execute_sh_in_container(sh_dest_file)
            else:
                return self._execute_code_in_container(code_dest_file)
        except docker.errors.ContainerError as e:
            if 'memory' in str(e).lower():  # Check if the error is related to memory
                self._increase_memory_and_rerun(code, unit_input, libraries)
            else:
                raise

    def _ensure_session_is_open(self):
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before running code."
            )

    def _install_libraries_if_needed(self, libraries):
        if libraries:
            if self.lang.upper() in NotSupportedLibraryInstallation:
                raise ValueError(
                    f"Library installation has not been supported for {self.lang} yet!"
                )
            self._install_libraries(libraries)

    def _install_libraries(self, libraries):
        if self.lang == Language.GO:
            self._prepare_go_environment()
        for library in libraries:
            command = get_libraries_installation_command(self.lang, library)
            install_feedback=self.execute_command(
                command, workdir="/example" if self.lang == Language.GO else None
            )

    def _prepare_go_environment(self):
        self.execute_command("mkdir -p /example")
        self.execute_command("go mod init example", workdir="/example")
        self.execute_command("go mod tidy", workdir="/example")

    def _prepare_code_file(self, code):
        code_file = f"/tmp/code.{get_code_file_extension(self.lang)}"
        code_dest_file = "/example/code.go" if self.lang == Language.GO else code_file
        with open(code_file, "w") as f:
            f.write(code)
        
        return code_file, code_dest_file

    def _prepare_sh_file(self, sh):
        sh_file = f"/tmp/run.sh"
        sh_dest_file = "/example/run.sh" if self.lang == Language.GO else sh_file
        with open(sh_file, "w") as f:
            f.write(sh)
        return sh_file, sh_dest_file

    def _copy_code_to_container(self, src, dest):
        self.copy_to_runtime(src, dest)
    
    def _execute_code_in_container(self, code_dest_file, unit_input=None):
        output = ConsoleOutput("")
        commands = get_code_execution_command(self.lang, code_dest_file)
        for command in commands:
            try:
                output = self.execute_command(
                    command, workdir="/example" if self.lang == Language.GO else None
                )
            except docker.errors.ContainerError as e:
                if 'memory' in str(e).lower():  # Check if the error is related to memory
                    self._increase_memory_and_rerun(code_dest_file, unit_input)
                    return self._execute_code_in_container(code_dest_file, unit_input)
                else:
                    raise
        return output

    def _execute_sh_in_container(self, sh_dest_file):
        output = ConsoleOutput("")
        source_bash = "chmod +x " + sh_dest_file
        output2 = ConsoleOutput("") 
        output2 = self.execute_command(
            source_bash, workdir="/example" if self.lang == Language.GO else None
        )
        run_bash = "/bin/bash\t" + sh_dest_file
        output = self.execute_command(
            run_bash, workdir="/example" if self.lang == Language.GO else None
        )

        return output

    def copy_from_runtime(self, src: str, dest: str):
        self._ensure_session_is_open()
        if self.verbose:
            print(f"Copying {self.container.short_id}:{src} to {dest}..")
        self._extract_file_from_container(src, dest)

    def _extract_file_from_container(self, src, dest):
        bits, stat = self.container.get_archive(src)
        if stat["size"] == 0:
            raise FileNotFoundError(f"File {src} not found in the container")
        tarstream = io.BytesIO(b"".join(bits))
        with tarfile.open(fileobj=tarstream, mode="r") as tar:
            tar.extractall(os.path.dirname(dest))

    def copy_to_runtime(self, src: str, dest: str):
        self._ensure_session_is_open()
        self._create_directory_if_needed(dest)
        self._copy_file_to_container(src, dest)

    def _create_directory_if_needed(self, dest):
        directory = os.path.dirname(dest)
        self.container.exec_run(f"mkdir -p {directory}")
        if directory and not self.container.exec_run(f"test -d {directory}")[0] == 0:
            self.container.exec_run(f"mkdir -p {directory}")
            print('create successfully')
            if self.verbose:
                print(f"Creating directory {self.container.short_id}:{directory}")

    def _create_directory_if_needed_tmp(self, dest):
        directory = dest
        if directory and not self.container.exec_run(f"test -d {directory}")[0] == 0:
            self.container.exec_run(f"mkdir -p {directory}")
            if self.verbose:
                print(f"Creating directory {self.container.short_id}:{directory}")

    def _copy_file_to_container(self, src, dest):
        # print(f"Copying {src} to {self.container.short_id}:{dest}..")
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            tar.add(src, arcname=os.path.basename(src))
        tarstream.seek(0)
        self.container.put_archive(os.path.dirname(dest), tarstream)
    def _add_directory_to_tar(self,tar, path, arcname):
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                tar.add(dir_path, arcname=os.path.join(arcname, os.path.relpath(dir_path, path)))

            for file in files:
                file_path = os.path.join(root, file)
                tar.add(file_path, arcname=os.path.join(arcname, os.path.relpath(file_path, path)))
    def copy_directory_to_container(self, src_dir):
        print("begin copy the tools to the container..")
        dest_dir = "/tmp/tools"
        self._ensure_session_is_open()
        self._create_directory_if_needed_tmp(dest_dir)
        exit_code, output = self.container.exec_run(f"cd {dest_dir}")
        # print(f"Copying {src_dir} to {self.container.short_id}:{dest_dir}..")
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode='w') as tar:
            self._add_directory_to_tar(tar, src_dir, arcname='')
        tarstream.seek(0)
        self.container.put_archive(dest_dir, tarstream)
        command = f"ls {dest_dir}/java"
        exit_code, output = self.container.exec_run(command)
        
    
    def execute_command(
        self, command: Optional[str], workdir: Optional[str] = None
    ) -> ConsoleOutput:
        self._validate_command(command)
        self._ensure_session_is_open()
        if self.verbose:
            print(f"Executing command: {command}")
        return self._run_command_in_container(command, workdir)

    def _validate_command(self, command):
        if not command:
            raise ValueError("Command cannot be empty")

    def _run_command_in_container(self, command, workdir):
        if workdir:
            exit_code, exec_log = self.container.exec_run(
                command, stream=True, tty=True, workdir=workdir
            )
        else:
            exit_code, exec_log = self.container.exec_run(
                command, stream=True, tty=True
            )

        output = ""
        if self.verbose:
            print("Output:", end=" ")

        for chunk in exec_log:
            chunk_str = chunk.decode("utf-8")
            output += chunk_str
            if self.verbose:
                print(chunk_str, end="")

        return ConsoleOutput(output)
    
    def _increase_memory_and_rerun(self, code, unit_input, libraries):
        self.close()
        self.open()
        self.run(code, unit_input, libraries)

    
    def call_tool_python(self, code, unit_inputs, analysis) -> str:
        self._ensure_session_is_open()
        # print(self.container.exec_run("pip list coverage")[1].decode('utf-8'))
        analysis_info = analysis.replace("_"," ")
        print(f"Executing Python {analysis_info}...")
        self._install_libraries_if_needed(["coverage", "bandit", "pylint"])
        commands = []
        tmp_outputs = {}
        code_file, code_dest_file = self._prepare_code_file(code) 
        if analysis == "code_smell_analysis":
            command = "pylint "+code_dest_file+"\n"
            commands.append(command)
        elif analysis == "unit_test_analysis":
            for unit_input in unit_inputs:
                unit_input = unit_input.replace("\n", "\\n")
                command = f'echo "{unit_input}"' +f" | coverage run {code_dest_file}"+"\n"+"coverage "+"report"+"\n" 
                commands.append(command)
        elif analysis == "code_efficiency_evaluation":
            for unit_input in unit_inputs:
                unit_input = unit_input.replace("\n", "\\n")
                command = f'echo "{unit_input}"' +f" | python -m cProfile {code_dest_file}"+"\n" 
                commands.append(command)
        elif analysis == "code_bug_analysis":
            command = "bandit "+"-r "+f"{code_dest_file}"+"\n"
            commands.append(command)
        elif analysis == "code_basic_analysis":        
            command = ""
            try:
                tree = ast.parse(code)
                captured_output = io.StringIO()
                original_stdout = sys.stdout
                sys.stdout = captured_output
                astpretty.pprint(tree)
                sys.stdout = original_stdout
                captured_output.seek(0)  
                ast_pretty_printed = captured_output.read()
                fc = Flowchart.from_code(code)
                cfg_printed = fc.flowchart()
            except Exception as e:
                ast_pretty_printed = str(e)
            try:
                fc = Flowchart.from_code(code)
                cfg_printed = fc.flowchart()
            except Exception as e:
                cfg_printed = str(e)
            tmp_outputs = {"ast":ast_pretty_printed, "cfg":cfg_printed}
        outputs = []
        for i, command in enumerate(commands):
            sh_file, sh_dest_file = self._prepare_sh_file(command)
            self._copy_code_to_container(code_file, code_dest_file)
            self._copy_code_to_container(sh_file, sh_dest_file)
            output = self._execute_sh_in_container(sh_dest_file).text
            if analysis == "unit_test_analysis":
                output = re.sub(r'^\d+\\r\\n', '', output, count=1)
            outputs.append(output)
           
        if analysis == "code_basic_analysis":
            return tmp_outputs
        else:
            if len(outputs) == 1:
                return outputs[0]
            else:
                outputs_dict = {}
                if analysis == "unit_test_analysis":
                    for unit_input, output in zip(unit_inputs, outputs):
                        outputs_dict.update({unit_input : output})
                    result = []
                    
                    for input_key, output_str in outputs_dict.items():
                        cleaned_str = re.sub(r'^\d+\r\n', '', output_str)  
                        lines = cleaned_str.split('\r\n')
                        
                        total_line = next(line for line in lines if "TOTAL" in line)
                        parts = re.split(r'\s+', total_line.strip())
                        
                        result.append({
                            "Unit Input": input_key,
                            "Total Lines": int(parts[1]),
                            "Miss": int(parts[2]),
                            "Cover Rate": parts[3]
                        })
                else:
                    for unit_input, output in zip(unit_inputs, outputs):
                        outputs_dict.update({unit_input : output})
                    result = []
                    for input_key, output_str in outputs_dict.items():
                        cleaned_str = re.sub(r'^\d+\r\n', '', output_str)
                        
                        lines = [
                            line.strip() 
                            for line in cleaned_str.split('\r\n') 
                            if line.strip() and not line.startswith(('Ordered by:', 'ncalls  tottime'))
                        ]
                        
                        func_data = []
                        for line in lines:
                            if re.match(r'^\d+\s+[\d.]+', line):  
                                parts = re.split(r'\s+', line)
                                func_data.append({
                                    "ncalls": parts[0],
                                    "tottime": f"{float(parts[1]):.6f} s",
                                    "percall": f"{float(parts[2]):.6f} s",
                                    "cumtime": f"{float(parts[3]):.6f} s",
                                    "function location": parts[4]
                                })
                        
                        result.append({
                            "Unit Input": input_key,
                            "Total Calls": len(func_data),
                            "Total Time": next((line for line in lines if "function calls in" in line), ""),
                            "Functions": func_data
                        })
                
                return result

    # def call_tool_java(self, code, tool_name):
    #     print("Executing Java tool...")
    #     tool_name = tool_name.lower()
    #     code_file, code_dest_file = self._prepare_code_file(code)
    #     script_dir = os.path.dirname(os.path.abspath(__file__))
    #     classname = os.path.splitext(os.path.basename(code_dest_file))[0]
        
    #     directory_path = os.path.dirname(code_dest_file)
    #     src_dir = os.path.join(script_dir, "tools")
    #     self.copy_directory_to_container(src_dir)
    #     if tool_name == "javalang" or tool_name == "basic-ast":
    #         print(f"Tool -{tool_name}- execution succeed:")
    #         commands = []
    #         tokens = javalang.tokenizer.tokenize(code)
    #         parser = javalang.parser.Parser(tokens)
    #         tree = parser.parse()
    #         important_node_types = (
    #             javalang.tree.ClassDeclaration,
    #             javalang.tree.MethodDeclaration,
    #             javalang.tree.IfStatement,
    #             javalang.tree.VariableDeclaration,
    #             javalang.tree.BinaryOperation,
    #             javalang.tree.MethodInvocation,
    #         )
    #         def node_to_dict(node):
    #             if not isinstance(node, javalang.ast.Node):
    #                 return str(node)  
    #             result = {
    #                 "type": type(node).__name__,  
    #                 "properties": {}  
    #             }
    #             for attr, value in node.__dict__.items():
    #                 if isinstance(value, list):
    #                     result["properties"][attr] = [node_to_dict(item) for item in value]
    #                 elif isinstance(value, javalang.ast.Node):
    #                     result["properties"][attr] = node_to_dict(value)
    #                 else:
    #                     result["properties"][attr] = str(value)
    #             return result
    #         ast_dict = node_to_dict(tree)
    #         formatted_ast = json.dumps(ast_dict, indent=4)
    #         print(formatted_ast)

    #     elif tool_name == "soot" or tool_name == "basic-cfg":
    #         commands = [
    #             "java","-cp","tmp/tools/java/soot-4.5.0-jar-with-dependencies.jar", 
    #             "soot.Main",                   
    #             "-pp",                         
    #             "-cp","/tmp",         
    #             "-process-dir",classname,    
    #             "-allow-phantom-refs",         
    #             "-w",                          
    #             "-p","cg","enabled:true",    
    #             "-p", "jb", "enabled:true",
    #             "-f", "J",                     
    #             "-d", "tmp/tools/temp/soot"   
    #         ]
    #         to_class_command = f"javac -d tmp tmp/{classname}.java"
    #         command = " ".join(commands)
    #         command = "\n".join([to_class_command, command])
    #         command = to_class_command

    #     elif tool_name == "pmd" or tool_name == "smell":
    #         commands = [
    #             'tmp/tools/java/pmd-bin-7.6.0/bin/pmd', 'check', 
    #             '-d', code_dest_file, 
    #             '-R', 'tmp/tools/java/pmd-bin-7.6.0/pmd-pmd_releases-7.6.0/pmd-java/src/main/resources/rulesets/java/quickstart.xml', 
    #             '-f', 'text'
    #         ]
    #         command = " ".join(commands)
    #     elif tool_name == "jacoco" or tool_name == "coverage":
    #         commands = [[
    #             "java",
    #             "-javaagent:tmp/tools/java/jacoco-0.8.12/lib/jacocoagent.jar=destfile=tmp/tools/temp/jacoco.exec",
    #             "-cp", "/tmp", classname
    #         ],
    #         [
    #             "java",
    #             "-jar", "tmp/tools/java/jacoco-0.8.12/lib/jacococli.jar",
    #             "report", "tmp/tools/temp/jacoco.exec",
    #             "--classfiles", "/tmp",
    #             "--sourcefiles", classname,
    #             "--csv", "/dev/stdout"
    #         ]
    #         ]
    #     sh_file, sh_dest_file = self._prepare_sh_file(command)
    #     self._copy_code_to_container(code_file, code_dest_file)
    #     self._copy_code_to_container(sh_file, sh_dest_file)
    #     output = self._execute_sh_in_container(sh_dest_file)
    #     print(output.text)
    #     return output.text
