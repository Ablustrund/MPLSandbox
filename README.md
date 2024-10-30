# ✨ MPLSandbox
MPLSandbox is an out-of-the-box multi-programming language sandbox designed to provide unified and comprehensive feedback from compiler and analysis tools for LLMs.


<img width="950" alt="image" src="https://github.com/user-attachments/assets/792e9800-ad98-472a-96ff-b78725f94597">

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-brightgreen.svg)](./LICENSE)


# 🔍 Introduction

we propose MPLSandbox, an out-of-the-box sandbox designed to provide unified compiler feedback across multiple programming languages.
Additionally, it integrates traditional code analysis tools, delivering comprehensive code information to LLMs from numerous perspectives.
MPLSandbox simplifies code analysis for researchers, and can be seamlessly integrated into LLM training and application processes to enhance the performance of LLMs in a range of code-related tasks.

MPLSandbox consists of three core modules: 

### Multi-Programming Language Sandbox Environment

This Module can provide unified compiler feedback by compiling and executing the code.
The code and unit test samples are sent to the sub-sandbox of the corresponding programming language for isolated execution to obtain compiler feedback. 
The sandbox ensures the program executes safely without jeopardizing the external environment or interrupting the training process

### Code Analysis Module

This module includes multiple traditional analysis tools to offer a comprehensive analysis report from numerous perspectives.
It provides a comprehensive code analysis from multiple perspectives, such as static analysis (i.e., potential bug detection} and code smell analysis) and dynamic analysis (i.e., fuzz testing and efficiency analysis).
Additionally, this module can also assess other input information besides the code, such as evaluating the coverage of unit tests for the code, aiding researchers in improving the quality of these unit tests.

### Information Integration Module

This module integrates compilation feedback and various analysis results to accomplish a range of complex code-related tasks.
It integrates these results for LLMs to improve the quality of generated code and enhance their performance on a range of code-related tasks.
    

# 🛠️ Setup

## Install MPLSandbox

The user can create and install MPLSandbox using the following command:

```bash
git clone git@github.com:Ablustrund/MPLSandbox.git
cd MPLSandbox
pip install .
# pip install -e . ## for editable mode
```

## Prepare the Docker Images

First, users need to deploy the Docker images addresses on the host machine. After extensive testing, we have installed the necessary dependencies in Docker containers for various languages and packaged these custom Docker containers into the corresponding images as follows. We hope that users can directly use our open-source images because this can, to some extent, reduce the hassle of installing dependencies for various languages.


**Python**: [mplsandbox-python-3.9.19-v1](https://drive.google.com/file/d/1kkwwj1HbODHi2-Ws0wbXCPSPt4GHr3No/view?usp=drive_link)

**Java**: [mplsandbox-java-11.0.12-v1](https://drive.google.com/file/d/1dtThSM-N93evTl5IRBongd3KyoNA-eUt/view?usp=drive_link)

**JavaScript**: [mplsandbox-javascript-22-v1](https://drive.google.com/file/d/1dtThSM-N93evTl5IRBongd3KyoNA-eUt/view?usp=drive_link)

**C++**: [mplsandbox-cpp-11.2.0-v1](https://drive.google.com/file/d/1gEGoiG2WYsJp1tDQNmBp5-q1zhctG4vD/view?usp=drive_link)

**Go**: [mplsandbox-golang-1.17.0-v1](https://drive.google.com/file/d/1CZGpnoJnSn2yHEPA4WOWWFjSge2z_5lQ/view?usp=drive_link)

**Ruby**: [mplsandbox-ruby-3.0.2-v1](https://drive.google.com/file/d/1VrOkLUF7P9zapvTDYE5PBLqLrwHdeunu/view?usp=drive_link)

**TypeScript**: [mplsandbox-typescript-1-22-v1](https://drive.google.com/file/d/1DPg_fQlwiSFG9wZpIKNB8UhC6AwdnlZn/view?usp=drive_link)  

**Bash**: [mplsandbox-bash-v1](https://drive.google.com/file/d/10WHK6vxipTf8Kq5qN6ZEWdWRIR0kXVZe/view?usp=drive_link)

We recommend that users manually download these image files and then use the following command to import them into Docker:

```bash
docker load < <path_to_downloaded_image>
```
If users wish to use custom images, we recommend modifying the `DefaultImage` class in `/mplsandbox/const.py` to define their own images.


# 📚 Usage

## Use in the Project

Users can start mplsandbox and run it with the following lines of code:
```python
from mplsandbox import MPLSANDBOX
data = {   
"question":"Define get_sum_of_two_numbers():\n    \"\"\"Write a function that takes two integers as input and returns their sum.\n\n    -----Input-----\n    \n    The input consists of multiple test cases. Each test case contains two integers $a$ and $b$ ($-10^9 \\le a, b \\le 10^9$).\n    \n    -----Output-----\n    \n    For each test case, print the sum of the two integers.\n    \n    -----Example-----\n    Input\n    3\n    1 2 ↵\n    -1 1 ↵\n    1000000000 1000000000\n    \n    Output\n    3\n    0\n    2000000000\n    \"\"\"",
"code": 'def get_sum_of_two_numbers():\n    a, b = map(int, input().split(" "))\n    print(a * b)\nget_sum_of_two_numbers()',
"unit_cases": {
"inputs": ["1 2", "3 4"],
"outputs": ["3", "7"]
},
"lang": "AUTO"
}  # or a JSON file path
executor = MPLSANDBOX(data)
result = executor.run(analysis_type="all")
```

The specific descriptions of all fields in the data are as follows:

| Field    | Description |
|----------------|-------------|
| `question` | (Required) Specifies the path to the code file to be executed. |
| `code` | (Required) Specifies the code to be executed. |
| `unit_cases` | (Required) Specifies the unit test cases, including `inputs` and expected `outputs`. |
| `lang` | (Optional) Specifies the language of the code. If not specified, it can be set to `"AUTO"` for automatic recognition. |
| `libraries` | (Optional) Specifies a list of dependency library names that need to be installed. |
| `client` | (Optional) Specifies the docker client instance to be used |
| `image` | (Optional) Specifies the docker image used to run the code. |
| `dockerfile` | (Optional) Specifies the path to the dockerfile used to build a custom docker image. |
| `keep_template` | (Optional) If it is set to `True`, the template files will be kept after the code is run. |
| `verbose` | (Optional) If it is set to `True`, verbose output will be enabled to assist with debugging and diagnosing issues. |
| `app` | (Optional) If it is set to `True`, app mode will be enabled, facilitating the deployment of services on the server. |


##  Use from the Command Line

We also provide the following command-line interface to scan the `data.json` file and output the report to the `report.txt` file:

```bash
mplsandbox --data /path/to/your/data.json --report /path/to/your/report.txt
```

##  Use as a Service

MPLSandbox often serves as a node for emitting code-related signals, so configuring the corresponding services is very important. We have provided a simple service demo in the `scripts` directory, and users can run this demo with the following command:

```bash
cd scripts
python ./app.py
```
Then, users can access the service using the curl command or other methods, and the format example is in `scripts/test_app.sh`
```bash
./test_app.sh
```

# 🧑‍💻 Developing
We are working hard to refactor and improve the open-source version of MPLSandbox to closely match the functionality of the version used internally by Meituan LLM Team.

# 👀 Citation

```bibtex
TMP
```

```bibtex
@article{dou2024s,
  title={What's Wrong with Your Code Generated by Large Language Models? An Extensive Study},
  author={Dou, Shihan and Jia, Haoxiang and Wu, Shenxi and Zheng, Huiyuan and Zhou, Weikang and Wu, Muling and Chai, Mingxu and Fan, Jessica and Huang, Caishuang and Tao, Yunbo and others},
  journal={arXiv preprint arXiv:2407.06153},
  year={2024}
}
```


