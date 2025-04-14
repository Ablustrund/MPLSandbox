import argparse
from mplsandbox.sandbox import Sandbox
from mplsandbox.analyzetools import AnalyzeTools
from mplsandbox.const import LanguageValues, CodeType
import docker
from flask import jsonify
import traceback
import logging
from mplsandbox.utils import *


class MPLSANDBOX:
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], str):
            data_path = args[0]   
            try:
                with open(data_path, 'r') as file:
                    self.args = json.load(file)
            except Exception as e:
                raise ValueError(f"Failed to read JSON from the provided path: {e}")
        elif isinstance(args[0], dict):
            self.args = args[0]
        else:
            raise ValueError("Only dictionary type arguments or string paths to JSON files are accepted")
        self.app = self.args.get("app", False)
    
    def process_config(self):
        code = self.args.get('code')
        unit_dict = self.args.get('unit_cases')
        libraries = self.args.get('libraries', [])
        question = self.args.get('question')       
        lang = self.args.get('lang', "AUTO")  
        if lang == "AUTO":
            lang = detect_language(code)
        client = docker.from_env() if self.args.get('client') else None    
        image = self.args.get('image', None)
        docker_file = self.args.get('docker_file', None)
        keep_template = self.args.get('keep_template', True)
        verbose = self.args.get('verbose', False)

        if not code:
            raise_error_templete("No code provided", 400, self.app)
        if len(unit_dict["inputs"]) != len(unit_dict["outputs"]):
            raise_error_templete("Input and output cases in unit-test should be same.", 400, self.app)           
        if libraries is not None:
            if not isinstance(libraries, list) or not all(isinstance(lib, str) for lib in libraries):
                raise_error_templete({"Libraries must be a list of strings"}, 400, self.app)
        if lang not in LanguageValues:
            raise_error_templete(f"Invalid language specified.", 400, self.app)

        return client, image, docker_file, lang, keep_template, verbose, code, unit_dict, libraries, question
    

    def get_basic_info(self, show_per_unit_feedback=False):
        client, image, docker_file, lang, keep_template, verbose, code, unit_dict, libraries, question = self.process_config()
        try:
            with Sandbox(client=client,
                        image=image,
                        dockerfile=docker_file,
                        lang=lang,
                        keep_template=keep_template,
                        verbose=verbose,
                        ) as session:
                results = []
                correct_num = 0
                for unit_input, unit_answer in zip(unit_dict["inputs"], unit_dict["outputs"]):
                    output = session.run(code, unit_input, libraries=libraries)
                    if_correct = output_answer_check(unit_answer.strip(), output.text.strip())
                    if if_correct:
                        correct_num += 1
                    tmp_text = remove_ansi_codes(output.text)
                    results.append(tmp_text)
                    reward = get_reward(tmp_text, lang, if_correct)
                correct_rate = correct_num / len(unit_dict["inputs"])
                compiler_feedback, compiler_feedback_per_unit = results, results
                if reward == -0.3:
                    compiler_feedback = f"AssertionError:\nInput:{unit_dict['inputs']}\nOutput:{results}\nRequired Output:{unit_dict['outputs']}"
                    compiler_feedback_per_unit = f"AssertionError:\nInput:{unit_dict['inputs'][0]}\nOutput:{results[0]}\nRequired Output:{unit_dict['outputs'][0]}"
                results_dict = {"reward": reward, 
                                "compiler_feedback": compiler_feedback, 
                                "correct_rate": correct_rate, 
                                "question": question, 
                                "code": code,
                                "inputs": unit_dict["inputs"], 
                                "required_outputs": unit_dict["outputs"],
                                "language": lang}
                if show_per_unit_feedback:
                    results_dict["compiler_feedback_per_unit"] = compiler_feedback_per_unit
                return results_dict
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logging.error(f"Error during code execution: {error_message}, Stack trace: {stack_trace}")
            error_dict = {"error": {"type": error_type, "message": error_message, "stack_trace": stack_trace}}
            return (jsonify(error_dict), 500) if self.app else error_dict

    def code_analyze_feedback(self, analysis_type):
        client, image, docker_file, lang, keep_template, verbose, code, unit_dict, libraries, question = self.process_config()
        try:
            with AnalyzeTools(client=client,
                        image=image,
                        dockerfile=docker_file,
                        lang=lang,
                        keep_template=keep_template,
                        verbose=verbose,
                        ) as session:
                output = None
                output_dict = dict()
                analysis_list = ["all","code_basic_analysis","code_smell_analysis","code_bug_analysis","unit_test_analysis","code_efficiency_evaluation"]
                assert analysis_type in analysis_list, f"Invalid analysis type. Available types are {analysis_list}"
                if lang == "python":
                    if analysis_type == "all":
                        for sub_type in analysis_list[1:]:
                            output = session.call_tool_python(code=code,unit_inputs=unit_dict["inputs"],analysis=sub_type)
                            output_dict[sub_type] = output
                    else:
                        output = session.call_tool_python(code=code,unit_inputs=unit_dict["inputs"],analysis=analysis_type)
                        output_dict[analysis_type] = output
                return output_dict
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logging.error(f"Error during code execution: {error_message}, Stack trace: {stack_trace}")
            error_dict = {"error": {"type": error_type, "message": error_message, "stack_trace": stack_trace}}
            return (jsonify(error_dict), 500) if self.app else error_dict
        

    # def get_ai_anlysis(self, openai_api_key, model):
    #     basic_info = self.get_basic_info(show_per_unit_feedback=True)
    #     prompt = f"""You are a very professional {basic_info["language"]} code analyst;\n
    #     Now, for the question: {basic_info["question"]};\n 
    #     I have written a piece of code: {basic_info["code"]}, Please note that the code here is used for individual unit testing;\n
    #     The given input is: {basic_info["input"][0]};\n 
    #     the required answer is: {basic_info["required_output"][0]};\n
    #     and the compiler's feedback is: {basic_info["compiler_feedback_per_unit"]};\n 
    #     Please analyze the problem, given code, given input, required answer, and compiler feedback comprehensively.
    #     Please revise the code according to the requirements of the problem to complete the correct code.\n
    #     """
    #     context = [{'role': 'user', "content": prompt}]
    #     anlysis_report = get_completion_from_messages(openai_api_key, context, model) if basic_info['correct_rate'] != 1 else "The code is correct."
    #     anlysis_info = basic_info.copy().updata({"anlysis_report": anlysis_report})
    #     return anlysis_info

    def run(self,analysis_type="all"):
        basic_info = self.get_basic_info()
        analysis_info = self.code_analyze_feedback(analysis_type)
        result = basic_info
        if analysis_info is not None:
            result.update(analysis_info)
        return result

def main():
    parser = argparse.ArgumentParser(description="MPLSandbox Code Executor for Command Lines")
    parser.add_argument("--data", type=str, help="Path to the JSON data file")
    parser.add_argument("--report", type=str, help="Path to the TXT report")
    args = parser.parse_args()
    executor = MPLSANDBOX(args.data)
    report = executor.run(analysis_type="all")
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write("Report\n")
        f.write("="*50 + "\n")
        for key, value in report.items():
            f.write(f"{key}:\n")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            elif isinstance(value, list):
                f.write(f"  {', '.join(map(str, value))}\n")
            else:
                f.write(f"  {value}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
