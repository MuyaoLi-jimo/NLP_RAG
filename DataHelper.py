"""
preparing the dataset and evaluation matrix
@author muyao
"""

from datasets import load_dataset,load_from_disk,DatasetDict
from typing import Dict
from pathlib import Path
from rich import print,console
import numpy as np

class DataHelper:
    def __init__(self):
        self.datasets = {}
        self.available_dataset = {"gsm8k"}
        
    def get_dataset(self,dataset_name:str):
        dataset = self.datasets.get(dataset_name)
        # 如果已经在datasets中，直接提取
        if dataset:
            return dataset
        
        if dataset_name == "gsm8k":
            dataset = self.get_gsm8k()
        elif dataset_name == "mbpp":
            dataset = self.get_mbpp()
        else:
            raise AssertionError(f"do not define dataset: {dataset_name}")
        
        self.datasets[dataset_name] = dataset
        return dataset

    def evaluate(self,dataset_name:str,answers:dict,method:str="main"):
        """Evaluate the performance using provided answers.
        :param dataset_name: Name of the dataset to evaluate.
        :param answers: A dictionary of answers where keys are the question IDs.
        :param method: Evaluation method (default is 'main').
        """
        assert len(answers)>0
        dataset = self.get_dataset(dataset_name)
        if dataset_name=="gsm8k":
            #使用accuracy
            correct_count = 0
            evaluated_count = 0
            for item in dataset["test"]:
                question_id = item['id']
                if question_id in answers:
                    print(question_id,answers)
                    user_answer = answers[question_id]
                    correct_answer = item['gt']
                    if user_answer == correct_answer:
                        correct_count += 1
                    evaluated_count += 1
            accuracy = correct_count / evaluated_count 
            console.Console().log(f"Accuracy: {accuracy:.2f}")
            return accuracy
        elif dataset_name=="mbpp":
            # 要求使用temperature=1.3 top p = 0.95的nucleus采样
            if method=="main" or method == "pass@1":
                # 用Pass@k来评价，这里我们限定k=1，n=1
                evaluated_count = 0
                pass_k_accumulate = 0
                for item in dataset["test"]:
                    question_id = item['id']
                    if question_id in answers:
                        evaluated_count += 1
                        text_list = item["test_list"]
                        user_answer = answers[question_id]
                        user_answer = self.program_filter(user_answer)
                        check_program = user_answer + "\n" + item["test_setup_code"] + "\n".join(text_list)
                        pass_k_accumulate+=pass_k(1,program_test_wrapper(check_program),)
                pass_k_score = pass_k_accumulate/evaluated_count
                console.Console().log(f"Pass@k score: {pass_k_score:.2f}")
                return pass_k_score
        else:
            raise ValueError(f"no such dataset: {dataset_name}")
        return 0
                     
    def program_filter(self,completion: str) -> str:
        # The program tends to overwrite, we only take the first function
        completion = completion.lstrip("\n")
        completion.replace("\t", "    ")
        return completion.split("\n\n")[0]
     
    def get_mbpp(self):
        dataset_path = Path(__file__).parent/"dataset"/"mbpp"
        if dataset_path.exists():
            return load_from_disk(dataset_path)
        # load from hf
        dataset = load_dataset("mbpp","sanitized")
        dataset = dataset.rename_column("task_id","id")
        pattern = r'(def .+?\)) *: *'
        import re
        def preprocess(example):
            code = example["code"]
            
            match = re.search(pattern, code)
            if match:
                # 提取定义以及定义之前的所有内容
                example["def"]= match.group(1)
                return example
            else:
                print(example)
                raise AssertionError("no way")
        dataset = dataset.map(preprocess,
            desc="getting final answer",
        )
        dataset.save_to_disk(dataset_path)
        return dataset
    
    def get_gsm8k(self):
        dataset_path = Path(__file__).parent/"dataset"/"gsm8k"
        if dataset_path.exists():
            return load_from_disk(dataset_path)
        # load from hf
        dataset = load_dataset("openai/gsm8k","main")
        # 预处理，提取答案
        import re
        import uuid
        pattern = r"####\s*(-?\d+)"  #'#### 数字'
        def preprocess(example):
            match = re.search(pattern, example["answer"])
            if match:
                example["gt"] = int(match.group(1))
            else:
                print(example)
                raise AssertionError("no way")
            example["id"] = str(uuid.uuid4())
            return example
        dataset = dataset.map(preprocess,
            desc="getting final answer",
        )
        test_prompt_dataset = dataset["test"].train_test_split(test_size=5)
        dataset = DatasetDict({
            "train":dataset["train"],
            "test":test_prompt_dataset["train"],
            "prompt":test_prompt_dataset["test"],
        })
        dataset.save_to_disk(dataset_path)
        return dataset

def pass_k(num_samples:int,num_correct:int,k:int=1):
    """ 
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if num_samples - num_correct < k:
        return 1.0
    return  1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))  

def program_test_wrapper(check_program: str):
    # 创建一个进程来运行代码
    import multiprocessing
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=program_test, args=(check_program, result_queue))
    process.start()
    process.join(timeout=10)  # 设置最大运行时间为10秒
    if process.is_alive():
        process.terminate()  # 如果代码仍在运行，则终止进程
        process.join()
        return False
    return  result_queue.get()

def program_test(check_program, result_queue=None):
    try: 
        exec(check_program)
        if result_queue:
            result_queue.put(True)  # 将结果放入队列
        return True
    except Exception as e:
        if result_queue:
            result_queue.put(False)  # 将结果放入队列
        return False
    
if __name__ == "__main__":
    dh = DataHelper()
    dataset = dh.get_dataset("mbpp")
    item = dataset["test"][0]
    answers = {
        dataset["test"][0]['id']:dataset["test"][0]["code"],
        dataset["test"][1]['id']:"1",
        dataset["test"][2]['id']:"def mlgb(): return False"
    }
    print(dh.evaluate(dataset_name="mbpp",answers=answers))
    