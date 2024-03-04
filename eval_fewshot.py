from datasets import load_dataset
from Evaluator import QwenEvaluator

import os
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
import time
#os.environ['VLLM_USE_MODELSCOPE'] = "True" #dont set this environment variable if you use the local path or huggingface repo. this variable is only set for modelscope.
batch_size = 16 
OUT_DIR_ZH = "eval_result/zh_qwen_few_shot.jsonl" # the path that saves the response from model
OUT_DIR_EN = "eval_result/en_qwen_few_shot.jsonl"
en_test = "test_data/med_qa_en_4options_bigbio_qa_test_0000.parquet"
zh_test = "test_data/med_qa_zh_4options_bigbio_qa_test_0000.parquet"
model = "qwen/Qwen1.5-7B" # you may change it to your local checkpoint path
num_gpu=4 # the number of gpus used in the inference. currently only one-node multi-gpu is supported
num_fewshot=5 #the number of examples used as few shots

def run_vllm(test_data_list, language, llm, sampling_params):
    if language=="en":
        out_dir = OUT_DIR_EN
    elif language=="zh":
        out_dir = OUT_DIR_ZH
    else:
        raise NotImplementedError("please add other language support!")
    with open(out_dir, "a") as writer:
        for batch in tqdm(test_data_list):
            responses = llm.generate([item["full_prompt"] for item in batch], sampling_params)
            for i in range(len(responses)):
                prompt = responses[i].prompt
                assert prompt == batch[i]["full_prompt"]
                generated_text = responses[i].outputs[0].text
                output_dict = batch[i]
                output_dict["bot_ans"] = generated_text
                writer.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
    
# evaluated models:
# qwen/Qwen1.5-7B-Chat
def run_model():
    batch_size = 16
    dataset = load_dataset("parquet",data_files={"en_test":en_test,"zh_test":zh_test})
    # set en and zh evaluators seperately
    zh_eval_data = dataset["zh_test"]
    zh_evaluator = QwenEvaluator(choices=["A","B","C","D"], k=num_fewshot, data=zh_eval_data)
    zh_test_data_list = zh_evaluator.construct_zh_few_shot_test()
    zh_test_data_list = [zh_test_data_list[i:i+batch_size] for i in range(0, len(zh_test_data_list), batch_size)]
    
    en_eval_data = dataset["en_test"]
    en_evaluator = QwenEvaluator(choices=["A","B","C","D"], k=num_fewshot, data=en_eval_data)
    en_test_data_list = en_evaluator.construct_en_few_shot_test()
    en_test_data_list = [en_test_data_list[i:i+batch_size] for i in range(0, len(en_test_data_list), batch_size)]
    
    llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=num_gpu)  # Name or path of your model
    sampling_params = SamplingParams(temperature=1.0, top_p=0.8)
    run_vllm(zh_test_data_list, language="zh", llm=llm, sampling_params=sampling_params)
    run_vllm(en_test_data_list, language="en", llm=llm, sampling_params=sampling_params)

    
    
def eval_results():
    with open(OUT_DIR_ZH, "r") as reader:
        zh_data_list = [json.loads(i) for i in reader]
    with open(OUT_DIR_EN, "r") as reader:
        en_data_list = [json.loads(i) for i in reader]
    zh_evaluator = QwenEvaluator(choices=["A","B","C","D"], k=5, data=None)
    en_evaluator = QwenEvaluator(choices=["A","B","C","D"], k=5, data=None)
    en_acc = en_evaluator.eval_subject_en(en_data_list)
    print("The Accuracy of English Evaluation: ", en_acc)
    zh_acc = zh_evaluator.eval_subject_zh(zh_data_list)
    print("The Accuracy of Chinese Evaluation: ", zh_acc)
    

if __name__ == "__main__":
    run_model()
    time.sleep(2) # just in case 
    eval_results()
