from datasets import load_dataset
from Evaluator import QwenEvaluator

import os
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
os.environ['VLLM_USE_MODELSCOPE'] = "True"
batch_size = 16
OUT_DIR = "eval_result/en_qwen_few_shot.jsonl"
en_test = "test_data/med_qa_en_4options_bigbio_qa_test_0000.parquet"
zh_test = "test_data/med_qa_zh_4options_bigbio_qa_test_0000.parquet"

model = "qwen/Qwen1.5-7B"
num_gpu=4

# evaluated models:
# qwen/Qwen1.5-7B-Chat
def run_model():
    batch_size = 16
    dataset = load_dataset("parquet",data_files={"en_test":en_test,"zh_test":zh_test})
    #zh_eval_data = dataset["zh_test"]
    en_eval_data = dataset["en_test"]
    evaluator = QwenEvaluator(choices=["A","B","C","D"], k=5, data=en_eval_data)
    #test_data_list = evaluator.construct_zh_few_shot_test()
    test_data_list = evaluator.construct_en_few_shot_test()
    test_data_list = [test_data_list[i:i+batch_size] for i in range(0, len(test_data_list), batch_size)]
    llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=num_gpu)  # Name or path of your model
    sampling_params = SamplingParams(temperature=1.0, top_p=0.8)
    with open(OUT_DIR, "a") as writer:
        for batch in tqdm(test_data_list):
            responses = llm.generate([item["full_prompt"] for item in batch], sampling_params)
            for i in range(len(responses)):
                prompt = responses[i].prompt
                assert prompt == batch[i]["full_prompt"]
                generated_text = responses[i].outputs[0].text
                output_dict = batch[i]
                output_dict["bot_ans"] = generated_text
                writer.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
    
def eval_results():
    with open(OUT_DIR, "r") as reader:
        data_list = [json.loads(i) for i in reader]
    evaluator = QwenEvaluator(choices=["A","B","C","D"], k=5, data=None)
    acc = evaluator.eval_subject(data_list)
    print(acc)

if __name__ == "__main__":
    run_model()
    #eval_results()