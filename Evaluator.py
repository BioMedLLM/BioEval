import os
from tqdm import tqdm
import re
import string

class Evaluator:
    def __init__(self, choices, model_name, k=-1):
        self.choices = choices
        self.model_name = model_name
        self.k = k
        self.puncs = list(string.punctuation)

    def format_example(self, line, include_answer=True):
        # print(example)
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += '\n答案：'
        if include_answer:
            example += f'{line["answer"]}\n\n'
        return example
    
    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None):
        pass

    def normalize_answer(self,s):

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude=set(self.puncs)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def exact_match(self,pred, target):
        return self.normalize_answer(pred)==self.normalize_answer(target)

class QwenEvaluator(Evaluator):
    def __init__(self, choices, k, data):
        super(QwenEvaluator, self).__init__(choices, k)
        self.data = data
        self.k = k
        #choices["A","B","C","D"]

    def format_example_zh(self,line,include_answer=True,cot=False):
        question, can_ans, ans_str= line['question'], line["choices"], line["answer"][0]
        ans = None
        example = question
        for i in range(len(self.choices)):
            if can_ans[i] == ans_str:
                ans = self.choices[i]+". "+ans_str
                chosen = self.choices[i]
            can_str = "\n"+self.choices[i]+". "+can_ans[i]
            example += can_str
        assert ans!= None, print(line)
        example += "\n答案："
        if include_answer:
            example += ans
        return {"full_prompt":example, "question":question, "ans":ans, "ans_str":ans_str,"choices":can_ans,"chosen":chosen}

    def generate_few_shot_prompt_zh(self, cot=False):
        sys_prompt="你是一个中文人工智能助手，以下是中国关于医学考试的单项选择题，请选出其中的正确答案。\n"
        k=self.k
        # we use the top k examples as the few shot prompt
        prompt = sys_prompt
        for i in range(k):
            tmp=self.format_example_zh(self.data[i],include_answer=True,cot=cot)
            prompt+=tmp["full_prompt"] + "\n"
        return prompt
    
    def construct_zh_few_shot_test(self):
        data_list = []
        fewshot_prompt = self.generate_few_shot_prompt_zh()
        print(fewshot_prompt,"///")
        for i in range(self.k, len(self.data)):
            tmp = self.format_example_zh(self.data[i], include_answer=False)
            tmp["full_prompt"] = fewshot_prompt +"\n"+ tmp["full_prompt"]
            data_list.append(tmp)
        return data_list

    def eval_subject(self, data_list, few_shot=False, save_result_dir=None,cot=False):
        correct_num = 0
        if save_result_dir:
            result = []
            score=[]
        for d in data_list:
            #print(response_str)
            response_str, answer, ans_str = d["bot_ans"], d["chosen"], d["ans"]
            if cot:
                ans_list=re.findall(r"答案是(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"答案为(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"选项(.+?)是正确的。",response_str)

                if len(ans_list)==0:
                    correct=0
                else:
                    if self.exact_match(ans_list[-1],ans_str):
                        correct_num+=1
                        correct=1
                    else:
                        correct=0
            else:
                response_str=response_str.strip()
                if few_shot:
                    if len(response_str)>0:
                        if self.exact_match(response_str,ans_str):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
                else:
                    if len(response_str)>0:
                        ans_list=self.extract_ans(response_str)
                        if len(ans_list)>0 and (ans_list[-1]==answer):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
            if save_result_dir:
                result.append(response_str)
                score.append(correct)
        print(correct_num)
        correct_ratio = 100*correct_num/len(data_list)
        
        return correct_ratio

    def extract_ans(self,response_str):
        pattern=[
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
            r"([A-D])\.\s?"
        ]
        ans_list=[]
        if response_str[0] in ["A",'B','C','D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list)==0:
                ans_list=re.findall(p,response_str)
            else:
                break
        return ans_list