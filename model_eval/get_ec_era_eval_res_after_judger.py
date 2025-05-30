from utils import get_F1Score, format_ms_ec_era_model_answer_tolist
import json
import os


def process_jsonl(file_path):
    overall_sample = 0
    acc_sample = 0

    with open(file_path, "r") as file:
        for line in file:

            sample = json.loads(line.strip())

            try:
                model_answer_front = sample.get("judge_res_1", [])
                model_answer_back = sample.get("judge_res_2", [])
                if model_answer_front[0] + model_answer_back[1] < model_answer_front[1] + model_answer_back[0]:
                    acc_sample += 1

                overall_sample += 1

            except:
                print(sample["id"])
    return acc_sample / overall_sample, overall_sample


task_name = "error_reason_analysis"  # error_reason_analysis error_correction
root_file_path = f"eval_res/{task_name}"

for model_name in os.listdir(root_file_path):
    if model_name.endswith("judger.jsonl"):

        file_path = f"eval_res/{task_name}/{model_name}" 

        ACC, counted_num = process_jsonl(file_path)

        # 输出结果
        print(task_name, model_name, f"ACC: {ACC}, counted_num:{counted_num}")
