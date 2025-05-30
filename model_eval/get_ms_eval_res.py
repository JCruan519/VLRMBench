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
                task_gt = sample["task_gt"]
                model_answer_front = sample.get("model_answer_front") or "[0,0]"
                model_answer_back = sample.get("model_answer_back") or "[0,0]"

                formatted_model_answer_front = format_ms_ec_era_model_answer_tolist(model_answer_front, task_gt)
                formatted_model_answer_back = format_ms_ec_era_model_answer_tolist(model_answer_back, task_gt)

                if (
                    formatted_model_answer_front[0] + formatted_model_answer_back[1]
                    > formatted_model_answer_front[1] + formatted_model_answer_back[0]
                ):
                    acc_sample += 1

                overall_sample += 1

            except:
                print(sample["id"])
    return acc_sample / overall_sample, overall_sample


# example
task_name = "multi_solution"
root_file_path = f"eval_res/{task_name}"


for model_name in os.listdir(root_file_path):
    if model_name.endswith(".jsonl"):

        try:

            file_path = f"eval_res/{task_name}/{model_name}"

            ms_acc, counted_num = process_jsonl(file_path)

            print(task_name, model_name, f"ms_acc: {ms_acc}, counted_num:{counted_num}")

        except:
            print("fffffffffffffffffff", model_name)
