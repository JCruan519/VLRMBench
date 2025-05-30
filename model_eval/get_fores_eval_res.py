import re
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
                model_answer = sample.get("model_answer", [])

                if task_gt == True:
                    if re.search(r"\b(yes|true)\b", model_answer, re.IGNORECASE):
                        acc_sample += 1

                elif task_gt == False:
                    if re.search(r"\b(no|false)\b", model_answer, re.IGNORECASE):
                        acc_sample += 1

                overall_sample += 1

            except:
                print(sample["id"])
    return acc_sample / overall_sample, overall_sample


# example
task_name = "foresight"
root_file_path = f"eval_res/{task_name}"


for model_name in os.listdir(root_file_path):
    if model_name.endswith(".jsonl") and "qvq" in model_name:

        file_path = f"eval_res/{task_name}/{model_name}"

        acc, counted_num = process_jsonl(file_path)

        print(task_name, model_name, f"acc: {acc}, counted_num:{counted_num}")
