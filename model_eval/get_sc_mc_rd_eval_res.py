from utils import get_F1Score, format_model_answer_tolist
import json, os


def process_jsonl(file_path):
    counted_num = 0
    gathered_model_answer = []
    gathered_task_gt = []

    with open(file_path, "r") as file:
        for line in file:

            sample = json.loads(line.strip())
            try:

                task_gt = sample["task_gt"]
                model_answer = sample.get("model_answer")

                formatted_model_answer = format_model_answer_tolist(model_answer, task_gt)

                gathered_task_gt.extend(task_gt)
                gathered_model_answer.extend(formatted_model_answer)
                counted_num += 1
            except:
                print(sample["id"])

    F1_pos, F1_neg, F1_w = get_F1Score(gathered_model_answer, gathered_task_gt)

    return F1_pos, F1_neg, F1_w, counted_num


task_name = "image_ref_error"
root_file_path = f"eval_res/{task_name}"

for model_name in os.listdir(root_file_path):
    if model_name.endswith(".jsonl"):

        file_path = f"eval_res/{task_name}/{model_name}"

        F1_pos, F1_neg, F1_w, counted_num = process_jsonl(file_path)

        print(task_name, model_name, f"F1_weighted: {F1_w}, counted_num:{counted_num}")
