import json
import base64
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import OpenAI
import time
import re
from tqdm import tqdm
from utils import encode_image
import argparse

parser = argparse.ArgumentParser(description="VLMRMBench_online_API")
parser.add_argument("--VLM_MODEL", type=str, default="qvq_72b", help="VLM model name to use, default: qvq_72b")
parser.add_argument(
    "--test_file_name", type=str, default="location_error", help="Test file name, default: location_error"
)
parser.add_argument("--num_threads", type=int, default=6, help="Thread pool size, default: 6")
parser.add_argument("--max_retries", type=int, default=1, help="Maximum number of retries, default: 3")
parser.add_argument("--timeout", type=int, default=240)
parser.add_argument("--max_token_len", type=int, default=256)
parser.add_argument("--scale_token_len", type=int, default=16)
args = parser.parse_args()

VLM_MODEL = args.VLM_MODEL
test_file_name = args.test_file_name
num_threads = args.num_threads
max_retries = args.max_retries
TIMEOUT = args.timeout
MAX_TOKEN_LEN = args.max_token_len


client = OpenAI(api_key="", base_url="")  # <Your API Key>  # <Your API Base Url>

group_1 = [
    "step_correctness",
    "redundant_det",
    "most_confidence",
    "foresight",
    "error_correction",
    "error_reason_analysis",
    "attribute_hallucination",
    "existence_hallucination",
    "detail_error",
    "image_ref_error",
    "location_error",
]
group_2 = ["multi_solution"]

if test_file_name == "step_correctness":
    from eval_prompt_files import step_correctness_eval_prompt

    eval_prompt = step_correctness_eval_prompt
elif test_file_name == "redundant_det":
    from eval_prompt_files import redundant_det_eval_prompt

    eval_prompt = redundant_det_eval_prompt
elif test_file_name == "most_confidence":
    from eval_prompt_files import most_confidence_eval_prompt

    eval_prompt = most_confidence_eval_prompt
elif test_file_name == "foresight":
    from eval_prompt_files import foresight_eval_prompt

    eval_prompt = foresight_eval_prompt
elif test_file_name == "error_correction":
    from eval_prompt_files import error_correction_eval_prompt

    eval_prompt = error_correction_eval_prompt
    MAX_TOKEN_LEN = MAX_TOKEN_LEN * args.scale_token_len
elif test_file_name == "error_reason_analysis":
    from eval_prompt_files import error_reason_analysis_eval_prompt

    eval_prompt = error_reason_analysis_eval_prompt
    MAX_TOKEN_LEN = MAX_TOKEN_LEN * args.scale_token_len
elif test_file_name == "attribute_hallucination":
    from eval_prompt_files import attribute_hallucination_eval_prompt

    eval_prompt = attribute_hallucination_eval_prompt
elif test_file_name == "existence_hallucination":
    from eval_prompt_files import existence_hallucination_eval_prompt

    eval_prompt = existence_hallucination_eval_prompt
elif test_file_name == "detail_error":
    from eval_prompt_files import detail_error_eval_prompt

    eval_prompt = detail_error_eval_prompt
elif test_file_name == "image_ref_error":
    from eval_prompt_files import image_ref_error_eval_prompt

    eval_prompt = image_ref_error_eval_prompt
elif test_file_name == "location_error":
    from eval_prompt_files import location_error_eval_prompt

    eval_prompt = location_error_eval_prompt
elif test_file_name == "multi_solution":
    from eval_prompt_files import multi_solution_eval_prompt

    eval_prompt = multi_solution_eval_prompt
else:
    raise ValueError(f"Invalid test_file_name: {test_file_name}.")


# File path configuration
input_file = f"benchmark_data/{test_file_name}.jsonl"  # path of the benchmark data
output_file = f"eval_res/{test_file_name}/{VLM_MODEL}.jsonl"  # path to save the eval results
os.makedirs(os.path.dirname(output_file), exist_ok=True)
image_base_path = "meta_data/Image"  # path of the image

write_lock = threading.Lock()

processed_ids = set()

stop_processing_event = threading.Event()


def process_line_group_1(line):
    if stop_processing_event.is_set():
        return
    try:
        data = json.loads(line)
        entry_id = data["id"]
        if entry_id in processed_ids:
            print(f"Entry {entry_id} already processed, skipping.")
            return
        if "error_info" in data:
            return

        # Handle image: read and convert to base64 encoding
        if "image" in data and isinstance(data["image"], list):
            base64_image_list = []
            for image_path in data["image"]:
                image_full_path = os.path.join(image_base_path, image_path)
                if os.path.exists(image_full_path):
                    base64_image = encode_image(image_full_path)
                    base64_image_list.append(base64_image)
                else:
                    print(f"Image {image_full_path} error, skipping entry {entry_id}.")
        else:
            return

        question = data["question"]
        task_gt = data["task_gt"]

        if test_file_name == "foresight":
            if "reasoning_error" in data:
                reasoning_process = data["reasoning_error"]
            elif "step_list" in data:
                reasoning_process = data["step_list"]
            else:
                print(f"step_list or reasoning_error are not in Entry {entry_id}, skipping.")
                return
        else:
            reasoning_process = data["reasoning_error"]

        content = []

        # Add images to content
        if base64_image_list:
            for base64_image in base64_image_list:
                content.append({"type": "image_url", "image_url": {"url": base64_image}})

        # Add text explanation
        content.append(
            {"type": "text", "text": eval_prompt.format(question=question, reasoning_process=reasoning_process)}
        )

        messages = [{"role": "user", "content": content}]

        retry_attempts = 0

        res_data = {}
        res_data["id"] = data["id"]
        if test_file_name == "error_correction":
            res_data["image"] = data["image"]
            res_data["question"] = question
        if test_file_name == "error_reason_analysis":
            res_data["image"] = data["image"]
            res_data["question"] = question
            res_data["reasoning_error"] = data["reasoning_error"]
        res_data["task_gt"] = task_gt

        while retry_attempts < max_retries:

            try:
                response = client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=messages,
                    temperature=0,
                    timeout=TIMEOUT,
                    max_tokens=MAX_TOKEN_LEN,
                )
                model_answer = response.choices[0].message.content
                res_data["model_answer"] = model_answer
                with write_lock:
                    with open(output_file, "a", encoding="utf-8") as outfile:
                        json.dump(res_data, outfile, ensure_ascii=False)
                        outfile.write("\n")
                    processed_ids.add(entry_id)
                break

            except Exception as e:
                retry_attempts += 1
                print(f"An error occurred while processing entry {entry_id} for {retry_attempts}/{max_retries}: {e}")

    except json.JSONDecodeError:
        print("Invalid JSON format, skipping line.")
        return


def process_line_group_2(line):
    if stop_processing_event.is_set():
        return
    try:
        data = json.loads(line)
        entry_id = data["id"]
        if entry_id in processed_ids:
            print(f"Entry {entry_id} already processed, skipping.")
            return
        if "error_info" in data:
            return

        # Handle image: read and convert to base64 encoding
        if "image" in data and isinstance(data["image"], list):
            base64_image_list = []
            for image_path in data["image"]:
                image_full_path = os.path.join(image_base_path, image_path)
                if os.path.exists(image_full_path):
                    base64_image = encode_image(image_full_path)
                    base64_image_list.append(base64_image)
                else:
                    print(f"Image {image_full_path} error, skipping entry {entry_id}.")
        else:
            return

        question = data["question"]
        ai_respond_1 = data["step_list"]
        ai_respond_2 = data["reasoning_error"]

        content_front = []
        content_back = []

        # Add images to content
        if base64_image_list:
            for base64_image in base64_image_list:
                content_front.append({"type": "image_url", "image_url": {"url": base64_image}})
                content_back.append({"type": "image_url", "image_url": {"url": base64_image}})

        # Add text explanation
        content_front.append(
            {
                "type": "text",
                "text": eval_prompt.format(question=question, ai_respond_1=ai_respond_1, ai_respond_2=ai_respond_2),
            }
        )
        content_back.append(
            {
                "type": "text",
                "text": eval_prompt.format(question=question, ai_respond_1=ai_respond_2, ai_respond_2=ai_respond_1),
            }
        )

        retry_attempts = 0

        res_data = {}
        res_data["id"] = data["id"]
        res_data["task_gt"] = [0, 1]

        while retry_attempts < max_retries:

            try:
                response = client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=[{"role": "user", "content": content_front}],
                    temperature=0,
                    timeout=TIMEOUT,
                    max_tokens=MAX_TOKEN_LEN,
                )
                model_answer = response.choices[0].message.content
                res_data["model_answer_front"] = model_answer

                response = client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=[{"role": "user", "content": content_back}],
                    temperature=0,
                    timeout=TIMEOUT,
                    max_tokens=MAX_TOKEN_LEN,
                )
                model_answer = response.choices[0].message.content
                res_data["model_answer_back"] = model_answer

                with write_lock:
                    with open(output_file, "a", encoding="utf-8") as outfile:
                        json.dump(res_data, outfile, ensure_ascii=False)
                        outfile.write("\n")
                    processed_ids.add(entry_id)

                break

            except Exception as e:
                retry_attempts += 1
                print(f"An error occurred while processing entry {entry_id} for {retry_attempts}/{max_retries}: {e}")

    except json.JSONDecodeError:
        print("Invalid JSON format, skipping line.")
        return


# Read existing entry_ids from output file to avoid reprocessing
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as outfile:
        for line in outfile:
            try:
                existing_data = json.loads(line)
                processed_ids.add(existing_data["id"])
            except json.JSONDecodeError:
                continue

# Read input file and filter out processed or rejected entries
filtered_lines = []
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            data = json.loads(line)
            entry_id = data["id"]
            if entry_id in processed_ids:
                continue  # Skip already processed or rejected entry
            filtered_lines.append(line)
        except json.JSONDecodeError:
            print("Invalid JSON format, skipping line.")
            continue

print(
    f"BEGIN TEST: {test_file_name}, MODEL: {VLM_MODEL}, Total lines to process after filtering: {len(filtered_lines)}"
)


if test_file_name in group_1:
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_line_group_1, line) for line in filtered_lines]
        with tqdm(total=len(filtered_lines), desc="Processing lines", unit="line") as pbar:
            for future in as_completed(futures):
                future.result()  # Wait for task completion
                pbar.update(1)
elif test_file_name in group_2:
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_line_group_2, line) for line in filtered_lines]
        with tqdm(total=len(filtered_lines), desc="Processing lines", unit="line") as pbar:
            for future in as_completed(futures):
                future.result()  # Wait for task completion
                pbar.update(1)

print("Processing complete.")
