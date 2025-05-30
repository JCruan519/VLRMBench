import re
import json
import openai
import os
import concurrent.futures
from openai import OpenAI
from eval_prompt_files import error_correction_judge_prompt, error_reason_analysis_judge_prompt
from utils import format_ms_ec_era_model_answer_tolist, encode_image
from tqdm import tqdm


client = OpenAI( # remember to set your VLLM url and api_key if needed
    base_url="http://localhost:12453/v1",
    api_key="token-abc123",
)
JUDGE_MODEL = "qwen2_5_vl_72b_vllm"

max_retries = 3
max_threads = 8  # Specify the number of threads


image_base_path = "meta_data/Image"

###############
for task_name in ['error_reason_analysis', 'error_correction']:
    root_file_path = f"eval_res/{task_name}" # replace eval_res with the actual path

    for model_name in ['minicpm_o_2_6', 'minicpm_v_2_6', 'ovis2_8b_localinf', 'ovis2_16b_localinf', 'ovis2_34b_localinf', 'qwen2_5_vl_72b_vllm']:

        print(model_name)

        if task_name == "error_correction":
            eval_prompt = error_correction_judge_prompt
        elif task_name == "error_reason_analysis":
            eval_prompt = error_reason_analysis_judge_prompt
        else:
            raise ValueError(f"Invalid task_name: {task_name}.")

        file_path = f"eval_res/{task_name}/{model_name}.jsonl"
        output_file_path = f"eval_res/{task_name}/{model_name}_wr_by_{JUDGE_MODEL}_judger.jsonl"
        save_acc_res_file_path = f"eval_res/{task_name}/{task_name}_wr_by_{JUDGE_MODEL}_judger_acc_log.txt"

        overall_sample = 0
        win_sample = 0
        tie_sample = 0
        lose_sample = 0

        def process_sample(sample, progress_bar):
            global overall_sample, win_sample, tie_sample, lose_sample
            entry_id = sample['id']

            # Load the existing output file to check processed IDs
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r') as outfile:
                    processed_ids = {json.loads(line.strip())['id'] for line in outfile}
                if entry_id in processed_ids:
                    progress_bar.update(1)
                    return

            try:
                # Get task_gt and model_answer
                task_gt = sample['task_gt']
                model_answer = sample['model_answer']
                question = sample['question']

                content_front = []
                content_back = []

                # Process images
                base64_image_list = []
                if 'image' in sample and isinstance(sample['image'], list):
                    for image_path in sample['image']:
                        image_full_path = os.path.join(image_base_path, image_path)
                        if os.path.exists(image_full_path):
                            base64_image = encode_image(image_full_path)
                            base64_image_list.append(base64_image)
                        else:
                            print(f"Image {image_full_path} error, skipping entry {entry_id}.")

                if base64_image_list:
                    for base64_image in base64_image_list:
                        content_front.append({
                            "type": "image_url",
                            "image_url": {'url': base64_image}
                        })
                        content_back.append({
                            "type": "image_url",
                            "image_url": {'url': base64_image}
                        })

                if task_name == "error_correction":
                    content_front.append({
                        "type": "text",
                        "text": eval_prompt.format(question=question,
                                                    ai_respond_1=task_gt,
                                                    ai_respond_2=model_answer)
                    })
                    content_back.append({
                        "type": "text",
                        "text": eval_prompt.format(question=question,
                                                    ai_respond_1=model_answer,
                                                    ai_respond_2=task_gt)
                    })
                elif task_name == "error_reason_analysis":
                    content_front.append({
                        "type": "text",
                        "text": eval_prompt.format(question=question,
                                                    reasoning_error=sample['reasoning_error'],
                                                    ai_respond_1=task_gt,
                                                    ai_respond_2=model_answer)
                    })
                    content_back.append({
                        "type": "text",
                        "text": eval_prompt.format(question=question,
                                                    reasoning_error=sample['reasoning_error'],
                                                    ai_respond_1=model_answer,
                                                    ai_respond_2=task_gt)
                    })
                else:
                    raise ValueError(f"Invalid test_file_name: {test_file_name}.")

                retry_attempts = 0
                while retry_attempts < max_retries:
                    try:
                        # Query model for response
                        response = client.chat.completions.create(
                            model=JUDGE_MODEL,
                            messages=[{
                                'role': 'user',
                                'content': content_front
                            }],
                            temperature=0,
                            timeout=240,
                        )
                        judge_res_1 = response.choices[0].message.content
                        judge_res_1 = format_ms_ec_era_model_answer_tolist(judge_res_1, [1, 1])

                        response = client.chat.completions.create(
                            model=JUDGE_MODEL,
                            messages=[{
                                'role': 'user',
                                'content': content_back
                            }],
                            temperature=0,
                            timeout=240,
                        )
                        judge_res_2 = response.choices[0].message.content
                        judge_res_2 = format_ms_ec_era_model_answer_tolist(judge_res_2, [1, 1])

                        # Update overall sample count
                        overall_sample += 1
                        gpt_4o_score = judge_res_1[0] + judge_res_2[1]
                        model_score = judge_res_1[1] + judge_res_2[0]
                        if model_score > gpt_4o_score:
                            win_sample += 1
                        elif model_score == gpt_4o_score:
                            tie_sample += 1
                        else:
                            lose_sample += 1

                        # Save results to JSONL
                        result = {
                            'id': entry_id,
                            'judge_res_1': judge_res_1,
                            'judge_res_2': judge_res_2
                        }
                        with open(output_file_path, 'a') as outfile:
                            outfile.write(json.dumps(result) + '\n')

                        break

                    except Exception as e:
                        retry_attempts += 1
                        print(f"An error occurred while processing entry {entry_id} for {retry_attempts}/{max_retries}: {e}")

            except Exception as e:
                print(f"Error processing sample {entry_id}: {e}")

            finally:
                progress_bar.update(1)

        def process_jsonl(file_path):
            with open(file_path, 'r') as file:
                samples = [json.loads(line.strip()) for line in file]

            # Initialize progress bar with the total number of samples
            total_samples = len(samples)
            print(f"Total samples to process: {total_samples}")

            # Track filtered and pending samples
            filtered_samples = 0
            pending_samples = total_samples

            # Filter out already processed samples
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r') as outfile:
                    processed_ids = {json.loads(line.strip())['id'] for line in outfile}

                samples = [sample for sample in samples if sample['id'] not in processed_ids]
                filtered_samples = total_samples - len(samples)
                pending_samples = len(samples)

            print(f"Filtered {filtered_samples} samples, {pending_samples} samples remaining to process.")

            # Create progress bar with total samples
            with tqdm(total=pending_samples, desc="Processing entries", unit="entry") as progress_bar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                    executor.map(lambda sample: process_sample(sample, progress_bar), samples)

            print(f"Win: {win_sample}/{overall_sample} = {win_sample / overall_sample:.4f}; Tie: {tie_sample}/{overall_sample} = {tie_sample / overall_sample:.4f}; Lose: {lose_sample}/{overall_sample} = {lose_sample / overall_sample:.4f}; ")
            return win_sample / overall_sample, tie_sample / overall_sample, lose_sample / overall_sample

        # Run the processing
        Win, Tie, Lose = process_jsonl(file_path)
        print(f"{task_name}, {model_name}, Win: {Win}, Tie: {Tie}, Lose: {Lose}")

        if not os.path.exists(save_acc_res_file_path):
            with open(save_acc_res_file_path, "w") as file:
                pass  

        with open(save_acc_res_file_path, "a") as file:
            file.write(f"{task_name}, {model_name}, Win: {Win}, Tie: {Tie}, Lose: {Lose}\n")
