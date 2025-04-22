import re
from sklearn.metrics import f1_score
import numpy as np


def format_model_answer_tolist(model_answer, task_gt):

    numbers = re.findall(r'\d+', model_answer)
    

    result = [int(num) for num in numbers]
    

    result = [num if num == 0 or num == 1 else 1 for num in result]
    

    if len(result) >= len(task_gt):
        return result[:len(task_gt)]
    else:

        return result + [0] * (len(task_gt) - len(result))


def format_ms_ec_era_model_answer_tolist(model_answer, task_gt):

    numbers = re.findall(r'\d+', model_answer)
    

    result = [int(num) for num in numbers]
    

    if len(result) >= len(task_gt):

        return result[-len(task_gt):]
    else:

        return result + [0] * (len(task_gt) - len(result))


def get_F1Score(gathered_model_answer, gathered_task_gt):

    model_answer = np.array(gathered_model_answer)
    task_gt = np.array(gathered_task_gt)


    pos_count = np.sum(task_gt == 1)
    neg_count = np.sum(task_gt == 0)


    F1_pos = f1_score(task_gt, model_answer, pos_label=1)
    F1_neg = f1_score(task_gt, model_answer, pos_label=0)

    w_pos = neg_count / (pos_count + neg_count)
    w_neg = pos_count / (pos_count + neg_count)


    F1_w = w_neg * F1_neg + w_pos * F1_pos


    return F1_pos, F1_neg, F1_w



from PIL import Image
import io
import base64

def compress_image(image, max_size):

    quality = 85  
    output_buffer = io.BytesIO()  

    while True:
        image.save(output_buffer, format=image.format.upper(), quality=quality)  
        current_size = output_buffer.getbuffer().nbytes  
        if current_size <= max_size or quality <= 10:
            break  
        quality -= 5  
        output_buffer.seek(0)  
        output_buffer.truncate(0)

    output_buffer.seek(0)
    return Image.open(output_buffer)


def encode_image(image_path, max_size_bytes=1024 * 1024):

    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            original_image = Image.open(io.BytesIO(image_data))
            image_type = original_image.format.lower()

            
            min_side = min(original_image.size)
            if min_side < 36:
                if original_image.width < original_image.height:
                    new_width = 36
                    new_height = int(original_image.height * (36 / original_image.width))
                else:
                    new_height = 36
                    new_width = int(original_image.width * (36 / original_image.height))
                
                original_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            
            if len(image_data) > max_size_bytes:
                compressed_image = compress_image(original_image, max_size_bytes)
                image_data = io.BytesIO()
                compressed_image.save(image_data, format=image_type.upper(), quality=70)
                image_data = image_data.getvalue()

            
            base64_image = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/{image_type};base64,{base64_image}"
    except IOError:
        raise ValueError("invalid image file")