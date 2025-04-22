# Eval Prompt

step_correctness_eval_prompt = """## Task Description
Based on the provided image, question and AI assistant's reasoning process, your task \
is to identify the steps that contains errors in the reasoning process.

## Instructions
- You need to assign an integral indicator to each step. 0 indicates that the step does not contain errors, \
and 1 indicates that the step contains errors. 

## Question
{question}

## AI assistant's reasoning process
{reasoning_process}

## Output Format
[Indicator for STEP 1, Indicator for STEP 2, ..., Indicator for STEP N]

## Your Output: 
"""

most_confidence_eval_prompt = step_correctness_eval_prompt


multi_solution_eval_prompt = """## Task Description
You are an expert judge specializing in assessing AI-generated responses. \

## Instructions
- Your role is to objectively evaluate the quality of two AI assistants' responses \
based on the given image and question. 
- Assign a score to each response on a scale of 0 to 10, where 0 indicates a response \
that is entirely illogical or irrelevant, and 10 signifies a response \
that is exceptionally well-reasoned, accurate, and directly addresses the question.

## Question
{question}

## AI Assistant's Response 1
{ai_respond_1}

## AI Assistant's Response 2
{ai_respond_2}

## Output Format
Return a List containing two scores:
["Score for Response 1", "Score for Response 2"]

## Your Output: 
"""


redundant_det_eval_prompt = """## Task Description
Based on the provided image, question and AI assistant's reasoning process, your task \
is to identify the steps that contains redundancy in the reasoning process.

## Instructions
- The redundancy may involve unnecessary elaboration, repetitive expressions, \
or redundant details that do not contribute to the correctness of the reasoning.
- You need to assign an integral indicator to each step. 0 indicates that the step does not contain redundancy, \
and 1 indicates that the step contains redundancy. 

## Question
{question}

## AI assistant's reasoning process
{reasoning_process}

## Output Format
[Indicator for STEP 1, Indicator for STEP 2, ..., Indicator for STEP N]

## Your Output: 
"""


detail_error_eval_prompt = """## Task Description
Based on the provided image, question and AI assistant's reasoning process, your task \
is to identify the steps that contains detail error in the reasoning process.

## Instructions
- There is at least one error in the reasoning process.
- Detail errors refer to errors in which numbers or calculated symbols are wrong \
- You need to assign an integral indicator to each step. 0 indicates that the step does not contain errors, \
and 1 indicates that the step contains errors. 

## Question
{question}

## AI assistant's reasoning process
{reasoning_process}

## Output Format
[Indicator for STEP 1, Indicator for STEP 2, ..., Indicator for STEP N]

## Your Output: 
"""


existence_hallucination_eval_prompt = """## Task Description
Based on the provided image, question and AI assistant's reasoning process, your task \
is to identify the steps that contains existence hallucination error in the reasoning process.

## Instructions
- There is at least one error in the reasoning process.
- The term "existence hallucination error" refers to errors in reasoning where a model either: \
Introduces entities that do not exist in the image (e.g., hallucinating objects, people, digit, or \
details not present), or Fails to acknowledge entities that are actually present in the image \
(e.g., ignoring critical objects or contextual elements).
- You need to assign an integral indicator to each step. 0 indicates that the step does not contain errors, \
and 1 indicates that the step contains errors. 

## Question
{question}

## AI assistant's reasoning process
{reasoning_process}

## Output Format
[Indicator for STEP 1, Indicator for STEP 2, ..., Indicator for STEP N]

## Your Output: 
"""


attribute_hallucination_eval_prompt = """## Task Description
Based on the provided image, question and AI assistant's reasoning process, your task \
is to identify the steps that contains Attribute hallucination error in the reasoning process.

## Instructions
- There is at least one error in the reasoning process.
- The term "Attribute hallucination error" refers to errors in reasoning where a model Misidentifies properties \
(e.g., size, color, shape) of entities in an image.
- You need to assign an integral indicator to each step. 0 indicates that the step does not contain errors, \
and 1 indicates that the step contains errors. 

## Question
{question}

## AI assistant's reasoning process
{reasoning_process}

## Output Format
[Indicator for STEP 1, Indicator for STEP 2, ..., Indicator for STEP N]

## Your Output: 
"""


location_error_eval_prompt = """## Task Description
Based on the provided image, question and AI assistant's reasoning process, your task \
is to identify the steps that contains location errors in the reasoning process.

## Instructions
- There is at least one error in the reasoning process.
- You need to assign an integral indicator to each step. 0 indicates that the step does not contain errors, \
and 1 indicates that the step contains errors. 

## Question
{question}

## AI assistant's reasoning process
{reasoning_process}

## Output Format
[Indicator for STEP 1, Indicator for STEP 2, ..., Indicator for STEP N]

## Your Output: 
"""


image_ref_error_eval_prompt = """## Task Description
Based on the provided image, question and AI assistant's reasoning process, your task \
is to identify the steps that contains image reference errors in the reasoning process.

## Instructions
- There is at least one error in the reasoning process.
- The term "Image reference error" refers to a mistake where the model incorrectly identifies an entity from one image as being present in another image.
- You need to assign an integral indicator to each step. 0 indicates that the step does not contain errors, \
and 1 indicates that the step contains errors. 

## Question
{question}

## AI assistant's reasoning process
{reasoning_process}

## Output Format
[Indicator for STEP 1, Indicator for STEP 2, ..., Indicator for STEP N]

## Your Output: 
"""


foresight_eval_prompt = """## Task Description
Based on the provided image, question and part of the AI assistant's reasoning process, \
Your task is to evaluate whether the given partial reasoning process will \
lead to a correct final answer.

## Question
{question}

## AI Assistant's Partial Reasoning Process
{reasoning_process}

## Only output Yes or No without additional explanation: 
"""


error_correction_eval_prompt = """## Task Description
You are an expert in reasoning process correction. Based on the given image, a question, an AI assistant's reasoning process \
(which contains errors), your task is to identify and correct the errors in the reasoning process, \
providing a fully corrected reasoning process.

## Instructions
- Correct any logical flaws or misinterpretations in the reasoning.
- Structure the corrected reasoning process clearly in a step-by-step manner.
- Ensure that your modified reasoning follows a logical flow that leads to the correct answer.

## Question
{question}

## AI Assistant's Reasoning Process
{reasoning_process}

## Your modified process must follow this format: ["STEP1: ...", "STEP2: ...", ...]

## Your output: 
"""

error_correction_judge_prompt = multi_solution_eval_prompt


error_reason_analysis_eval_prompt = """## Task Description
You are an expert in analyzing the causes of errors in the reasoning process. Based on the given image, question, and AI assistant's reasoning process \
(which contains errors), your task is to critically analyze the reasoning, identify incorrect steps, explain the errors in detail.

## Instructions
- Identify each incorrect step or assumption in the reasoning process.
- For each error, provide a detailed explanation of why it is incorrect, and describe the underlying cause of the error.

## Question
{question}

## AI Assistant's Reasoning Process
{reasoning_process}

## Your output: 
"""

error_reason_analysis_judge_prompt = """## Task Description
You are an experienced judge specializing in assessing AI-generated responses. 
Based on the given image, a question, and an incorrect reasoning process, along with two AI assistant's responses (which include an analysis of \
the reasoning errors), your task is to evaluate the quality of each AI assistant's response. 

## Instructions
- Judge which AI response provides a clearer and more logical analysis of the errors.
- Assign a score to each response from 0 to 10:
  - 0 represents a response that identifies no errors.
  - 10 represents a response that fully identifies all errors and provides a clear, logical, and reasonable analysis of the causes.

## Question
{question}

## Incorrect Reasoning Process
{reasoning_error}

## AI Assistant's Response 1
{ai_respond_1}

## AI Assistant's Response 2
{ai_respond_2}

## Output Format 
Return a list containing two scores: 
["Score for Response 1", "Score for Response 2"]

## Your Output:
"""
