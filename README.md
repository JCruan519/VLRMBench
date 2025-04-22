# VLRMBench

This is the official repository for the paper:

**"VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models"**

---

## 📦 Dataset

- Benchmark `.jsonl` files are located in the [`benchmark_data/`](benchmark_data) directory.
- Images can be downloaded from [this link](https://huggingface.co/datasets/Winston-Yuan/VLRMBench) and should be extracted to:

```
meta_data/Image
```

Each file in the `benchmark_data/` folder corresponds to one specific evaluation task in VLRMBench:

| File Name                         | Task Name                | Abbreviation |
|----------------------------------|---------------------------|--------------|
| `step_correctness.jsonl`         | Step Correctness          | SC           |
| `redundant_det.jsonl`            | Redundant Detection       | RD           |
| `most_confidence.jsonl`          | Confidence Misdirection   | CM           |
| `existence_hallucination.jsonl`  | Existence Hallucination   | EH           |
| `attribute_hallucination.jsonl`  | Attribute Hallucination   | AH           |
| `detail_error.jsonl`             | Detail Error              | DE           |
| `location_error.jsonl`           | Spatial Relationship      | SR           |
| `image_ref_error.jsonl`          | Image Confusion           | IRE          |
| `multi_solution.jsonl`           | Multi-Solution            | MS           |
| `foresight.jsonl`                | Forecasting Future        | FF           |
| `error_reason_analysis.jsonl`    | Error Reason Analysis     | ERA          |
| `error_correction.jsonl`         | Error Correction          | EC           |

Each `.jsonl` file contains multiple entries (one per line), each representing a benchmark instance for that task. Fields may vary depending on the task type.


## 🔍 Evaluation

> **Important:** Please make sure to update your API keys and dataset paths before running evaluations.

### 1. Configuration

Modify the model, dataset paths, and API credentials in the following files:
- `model_eval/run_vllm.sh`
- `model_eval/run_vllm_api_eval_with_metrices.sh`
- `model_eval/vllm_localapi_eval.py`
- `model_eval/run_online_api_eval_with_metrices.sh`
- `model_eval/online_api_eval.py`

### 2. Local Model Evaluation

Start your VLLM server using:
```bash
bash model_eval/run_vllm.sh
```

Then run:

```bash
bash model_eval/run_vllm_api_eval_with_metrices.sh
```

This will evaluate the local Vision-Language model using the VLRMBench benchmark.

### 3. Online Model Evaluation

To evaluate remote models via API (e.g., OpenAI, Claude, Gemini), run:

```bash
bash model_eval/run_online_api_eval_with_metrices.sh
```

---

## 📜 Citation

If you use this benchmark or codebase in your research, please cite:

**VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models**  
arXiv: [2503.07478](https://arxiv.org/abs/2503.07478)

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or contact the authors directly.
