#!/usr/bin/env bash

# Format duration as hours and minutes
format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    printf "%dh %dm" $hours $minutes
}

# Record the start time
total_start_time=$(date +%s)

############### Configuration ###############
VLM_MODEL='vqvq-72b-preview'

max_retries=1
num_threads=32
timeout=600
max_token_len=4096
scale_token_len=1
#############################################

test_file_names=(
  "multi_solution"
  "step_correctness"
  "redundant_det"
  "most_confidence"
  "foresight"
  "error_correction"
  "error_reason_analysis"
  "attribute_hallucination"
  "existence_hallucination"
  "detail_error"
  "image_ref_error"
  "location_error"
)

echo "Evaluation started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================"

for test_file_name in "${test_file_names[@]}"; do
    single_start=$(date +%s)
    
    echo "[Start] ${test_file_name}-${VLM_MODEL} at $(date '+%H:%M:%S')"
    
    python online_api_eval.py \
        --test_file_name ${test_file_name} \
        --VLM_MODEL ${VLM_MODEL} \
        --max_retries ${max_retries} \
        --num_threads ${num_threads} \
        --timeout ${timeout} \
        --max_token_len ${max_token_len} \
        --scale_token_len ${scale_token_len}

    single_end=$(date +%s)
    single_duration=$((single_end - single_start))
    echo "[Done] Duration: $(format_duration $single_duration)"
    echo "------------------------------"
done

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

echo "================================"
echo "Evaluation ended at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: $(format_duration $total_duration)"
