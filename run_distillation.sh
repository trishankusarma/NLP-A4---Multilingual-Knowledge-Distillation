# Shell script that runs the entire pipeline - from training data
# generation to evaluation of the distilled student

# dataset_generation.py: Script to query the teacher and build the training corpus.
# You can select the samples on which you want to infer the teacher.
# ○ --teacher_model: HuggingFace model ID or path to the teacher model
# ○ --num_samples: Comma-separated sample counts for English, Hindi, Bengali,
# Kannada, and Tamil
# ○ --output_file: Path to save the final train.jsonl

# this might not be a part of final pipeline
#!/bin/bash

#!/bin/bash
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3

export CUDA_VISIBLE_DEVICES=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

# python dataset_generation.py \
#     --teacher_model Qwen/Qwen2.5-7B-Instruct \
#     --num_samples 3500,2500,2000,1000,1000 \
#     --output_file outputs/train.jsonl \
#     --max_new_tokens 512 \
#     --gpu_memory_utilization 0.85 \
#     --tensor_parallel_size 1 \
#     --batch_size 10000

# # In-family student (Qwen)
# python train_distill.py \
#     --student_model Qwen/Qwen2.5-1.5B-Instruct \
#     --train_data outputs/train.jsonl \
#     --output_dir outputs/qwen_distilled \
#     --log_file_name qwen_distill \
#     --epochs 3 \
#     --early_stopping_patience 2 \
#     --val_max_new_tokens 512 \
#     --batch_size 2 \
#     --grad_accumulation_steps 4 \
#     --val_acc_samples_per_lang 10 \
#     --max_length  1702

# python train_distill.py \
#     --student_model "meta-llama/Llama-3.2-1B-Instruct"\
#     --train_data outputs/train.jsonl \
#     --output_dir outputs/llama_distilled \
#     --log_file_name llama_distill \
#     --epochs 3 \
#     --early_stopping_patience 2 \
#     --val_max_new_tokens 512 \
#     --batch_size 2 \
#     --grad_accumulation_steps 4 \
#     --val_acc_samples_per_lang 10 \
#     --max_length  1702

python inference_eval.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_path outputs/qwen_distilled/best \
    --test_data outputs/test.jsonl \
    --output_predictions outputs/predictions_qwen.jsonl \
    --report_file outputs/metrics_qwen.txt

# Llama distilled
# python inference_eval.py \
#     --base_model "meta-llama/Llama-3.2-1B-Instruct"\
#     --adapter_path outputs/llama_distilled/best \
#     --test_data outputs/test.jsonl \
#     --output_predictions outputs/predictions_llama.jsonl \
#     --report_file outputs/metrics_llama.txt