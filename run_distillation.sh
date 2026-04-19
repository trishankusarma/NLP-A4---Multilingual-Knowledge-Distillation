# Shell script that runs the entire pipeline - from training data
# generation to evaluation of the distilled student

# dataset_generation.py: Script to query the teacher and build the training corpus.
# You can select the samples on which you want to infer the teacher.
# ○ --teacher_model: HuggingFace model ID or path to the teacher model
# ○ --num_samples: Comma-separated sample counts for English, Hindi, Bengali,
# Kannada, and Tamil
# ○ --output_file: Path to save the final train.jsonl
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/scai/msr/aiy247541/.conda/envs/vllm_server_nlp/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export PATH=/home/scai/msr/aiy257584/anaconda3/envs/skylake_env_2/bin:$PATH

# this might not be a part of final pipeline
# python dataset_generation.py \
#     --teacher_model Qwen/Qwen2.5-7B-Instruct \
#     --num_samples 3500,2500,2000,1000,1000 \
#     --output_file outputs/train.jsonl \
#     --max_new_tokens 1024 \
#     --gpu_memory_utilization 0.85 \
#     --tensor_parallel_size 1

# In-family student (Qwen)
# python train_distill.py \
#     --student_model Qwen/Qwen2.5-1.5B-Instruct \
#     --train_data outputs/train.jsonl \
#     --output_dir outputs/qwen_distilled \
#     --log_file_name qwen_distill \
#     --epochs 10 \
#     --early_stopping_patience 2 \
#     --val_max_new_tokens 1024 \
#     --batch_size 4 \
#     --grad_accumulation_steps 2 \
#     --val_acc_samples_per_lang 50

python train_distill.py \
    --student_model /scratch/scai/phd/aiz248311/col772/a4/models/Llama-3.2-1B-Instruct \
    --train_data outputs/train.jsonl \
    --output_dir outputs/llama_distilled \
    --log_file_name llama_distill \
    --epochs 10 \
    --early_stopping_patience 2 \
    --val_max_new_tokens 1024 \
    --batch_size 4 \
    --grad_accumulation_steps 2 \
    --val_acc_samples_per_lang 50