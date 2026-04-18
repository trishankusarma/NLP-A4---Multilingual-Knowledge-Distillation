# Shell script that runs the entire pipeline - from training data
# generation to evaluation of the distilled student

# dataset_generation.py: Script to query the teacher and build the training corpus.
# You can select the samples on which you want to infer the teacher.
# ○ --teacher_model: HuggingFace model ID or path to the teacher model
# ○ --num_samples: Comma-separated sample counts for English, Hindi, Bengali,
# Kannada, and Tamil
# ○ --output_file: Path to save the final train.jsonl

# this might not be a part of final pipeline
# python dataset_generation.py \
#     --teacher_model Qwen/Qwen2.5-7B-Instruct \
#     --num_samples 3500,2500,2000,1000,1000 \
#     --output_file outputs/train.jsonl \
#     --max_new_tokens 1024 \
#     --gpu_memory_utilization 0.85 \
#     --tensor_parallel_size 1

# In-family student (Qwen)
python train_distill.py \
    --student_model Qwen/Qwen2.5-1.5B-Instruct \
    --train_data outputs/train.jsonl \
    --output_dir outputs/qwen_distilled \
    --log_file_name qwen_distill \
    --epochs 10 \
    --batch_size 6 \
    --grad_accumulation_steps 2 \
    --val_acc_samples_per_lang 50

python train_distill.py \
    --student_model meta-llama/Llama-3.2-1B-Instruct \
    --train_data outputs/train.jsonl \
    --output_dir outputs/llama_distilled \
    --log_file_name llama \
    --epochs 10 \
    --batch_size 6 \
    --grad_accumulation_steps 2 \
    --val_acc_samples_per_lang 50