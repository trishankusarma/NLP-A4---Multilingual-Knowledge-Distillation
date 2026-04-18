# Shell script that runs the entire pipeline - from training data
# generation to evaluation of the distilled student

# dataset_generation.py: Script to query the teacher and build the training corpus.
# You can select the samples on which you want to infer the teacher.
# ○ --teacher_model: HuggingFace model ID or path to the teacher model
# ○ --num_samples: Comma-separated sample counts for English, Hindi, Bengali,
# Kannada, and Tamil
# ○ --output_file: Path to save the final train.jsonl

# this might not be a part of final pipeline
python -m dataset_generation --teacher_model will_tell \
    --num_samples will_tell \
    --output_file will_tell