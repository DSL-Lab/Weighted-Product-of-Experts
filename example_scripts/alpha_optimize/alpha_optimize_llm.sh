# Set the mode of wandb, online to log the data of experiments
export WANDB_MODE=online
# export WANDB_MODE=disabled

for model_type in gpt2 llama3-1B llama3-3B llama3-8B; do
    for dataset in enwik8 enwik9 math code shakespeare; do
        echo "Running model: $model_type on dataset: $dataset"

        bash example_scripts/alpha_optimize/alpha_optimize.sh \
        --model_type $model_type \
        --dataset $dataset
    done
done