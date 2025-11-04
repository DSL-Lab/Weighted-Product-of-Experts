model_type=transformer
dim=64
dataset=code

# Set the mode of wandb, online to log the data of experiments
export WANDB_MODE=online
# export WANDB_MODE=disabled

# experiments of transformer
for dim in 64 128 256; do
    for dataset in enwik8 enwik9 math code shakespeare; do
        echo "Running model: $model_type $dim on dataset: $dataset"

        bash example_scripts/alpha_optimize/alpha_optimize.sh \
        --dim $dim \
        --model_type $model_type \
        --dataset $dataset
    done
done



