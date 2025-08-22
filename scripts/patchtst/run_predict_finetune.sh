#!/bin/bash
# read debug flag
DEBUG=0
while getopts "d" flag; do
        case "${flag}" in
                d) DEBUG=1;;
        esac
done
shift $((OPTIND - 1))

train_data_dirs=(
#     ./data/improved/skew_mixedp_ic16/train
#     ./data/improved/final_skew40/train
#     ./data/improved/final_skew40/train_z5_z10
#     ./data/improved/final_base40/train
#     ./data/improved/final_base40/train_z5_z10
      ./data/new_skew40/train
)

# --- START: Modified section (replaces jq) ---
# Goal: Convert the bash array to a JSON array string.
# Example: (path1 path2) -> ["path1","path2"]

train_data_dirs_json="[" # Start the JSON array string
num_dirs=${#train_data_dirs[@]} # Get the total number of directories
i=0
for dir in "${train_data_dirs[@]}"; do
    train_data_dirs_json+="\"$dir\"" # Add the directory path in quotes
    i=$((i + 1))
    if [ "$i" -lt "$num_dirs" ]; then
        train_data_dirs_json+="," # Add a comma if it's not the last element
    fi
done
train_data_dirs_json+="]" # Close the JSON array string
# --- END: Modified section ---

echo "train_data_dirs: $train_data_dirs_json"

ulimit -n 999999
if [ "$DEBUG" -eq 0 ]; then

        TOTAL_CORES=$(nproc)
        CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
        CORES_PER_JOB=$(( $CORES_PER_GROUP / 4 ))

        CUDA_DEVICES=4,5,6,7
        # CUDA_DEVICES=4,5,6,7
        NUM_DEVICES=$(echo "$CUDA_DEVICES" | tr -d ' ' | tr ',' '\n' | wc -l)

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$CORES_PER_JOB torchrun \
                --nproc-per-node $NUM_DEVICES \
                --master-port 29501 \
                scripts/patchtst/train.py \
                shuffle_buffer_length=100_000 \
                train_data_dirs=$train_data_dirs_json \
                patchtst.mode=predict \
                patchtst.use_dynamics_embedding=true \
                patchtst.pretrained_encoder_path=null \
                patchtst.context_length=512 \
                patchtst.prediction_length=128 \
                patchtst.patch_length=16 \
                patchtst.patch_stride=16 \
                patchtst.num_hidden_layers=10 \
                patchtst.num_attention_heads=8 \
                patchtst.d_model=256 \
                patchtst.num_rff=128 \
                patchtst.rff_scale=1.0 \
                patchtst.rff_trainable=false \
                patchtst.num_poly_feats=56 \
                patchtst.poly_degrees=2 \
                patchtst.channel_attention=true \
                patchtst.max_wavelength=500 \
                patchtst.rope_percent=0.75 \
                patchtst.pooling_type=mean \
                patchtst.loss=mse \
                patchtst.distribution_output=null \
                train.per_device_train_batch_size=512 \
                train.max_steps=20000 \
                train.save_steps=200 \
                train.log_steps=100 \
                train.warmup_ratio=0.1 \
                train.torch_compile=false \
                train.weight_decay=0.0 \
                train.output_dir=./checkpoints/ \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/patchtst/train.py \
                run_name=DEBUG \
                patchtst.pretrained_encoder_path=null \
                shuffle_buffer_length=100 \
                patchtst.mode=predict \
                train.ddp_backend=null \
                train.torch_compile=false \
                "$@"
fi

