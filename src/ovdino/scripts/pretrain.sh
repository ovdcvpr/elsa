#!/usr/bin/env bash
# set -x

# project config
root_dir="$(realpath $(dirname $0)/../../)"
code_dir=$root_dir/ovdino
time=$(date "+%Y%m%d-%H%M%S")

config_file=$1
config_name=$(basename $config_file .py)
output_dir=$root_dir/wkdrs/$config_name

# multi nodes distrubuted envs
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}
export NNODES=${NNODES}
export NODE_RANK=${NODE_RANK}

# env config
export DETECTRON2_DATASETS="$root_dir/datas/"
export MODEL_ROOT="$root_dir/inits/"
export HF_HOME="$root_dir/inits/huggingface"
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Distributed Pre-Training with $config_name"
evaluation_dir="$output_dir/eval_coco_$time"
mkdir -p $evaluation_dir
cd $code_dir
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    python ./tools/train_net.py \
    --config-file $config_file \
    --resume \
    train.output_dir=$output_dir \
    dataloader.evaluator.output_dir="$evaluation_dir" | tee $output_dir/train_$time.log
    