source activate scalediff_test

# Set your GPU id here. Currently, the evaluation script only supports single GPU.
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=QizhiPei/ScaleDiff-7B

bash eval.sh