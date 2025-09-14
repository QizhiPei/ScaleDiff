source activate scalediff_train

# export DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=8

[ -z "$EPOCH" ] && EPOCH=1
[ -z "$DATASET" ] && DATASET=math_difficult_qft
[ -z "$SEED" ] && SEED=42
[ -z "$RUN_NAME" ] && RUN_NAME=qwen3_7b_qft_${DATASET//,/_}_eps${EPOCH}_seed${SEED}_v1

DS_CONFIG_PATH=examples/deepspeed/ds_z2_config.json
MODEL_PATH=Qwen/Qwen3-8B-Base
OUTPUT_PATH=saves/${RUN_NAME}

set -x
cd ../LLaMA-Factory

llamafactory-cli train \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn fa2 \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --template qwen_qft \
    --seed $SEED \
    --finetuning_type full \
    --preprocessing_num_workers 8 \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --weight_decay 0.0001 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --ddp_timeout 180000000 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --logging_steps 5 \
    --cutoff_len 1024 \
    --save_steps 180000000 \
    --plot_loss \
    --num_train_epochs $EPOCH \
    --bf16 \
    --save_only_model \
    --report_to none \
    --run_name $RUN_NAME \
    --use_liger_kernel
