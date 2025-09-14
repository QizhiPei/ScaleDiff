source activate scalediff_train

export OMP_NUM_THREADS=8

set -x
cd ../LLaMA-Factory

[ -z "$EPOCH" ] && EPOCH=3
[ -z "$PER_DEVICE_BATCH_SIZE" ] && PER_DEVICE_BATCH_SIZE=2
[ -z "$GRADIENT_ACCUMULATION_STEPS" ] && GRADIENT_ACCUMULATION_STEPS=2
[ -z "$DATASET" ] && DATASET=ScaleDiff-Math-generated,ScaleDiff-Math-original
[ -z "$SEED" ] && SEED=42
[ -z "$RUN_NAME" ] && RUN_NAME=qwen25_math_7b_inst_${DATASET//,/_}_eps${EPOCH}_seed${SEED}

DS_CONFIG_PATH=examples/deepspeed/ds_z2_config.json
MODEL_PATH=QizhiPei/Qwen2.5-Math-7B-Instruct-RoPE-300k
OUTPUT_PATH=saves/${RUN_NAME}

set -x

llamafactory-cli train \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn fa2 \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --template qwen_nosystem \
    --seed $SEED \
    --finetuning_type full \
    --preprocessing_num_workers 16 \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --weight_decay 0.0001 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ddp_timeout 180000000 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --logging_steps 5 \
    --cutoff_len 32768 \
    --save_steps 180000000 \
    --plot_loss \
    --num_train_epochs $EPOCH \
    --bf16 \
    --save_only_model \
    --report_to none \
    --run_name $RUN_NAME \
    --use_liger_kernel \
    --neat_packing