echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_PATH: ${MODEL_PATH}"

for task in aime24 aime25
    do
    for seed in {0..9}
    do
        python main.py \
            --model ${MODEL_PATH} \
            --task "custom|${task}|0|0" \
            --temperature 0.6 \
            --top_p 0.95 \
            --seed ${seed} \
            --output_dir eval_outputs \
            --max_new_tokens 32768 \
            --max_model_length 32768 \
            --custom_tasks_directory lighteval_tasks.py \
            --use_chat_template \
            --gpu_memory_utilization 0.8
    done
done

task=math_500
for seed in {0..2}
do
    python main.py \
        --model ${MODEL_PATH} \
        --task "custom|${task}|0|0" \
        --temperature 0.6 \
        --top_p 0.95 \
        --seed ${seed} \
        --output_dir eval_outputs \
        --max_new_tokens 32768 \
        --max_model_length 32768 \
        --custom_tasks_directory lighteval_tasks.py \
        --use_chat_template \
        --gpu_memory_utilization 0.8
done