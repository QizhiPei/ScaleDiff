import os
from tqdm import tqdm
import os
from utils_jsonl import read_jsonl, save_jsonl
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds_path', type=str, default='your_original_dataset_path.jsonl')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

ds = read_jsonl(args.ds_path)

prompts = [item['conversations'][0]['value'] for item in ds]

model_path = "THU-KEG/AdaptThink-7B-delta0.05"

tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [
    [{"role": "user", "content": item}] for item in prompts
]

text = [
    tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    for msg in tqdm(messages)
]

from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1,
    stop = ['</think>'],
    include_stop_str_in_output=True,
    logprobs=20,
)

device_count = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
print(f"Using {device_count} GPUs")
llm = LLM(model=model_path, tensor_parallel_size=device_count)

res_list = []
think_list = []
for i in tqdm(range(0, len(text), args.batch_size)):
    end = min(i + args.batch_size, len(text))
    batch_text = text[i:end]

    outputs = llm.generate(batch_text, sampling_params)

    for j, output in enumerate(outputs):        
        if list(output.outputs[0].logprobs[0].keys())[0] == 151649:
            think_flag = 0
        else:
            think_flag = 1

        res_list.append({
            "think_flag": think_flag,
        })
        think_list.append(think_flag)
    
print(f"Think (Difficult) ratio is {sum(think_list) / len(think_list)}")

save_jsonl(res_list, "math_qwen3_think_difficulty.jsonl")