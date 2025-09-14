import os
import json
from tqdm import tqdm
import os
import argparse
from utils_jsonl import read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('--ds_path', type=str, default='diffgen_outputs/diffgen_vllm_split0.jsonl')
parser.add_argument('--teacher_model_path', type=str, default='Qwen/Qwen3-8B')
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--split_id', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

ds = read_jsonl(args.ds_path)

total_len = len(ds)
chunk_size = total_len // args.split_num
remainder = total_len % args.split_num

if args.split_id < remainder:
    start_idx = args.split_id * (chunk_size + 1)
    end_idx = start_idx + chunk_size + 1
else:
    start_idx = remainder * (chunk_size + 1) + (args.split_id - remainder) * chunk_size
    end_idx = start_idx + chunk_size

os.makedirs(f'solution_distill_outputs/{args.ds_path.replace("/", "_").replace(".jsonl", "")}_solution', exist_ok=True)
res_path = f'solution_distill_outputs/{args.ds_path.replace("/", "_").replace(".jsonl", "")}_solution/res_{args.split_id}.jsonl'

if os.path.exists(res_path):
    with open(res_path, 'r', encoding='utf-8') as f:
        generated_num = sum(1 for _ in f)
    start_idx = start_idx + generated_num

ds = ds[start_idx:end_idx]
problems = [ds[i]['problem'] for i in range(len(ds))]
print(f"start_idx: {start_idx}, end_idx: {end_idx}, len(ds): {len(ds)}")
print(f"example problem: {problems[0]}")

# load model
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)

messages = [
    [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{problem}"}] for problem in problems
]

text = [
    tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    for msg in tqdm(messages)
]

print("-" * 100)
print(text[0])
print("-" * 100)

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

device_count = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
print(f"Using {device_count} GPUs")
llm = LLM(model=args.teacher_model_path, tensor_parallel_size=device_count)

with open(res_path, 'a', encoding='utf-8') as f:
    for i in range(0, len(text), args.batch_size):
        end = min(i + args.batch_size, len(text))
        batch_text = text[i:end]
        outputs = llm.generate(batch_text, sampling_params)
        generated_text = [output.outputs[0].text for output in outputs]

        if len(generated_text) != len(batch_text):
            print(f"Mismatched output: got {len(generated_text)} for {len(batch_text)}")

        saved = [{'problem': ds[i + j]['problem'], 'answer': generated_text[j]} for j in range(len(generated_text))]

        for s in saved:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
        f.flush()