import os
import json
from tqdm import tqdm
import os
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--model_path', type=str, default="QizhiPei/DiffGen-8B")
parser.add_argument('--temperature', type=float, default=0.6)
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--split', type=str, default="0")
args = parser.parse_args()

split = args.split
temperature = args.temperature
top_p = 0.95
top_k = 20
max_tokens = 1024

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

os.makedirs("diffgen_outputs", exist_ok=True)
generated_line_num = 0
if os.path.exists(f"diffgen_outputs/diffgen_vllm_split{split}.jsonl"):
    generated_line_num = sum(1 for _ in open(f"diffgen_outputs/diffgen_vllm_split{split}.jsonl", "r", encoding="utf-8"))

text = [
    "<|im_start|>user\n" for _ in range(args.num_samples - generated_line_num)
]

sampling_params = SamplingParams(
    temperature=temperature, 
    top_p=top_p, 
    top_k=top_k, 
    max_tokens=max_tokens,
)

device_count = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
print(f"Using {device_count} GPUs")

llm = LLM(model=args.model_path, tensor_parallel_size=device_count, seed=int(split))

with open(f"diffgen_outputs/diffgen_vllm_split{split}.jsonl", "a", encoding="utf-8") as f:
    for i in tqdm(range(0, len(text), args.batch_size)):
        end = min(i + args.batch_size, len(text))
        batch_text = text[i:end]
        outputs = llm.generate(batch_text, sampling_params)
        generated_text = [output.outputs[0].text for output in outputs]
        for j in range(len(generated_text)):
            f.write(json.dumps({"problem": generated_text[j]}) + "\n")
        f.flush()
    
    