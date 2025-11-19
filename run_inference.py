from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "Qwen/Qwen3-4B-Instruct-2507"
adapter_name = "HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890"

print(f"Loading base model: {base_model_name}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"Loading tokenizer: {base_model_name}")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print(f"Loading adapter: {adapter_name}")
model = PeftModel.from_pretrained(model, adapter_name)

prompt = "Translate 'my elder brother' to Dakota using the correct possessive suffix."
messages = [
    {"role": "system", "content": "You are a Dakota language expert."},
    {"role": "user", "content": prompt},
]

print("Generating response...")
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


