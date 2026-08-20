import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Base model path or HF ID")
    parser.add_argument("--adapter-path", default="tmp_publish", help="Path to the adapter weights")
    parser.add_argument("--prompt", type=str, default="Translate 'my elder brother' to Dakota.")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}...")
    # Quantization config for 30B model to fit in consumer GPU (if possible)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: Qwen3-30B might require specific access or not be public yet.")
        return

    print(f"Loading adapter from: {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant translating English to Dakota."},
        {"role": "user", "content": args.prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\nGenerating response...")
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=256,
        temperature=0.7
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*50)
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    print(f"Response: {response}")
    print("="*50)

if __name__ == "__main__":
    main()

