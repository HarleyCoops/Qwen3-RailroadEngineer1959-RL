# app.py

import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)


SYSTEM_PROMPT = (
    "You are a Dakota language expert who uses concise, grammatical Dakota sentences. "
    "Translate or explain the input faithfully while preserving Dakota orthography."
)


def generate(prompt, max_tokens, temperature, top_p):
    if not prompt.strip():
        return "Please provide a prompt."
    
    try:
        user_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract assistant response - try multiple methods
        if "Assistant:" in decoded:
            response = decoded.split("Assistant:")[-1].strip()
        else:
            # Fallback: extract everything after the user prompt
            response = decoded[len(user_prompt):].strip()
        
        # Clean up stop sequences
        stop_sequences = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\nUser:", "\nUser:"]
        for stop in stop_sequences:
            if stop in response:
                response = response.split(stop)[0].strip()
        
        # Ensure we return something
        if not response or len(response) < 1:
            return f"[Model generated empty response. Full output: {decoded[:200]}...]"
        
        return response
    
    except Exception as e:
        return f"Error during generation: {str(e)}"


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=4, label="Prompt", placeholder="Translate into Dakota: ..."),
        gr.Slider(32, 256, value=120, step=8, label="Max new tokens"),
        gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p"),
    ],
    outputs=gr.Textbox(lines=6, label="Model output"),
    title="Dakota Grammar RL – Qwen3‑0.6B",
    description="Composite-reward RL model trained on Dakota grammar tasks. "
                "Use concise prompts (translations, grammar explanations, etc.).",
)


if __name__ == "__main__":
    demo.launch()
