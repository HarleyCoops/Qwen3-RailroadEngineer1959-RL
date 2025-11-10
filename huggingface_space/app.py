# app.py

import os
import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# Fix OMP_NUM_THREADS warning
os.environ["OMP_NUM_THREADS"] = "1"

MODEL_ID = "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL"

SYSTEM_PROMPT = (
    "You are a Dakota language expert who answers in concise Dakota sentences, "
    "preserving orthography and explaining grammar when asked."
)


def build_prompt(user_text: str) -> str:
    """Build the full prompt with system message and User/Assistant formatting."""
    return f"{SYSTEM_PROMPT}\n\nUser: {user_text.strip()}\nAssistant:"


class StopSequenceCriteria(StoppingCriteria):
    """Stop generation when specific sequences are detected."""
    def __init__(self, tokenizer, stop_sequences):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.stop_ids = []
        for seq in stop_sequences:
            ids = tokenizer.encode(seq, add_special_tokens=False)
            if ids:
                self.stop_ids.append(ids)
    
    def __call__(self, input_ids, scores, **kwargs):
        # Check if any stop sequence appears in the generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        for stop_seq in self.stop_sequences:
            if stop_seq in generated_text:
                return True
        return False


@spaces.GPU(duration=60)
def infer(prompt, max_tokens, temperature, top_p):
    # Load weights the first time the GPU worker spins up
    if "tokenizer" not in infer.__dict__:
        infer.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        infer.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    
    tok = infer.tokenizer
    model = infer.model
    
    if not prompt.strip():
        return "Please provide a prompt."
    
    try:
        # Build the full prompt internally before sending to model
        full_prompt = build_prompt(prompt)
        inputs = tok(full_prompt, return_tensors="pt").to(model.device)
        
        # Add stopping criteria to prevent "Answer:" loops
        stopping_criteria = StoppingCriteriaList([
            StopSequenceCriteria(tok, ["Assistant:", "Answer:"])
        ])
        
        # Use more conservative generation parameters to match training
        output = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, 60),  # Cap at 60 for concise responses
            temperature=max(temperature, 0.3),  # Minimum 0.3 to avoid too deterministic
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        
        # Decode only the assistant response
        decoded = tok.decode(output[0], skip_special_tokens=True)
        
        if "Assistant:" in decoded:
            decoded = decoded.split("Assistant:")[-1]
        
        decoded = decoded.strip() or "[Model returned an empty string]"
        
        return decoded
    
    except Exception as e:
        return f"Error during generation: {str(e)}"


demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(lines=4, label="Prompt", placeholder="Translate into Dakota: ..."),
        gr.Slider(32, 128, value=60, step=8, label="Max new tokens"),
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
