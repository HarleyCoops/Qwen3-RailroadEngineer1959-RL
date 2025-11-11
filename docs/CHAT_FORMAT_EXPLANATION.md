# Chat Format Explanation for Dakota Grammar RL Model

## How Chat Formatting Works

### Built-in Chat Template (Not from Training Data)

The chat formatting is **NOT** learned from your training data. Instead, it comes from the **Qwen3 model's built-in chat template** that's part of the model architecture.

### During Training

1. **Environment Setup**: The `DakotaGrammarEnv` uses `message_type="chat"` (see `environments/dakota_grammar_translation/environment.py:429`)

2. **Verifiers Framework**: When `message_type="chat"` is set, the verifiers framework automatically calls `tokenizer.apply_chat_template()` to format messages

3. **Message Structure**: Messages are structured as:
   ```python
   [
       {"role": "system", "content": system_prompt},
       {"role": "user", "content": user_prompt}
   ]
   ```

4. **Template Application**: The tokenizer's built-in chat template converts this to the exact format the model expects (with special tokens, formatting, etc.)

### During Inference

The inference script uses the **exact same process**:

1. **Same Message Structure**: Creates messages with `{"role": "system"}` and `{"role": "user"}`

2. **Same Template**: Calls `tokenizer.apply_chat_template()` with the same parameters

3. **Same System Prompt**: Uses the same `DEFAULT_SYSTEM_PROMPT` from the environment

### Why This Works

- The Qwen3 model has a **built-in chat template** defined in its tokenizer config
- This template knows how to format system/user/assistant messages
- The verifiers framework uses this template during training
- Our inference script uses the same template, ensuring perfect format matching

### Verification

You can verify the chat template is correct by checking:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL")
print(tokenizer.chat_template)  # Shows the template function
```

The template is part of the model files on HuggingFace, so it's automatically downloaded with the model.

### Key Takeaway

**You don't need to worry about format matching** - as long as you:
1. Use the same message structure (`[{"role": "system"}, {"role": "user"}]`)
2. Use `tokenizer.apply_chat_template()` (which the script does)
3. Use the same system prompt (which the script does)

The format will automatically match training because both use the model's built-in chat template.

