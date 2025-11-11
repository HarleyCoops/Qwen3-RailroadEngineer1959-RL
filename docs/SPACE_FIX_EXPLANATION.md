# Why the Space Wasn't Working - Detailed Explanation

## The Problem

Your Space was returning just "Dakota" instead of proper translations because it was using the **wrong prompt format**. The model was trained with a specific chat template format, but the Space code was using a simple string concatenation.

## What Was Wrong

### Original (Broken) Code:
```python
def build_prompt(user_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser: {user_text.strip()}\nAssistant:"

# This creates a plain string like:
# "You are a Dakota expert...\n\nUser: Translate to Dakota: Hello\nAssistant:"
```

**Problem**: This is just a plain text string. The model doesn't understand this format because it was trained with a **structured chat format** using special tokens and formatting.

### What the Model Expects (Training Format):

During RL training, the verifiers framework used:
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# Then applied the chat template:
formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
```

This creates something like:
```
<|im_start|>system
You are a Dakota language expert...
<|im_end|>
<|im_start|>user
Translate to Dakota: Hello
<|im_end|>
<|im_start|>assistant
```

## Why Format Matters So Much

### 1. **Token Alignment**
- LLMs process text as **tokens** (sub-word pieces)
- The chat template adds special tokens (`<|im_start|>`, `<|im_end|>`, etc.)
- These tokens tell the model: "This is a system message", "This is user input", "Now generate assistant response"
- Without these tokens, the model doesn't know where instructions end and generation should begin

### 2. **Training Distribution Mismatch**
- Your model was trained on prompts formatted with `apply_chat_template()`
- When you give it a different format, it's like speaking a different language
- The model learned: "When I see `<|im_start|>assistant`, I should generate a response"
- Without that token, it doesn't know what to do

### 3. **Why It Returned Just "Dakota"**
When the format is wrong:
- The model gets confused about what it's supposed to do
- It might think "Dakota" is part of the prompt, not something to generate
- Or it starts generating but stops early because it doesn't recognize the expected format
- The response extraction logic also fails because it's looking for the wrong markers

## The Fix

### New (Correct) Code:
```python
def format_chat_messages(system_prompt: str, user_message: str) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

# Then apply the chat template:
messages = format_chat_messages(SYSTEM_PROMPT, prompt)
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

**Why This Works**:
1. ✅ Uses the same message structure as training
2. ✅ Applies the model's built-in chat template (same one used during training)
3. ✅ Adds the generation prompt token (`<|im_start|>assistant`)
4. ✅ Model recognizes the format and generates properly

## How Chat Templates Work

### Built-in Template (Part of Model)
The Qwen3 model has a chat template defined in `chat_template.jinja`:
```jinja
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        <|im_start|>system
        {{ message['content'] }}
    {%- elif message['role'] == 'user' %}
        <|im_start|>user
        {{ message['content'] }}
    {%- elif message['role'] == 'assistant' %}
        <|im_start|>assistant
        {{ message['content'] }}
    {%- endif %}
    <|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
    <|im_start|>assistant
{%- endif %}
```

When you call `tokenizer.apply_chat_template()`, it:
1. Takes your message structure `[{"role": "system"}, {"role": "user"}]`
2. Applies the template to format it correctly
3. Adds special tokens the model understands
4. Returns the formatted string ready for tokenization

## Why It Worked Once Then Stopped

This is interesting - it suggests:
1. **First time**: Maybe the model state was different, or there was some caching
2. **After that**: The model settled into a pattern where it recognized the wrong format wasn't working
3. **Or**: The response extraction logic was accidentally working once but failing consistently after

The key point: **It was never really working correctly** - it was just lucky once.

## Key Takeaways

1. **Format Matching is Critical**: LLMs are very sensitive to input format
2. **Use Built-in Templates**: Always use `apply_chat_template()` for chat models
3. **Match Training Format**: Inference format must match training format exactly
4. **Special Tokens Matter**: Those `<|im_start|>` tokens aren't decoration - they're instructions to the model

## Testing the Fix

After updating your Space with the fixed code:

1. **Upload the new `app.py`** to your Space
2. **Wait for rebuild** (usually 2-5 minutes)
3. **Test with**: "Translate to Dakota: I am not feeling well today"
4. **Expected**: Should return a proper Dakota translation, not just "Dakota"

The fix ensures the model receives prompts in the exact format it was trained on, so it can generate proper responses.

