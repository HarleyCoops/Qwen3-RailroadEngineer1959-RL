# Where to Find Debug Output in HuggingFace Spaces

## The Problem

Print statements (`print()`) in HuggingFace Spaces don't always show up where you expect. They can appear in:
1. **Logs tab** - But might be buried or not visible
2. **Terminal output** - If you have terminal access
3. **Nowhere** - Sometimes Gradio suppresses them

## Solution: Return Debug Info in Response

I've updated the code to **return debug information directly in the response** when `DEBUG_INFERENCE=true` is enabled. This way you'll see it in the UI itself!

## How It Works Now

When `DEBUG_INFERENCE=true` is set:

1. **If generation succeeds**: You'll see debug info + the response
2. **If generation fails**: You'll see detailed debug info showing what went wrong

## What You'll See

When you submit a prompt with debug enabled, the response will show:

```
[DEBUG INFO]
=== DEBUG MODE ENABLED ===
Formatted prompt length: 245
Output shape: torch.Size([1, 312])
Input length: 248
Output length: 312
Prompt tokens: 248
Generated tokens: 64
Decoded from tokens: 'Háu'
After cleanup: 'Háu'
Final length: 3

[RESPONSE]
Háu
```

Or if there's an issue:

```
[DEBUG: Empty result]

=== DEBUG MODE ENABLED ===
Formatted prompt length: 245
Generated tokens: 0
WARNING: No new tokens generated!
Full decoded text: [shows what model actually output]
```

## Steps to Debug

1. **Make sure DEBUG_INFERENCE=true is set**:
   - Space → Settings → Variables
   - Check `DEBUG_INFERENCE` = `true`
   - Restart if you just added it

2. **Submit a test prompt**:
   - Go to the App tab
   - Enter: `Translate to Dakota: Hello`
   - Click Submit

3. **Check the response box**:
   - The debug info will appear **in the response output box**
   - Scroll down if it's long
   - Look for "DEBUG MODE ENABLED" at the top

## What to Look For

**If you see "Generated tokens: 0":**
- Model isn't generating anything
- Check generation parameters
- Might need to increase temperature or max_new_tokens

**If you see tokens but empty result:**
- Extraction is failing
- Check "Decoded from tokens" to see raw output
- Check "After cleanup" to see what got removed

**If you see errors:**
- Check the error message
- Might be a model or tokenization issue

## Alternative: Check Logs Tab

If you still want to check the Logs tab:

1. Go to your Space
2. Click **"Logs"** tab (next to Files, Settings)
3. Look for lines starting with `DEBUG -`
4. They appear **during inference**, not just at startup

But the response box method is easier - you'll see everything right there!

