# How to Get Debug Logs from HuggingFace Space

## The Issue

Your model loads successfully (we can see that from startup logs), but you're getting empty strings. We need to see what happens **during inference**, not just startup.

## Step-by-Step Debugging

### 1. Enable Debug Mode

1. Go to your Space page on HuggingFace
2. Click **Settings** (gear icon)
3. Scroll to **Variables** section
4. Click **"Add new variable"**
5. Set:
   - **Key**: `DEBUG_INFERENCE`
   - **Value**: `true`
6. Click **Save**
7. **Restart the Space** (important!)

### 2. Check Logs During Inference

The startup logs you showed are just the initial load. To see inference logs:

1. **After restart**, go to your Space
2. Click the **"Logs"** tab (next to "Files", "Settings", etc.)
3. **Keep the Logs tab open**
4. Go to the **"App"** tab
5. Enter a test prompt: `Translate to Dakota: Hello`
6. Click **Submit**
7. **Immediately go back to Logs tab**

You should see debug output like:
```
DEBUG - Formatted prompt length: 245
DEBUG - Generated tokens: 5
DEBUG - Decoded from tokens: 'Háu'
DEBUG - After cleanup: 'Háu'
```

### 3. What to Look For

**If you see `Generated tokens: 0`:**
- Model isn't generating anything
- Check generation parameters
- Model might be stuck

**If you see tokens but empty result:**
- Extraction/cleanup is removing everything
- Check "Decoded from tokens" to see raw output

**If you see errors:**
- Check the error message
- Might be a model loading issue

### 4. Alternative: Add Print Statements

If debug mode doesn't work, you can temporarily add print statements that will always show:

```python
# Add this right after generation
print(f"=== INFERENCE DEBUG ===")
print(f"Generated tokens: {len(generated_tokens)}")
print(f"Decoded text: {repr(generated_text[:100])}")
print(f"========================")
```

These will always appear in logs, even without DEBUG_INFERENCE.

## Quick Test

Try the simplest possible prompt first:
```
Hello
```

If this also returns empty, it's a fundamental generation issue, not prompt-specific.

## Common Issues

1. **Model not generating**: Check if `max_new_tokens` is too low or temperature too low
2. **Token extraction wrong**: The debug will show if tokens exist but extraction fails
3. **Cleanup too aggressive**: Debug will show what gets removed

The debug output will tell us exactly where the problem is!

