# Step-by-Step: Running Your First Hosted Eval

## Step 1: Access Your Environment

1. Go to: https://app.primeintellect.ai/dashboard/environments/harleycooper/dakota1890
2. You should see your Dakota1890 environment page

## Step 2: Start a New Evaluation

1. Click the **"Run Evaluation"** or **"New Hosted Evaluation"** button
2. You'll see a wizard with steps: "Select Model" → "Configuration"

## Step 3: Select Model (Step 1)

1. In the model dropdown/selector, find and select:
   - **`qwen/qwen3-235b-a22`** (or whichever Qwen 235B variant is available)
   - If you see multiple variants, pick the cheapest one ($0.22/$0.88 per 1M tokens)
2. Click **"Next"** or proceed to Configuration

## Step 4: Configuration (Step 2)

### Number of Examples
- **Value**: `5`
- **Checkbox**: Uncheck "Use all examples" (if it exists)

### Rollouts per Example
- **Value**: `1`
- This means each example runs once (fastest for testing)

### Environment Arguments (IMPORTANT!)

Click **"+ Add Entry"** button to add environment arguments:

**Entry 1:**
- **Key**: `dataset_path`
- **Value**: This depends on how your dataset is accessible:
  - If packaged with environment: Check environment docs for the path
  - If using a URL: Your dataset URL
  - If using a file path: The path format your environment expects
  
**Entry 2 (Optional but recommended):**
- **Key**: `max_examples`
- **Value**: `10`

**How to find the correct dataset_path:**
- Check your environment's README or documentation
- The environment might auto-detect it if packaged
- Or you may need to upload/package the dataset with the environment

### Other Settings (Optional)
- **Max tokens**: Set to `256` if there's a field for it
- **Temperature**: `0.7` (if available)

## Step 5: Review and Launch

1. Review your settings:
   - Model: qwen/qwen3-235b-a22
   - Examples: 5
   - Rollouts: 1
   - Environment args: dataset_path set

2. Click **"Launch"** or **"Start Evaluation"**

## Step 6: Monitor Progress

- You'll see a progress indicator
- Expected time: **3-5 minutes**
- Results will appear in the dashboard when complete

## Troubleshooting

**If dataset_path is unclear:**
- Check: https://app.primeintellect.ai/dashboard/environments/harleycooper/dakota1890
- Look for documentation or examples
- The environment might auto-detect the dataset if it's packaged

**If model not found:**
- Use the dropdown to select from available models
- The dropdown shows the exact names Prime Inference recognizes

**If you need help:**
- Check environment README: `environments/dakota_grammar_translation/README.md`
- Or use the CLI version (we can set that up if UI doesn't work)

## Expected Results

- **Time**: 3-5 minutes
- **Cost**: < $0.01 (essentially free)
- **Output**: Evaluation metrics (character preservation, affix accuracy, semantic accuracy, composite reward)

Once complete, you can:
- View detailed results
- Scale up to more examples
- Run full evaluations

