# KL Divergence Manim Animation

Epic animated visualization of KL divergence metrics during Dakota Grammar RL training.

## Features

The animation includes multiple scenes:

1. **KLDivergenceEpicComplete** (Recommended): Full-featured animation with:
   - Title sequence
   - Animated curve drawing
   - Real-time value displays
   - Statistical highlights
   - Key insights

2. **KLDivergenceEpic**: Single-panel view with all three KL metrics

3. **KLDivergenceMultiPanel**: Three-panel side-by-side comparison

4. **KLDivergence3D**: 3D visualization (requires ThreeDScene)

## Installation

```bash
# Install Manim
pip install manim

# Optional: For smoother curves
pip install scipy

# Required: pandas and numpy
pip install pandas numpy
```

## Usage

### Quick Preview (Low Quality, Fast)
```bash
manim -pql scripts/visualization/kl_divergence_animation.py KLDivergenceEpicComplete
```

### High Quality (Slow, Beautiful)
```bash
manim -pqh scripts/visualization/kl_divergence_animation.py KLDivergenceEpicComplete
```

### Medium Quality (Balanced)
```bash
manim -pqm scripts/visualization/kl_divergence_animation.py KLDivergenceEpicComplete
```

### Other Scenes
```bash
# Single panel epic view
manim -pql scripts/visualization/kl_divergence_animation.py KLDivergenceEpic

# Multi-panel view
manim -pql scripts/visualization/kl_divergence_animation.py KLDivergenceMultiPanel

# 3D Trajectory (Recommended 3D visualization)
manim -pql scripts/visualization/kl_divergence_animation.py KLDivergence3DTrajectory

# 3D Basic Trajectory
manim -pql scripts/visualization/kl_divergence_animation.py KLDivergence3D

# 3D Surface Landscape
manim -pql scripts/visualization/kl_divergence_animation.py KLDivergence3DSurface
```

**Note**: 3D scenes require `ThreeDScene` and may take longer to render. Use `-pql` for quick previews, `-pqm` for medium quality, or `-pqh` for high quality.

## Output

The animation will be saved to:
- `media/videos/kl_divergence_animation/<quality>/KLDivergenceEpicComplete.mp4`

## 3D Visualizations

The script includes **three different 3D visualization approaches**:

### 1. KLDivergence3D - Trajectory Through Policy Space
Shows the policy evolution as a **3D trajectory** where:
- **X-axis**: Masked KL Divergence (0-12)
- **Y-axis**: Overall KL Divergence (0-5)  
- **Z-axis**: Unmasked KL Divergence (0-0.2)

Each point in 3D space represents the policy state at a training step. The trajectory shows how the policy moves through KL divergence space, starting near the origin (low divergence) and moving toward higher masked KL while keeping unmasked KL low.

**Key Insight**: The trajectory moves primarily along the X-axis (masked KL increases) while staying low on the Z-axis (unmasked KL stays low), showing targeted adaptation.

### 2. KLDivergence3DSurface - Surface Landscape
Creates a **3D surface** showing the relationship between:
- **X-axis**: Training Steps (0-1000)
- **Y-axis**: Masked KL Divergence (0-12)
- **Z-axis**: Overall KL Divergence (0-5)

The surface represents the "landscape" of policy states, with a trajectory line showing the actual path taken during training.

### 3. KLDivergence3DTrajectory - Enhanced Trajectory
An enhanced version with:
- **Color-coded trajectory**: Green (start) → Yellow → Orange → Red (end)
- **Milestone markers**: Spheres at key training steps (0, 250, 500, 750, 999)
- **Step labels**: Shows which training step each milestone represents
- **Smooth camera rotation**: Rotates to show the 3D structure from all angles

**Best for**: Understanding the full 3D evolution of the policy.

## What It Shows (2D Visualizations)

The 2D animations visualize three key KL divergence metrics:

1. **Masked KL (Mean)** - Red curve
   - Shows divergence for Dakota-specific masked tokens
   - Final value: ~9.32 (high adaptation)
   - Indicates significant policy changes for Dakota patterns

2. **Overall KL (Mean)** - Orange curve
   - Overall policy divergence
   - Final value: ~3.83 (moderate adaptation)
   - Shows controlled policy drift

3. **Unmasked KL (Mean)** - Green curve (scaled ×100 for visibility)
   - Divergence for general language tokens
   - Final value: ~0.042 (very low)
   - Confirms preservation of general language understanding

## Key Insights Visualized

- **Policy Adaptation**: Masked KL increases significantly, showing the model learned Dakota-specific patterns
- **Stability**: Overall KL remains moderate, indicating stable training
- **Preservation**: Unmasked KL stays extremely low, confirming no catastrophic forgetting

## Customization

Edit `scripts/visualization/kl_divergence_animation.py` to:
- Change colors
- Adjust animation speed
- Add more statistics
- Modify layout
- Add additional metrics

## Troubleshooting

**Error: File not found**
- Ensure `wandb_analysis/kl_divergence_curve.csv` exists
- Run `python scripts/analysis/export_comprehensive_analysis.py` first

**Error: scipy not found**
- The script will fall back to linear interpolation
- Install scipy for smoother curves: `pip install scipy`

**Manim not found**
- Install Manim: `pip install manim`
- Or use conda: `conda install -c conda-forge manim`

## Example Output

The animation creates a professional visualization showing:
- Smooth animated curves drawing over time
- Color-coded metrics with labels
- Real-time value displays
- Statistical summaries
- Key insights in text overlays

Perfect for presentations, papers, or social media!

