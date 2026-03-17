# I2V Fine-Tuning Guide

## Overview
Fine-tune the Wan I2V (Image-to-Video) model with high-quality datasets to improve temporal coherence, motion quality, and overall video generation performance.

## Recommended Datasets

### 1. **MSR-VTT Enhanced** (⭐⭐⭐⭐⭐ BEST STARTING POINT)
- **Size**: 6.5K high-quality video-caption pairs
- **Quality**: Well-annotated, diverse content
- **Best for**: Initial I2V fine-tuning, temporal coherence
- **Usage**: `AlexZigma/msr-vtt`

### 2. **VideoFeedback Annotated** (⭐⭐⭐⭐)
- **Size**: Curated video feedback dataset
- **Quality**: Human-annotated quality feedback
- **Best for**: Quality improvement and evaluation
- **Usage**: `TIGER-Lab/VideoFeedback` (config: annotated)

### 3. **LLaVA-Video-178K** (⭐⭐⭐⭐)
- **Size**: 178K video instruction samples
- **Quality**: Academic-grade video understanding
- **Best for**: Advanced video comprehension
- **Usage**: `lmms-lab/LLaVA-Video-178K` (config: 0_30_s_academic_v0_1)

## Quick Start

### 1. Download Datasets
```bash
# Download all recommended datasets
python scripts/download_i2v_datasets.py

# Or download specific datasets
python scripts/download_i2v_datasets.py --datasets msr_vtt_enhanced video_feedback_annotated

# For testing with small samples
python scripts/download_i2v_datasets.py --max_samples 1000 --datasets msr_vtt_enhanced
```

### 2. Fine-tune Model
```bash
# Basic fine-tuning with MSR-VTT
python scripts/finetune_i2v_model.py \
    --dataset AlexZigma/msr-vtt \
    --output_dir outputs/i2v_finetune_msr \
    --num_epochs 1 \
    --max_samples 1000 \
    --batch_size 1 \
    --learning_rate 1e-5

# Advanced fine-tuning with VideoFeedback
python scripts/finetune_i2v_model.py \
    --dataset TIGER-Lab/VideoFeedback \
    --output_dir outputs/i2v_finetune_feedback \
    --num_epochs 2 \
    --max_samples 5000 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --mixed_precision fp16
```

### 3. Test Fine-tuned Model
```bash
# Test with the fine-tuned model
python run_story_simple.sh \
    --video_model_id outputs/i2v_finetune_hdvg \
    --test_prompt "A cat playing in a garden"
```

## Expected Improvements

### Temporal Coherence
- **Before**: Jittery motion, inconsistent object movement
- **After**: Smooth, natural motion with consistent object trajectories

### Motion Quality
- **Before**: Robotic or unnatural movements
- **After**: Fluid, realistic motion patterns

### Content Consistency
- **Before**: Objects appearing/disappearing randomly
- **After**: Stable object presence and behavior

### Frame Continuity
- **Before**: Abrupt changes between frames
- **After**: Seamless transitions and logical progression

## Training Tips

### Dataset Selection
- Start with **HD-VG-130M** for best quality-to-time ratio
- Use **Panda-70M** for maximum diversity
- Combine datasets for specialized domains (e.g., +YouCook2 for procedural content)

### Hyperparameters
- **Learning Rate**: 1e-5 to 2e-5 (start low)
- **Batch Size**: 1-2 (memory constraints)
- **Epochs**: 1-3 (avoid overfitting)
- **Max Samples**: 10K-50K for initial testing

### Memory Optimization
- Use `--mixed_precision fp16` for faster training
- Enable model CPU offload in the script
- Monitor GPU memory usage

### Quality Checks
- Generate test videos every 1K steps
- Compare temporal coherence metrics
- Validate on held-out prompts

## Integration with SceneWeaver

After fine-tuning, update your model registry:

```json
{
  "video_backbone": {
    "selected": {
      "key": "wan_i2v_finetuned",
      "local_path": "outputs/i2v_finetune_hdvg",
      "exists": true
    }
  }
}
```

Then use in your pipeline:
```bash
python scripts/run_story_pipeline.py --video_model_id outputs/i2v_finetune_hdvg
```

## Troubleshooting

### Out of Memory
- Reduce batch size to 1
- Use gradient checkpointing
- Enable VAE slicing

### Poor Quality
- Check dataset quality
- Reduce learning rate
- Increase training samples
- Validate data preprocessing

### Slow Training
- Use mixed precision
- Enable model offloading
- Use smaller datasets for testing

## Advanced Techniques

### Curriculum Learning
Train on easy samples first, then complex ones.

### Multi-Dataset Training
Combine multiple datasets for better generalization.

### LoRA Fine-tuning
For memory efficiency, consider LoRA adapters instead of full fine-tuning.

### Evaluation Metrics
- Temporal coherence scores
- Motion smoothness metrics
- Content consistency checks
- Human preference studies