# Memory Management Tips for Distillation Training

## Problem: CUDA Out of Memory

When training distillation, you may encounter:
```
torch.OutOfMemoryError: CUDA out of memory
```

This happens because:
1. Teacher model (Qwen2-VL 2B) uses ~4-8GB GPU memory
2. Student model needs additional memory
3. Training requires even more memory for gradients and activations

## Solutions (in order of effectiveness)

### 1. Use CPU Offloading (Recommended)
Load teacher model on CPU instead of GPU:

```bash
python train.py \
    --data_root ../data \
    --output_dir outputs \
    --teacher_type qwen2vl \
    --student_type tiny \
    --use_cpu_offload \
    --batch_size 2
```

**Pros**: Saves ~4-8GB GPU memory  
**Cons**: Teacher inference is slower (but only needed during training)

### 2. Reduce Batch Size
Smaller batches use less memory:

```bash
python train.py \
    --batch_size 2  # or even 1
```

### 3. Use Smaller Student Model
Phi2 is smaller than TinyVLM:

```bash
python train.py \
    --student_type phi2  # Instead of tiny
```

### 4. Enable Gradient Checkpointing (Already Enabled)
Saves memory by trading compute for memory:
- Already enabled by default
- Reduces memory by ~30-50%

### 5. Use Mixed Precision (Already Enabled)
Uses FP16/BF16 instead of FP32:
- Already enabled by default
- Reduces memory by ~50%

### 6. Reduce Number of Samples
Train on fewer scenes initially:

```bash
python train.py \
    --max_samples 50  # Start small
```

## Recommended Settings for 8GB GPU

```bash
python train.py \
    --data_root ../data \
    --output_dir outputs \
    --teacher_type qwen2vl \
    --student_type tiny \
    --num_epochs 10 \
    --batch_size 2 \
    --use_cpu_offload \
    --use_gradient_checkpointing \
    --use_mixed_precision \
    --max_samples 100
```

## Recommended Settings for 16GB+ GPU

```bash
python train.py \
    --data_root ../data \
    --output_dir outputs \
    --teacher_type qwen2vl \
    --student_type tiny \
    --num_epochs 10 \
    --batch_size 4 \
    --use_mixed_precision
```

## Monitor Memory Usage

Add this to see memory usage:

```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

## Alternative: Pre-generate Teacher Outputs

If memory is still an issue, you can:
1. Run teacher on all data and save outputs
2. Train student separately using saved outputs

This avoids loading both models simultaneously.

## Environment Variables

You can also try:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py ...
```

This helps with memory fragmentation.

