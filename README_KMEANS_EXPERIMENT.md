# K-means SPM Classification Experiment

## Research Question

**Does representing each class with multiple prototypes (via K-means clustering) improve classification accuracy and/or unlearning behavior compared to a single mean prototype?**

This experiment tests whether the SPM classifier can benefit from richer class representations through multi-prototype aggregation.

---

## Overview

The SPM (Semi-Parametric) classifier traditionally uses a single mean prototype per class during training/validation. This experiment replaces that with **K-means clustering** to generate **K prototypes per class**, allowing the model to capture multi-modal class distributions.

### Key Hypothesis
Multi-prototype representation → Better class modeling → Improved accuracy and unlearning robustness

---

## Critical Implementation Details

### 1. PyTorch-based K-means (Preserves Gradients)

The previous sklearn-based approach with `.detach()` broke the gradient flow, preventing encoder fine-tuning. The new PyTorch implementation preserves gradients:

```python
def pytorch_kmeans(embeddings, K, num_iterations=10):
    """All operations are PyTorch tensors → gradients flow through centroids"""
    centroids = embeddings[torch.randperm(len(embeddings))[:K]].clone()
    
    for _ in range(num_iterations):
        distances = torch.cdist(embeddings, centroids)
        assignments = torch.argmin(distances, dim=1)
        
        # This operation is differentiable!
        centroids = torch.stack([
            embeddings[assignments == k].mean(dim=0) if (assignments == k).any()
            else centroids[k]
            for k in range(K)
        ])
    
    return centroids  # Gradients propagate back to embeddings
```

**Why this matters:**
- In training: Loss → MoE → centroids → encoder (now possible!)
- Encoder parameters can be optimized with respect to aggregated prototypes
- All K values (K=1, K=3, K=5) benefit from proper gradient flow

### 2. Consistent Evaluation Methodology

**THE MAIN CHANGE:** Evaluation now uses the same aggregation pathway as training.

**Before (Inconsistent):**
```
Training:   support_samples → K-means → prototypes → MoE
Evaluation: k-NN → (raw embeddings, no aggregation) → MoE
```
Problem: Training and evaluation use different pathways

**After (Consistent):**
```
Training:   support_samples → K-means → prototypes → MoE
Evaluation: k-NN → K-means aggregation → prototypes → MoE
```
Benefit: Fair comparison; both pathways test the multi-prototype hypothesis

### 3. Always Apply Aggregation

Removed the condition that skipped aggregation when `support_samples == num_classes`. Now aggregation is always applied, ensuring K-means is used throughout training and evaluation.

---

## Evaluation Methodology (MAIN PATHWAY)

The evaluation uses an **aggregation + MoE pathway** that mirrors training:

### Step-by-Step Data Flow

```
Input:
  - Query images: (B, 3, 32, 32)
  - Training data: all training images

1. Encode Query
   query_emb = encoder(query_imgs)  # (B, 512)

2. FAISS k-NN Search (get diverse support set)
   k_neighbors_embs, k_neighbors_labels = FAISS.search(query_emb, k=3072)
   # Returns: (B, k, 512), (B, k)
   # Note: These are RAW embeddings from training set

3. Apply K-means Aggregation (CONSISTENT WITH TRAINING)
   # Reshape for processing
   k_neighbors_flat = k_neighbors.reshape(B*k, 512)
   labels_flat = k_neighbors_labels.reshape(B*k)
   
   # Apply same aggregation as training
   aggregated_prototypes, proto_labels = Feature_Aggregation_KMeans(
       k_neighbors_flat, 
       labels_flat
   )
   # For each class in k-NN results:
   #   Run K-means to get K prototypes
   # Output: (K*C, 512), (K*C,) where C = num_classes
   
   Example:
   - Input: ~307 k-NN embeddings with labels
   - CIFAR-10: ~30 from each class (307/10)
   - K=3: Run K-means on 30 samples → 3 prototypes per class
   - Output: 30 prototypes total (3 × 10 classes)

4. MoE Forward Pass (same as training)
   moe_output = moe(query_emb, aggregated_prototypes, proto_labels)
   # (B, 10) logits

5. Prediction
   preds = torch.argmax(moe_output, dim=-1)  # (B,)
   accuracy = (preds == query_labels).float().mean()

Output:
  - Test accuracy
  - Test loss
```

### Why This Evaluation is Correct

1. **Hypothesis-specific**: Directly tests whether multiple prototypes improve SPM
2. **Consistent with training**: Both use aggregation + MoE
3. **Fair comparison**: K=1, K=3, K=5 evaluated with same methodology
4. **Gradient-aware**: Training optimization tested with actual eval pathway

### Difference from Original SPM Evaluation

| Aspect | Original SPM | K-means SPM |
|--------|--------------|------------|
| k-NN search | ✓ Get k neighbors | ✓ Get k neighbors (same) |
| Aggregation | None (raw embeddings) | ✓ K-means per class (NEW) |
| Prototypes | k raw embeddings | K prototypes per class |
| MoE input | Variable # per query | Consistent # per query |
| Test how | Different at test/train | Same at test/train |

---

## Expected Results

### Baseline Comparison
- **K=1**: Single K-means cluster per class (should approximate baseline)
  - Expected: Slightly different from original mean due to K-means algorithm
  - Gradient flow: ✓ Now working correctly
- **K=3**: Three prototypes per class (MAIN EXPERIMENT)
  - Expected: Improved accuracy over K=1 if classes are multi-modal
- **K=5**: Five prototypes per class (ABLATION)
  - Expected: Further improvement or saturation/overfitting

### Key Metrics
```
Compare across K=1, K=3, K=5:
1. Training accuracy (convergence speed)
2. Validation accuracy (generalization)
3. Test accuracy (final performance)
4. Training time (K-means overhead)
5. Gradient magnitude (to verify backprop works)
```

### Positive Indicators
- K=3 > K=1 significantly → Multiple prototypes help!
- K=5 > K=3 moderately → Diminishing returns (expected)
- Encoder loss decreases → Gradients flowing correctly
- K=1 ≈ Baseline → PyTorch K-means working correctly

### Warning Signs
- K=3 ≈ K=1 → Classes well-separated, multi-modal modeling not needed
- K=5 << K=3 → Overfitting, K too large for support set size
- Encoder loss stagnant → Gradient flow issue or learning rate problem
- NaN loss → Numerical issues (unlikely, but check empty clusters)

---

## Experimental Setup

### Models to Train

```yaml
1. Baseline:           cifar_resnet18.yaml
   └─ Single mean prototype per class (original)

2. K-means K=1:        cifar_resnet18_kmeans_k1.yaml
   └─ Single K-means cluster per class (comparison)

3. K-means K=3:        cifar_resnet18_kmeans_k3.yaml
   └─ THREE prototypes per class (MAIN EXPERIMENT)

4. K-means K=5:        cifar_resnet18_kmeans_k5.yaml
   └─ Five prototypes per class (ABLATION)
```

### Training Command

```bash
# Train main experiment
python main.py --config config/classification/cifar_resnet18_kmeans_k3.yaml

# Or train all variants
for K in 1 3 5; do
    python main.py --config config/classification/cifar_resnet18_kmeans_k${K}.yaml
done
```

### Monitoring

- **W&B Dashboard:** Track validation/test accuracy over epochs
- **Gradient Flow:** Check that encoder loss decreases
- **Training Time:** Monitor K-means computation overhead
- **Memory:** Should be similar to or less than baseline

---

## Configuration Details

### Key YAML Parameters

```yaml
model:
  target: models.spm_classifier_kmeans.SemiParametricClassifier_KMeans
  params:
    K: 3                           # ← PRIMARY VARIABLE
    encoder_type: resnet18
    num_classes: 10
    learning_rate: 0.1
    pretrained_imagenet: false

data:
  params:
    support_size_train: 256        # Support set size during training
    support_size_eval: 3072        # k for FAISS k-NN during evaluation
```

### Understanding K Parameter

```python
# For each class during aggregation:
actual_k = min(K, num_samples_in_class)

Example: CIFAR-10, 256 support samples, K=3
- Each class has ~25 samples
- actual_k = min(3, 25) = 3
- Result: 3 prototypes × 10 classes = 30 total

Example: K=5 with imbalanced classes
- Class A: 50 samples → 5 prototypes
- Class B: 2 samples → 2 prototypes (not forced to 5)
- Class C: 20 samples → 5 prototypes
```

---

## Running the Experiment

### Step 1: Train Models

```bash
cd /path/to/spm_unlearning

# Train K-means K=3 (main experiment)
python main.py --config config/classification/cifar_resnet18_kmeans_k3.yaml

# For ablation studies
python main.py --config config/classification/cifar_resnet18_kmeans_k1.yaml
python main.py --config config/classification/cifar_resnet18_kmeans_k5.yaml
```

### Step 2: Monitor Training

- Open W&B project: `https://wandb.ai/[your-entity]/Spm`
- Track runs: `cifar_resnet18_kmeans_k1`, `cifar_resnet18_kmeans_k3`, `cifar_resnet18_kmeans_k5`
- Compare metrics: val_acc, test_acc, train_loss

### Step 3: Evaluate Results

```bash
jupyter notebook evaluation/kmeans_experiment.ipynb
```

In the notebook:
1. Load checkpoint paths
2. Plot training curves
3. Compare test accuracies
4. Analyze interpretation of results
5. Export results to JSON

### Step 4: Interpretation

Use the notebook's interpretation guide to determine if:
- Multiple prototypes helped (K=3 > K=1)
- Diminishing returns occurred (K=5 vs K=3)
- Gradient flow worked (encoder loss decreased)

---

## Computational Considerations

### K-means Overhead

```
Per batch (256 support samples):
- Encode: ~50ms (standard ResNet18 forward)
- K-means per class: ~10-20ms (10 iterations × 10 classes)
- MoE forward: ~5ms (same as baseline)
- Total overhead: ~10-30% (depends on K)

Memory: ~same as baseline (prototypes replace single mean)
```

### Optimization Tips

```python
# If training too slow:
# 1. Reduce support_size_train in config
# 2. Reduce K (use K=1 for debugging)
# 3. Reduce kmeans_iterations in Feature_Aggregation_KMeans

# If overfitting:
# 1. Reduce K
# 2. Add data augmentation
# 3. Increase regularization (weight decay, dropout)
```

---

## Data Files and Paths

### Model Checkpoints
```
results/classification/
├── cifar_resnet18/                 (Original baseline)
├── cifar_resnet18_kmeans_k1/       (K=1)
├── cifar_resnet18_kmeans_k3/       (K=3 - main)
└── cifar_resnet18_kmeans_k5/       (K=5 - ablation)
```

### Config Files
```
config/classification/
├── cifar_resnet18.yaml             (Original)
├── cifar_resnet18_kmeans_k1.yaml   (K=1)
├── cifar_resnet18_kmeans_k3.yaml   (K=3)
└── cifar_resnet18_kmeans_k5.yaml   (K=5)
```

### Evaluation Notebook
```
evaluation/kmeans_experiment.ipynb  (Comprehensive evaluation)
```

---

## Technical Notes

### PyTorch K-means Algorithm

```
Input: embeddings (N, D), K, num_iterations=10

1. Initialize centroids
   idx = torch.randperm(N)[:K]
   centroids = embeddings[idx].clone()  # (K, D)

2. Iterative clustering
   for iter in range(num_iterations):
     a. distances = ||embeddings - centroids||²  # (N, K)
     b. assignments = argmin(distances, dim=1)   # (N,)
     c. centroids = stack([
          mean(embeddings[assignments==k])
          for k in range(K)
        ])  # (K, D)

3. Return centroids
   All operations preserve gradients!
```

### Gradient Backpropagation

```
Forward pass:
  support_emb (N, D)
      ↓ [PyTorch K-means]
  centroids (K, D)  ← differentiable w.r.t. support_emb
      ↓ [MoE]
  logits (B, C)
      ↓ [Loss]
  loss (scalar)

Backward pass:
  loss.backward()
  ∂loss/∂logits → ∂loss/∂centroids → ∂loss/∂support_emb
                                   → ∂loss/∂encoder
  
Encoder parameters updated! ✓
```

---

## Troubleshooting

### Issue: Loss is NaN
**Cause:** Empty clusters during K-means  
**Check:** Already handled in code - empty clusters keep previous centroid  
**Solution:** Reduce K or check for data issues

### Issue: Training is very slow
**Cause:** K-means overhead on large support sets  
**Solutions:**
- Reduce `support_size_train` (e.g., 256 → 128)
- Reduce `K` (e.g., 3 → 1)
- Reduce K-means iterations (modify code: `kmeans_iterations=5`)

### Issue: K=3 not better than K=1
**Cause:** Classes might be well-separated or need more samples  
**Investigation:**
- Check class separability in embedding space
- Visualize k-NN distributions
- Try K=5 or K=7
- Increase `support_size_train`

### Issue: K=5 worse than K=3
**Cause:** Overfitting - K too large for support set  
**Solution:**
- Increase `support_size_train`
- Reduce K back to 3
- Add regularization

---

## Interpreting the Evaluation Methodology

### Why Use Aggregation in Evaluation?

**Original SPM:**
- Training: Uses support set as-is (or computes mean)
- Evaluation: Uses k-NN neighbors as-is
- Both pathways similar but not identical

**K-means SPM:**
- Training: Aggregates support set via K-means
- Evaluation: Aggregates k-NN results via K-means
- Both pathways identical ✓
- Tests: "Do multiple prototypes help SPM?"

### Why Not Use Raw k-NN?

If we used raw k-NN in evaluation:
- Training uses aggregation
- Evaluation uses raw embeddings
- Unfair comparison
- Doesn't test the hypothesis properly

### Why k-NN in Evaluation at All?

We use k-NN to get a diverse support set for each query:
- Ensures representatives from all classes
- Simulates having labeled data at test time
- Consistent with original SPM evaluation
- Then we aggregate via K-means (new)

---

## Expected Paper Results

### Headline Finding
"Multi-prototype SPM representation improves classification accuracy by ~X% and unlearning robustness by ~Y% compared to single-prototype baseline."

### Figure 1: Training Progress
- x-axis: Epoch
- y-axis: Validation Accuracy
- Lines: Original, K=1, K=3, K=5
- Expected: K=3 > K=1 ≈ Baseline, K=5 > K=3

### Figure 2: Test Accuracy Comparison
- Bar chart: K=1, K=3, K=5
- Expected: K=3 >> K=1, K=5 slightly > K=3

### Figure 3: Analysis
- Class prototypes visualization
- K-means centroids in embedding space
- Multi-modal class distributions

---

## References & Future Work

### Current Implementation
- PyTorch K-means: Lloyd's algorithm, 10 iterations
- Evaluation: FAISS k-NN + aggregation + MoE
- K configurable: 1-5 recommended

### Future Improvements
1. Learned prototype generation (end-to-end learnable)
2. Adaptive K per class (learn optimal K)
3. Soft K-means (probabilistic cluster assignment)
4. GPU-accelerated K-means (faster training)
5. Prototype distillation (K-means in training, single mean in eval)

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python main.py --config config/classification/cifar_resnet18_kmeans_k3.yaml` | Train main experiment |
| `jupyter notebook evaluation/kmeans_experiment.ipynb` | Evaluate and visualize |
| Check W&B | Monitor training progress |
| Compare checkpoints | Analyze model differences |

---

## Summary

This experiment tests whether multiple prototypes per class (via K-means clustering) improve the SPM classifier. Key improvements over previous work:

1. ✓ PyTorch K-means preserves gradients
2. ✓ Evaluation uses aggregation (consistent with training)
3. ✓ Always aggregate (K parameter always active)
4. ✓ Fair comparison: K=1 vs K=3 vs K=5

Expected outcome: **K=3 provides better class representation → improved accuracy**
