# K-means SPM Classifier: Implementation Summary (v2 - Fixed)

## Quick Overview

Modified the SPM classifier to use **K-means clustering** to generate **K prototypes per class** instead of a single mean prototype. **CRITICAL FIX: Now uses PyTorch K-means with preserved gradients and consistent evaluation methodology.**

## Critical Fixes (v2)

### ✅ 1. PyTorch-based K-means with Preserved Gradients
**Problem:** Previous implementation used `.detach()` and sklearn K-means, breaking the autograd path
**Solution:** Implemented native PyTorch K-means algorithm
```python
def pytorch_kmeans(embeddings, K, num_iterations=10):
    centroids = random K embeddings
    for iteration in range(10):
        # Hard assign points to nearest centroid
        assignments = argmin(||embeddings - centroids||)
        # Recompute centroids (DIFFERENTIABLE)
        centroids = stack([mean(embeddings[assignments==k]) for k in range(K)])
    return centroids
```
**Impact:** Loss now backpropagates through centroids to encoder! ✓

### ✅ 2. Consistent Aggregation in Training AND Evaluation
**Problem:** Evaluation used raw k-NN embeddings while training used aggregation (inconsistent)
**Solution:** Apply K-means aggregation in both training and evaluation
```
Training:   support_samples → K-means → prototypes → MoE
Evaluation: k-NN results → K-means → prototypes → MoE  ← CHANGED
```
**Impact:** Fair hypothesis testing; both test if multiple prototypes help SPM

### ✅ 3. Always Apply Aggregation
**Problem:** Condition meant aggregation skipped when `support_imgs.shape[0] == num_classes`
**Solution:** Always apply Feature_Aggregation_KMeans
**Impact:** K-means always used; K parameter always matters

---

## What Changed: File-by-File

### ✅ Updated: `models/spm_classifier_kmeans.py` (~500 lines)

**Key changes:**
1. Removed sklearn, implement PyTorch K-means
2. Feature_Aggregation_KMeans.pytorch_kmeans() static method
3. Forward(): Always aggregate (removed condition)
4. test_step(): Apply aggregation to k-NN results before MoE

### ✅ Created: `evaluation/kmeans_experiment.ipynb`
Comprehensive Jupyter notebook with:
- Evaluation methodology explanation with data flow diagrams
- Expected results analysis and interpretation guide
- Plotting framework for training progress
- Comparison of K=1, K=3, K=5 results

### ✅ Unchanged
- Original `models/spm_classifier.py`
- All dataset/datamodule files
- All training/validation logic

---

## Data Flow: Before vs After

### Training Forward Pass
```
BEFORE (Broken Gradients):
  support_emb → [.detach() + sklearn] → numpy → K-means → tensor → prototypes
                ✗ GRADIENTS LOST HERE

AFTER (Fixed):
  support_emb → [PyTorch K-means] → prototypes
               ✓ GRADIENTS FLOW THROUGH
                 ↓ Backprop to encoder
```

### Evaluation
```
BEFORE (Inconsistent):
  k-NN → raw embeddings → MoE
  (No aggregation, different from training)

AFTER (Consistent):
  k-NN → [PyTorch K-means aggregation] → prototypes → MoE
  (Same pathway as training)
```

---

## Configuration

```bash
# K=3 (main experiment)
python main.py --config config/classification/cifar_resnet18_kmeans_k3.yaml

# K=1 (baseline comparison)
python main.py --config config/classification/cifar_resnet18_kmeans_k1.yaml

# K=5 (ablation)
python main.py --config config/classification/cifar_resnet18_kmeans_k5.yaml

# Original baseline
python main.py --config config/classification/cifar_resnet18.yaml
```

---

## How the Evaluation Works (MAIN PATHWAY)

### Step-by-Step
```
1. Query image → encode → query_embedding (B, 512)

2. FAISS k-NN search
   → Get k neighbors from training set (B*k, 512) + labels

3. Apply K-means aggregation ← KEY CHANGE
   For each class in k-NN results:
     Run K-means to get K prototypes per class
   → Get aggregated prototypes (K*C, 512) + labels

4. MoE forward pass
   moe_output = moe(query_embedding, prototypes, labels)
   → (B, num_classes) scores

5. Predict
   preds = argmax(moe_output)
   accuracy = mean(preds == query_labels)
```

### Why This is Correct
- **Consistent with training:** Both use aggregation + MoE pathway
- **Tests the hypothesis:** Multiple prototypes improve SPM (K=1 vs K=3)
- **Fair comparison:** Same evaluation for all K values
- **Gradient flow:** Training can optimize aggregated prototypes

---

## PyTorch K-means Implementation Detail

```python
@staticmethod
def pytorch_kmeans(embeddings, K, num_iterations=10):
    N, D = embeddings.shape
    device = embeddings.device
    
    # Init: random K points
    indices = torch.randperm(N, device=device)[:K]
    centroids = embeddings[indices].clone()  # (K, D)
    
    for iteration in range(num_iterations):
        # Compute distances
        distances = torch.cdist(embeddings, centroids)  # (N, K)
        
        # Hard assignment
        assignments = torch.argmin(distances, dim=1)  # (N,)
        
        # Recompute centroids (fully differentiable!)
        new_centroids = []
        for k in range(K):
            mask = (assignments == k)
            if mask.sum() > 0:
                new_centroids.append(embeddings[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[k])
        
        centroids = torch.stack(new_centroids)
    
    return centroids  # Gradients flow back!
```

**Gradient Flow:**
- `centroids = mean(embeddings[assignments==k])` is differentiable
- Backward pass computes `∂loss/∂embeddings`
- Encoder receives gradients during training

---

## Expected Results

### Good Signs
- **K=3 >> K=1**: Multiple prototypes help! (main hypothesis validated)
- **K=1 ≈ baseline**: PyTorch K-means working correctly
- **K=5 > K=3**: Further improvements possible
- **Gradients flowing**: Encoder loss decreases during training

### Potential Issues & Fixes
| Issue | Cause | Fix |
|-------|-------|-----|
| K=3 ≈ K=1 | Classes already well-separated | Try K=5 or K=7 |
| K=5 < K=3 | Overfitting, K too large | Reduce K or increase support_size |
| NaN loss | Empty clusters | Already handled (keeps old centroid) |
| Slow training | K-means overhead | Reduce support_size or K |

---

## Files Summary

### New/Updated
```
models/spm_classifier_kmeans.py      (PyTorch K-means, fixed test_step)
evaluation/kmeans_experiment.ipynb   (comprehensive evaluation notebook)

config/classification/
├── cifar_resnet18_kmeans_k1.yaml
├── cifar_resnet18_kmeans_k3.yaml
├── cifar_resnet18_kmeans_k5.yaml
└── imagenet_resnet18_kmeans_k3.yaml
```

### Original (Unchanged)
```
models/spm_classifier.py             (baseline)
```

---

## Key Improvements Over v1

| Aspect | v1 (Broken) | v2 (Fixed) | Benefit |
|--------|----------|-----------|---------|
| K-means | sklearn + detach | PyTorch native | Gradients flow! |
| Evaluation | Raw k-NN | Aggregation + MoE | Consistent with training |
| Aggregation | Conditional | Always applied | K parameter always active |
| Test_step | No aggregation | Aggregation applied | Fair evaluation |

---

## Run Experiments

```bash
# Main experiment (K=3)
python main.py --config config/classification/cifar_resnet18_kmeans_k3.yaml

# After training, evaluate in notebook
jupyter notebook evaluation/kmeans_experiment.ipynb
```

Monitor in W&B dashboard: https://wandb.ai/your-username/Spm

---

## Next Steps

1. ✅ PyTorch K-means implemented (preserved gradients)
2. ✅ Consistent evaluation (aggregation in test_step)
3. ✅ Experiment notebook created
4. → Train models: `python main.py --config ...`
5. → Monitor W&B: Training curves, test accuracy
6. → Load results in notebook
7. → Analyze: Which K is best?
