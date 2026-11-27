# SPM Methodology and Dataset Reduction Report

## Scope

This report explains the mathematical structure of the models and optional
methods implemented in this repository:

- Base semi-parametric classifier
- Deletion-aware classifier variants
- K-means prototype classifier
- Overfitting-mitigation methods
- Dataset reduction methods

The document describes what the code implements; it does not claim that an
unimplemented or untested experiment will improve accuracy.

## 1. SPM episode and base model

### Episodic representation

An episode consists of a query batch $Q$ and a support set $S$:

$$
Q=\{(x_i,y_i)\}_{i=1}^{B},
\qquad
S=\{(s_j,t_j)\}_{j=1}^{N}.
$$


The base classifier encodes the query and support images through a shared
encoder $f_\phi$:

$$
q_i=f_\phi(x_i),\qquad z_j=f_\phi(s_j).
$$

The base model encodes query and support images together before splitting the
joint embedding tensor, allowing BatchNorm to see a shared episode context.
For a support set larger than the number of classes, support embeddings are
aggregated by class:

$$
\mu_c=\frac{1}{|S_c|}\sum_{j:t_j=c}z_j.
$$

This is the mean-prototype representation.

### Pairwise relation expert

For each support prototype $z_c$, the pairwise relation expert computes a
scaled similarity with query $q_i$:

$$
h_{ic}=q_i^\mathsf{T}W_q^\mathsf{T}W_s z_c
\qquad
\alpha_{ic}=
\frac{\exp(h_{ic}/\sqrt{d})}
{\sum_k\exp(h_{ik}/\sqrt{d})}.
$$

One-hot class membership of the support prototypes is denoted $v_c$. The
expert output is

$$
r_i(c)=\sum_k\alpha_{ik}v_k(c).
$$

Thus classes absent from the episode receive zero probability. $r_i(c)$ is
mapped back to the full class dimension.

### Mixture of experts

The repository supports multiple experts. A gating network uses both the query
and support mean:

$$
g_i=\operatorname{softmax}
\left(W_g[q_i;\operatorname{mean}(Z)]\right).
$$

The final class probability is

$$
p_i=\sum_mg_{im}r_i^{(m)}.
$$

The model returns $\log p_i+\epsilon$. Training minimizes negative log
likelihood:

$$
\mathcal L_{\mathrm{NLL}}
=-\frac{1}{B}\sum_i\log p_i[y_i].
$$

### Test-time retrieval

During test, query embeddings are used to search a FAISS index over training
embeddings. The $k$ nearest training embeddings become the episode support
set. In this sense the unlearning operation is semantically local: removing
training support examples removes them from the support set and embedding index
rather than changing every model parameter. The specific FAISS index construction
and support retrieval are implemented in the test/evaluation path.

### Optimizers

For a non-pretrained scratch ResNet-18, SGD with momentum and weight decay is
used. A pretrained ImageNet encoder can use separate encoder and MoE learning
rates. The repository may use cosine annealing, a multi-step schedule, or
`ReduceLROnPlateau`, depending on the model/config variant.

## 2. Deletion-aware model

### Primary deletion-aware classifier

The deletion-aware model reuses the base pairwise-expert architecture and
support-conditioned forward pass. It does not introduce a second parameter
family for deletion itself; instead it changes the training episodes and loss
so the model is exposed to support sets with some information removed.

The alternate deletion-aware class, `SemiParametricClassifierAlt`, uses a
full-support episode and a deleted-support episode:

$$
\mathcal L_{\mathrm{full}}
=\mathrm{NLL}(p(Q,S),y),
$$

$$
\mathcal L_{\mathrm{deleted}}
=\mathrm{NLL}(p(Q,S\setminus D),y),
$$

$$
\mathcal L
=\mathcal L_{\mathrm{full}}
+\lambda_{\mathrm{deleted}}
\mathcal L_{\mathrm{deleted}},
$$

where $D$ is the random fraction removed from the support episode and
$\lambda_{\mathrm{deleted}}$ defaults to 1.0.

The dataloader constructs both support sets in the same batch. Validation and
test retain the normal full-support behavior unless an explicit robustness
evaluation deletes support entries.

### Interpretation

This objective encourages support invariance: predictions should remain stable
for the same query as the support set changes. This is theoretically relevant
to exact or approximate unlearning because deletion removes support rows rather
than requiring direct parameter updates.

The standard deletion-aware base model and alternate variants should not be
interpreted as an exact removal guarantee: exact support deletion is a
test-time and data-index property, whereas this auxiliary objective only
improves robustness.

## 3. K-means prototype model

### K-means support aggregation

The K-means variant replaces the class mean by up to $K$ class-local
prototypes. For class $c$, let

$$
Z_c=\{z_j:t_j=c\}.
$$

The code minimizes the within-cluster squared distance:

$$
\min_{\{C_{c,k}\}}
\sum_{z\in Z_c}
\min_k\|z-C_{c,k}\|_2^2.
$$

It uses:

- A deterministic first centroid
- A farthest-distance K-means++-style initialization
- 10 Lloyd iterations
- A hard cluster assignment
- A centroid mean update
- Differentiable centroid computation through the embedding tensor

If a class has fewer than $K$ embeddings, the implementation uses the
available embedding count as the actual number of prototypes.

### Prototype forward pass

The output support tensor contains prototypes rather than every support row:

$$
\tilde Z_c=
\{C_{c,1},\ldots,C_{c,K_c}\},
\qquad
K_c=\min(K,|Z_c|).
$$

The pairwise expert maps prototype similarities through the prototype labels,
then aggregates probabilities that belong to the same class. The resulting
log-probability objective remains the same NLL as the base model.

### K-means label smoothing

The optional K-means variant can replace a one-hot target with

$$
\tilde y=(1-\epsilon)y+\frac{\epsilon}{C}\mathbf 1.
$$

The loss is

$$
\mathcal L_{\mathrm{LS}}
=(1-\epsilon)\mathcal L_{\mathrm{NLL}}
+\epsilon\frac{1}{C}\sum_c-\log p_i(c).
$$

The K-means implementation uses 10 hard-clustering iterations. Their
centroid calculations are differentiated through PyTorch operations, but the
hard assignment is not itself continuous.

## 4. Overfitting-mitigation framework

### Weight decay

Weight decay adds $\lambda\|\theta\|_2^2/2$ through the existing optimizer
groups. It reduces parameter norms and limits model memorization. The framework
does not silently replace SGD by AdamW.

### Feature dropout

Feature dropout applies

$$
\tilde z=\frac{m\odot z}{1-p},\qquad m_j\sim\mathrm{Bernoulli}(1-p),
$$

during training. It regularizes encoder representations and is architecturally
appropriate because support and query embeddings share the encoder output.

### RandAugment and AugMix

RandAugment applies a random subset of image transforms with configurable
operation count and magnitude. AugMix mixes a set of augmented views and
encourages consistency. Random Erasing removes rectangular regions.

In this repository, these transforms affect training episodes. Validation query
images remain deterministic. Validation support datasets are handled separately
so that test evaluation is not accidentally treated as a training augmentation.

### Mixup

For $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$, query Mixup produces

$$
\tilde x=\lambda x_i+(1-\lambda)x_j,
$$

$$
\mathcal L_{\mathrm{mix}}
=\lambda\mathrm{NLL}(p(\tilde x),y_i)
+(1-\lambda)\mathrm{NLL}(p(\tilde x),y_j).
$$

Mixup is applied to query images only. Support images and support labels remain
unmixed, preserving the episode’s support-conditioned meaning.

### CutMix

CutMix replaces a rectangular region $R$ in a query:

$$
\tilde x=x_i\odot M+x_j\odot(1-M),
$$

$$
\lambda=1-\frac{|R|}{|x|}.
$$

The loss is computed as
$\lambda \mathrm{NLL}(y_i)+(1-\lambda)\mathrm{NLL}(y_j)$.

As with Mixup, support sets are not mixed.

### EMA

EMA maintains

$$
\theta_t^{EMA}
=\beta\theta_{t-1}^{EMA}
+(1-\beta)\theta_t.
$$

This implements a low-pass average of training parameters. A common starting
value is $\beta=0.999$.

### SWA

SWA maintains an equal-weight running mean of checkpoints after the configured
start epoch:

$$
\theta_t^{SWA}
=\theta_{t-1}^{SWA}
+\frac{\theta_t-\theta_{t-1}^{SWA}}{t}.
$$

The implementation supports SWA BatchNorm recalculation over up to 64 training
episodes. Recalculation resets BatchNorm running statistics and updates them
in training mode. For EMA+SWA, the repository currently averages EMA and SWA
parameter tensors equally.

### Early stopping and gradient clipping

Early stopping monitors `val_loss` or `val_acc` for a configurable patience.
Gradient clipping bounds the configured Lightning gradient norm/value before
the optimizer step.

### Combined interactions

Recommended ablations proceed in stages:

1. Weight decay or label smoothing
2. Query Mixup/CutMix
3. RandAugment/erasing
4. EMA
5. SWA with BatchNorm recalculation
6. Early stopping and clipping

Mixup/CutMix and label smoothing can both smooth targets, so simultaneous
strong values may be redundant. RandAugment and Mixup can also combine
stochastic training difficulty; conservative magnitudes are preferable first.
The support set should remain intact in all variants.

## 5. Dataset reduction

All reduction artifacts use:

```json
{
  "artifact_type": "spm_dataset_reduction",
  "artifact_version": 1,
  "selected_indices": [],
  "selection": {},
  "original_training_size": 0,
  "retained_count": 0,
  "excluded_count": 0,
  "overall_retention_pct": 0.0,
  "per_class_retained": {},
  "identifier": ""
}
```

The existing datamodule creates a `Subset` from `selected_indices`. Validation
and test sets are not reduced.

### EL2N

EL2N-style filtering ranks each example by a norm of prediction error. In the
repository, the score is

$$
\mathrm{EL2N}(x_i,y_i)
=\|p_\theta(x_i)-e_{y_i}\|_2.
$$

The script supports class-wise and global exclusion:

- Lowest-score exclusion
- Highest-score exclusion

Class-wise filtering removes low/high-score fractions within each class,
preserving class proportions. Global filtering ranks all examples together.
EL2N requires a pilot model or cached EL2N scores.

### K-center greedy

After EL2N filtering, K-center greedily expands a class-specific subset.
Let $Z_c$ be filtered class embeddings and $S_c$ the selected set. The first
center is sampled with a NumPy seed. Subsequent centers maximize the distance
to the closest selected center:

$$
x_{t+1}
=\arg\max_{x\in Z_c\setminus S_c}
\min_{s\in S_c}\|f(x)-f(s)\|_2^2.
$$

The implementation uses class-wise selection and can preserve coverage over
the embedding manifold.

### Random selection

Random reduction has global or class-wise modes. For class $c$, class-wise
retention keeps approximately
$\rho|Z_c|$ examples selected uniformly. Global selection samples
$\rho N$ examples from all eligible training indices. Seeded NumPy sampling
makes the result reproducible.

### Herding

For class $c$, Herding greedily selects examples whose running sum matches
$m|Z_c|\mu_c$, where $\mu_c$ is the class mean:

$$
x_{t+1}
=\arg\min_x
\left\|
\frac{t+1}{|Z_c|}\sum_{z\in Z_c}z
-\left(\sum_{s\in S_c}z_s+z_x\right)
\right\|_2.
$$

Equivalently, each addition minimizes the discrepancy between the selected
empirical mean and the target class mean. The repository implements this
class-wise and starts with a seeded random first point.

### Moderate Coreset (Xia et al., ICLR 2023)

For class $c$, the implementation computes

$$
\mu_c=\frac{1}{|Z_c|}\sum_{z\in Z_c}z,
\qquad
d_j=\|z_j-\mu_c\|_2.
$$

It determines the median $d^{med}_c$ and retains the requested class fraction
with the smallest distance from the median:

$$
S_c=
\operatorname{argmin}_{A:|A|=k}
\sum_{j\in A}|d_j-d^{med}_c|.
$$

The implementation approximates this criterion by sorting all samples by
$|d_j-d^{med}_c|$ and taking the first $k$. Selection is class-wise and
seeded. It is feature-space Moderate Coreset selection and should not be
claimed as an exact reproduction of every optimization detail in the original
paper.

### EL2N + K-center combination

The existing EL2N/K-center method is a staged pipeline:

1. Use an EL2N pilot or cached EL2N scores to remove low/high-score examples.
2. Use class-wise farthest-first K-center to retain a coverage-oriented subset.
3. Write the selected indices to a compatible artifact.

This combination is different from moderate/near-median sampling and Herding:
EL2N is difficulty-driven, K-center is coverage-driven, Herding matches class
feature moments, and Moderate Coreset selects central distance profiles.

## 6. Practical interactions

Dataset reduction changes the support distribution and therefore the base
episode objective. Each query can only sample support examples from the
selected training subset. Consequently:

- Validation/test support sets may still use the full training support source.
- Evaluation support size and query sampling must remain fixed for fair
  reductions comparisons.
- A pilot checkpoint, feature checkpoint, or model initialization must be
  documented because it determines feature-space reduction.
- Random seeds should cover artifact generation and final training.
- Full-dataset and reduced-dataset artifacts should use distinct filenames.

Overfitting mitigation can interact with reduction: Mixup and augmentation
change training queries, while support distribution is controlled by the
selected subset. Keep the support set episode-compatible and avoid applying
query mixing to support labels.

## 7. Main implementation references

- Base SPM:
  [models/spm_classifier.py](../models/spm_classifier.py)
- Deletion-aware:
  [models/spm_classifier_deletion.py](../models/spm_classifier_deletion.py),
  [models/spm_classifier_deletion_alt.py](../models/spm_classifier_deletion_alt.py)
- K-means:
  [models/spm_classifier_kmeans.py](../models/spm_classifier_kmeans.py)
- Episode dataloaders:
  [utils/dataset.py](../utils/dataset.py)
- Deletion dataloaders:
  [utils/dataset_deletion.py](../utils/dataset_deletion.py)
- Overfitting mitigation:
  [overfitting_mitigation/README.md](../overfitting_mitigation/README.md)
- Existing reduction utilities:
  [utils/dataset_reduction.py](../utils/dataset_reduction.py)
- Herding/Moderate:
  [reduction_comparison/README.md](../reduction_comparison/README.md)
