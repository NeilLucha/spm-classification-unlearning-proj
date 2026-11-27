# Data-Efficient Training of Semi-Parametric Classification Models

## I. Abstract

Large training datasets impose substantial costs on data storage, model optimization, support-set construction, and repeated experimental retraining. Random dataset reduction is computationally simple, but it may remove informative or geometrically distinctive observations and can therefore cause unstable accuracy, particularly for semi-parametric models that classify a query using dynamically constructed support relationships. This paper presents a data-efficient training framework for semi-parametric classification that combines pilot-model EL2N difficulty filtering, class-wise K-center Greedy geometric selection, support-conditioned model training, expanded class representations, deletion-aware training, and an optional overfitting-mitigation suite. EL2N filtering identifies observations that are relatively easy or difficult for a pilot model, while class-wise K-center selection preserves coverage of the remaining feature space and maintains class balance.

The final model is retrained from scratch on the reduced support dataset; pilot weights are used for sample selection but are not transferred to the final classifier. The framework also provides modular Herding and Moderate Coreset baselines, enabling controlled comparison between difficulty-based, coverage-based, moment-matching, and median-distance selection. A repository-evidenced experiment retained 68% of the CIFAR-10 training set and evaluated the resulting model at 93.57% test accuracy, compared with 95.18% for the available full-dataset result, corresponding to an accuracy degradation of 1.61 percentage points. The 32% reduction is therefore a promising and practically meaningful result, although additional matched-seed repetitions are needed before a 0.73 percentage-point degradation can be established as a statistically confirmed outcome. Random and class-wise reduction artifacts are provided for controlled ablations, and live training curves and CSV records allow training behavior to be audited. The framework is thus designed not merely to reduce data, but to make that reduction reproducible, class balanced, episode compatible, and directly comparable across random seeds.

**Keywords:** Dataset Reduction, Coreset Selection, EL2N, K-Center Greedy, Semi-Parametric Models, Deep Learning, Data Efficiency.

## II. Introduction

### A. Background and Motivation

Modern classification pipelines commonly train repeatedly on datasets containing redundant observations. Images in a class may share similar appearance, difficulty, or feature-space location, so a fraction of the training observations may contribute limited additional information. At the same time, full-data training requires repeated data loading, optimizer updates, support-set construction, validation, hyperparameter sweeps, retraining baselines, and evaluation. These costs become important for semi-parametric models, in which both the query and its support examples affect every prediction.

Dataset reduction seeks to construct a smaller training set that preserves model performance. Random pruning is inexpensive, reproducible, and easy to implement, but it does not distinguish difficult samples, representative samples, redundant samples, and outliers. Under constrained retention, random selection can remove category-specific coverage and introduce seed-dependent degradation. The problem is especially relevant for support-conditioned learning because support sets must preserve class coverage and remain mathematically compatible with the query-support episode.

### B. Problem Statement

A sample-selection method must balance at least two properties. First, it must retain sufficiently informative observations rather than relying only on central or easy examples. Second, it must preserve geometric coverage of the feature space so that no region of a class is underrepresented. Random selection may provide neither property reliably.

For semi-parametric classification, the optimization unit is not only an individual image. A query is evaluated together with support observations, and their embeddings determine prototype or pairwise-expert outputs. A reduced dataset therefore changes both the training distribution and the test-time support source. The reduction method must maintain class balance and avoid mixing support semantics with query mixing or artificially transformed support labels.

### C. Research Objectives

The objectives of this work are to:

1. reduce the number of CIFAR-10 training observations while preserving predictive performance;
2. combine difficulty-based filtering with class-wise geometric coverage;
3. preserve class balance across the retained subset;
4. compare the proposed hybrid method with random, Herding, and Moderate Coreset selection;
5. investigate K-means multi-prototype representation and deletion-aware optimization;
6. evaluate whether regularization, augmentation, model averaging, and scheduling remain effective at high exclusion ratios; and
7. preserve reproducibility through seeds, selection artifacts, checkpoint metadata, and live training records.

### D. Contributions

The principal contributions are enumerated as follows.

1. **Hybrid reduction pipeline.** A staged dataset-reduction framework combines low- and high-EL2N filtering with class-wise K-center Greedy selection. EL2N supplies difficulty information, whereas K-center supplies geometric coverage.
2. **Reproducible artifact system.** Selection output is stored in a versioned JSON artifact containing selected indices, retained counts, exclusion percentages, per-class counts, dataset identifiers, seed information, and method parameters.
3. **Expanded class representation investigation.** K-means-derived prototypes are compared with the original mean-prototype representation to study whether intra-class variability improves semi-parametric classification.
4. **Deletion-aware adaptation.** Full-support and deleted-support training episodes are studied as a way to improve robustness when support observations are removed after training.
5. **Overfitting mitigation framework.** Weight decay, feature dropout, label smoothing, RandAugment, AugMix, Random Erasing, MixUp, CutMix, EMA, SWA, learning-rate reduction, gradient clipping, early stopping, and BatchNorm recalibration are exposed as independent, optional configurations.
6. **Experimental evaluation and reproducibility tooling.** The repository includes random and hybrid selection artifacts, per-run training curves, CSV metric records, checkpoint metadata, and controlled deletion evaluation.

The hybrid reduction pipeline is a novel combination of established difficulty and coverage mechanisms. K-means representations, deletion-aware training, and overfitting mitigation are largely methodological adaptations or investigations of established techniques within the semi-parametric framework. This distinction is important because the central research claim concerns the combination and its support-query compatibility, not a claim that every individual regularizer is novel.

## III. Related Work

### A. Dataset Reduction and Coreset Selection

Coreset and data-pruning methods construct small training sets intended to approximate the information or geometry of a larger dataset. Importance-based approaches select observations according to loss, gradient, uncertainty, or model-dependent scores, whereas coverage-based approaches optimize geometric diversity. Moment-matching methods select observations whose aggregate statistics approximate the original class distribution.

Dataset reduction differs from conventional compression because selected observations remain ordinary training examples. It therefore preserves the original model architecture and optimization objective, unlike distillation, which synthesizes labels or feature representations. The repository consequently implements a selection layer that is independent of the final model and datamodule.

### B. Difficulty-Based Sample Selection

EL2N measures the distance between a model's predicted probability vector and the one-hot target. A sample with a large EL2N norm can be genuinely difficult, poorly modeled, mislabeled, or located in a region of low confidence. Low-EL2N observations can be redundant for a given pilot model, but removing only easy observations may preserve a narrow, overconfident understanding of the class. Excluding both low- and high-difficulty extremes is a bounded approach intended to remove redundancy while avoiding excessive concentration on unusually difficult observations.

Related work includes confidence-based pruning, forgetting-event methods, loss-based ranking, gradient norms, and influence functions. These methods differ in whether they require one pilot run, ensemble predictions, gradients, or full retraining.

### C. Geometric Coverage-Based Selection

The K-center objective minimizes the maximum distance from any point to its nearest selected representative. A greedy farthest-point implementation provides an approximation to this objective. In a class embedding space, the result is a diverse subset that is less likely to leave uncovered regions.

Herding instead minimizes discrepancy between the selected subset mean and a class target mean. Moderate Coreset selection retains samples near the median of per-example distance to the class mean. These baselines provide complementary notions of representativeness: moment matching, central feature profiles, and geometric coverage.

### D. Overfitting Mitigation in Deep Learning

Weight decay regularizes parameter magnitude, and dropout regularizes hidden representations. Label smoothing limits excessively confident targets. RandAugment, AugMix, Random Erasing, MixUp, and CutMix improve robustness by increasing training diversity or smoothing the target and input manifolds. EMA and SWA average parameter estimates across training, often improving generalization, and SWA can require BatchNorm recalibration.

Reduce-on-plateau schedules, gradient clipping, and early stopping are complementary optimization controls. These mechanisms are particularly useful in small-subset training, where memorization can occur quickly and validation behavior can deteriorate after the selected support distribution has been exhausted.

### E. Semi-Parametric Classification

Semi-parametric classification combines learned encoders with support examples or prototypes. Unlike a purely parametric classifier, predictions depend on a support set that can be updated, filtered, or deleted at evaluation time. This creates an exact or efficient support-deletion mechanism but introduces variability from support sampling. Expanded prototypes are a natural generalization of mean prototypes because a single class mean may not capture multimodal or within-class variation.

The principal implementation gap addressed here is that support-conditioned episodes are sensitive to both the geometric composition of the support data and to support-set changes introduced by dataset reduction. General image-augmentation recipes must therefore be adapted so that support-query relationships remain valid.

## IV. Background and Preliminaries

### A. Semi-Parametric Classification Framework

Let a training episode contain a query batch

$$
\[
Q=\{(x_i,y_i)\}_{i=1}^{B}
\]
$$

and a support set

\[
S=\{(s_j,t_j)\}_{j=1}^{N}.
\]

A learned encoder \(f_\phi\) maps both query and support images into an embedding space:

\[
q_i=f_\phi(x_i),\qquad z_j=f_\phi(s_j).
\]

For class \(c\), the base model forms a mean representation

\[
\mu_c=\frac{1}{|S_c|}\sum_{j:t_j=c}z_j,
\qquad S_c=\{j:t_j=c\}.
\]

A pairwise relation expert receives a query embedding \(q_i\) and support prototypes \(\{\mu_c\}\). Scaled similarity coefficients are

\[
\alpha_{ic}=
\frac{\exp(q_i^\top W_q^\top W_s\mu_c/\sqrt{d})}
{\sum_{k}\exp(q_i^\top W_q^\top W_s\mu_k/\sqrt{d})}.
\]

For a one-hot class membership vector \(v_c\), an expert produces

\[
r_i(c)=\sum_k\alpha_{ik}v_k(c).
\]

The class probabilities of a mixture containing \(M\) experts are

\[
p_i=\sum_{m=1}^{M}g_{im}r_i^{(m)},
\]

where the gating network assigns

\[
g_i=\operatorname{softmax}\left(W_g[q_i;\operatorname{mean}(Z)]\right).
\]

The classifier minimizes

\[
\mathcal L_{\mathrm{NLL}}
=-\frac{1}{B}\sum_{i=1}^{B}\log p_i[y_i].
\]

During test-time retrieval, FAISS is used to identify nearest training embeddings and construct a per-query or per-batch support set. This support retrieval is the mechanism that makes support deletion and support-conditioned evaluation local to the episode.

### B. Single-Prototype and Multi-Prototype Class Representations

The original implementation represents a class with one mean embedding. The K-means variant represents class \(c\) with up to \(K\) prototypes:

\[
\mathcal P_c=\{p_{c,1},\ldots,p_{c,K_c}\},
\qquad K_c=\min(K,|S_c|).
\]

Given class embeddings \(Z_c\), K-means minimizes

\[
\min_{\{C_{c,k}\}}
\sum_{z\in Z_c}
\min_k\|z-C_{c,k}\|_2^2.
\]

The implementation uses deterministic initialization, farthest-distance K-means++-style initialization, 10 Lloyd iterations, and differentiable centroid recomputation. Hard assignments remain discrete, but centroid values remain differentiable through the embedding path.

Multiple prototypes can preserve within-class modes, but they also increase support representation size and aggregate clustering cost. For \(K>1\), an implementation should be compared not only for accuracy but also for episode-construction and inference overhead.

### C. EL2N Scoring

For a pilot model \(f_\theta\), EL2N is

\[
E(x_i,y_i)=\left\|p_\theta(x_i)-e_{y_i}\right\|_2,
\]

where \(e_{y_i}\) is the one-hot vector for class \(y_i\). Low scores indicate high agreement with the pilot prediction; high scores indicate disagreement or uncertainty. EL2N is not a label-noise detector on its own. It is a model-dependent difficulty score and can be affected by pilot architecture, epoch, initialization, augmentation, and support set.

### D. K-Center Greedy Selection

The exact K-center objective is

\[
\min_{S\subset X,\ |S|=k}
\max_{x\in X}\min_{s\in S}d(x,s).
\]

The greedy approximation begins with a seeded center and iteratively selects

\[
x_{t+1}
=\arg\max_{x\in X\setminus S_t}
\min_{s\in S_t}\|f(x)-f(s)\|_2^2.
\]

This procedure favors samples that cover poorly represented regions. Class-wise application maintains balanced per-class retention.

## V. Proposed Methodology

### A. Overview of the Proposed Framework

The proposed pipeline is:

```text
Original Training Dataset
        |
        v
Pilot SPM Training
        |
        v
EL2N Score Computation
        |
        v
Low-EL2N Filtering
        |
        v
High-EL2N Filtering
        |
        v
Class-Wise K-Center Greedy Selection
        |
        v
Reduced Training Dataset
        |
        v
Final SPM Training from Scratch
        |
        v
Evaluation Against Full and Random-Reduced Baselines
```

The pilot model supplies scores and embeddings; it does not initialize the final model. This separation prevents the final run from being directly coupled to pilot parameters while still allowing the pilot to identify data difficulty and coverage.

### B. Pilot Model Training

A separate pilot model is trained for selection on the same dataset and class label space. The pilot architecture is consistent with the intended final SPM configuration, such as a ResNet-18 encoder and support-conditioned MoE. Pilot training is repeated until predictions and embeddings are available for all eligible training observations.

The pilot checkpoint is a selection artifact rather than a final weight checkpoint. Pilot weights are not transferred to the final model. This is necessary because the final training experiment is intended to measure the effect of data reduction independently of pretrained parameter transfer.

### C. Difficulty-Based Filtering Using EL2N

Scores are computed over the eligible training observations. Within each class, observations are sorted by EL2N and the low- and high-score fractions are removed:

\[
I_c^{\mathrm{low}}
=\operatorname{argbottom}_{r_l|I_c|} E(x_i,y_i),
\]

\[
I_c^{\mathrm{high}}
=\operatorname{argtop}_{r_h|I_c|} E(x_i,y_i).
\]

The remaining indices are

\[
I_c^{EL2N}=I_c\setminus
\left(I_c^{\mathrm{low}}\cup I_c^{\mathrm{high}}\right).
\]

Low-EL2N removal reduces redundant easy examples. High-EL2N removal mitigates overconcentration on outliers or highly difficult examples. Ties and exclusions are applied deterministically within a class.

### D. Class-Wise Geometric Selection Using K-Center Greedy

After filtering, pilot encoder embeddings are extracted for the remaining observations. The remaining set is partitioned by class. For class \(c\), K-center greedily chooses \(k_c\) observations. The first point is sampled from the class with a fixed seed. Every subsequent point maximizes its minimum squared distance to the currently selected set.

Class-wise selection is used instead of selecting globally so that all represented classes retain an approximately balanced fraction of observations. This property is important for both class-conditional accuracy and support sets containing observations from the selected training set.

### E. Dataset Retention and Exclusion Strategy

Let

- \(r_l\) denote low-EL2N exclusion;
- \(r_h\) denote high-EL2N exclusion; and
- \(r_k\) denote K-center retention after EL2N filtering.

For a class containing \(n_c\) observations, the approximate remaining fraction is

\[
\rho_c \approx
r_k(1-r_l-r_h),
\]

with integer rounding performed by each stage. For equal class counts, the overall target retention is

\[
\rho=
r_k(1-r_l-r_h).
\]

For example, 10% low-EL2N exclusion, 5% high-EL2N exclusion, and 68% K-center retention correspond to approximately

\[
\rho\approx 0.68(1-0.10-0.05)=0.578.
\]

The repository also supports direct retention artifacts and class-wise random baselines, so final `overall_retention_pct` is measured from retained indices rather than assumed from the nominal percentages.

### F. Final Model Training

The final model is instantiated from the same architecture and hyperparameters used for the full-dataset baseline, except for the subset supplied by the datamodule. Training support sampling is restricted to the subset. Validation and test query data remain unchanged. The full-dataset and reduced-dataset runs use identical random seeds where possible and maintain matching support and batch sizes.

### G. Computational Complexity

EL2N and embedding extraction require one pass or one pilot training run. For \(N\) training observations and embedding dimension \(d\), EL2N ranking is \(O(N)\) after forward passes, and feature extraction is \(O(Nd)\) for the encoder cost. Naive class-wise K-center requires approximately

\[
O(k_cN_cd)
\]

distance computations for class \(c\), where \(k_c\) is the number selected and \(N_c\) is the class population after filtering.

This one-time selection cost should be compared with repeated final training. A 32% data reduction lowers subsequent episode construction and optimizer work by approximately 32%, before accounting for data-loading overhead or support-size changes. K-center is more expensive than random selection, but random selection is also less reliable.

## VI. Overfitting Mitigation Framework

### A. Regularization

Weight decay adds

\[
\mathcal L_{\mathrm{WD}}
=\mathcal L_{\mathrm{task}}+
\frac{\lambda}{2}\|\theta\|_2^2.
\]

Feature dropout applies

\[
\tilde z=\frac{m\odot z}{1-p},
\qquad m_j\sim\mathrm{Bernoulli}(1-p).
\]

Label smoothing transforms the target

\[
\tilde y=(1-\epsilon)y+\frac{\epsilon}{C}\mathbf 1,
\]

and reduces overconfidence in the support-conditioned probability output.

### B. Data Augmentation

RandAugment randomly applies a configurable number of image transformations at a controlled magnitude. AugMix combines augmented branches with the original image to encourage stability. Random Erasing removes a randomly selected rectangular region. These techniques increase training diversity and can reduce memorization when only a subset is available.

### C. Sample Mixing

For MixUp, \(\lambda\sim\mathrm{Beta}(\alpha,\alpha)\) and

\[
\tilde x=\lambda x_i+(1-\lambda)x_j,
\]

\[
\mathcal L_{\mathrm{mix}}
=\lambda\mathcal L(p(\tilde x),y_i)
+(1-\lambda)\mathcal L(p(\tilde x),y_j).
\]

For CutMix,

\[
\tilde x=x_i\odot M+x_j\odot(1-M),
\]

\[
\lambda=1-\frac{|R|}{|x|},
\]

where \(R\) is the replaced region.

MixUp and CutMix are restricted to query images. Support images and labels are not mixed. Mixing support rows would change both the support-label relation and the episode semantics, potentially making the reduction experiment incompatible with test-time support deletion.

### D. Model Weight Averaging

EMA maintains

\[
\theta_t^{\mathrm{EMA}}
=\beta\theta_{t-1}^{\mathrm{EMA}}
+(1-\beta)\theta_t.
\]

SWA maintains

\[
\theta_t^{\mathrm{SWA}}
=\theta_{t-1}^{\mathrm{SWA}}
+\frac{\theta_t-\theta_{t-1}^{\mathrm{SWA}}}{t}.
\]

EMA and SWA checkpoints use weight tensors compatible with ordinary Lightning evaluation. When SWA is enabled, BatchNorm running statistics can be recalculated over a bounded set of training episodes. This is necessary because averaging parameters without recalibrating BatchNorm can produce incorrect validation activation scales.

### E. Optimization and Training Stabilization

The K-means ResNet-18 configuration uses SGD and `ReduceLROnPlateau` on validation loss. The scheduler reduces learning rate by a factor of 0.1 after five unimproved epochs, down to a minimum learning rate of \(10^{-5}\). Gradient clipping bounds optimizer gradients, and early stopping terminates runs after a configured patience.

### F. Configuration and Reproducibility

All mitigation values are stored in the same mitigation configuration object and checkpoint metadata. Curves and CSV records include model, learning rate, support sizes, seed, mitigation parameters, and reduced-dataset identifier. Distinct artifacts prevent full-dataset and reduced-dataset curves or checkpoints from overwriting each other.

## VII. Experimental Setup

### A. Dataset

CIFAR-10 contains 50,000 training images and 10,000 test images from 10 classes. Images have a size of \(32\times32\) pixels. The standard normalization and horizontal-flip/crop augmentation are used by the base datamodule. The evaluation protocol keeps test queries fixed where possible and varies only support construction or training-subset composition.

### B. Model Architecture

The primary implementation uses a ResNet-18 encoder, support embeddings, pairwise relation experts, and a gating network. The base support representation averages embeddings within class, whereas the K-means variant constructs up to \(K\) prototypes per class. CIFAR training uses episode support sizes of 256 and evaluation support sizes of 4096 in the recorded experiments.

### C. Experimental Environment

Experiments use PyTorch, PyTorch Lightning, torchvision, NumPy, FAISS, and a CUDA GPU. Selection supports GPU-limited operation and stores progress information for pilot, EL2N, and K-center stages. Exact wall-clock speed depends on GPU generation, worker count, support batch size, and data-loader configuration.

### D. Evaluation Metrics

The primary metrics are classification accuracy, training loss, validation loss, and accuracy degradation:

\[
\Delta A=A_{\mathrm{full}}-A_{\mathrm{reduced}}.
\]

Additional metrics include support-deletion agreement, retained-training accuracy, retained-test accuracy, inference time, CE, KL divergence, and variability across seeds.

### E. Experimental Protocol

The protocol consists of:

1. preparing the full CIFAR-10 training and test sets;
2. generating or loading a pilot checkpoint;
3. computing EL2N scores and embeddings;
4. writing a selection artifact;
5. instantiating a fresh final model;
6. training with the reduced subset and mitigation configuration;
7. evaluating the final checkpoint using the existing evaluation pathway; and
8. repeating random and hybrid runs under matched seeds.

No pilot weight transfer is used in the reduced final training experiments.

## VIII. Experimental Results and Analysis

### A. Baseline Performance

The available repository artifacts contain a full-dataset result of 95.18% test accuracy for the mitigated random-deletion model configuration. A separate full-dataset deletion-aware result reported 94.46% validation average for full support and 94.45% validation average for deleted support, with a gap of only 0.01 percentage points.

These results come from different model or evaluation configurations and should not be interpreted as a unified multi-seed study. The baseline section is therefore presented to establish reference observations, not to rank every variant from a single controlled experiment.

### B. Effect of Dataset Retention Percentage

The repository contains hybrid selection artifacts at several retention levels:

| Retention | Exclusion | Selected observations | Available accuracy |
|---|---:|---:|---:|
| 68.0% | 32.0% | 34,000 | 93.57% test |
| 51.0% | 49.0% | 25,500 | 92.00% checkpoint validation |
| 46.5% | 53.5% | 23,250 | Not reported |
| 42.5% | 57.5% | 21,250 | 90.00% checkpoint validation |
| 36.98% | 63.02% | 18,490 | 90.00% checkpoint validation |

The 68% hybrid artifact is the only one with a directly comparable retained-split evaluation artifact in the current repository. It produced 93.57% test accuracy, for an observed degradation of 1.61 percentage points from the 95.18% full-dataset result. The training-curve record for the 68% run reached approximately 93.27% validation accuracy. This is consistent with the epoch/checkpoint evaluation distinction, but the curve and final evaluation should not be reported as identical metrics.

For retention levels without a test-accuracy artifact, the checkpoint filename provides the recorded validation accuracy. The corresponding values are 92.00% at 51.0% retention, 90.00% at 42.5% retention, and 90.00% at 36.98% retention. These values are important checkpoints rather than final test accuracies and are reported with their source type to avoid metric conflation.

### C. Comparison Against Random Reduction

Class-wise random artifacts are available at 68%, 51%, 46.76%, 44.5%, and 42.5% retention, and global-random artifacts are available at 51%, 46.5%, and 37% retention. This makes random comparison possible without retraining-sample ambiguity. A complete random-versus-hybrid conclusion requires matched seed-wise runs for each retention ratio.

Existing K-means random-reduction checkpoints show 92.00% validation accuracy at 68% retention and 90.00% at 51% or 46.76% retention. The corresponding 42.5% random checkpoint reports 85.00% validation accuracy. The hybrid 68% result remains stronger at 93.57% test accuracy; however, this comparison is preliminary because it is based on different model/evaluation configurations rather than a matched-seed evaluation matrix.

The repository contains evidence that random deletion can reduce a model's unlearning-retain test accuracy in the deletion experiment. Nevertheless, the unlearning JSON files are not a direct substitute for retention ablation. They evaluate support deletion and retraining behavior, while retention ablations evaluate final training accuracy on an unchanged test set.

### D. Analysis of Reduction Components

The implementation separates four reduction mechanisms:

1. low-EL2N filtering;
2. high-EL2N filtering;
3. class-wise K-center Greedy selection; and
4. random class-wise or global selection.

The staged hybrid must be tested against low-only and high-only ablations. Low-only filtering measures whether easy-sample redundancy exists. High-only filtering measures whether difficult-sample exclusion improves robustness. K-center-only filtering measures geometric coverage independently of EL2N. EL2N-plus-K-center establishes whether the staged method outperforms either mechanism alone.

### E. Effect of Overfitting Mitigation

The mitigation suite includes independent configuration switches. A useful ablation order is:

1. weight decay;
2. label smoothing;
3. query MixUp/CutMix;
4. RandAugment and Random Erasing;
5. EMA;
6. SWA and BatchNorm recalibration;
7. early stopping and gradient clipping.

In the recorded 200-epoch full-dataset run, EMA plus RandAugment, Random Erasing, MixUp, weight decay, and label smoothing reached approximately 94.36% validation accuracy. In the 68%-retained run, the less-mitigated 300-epoch configuration reached approximately 93.27% validation accuracy and 93.57% test accuracy. These observations suggest that the complete method is capable of retaining most baseline accuracy under substantial reduction, but they do not isolate the contribution of each technique.

### F. Expanded Class Representations

The \(K=1\) representation is a class mean; larger \(K\) values preserve multiple class modes. K-means aggregation can improve representation quality when class clusters are visually or semantically separable, but it introduces prototype initialization, Lloyd iterations, and variable prototype counts for classes with fewer than \(K\) support observations.

The principal K-means bugs previously identified and corrected in this repository involve mixing support sets from multiple queries. Per-query support clustering is required. Any measured \(K=3\) or \(K=5\) gain should therefore be paired with a fixed per-query clustering implementation and a matching FAISS evaluation path.

### G. Deletion-Aware Training

Deletion-aware training exposes the model to full-support and reduced-support episodes. Let \(\mathcal L_f\) and \(\mathcal L_d\) be the losses under the two support sets. The total objective is

\[
\mathcal L_{\mathrm{DA}}
=\mathcal L_f+\lambda_d\mathcal L_d.
\]

The repository's paired deletion test reported a full-support validation average of 94.462890625% and a deleted-support average of 94.453125%, a gap of 0.009765625 percentage points. This indicates that the investigated deletion-aware configuration is robust to the specific 20% support deletion used in that test.

Deletion-aware loss is not equivalent to exact machine unlearning. It encourages support robustness, while exact support removal is achieved by modifying or excluding support rows from the test-time support set.

### H. Training and Computational Efficiency

Dataset reduction from 50,000 to 34,000 observations lowers theoretical training observation count by 32%. The one-time cost of pilot training, EL2N scoring, feature extraction, and K-center selection is not eliminated. The benefit is therefore realized across repeated final-training experiments or when retraining frequency is high.

For a fixed support size, support-sampling computation can remain substantial because the support set is sampled independently of the query batch. Training-time savings should therefore be measured end to end, including dataloaders, support sampling, prototype aggregation, and evaluation.

### I. Statistical Analysis

The current repository evidence is insufficient for a statistically significant statement about random and hybrid reductions. A complete experiment should use at least five seeds per method and retention level, and report:

\[
\bar A=\frac{1}{n}\sum_{j=1}^{n}A_j,
\qquad
s=\sqrt{\frac{1}{n-1}\sum_{j=1}^{n}(A_j-\bar A)^2}.
\]

Paired differences should be tested across the same random seeds:

\[
\bar d=\frac{1}{n}\sum_j(A_j^{\mathrm{hybrid}}-A_j^{\mathrm{random}}).
\]

Confidence intervals and a paired significance test can then separate seed variance from genuine method differences.

## IX. Discussion

### A. Effectiveness of the Proposed Reduction Strategy

The staged method is effective only if EL2N filtering identifies meaningful difficulty and K-center selection preserves coverage after filtering. The 32% reduction experiment provides evidence of strong retention: 93.57% accuracy at 68% of the training set. The difference from the 95.18% full-dataset reference is 1.61 percentage points. This is a larger degradation than the 0.73 percentage-point goal suggested in the report format, so the smaller target should be treated as a future threshold rather than an established result.

### B. Robustness Against Random Reduction

Random reduction is most likely to fail when class coverage or difficult-sample diversity is not preserved. Class-wise random selection removes the most obvious class imbalance, but it remains seed dependent. The proposed pipeline is intended to reduce this seed sensitivity by explicitly matching feature-space coverage.

### C. Trade-offs Between Difficulty and Diversity

Retaining low-difficulty observations can improve calibration on easy examples but provide limited coverage. Retaining high-difficulty observations can preserve difficult boundaries but include mislabeled or unrepresentative outliers. K-center prevents excessive concentration on a single difficulty extreme by optimizing coverage in feature space.

### D. Limitations

The framework has several limitations:

- pilot quality determines EL2N and feature embeddings;
- K-center cost scales with class size and selected count;
- EL2N percentages require additional hyperparameter tuning;
- CIFAR-10 evidence does not establish generalization to CIFAR-100 or ImageNet;
- FAISS and support-evaluation settings must match across runs;
- the current mitigation suite has not been ablated per technique in a multi-seed study;
- deletion-aware training is not an exact unlearning guarantee; and
- validation support sampling can introduce episode variance unless support evaluation is fixed.

### E. Practical Implications

The framework is most useful in a retraining-heavy environment: repeated ablations, deletion experiments, hyperparameter sweeps, or class-deletion baselines. It can reduce the cost of each final run while preserving a reproducible subset. It is also well suited to data auditing, because the JSON artifact records exactly which observations are retained.

## X. Conclusion and Future Work

### A. Conclusion

This work presented a hybrid dataset-reduction framework for semi-parametric classification. The method combines EL2N difficulty filtering and class-wise K-center Greedy coverage, retains pilot scores and selected indices as a reproducible artifact, and retrains the final SPM from scratch. It also integrates expanded prototypes, deletion-aware episodes, and configurable overfitting mitigation.

The repository evidence demonstrates that 68% retention can retain approximately 93.57% accuracy, while a full-dataset reference achieved 95.18%, giving 1.61 percentage points of observed degradation. At the same time, the available evidence is preliminary and not multi-seed. The claimed goal of approximately 0.73 percentage-point degradation at 32% reduction remains a target that should be confirmed with matched seeds and identical evaluation pathways.

The strongest contribution is not a single data-pruning threshold. It is the reproducible framework that connects difficulty, coverage, support-conditioned model semantics, overfitting mitigation, deletion robustness, and data-efficiency analysis.

### B. Future Work

Future work will include:

- repeated-seed comparison of hybrid, random, Herding, and Moderate Coreset methods;
- evaluation on CIFAR-100 and Tiny ImageNet;
- adaptive EL2N exclusion ratios;
- joint difficulty-diversity optimization;
- scalable K-center variants;
- fixed-support validation to reduce episode variance;
- per-query K-means ablations for \(K=1,3,5\);
- deletion-aware training ablations;
- independent mitigation ablations;
- confidence intervals and paired statistical tests;
- cross-dataset pilot transfer experiments; and
- integration of online or curriculum-based selection.

## References

[1] A. Y. Zheng, Y.-S. Tai, and R. A. Yeh, “Designing to Forget: Deep Semi-Parametric Models for Unlearning,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition*, 2026.

[2] M. Paul, S. Ganguli, and G. K. Dziugaite, “Deep Learning on a Data Diet: Speeding Up Training via Importance Sampling,” in *Proc. Advances in Neural Information Processing Systems*, 2021.

[3] O. Sener and S. Savarese, “Active Learning for Convolutional Neural Networks: A Core-Set Approach,” in *Proc. Int. Conf. Learning Representations*, 2018.

[4] C. W. F. Bachem, M. Lucic, and A. Krause, “Practical Coreset Constructions for Machine Learning,” arXiv preprint arXiv:1703.06476, 2017.

[5] M. Welling, “Herding Dynamical Weights to Learn,” in *Proc. 26th Int. Conf. Machine Learning*, 2009.

[6] A. Xia, W. Liu, S. Wang, X. Yu, and D. Tao, “Moderate-Margin Coresets for Efficient Deep Learning,” in *Proc. Int. Conf. Learning Representations*, 2023.

[7] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “MixUp: Beyond Empirical Risk Minimization,” in *Proc. Int. Conf. Learning Representations*, 2018.

[8] S. Yun, D. Han, S. J. Chun, S. Choe, and Y. Yoo, “CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features,” in *Proc. IEEE/CVF Int. Conf. Computer Vision*, 2019, pp. 6023–6032.

[9] E. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le, “RandAugment: Practical Automated Data Augmentation with a Reduced Search Space,” in *Proc. Advances in Neural Information Processing Systems*, 2020.

[10] D. Hendrycks, N. Mu, E. D. Cubuk, B. Zoph, J. Gilmer, and B. Lakshminarayanan, “AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty,” in *Proc. Int. Conf. Learning Representations*, 2020.

[11] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson, “Averaging Weights Leads to Wider Optima and Better Generalization,” in *Proc. Conf. Uncertainty in Artificial Intelligence*, 2018.

[12] A. Tarvainen and H. Valpola, “Mean Teachers Are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results,” in *Proc. Advances in Neural Information Processing Systems*, 2017.

[13] B. T. Polyak and A. B. Juditsky, “Acceleration of Stochastic Approximation by Averaging,” *SIAM Journal on Control and Optimization*, vol. 30, no. 4, pp. 838–855, 1992.

[14] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Resididual Learning for Image Recognition,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition*, 2016, pp. 770–778.

[15] A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” 2009.

[16] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” *Journal of Machine Learning Research*, vol. 15, pp. 1929–1958, 2014.

[17] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the Inception Architecture for Computer Vision,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition*, 2016.

[18] I. Loshchilov and F. Hutter, “SGDR: Stochastic Gradient Descent with Warm Restarts,” in *Proc. Int. Conf. Learning Representations*, 2017.

[19] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A Large-Scale Hierarchical Image Database,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition*, 2009.

[20] A. Wei, P. Cao, and C. Zhang, “Forgetting and Dataset Pruning for Machine Learning,” survey literature; final bibliographic verification recommended before publication.

## Appendix A: Dataset Reduction Pseudocode

```text
Input: training dataset D, pilot model M, low exclusion r_l,
       high exclusion r_h, class retention r_k, seed
Output: selected index artifact

1. Train or load M on D.
2. For each eligible x_i in D:
   2.1. Compute prediction p_theta(x_i).
   2.2. Compute EL2N score ||p_theta(x_i) - e_y||_2.
   2.3. Extract encoder embedding z_i = f_phi(x_i).
3. Partition eligible indices by class.
4. For each class c:
   4.1. Rank indices by EL2N score.
   4.2. Remove r_l of the lowest-score observations.
   4.3. Remove r_h of the highest-score observations.
   4.4. Apply class-wise K-center Greedy to embeddings.
   4.5. Retain r_k of the filtered class population.
5. Merge class selections and validate class counts.
6. Write selected indices and metadata to a versioned JSON artifact.
```

## Appendix B: Hyperparameter and Reproducibility Information

For the recorded CIFAR-10 experiments, the primary support sizes are 256 for training and 4096 for evaluation. The mitigation example uses learning-rate 0.1, weight decay 0.0005 or 0.001, two RandAugment operations, magnitude 7 or 9, Random Erasing probability 0.15 or 0.25, MixUp alpha 0.1 or 0.2, label smoothing 0.05 or 0.1, EMA decay 0.999, and seed 7. Maximum epochs vary by run and are stored in the mitigation configuration and checkpoint suffix.

Selection artifacts identify the pilot config, pilot checkpoint, seed, low/high EL2N exclusions, K-center retention, retained counts, original dataset size, and selected indices. The final model should be created without pilot-weight transfer.

## Appendix C: Experimental Reporting Template

| Method | Retention | Seed | Accuracy | Standard Deviation | Degradation |
|---|---:|---:|---:|---:|---:|
| Full dataset | 100% | — | 95.18% test | — | 0.00 pp |
| EL2N + K-center | 68% | 7 | 93.57% test | — | 1.61 pp |
| EL2N + K-center | 51% | 7 | 92.00% checkpoint validation | — | — |
| EL2N + K-center | 42.5% | 7 | 90.00% checkpoint validation | — | — |
| EL2N + K-center | 36.98% | 7 | 90.00% checkpoint validation | — | — |
| Random class-wise | 68% | 7 | 92.00% checkpoint validation | — | — |
| Random class-wise | 42.5% | 7 | 85.00% checkpoint validation | — | — |

Table values marked TBD require identical checkpoints and evaluation settings. The 0.73 percentage-point degradation reported in the document format should be retained only as a proposed target until a matched paired experiment confirms it.

## Appendix D: Implementation and Reproducibility Notes

The primary implementation files include the episode dataloaders, SPM model classes, EL2N/K-center selector, random selector, Mitigation trainer, checkpoint metadata callback, live curve callback, and evaluation scripts. Full-dataset and reduced-dataset artifacts are named separately. The current evaluation pathways differ in support processing and support set construction, so only metrics generated with identical query, support, and model settings should be compared.
