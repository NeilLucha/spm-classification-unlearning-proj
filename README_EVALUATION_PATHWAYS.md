# Checkpoint Evaluation Pathways

This document describes how trained checkpoints are evaluated in this repository, which evaluation entry points exist, and how retrained-reference comparisons are produced.

## Overview

There are three main evaluation families:

1. **Lightning `test_step` evaluation** inside the model.
2. **General classifier evaluation** through `evaluate.py` and `evaluation/comp_classification_metrics.py`.
3. **Paired/deletion evaluation** through `scripts/eval_checkpoint.py` and `scripts/eval_checkpoint_paired.py`.

These pathways do not all use the same support representation. Their results should therefore only be compared when their evaluation settings match.

---

## 1. Training and Validation Episodes

The standard CIFAR data module is configured in files such as:

```text
config/classification/cifar_resnet18_kmeans_k1.yaml
```

During fitting:

```text
query dataset + training support dataset
                |
                v
QuerySupportTrainDataset
                |
                v
query_support_train_collate_fn
                |
                +--> query images and labels
                |
                +--> sampled support images and labels
                v
model.forward(query, support, support_labels)
                |
                v
encoder -> support aggregation -> MoE -> log probabilities
                |
                v
NLL loss, accuracy, optimizer update
```

The standard training collate function samples support examples by the labels present in the current query batch. It also applies a random permutation to the class labels for the episode.

For K-means models, `model.forward()` performs:

```text
support images
    -> encoder embeddings
    -> Feature_Aggregation_KMeans
    -> K prototypes per class
    -> MoE
    -> log probabilities
```

For `K=1`, aggregation uses the mean embedding for each class.

Validation uses `QuerySupportTestDataset` and `query_support_test_collate_fn`. The query set is the test split, while the support set is sampled from the training split. The current standard collate function samples support examples randomly each time a validation batch is constructed.

This means validation accuracy is support-set dependent. Unless a fixed support set is introduced, two validation epochs can evaluate the same checkpoint with different support images.

---

## 2. Lightning `test_step` Path

The K-means model contains a specialized `test_step` in:

```text
models/spm_classifier_kmeans.py
```

Its intended data flow is:

```text
Test query batch
    -> query encoder embeddings
    -> FAISS search over training embeddings
    -> per-query kNN support set
    -> K-means aggregation separately for each query
    -> MoE prediction
    -> test loss and accuracy
```

For each query `q_i`:

```text
q_i
  -> FAISS retrieves support S_i
  -> K-means clusters S_i by class
  -> MoE(q_i, prototypes_i)
  -> prediction_i
```

The support set is clustered independently for each query. Support sets from different queries must not be flattened together.

The model caches the FAISS index and training embeddings during the first test batch. The embeddings are either:

- loaded from `model.embedding_file`, or
- computed by running the encoder over `trainer.datamodule.train_dataloader()`.

This path is the most faithful evaluation path for the current K-means implementation because it uses per-query retrieval followed by per-query K-means aggregation.

### Important caveat

The regular `SemiParametricClassifier` has a different `test_step`: it retrieves kNN embeddings and feeds the flattened support embeddings directly to the MoE. It does not use the K-means aggregation path.

Therefore, a K-means checkpoint and an original SPM checkpoint evaluated through Lightning `test_step` are not necessarily using identical support processing.

---

## 3. `evaluate.py` Entry Point

The main command-line entry point is:

```text
evaluate.py
```

It loads a checkpoint using:

```python
ModelClass.load_from_checkpoint(args.ckpt_path, strict=True)
```

If `--retrained_path` is provided, a second checkpoint is loaded using the same model class.

The data module is then created from the config, with several command-line arguments overriding the config:

- `batch_size`
- `support_size_train`
- `support_size_eval`
- `num_classes`
- `dataset_name`
- `model_type`
- query class selection
- support class selection

For classifier evaluation, the selected function depends on the flags:

| Condition | Function | Purpose |
|---|---|---|
| `--retrain` | `evaluate_retrain` | Compare the checkpoint with a retrained reference and optionally perform unlearning analysis |
| query/unlearning flags | `evaluate_unlearn` | Evaluate forget/retain train and test splits |
| otherwise | `evaluate_all_test_data` | Evaluate all test samples |

Results are printed and optionally saved as JSON under the requested output directory. When `--knn` is enabled, output is placed under a `knn` subdirectory.

Example general evaluation command:

```bash
python evaluate.py \
  --model_type classifier \
  --config config/classification/cifar_resnet18_kmeans_k1.yaml \
  --ckpt_path results/classification/cifar_resnet18_kmeans_k1/<checkpoint>.ckpt \
  --dataset_name cifar10 \
  --num_classes 10 \
  --support_size_eval 4096 \
  --out_file results/eval_k1.json
```

---

## 4. `evaluate_all_test_data`

Implementation:

```text
evaluation/comp_classification_metrics.py
```

This function evaluates every sample from `datamodule.test_dataloader()`.

### Intended shared-support MoE path

The support loader is used to construct a single shared support representation. Query batches then use:

```text
all support images
    -> encoder embeddings
    -> support representation
query images
    -> encoder embeddings
query embeddings + shared support representation
    -> model.moe
    -> predictions
```

### `--knn` path

When `--knn` is enabled, the function retrieves nearest support embeddings and creates class probabilities by counting labels:

```text
query embedding
    -> nearest support embeddings
    -> class label counts
    -> normalized class probabilities
```

This is a retrieval/counting baseline, not the learned MoE path.

When `--knn_c` is enabled, the implementation uses a nearest-support operation rather than the normal k-neighbor search.

### Cluster portion

`cluster_portion < 1.0` samples a portion of the support set and computes class mean embeddings. This is a class-mean support representation, not K-means multi-prototype aggregation.

### Comparison warning

This helper is not equivalent to the K-means model's Lightning `test_step`:

- It builds one shared support representation for the test run.
- It may use class means.
- `--knn` bypasses the MoE and uses label counts.
- It does not perform per-query K-means clustering.

Use this path for legacy/general metrics only when that distinction is intentional.

---

## 5. `get_predict`

`get_predict` is used by retraining and unlearning evaluation.

It builds a shared support representation by averaging encoder embeddings for each support class:

```text
support images
    -> encoder
    -> running mean per class
    -> one support embedding per class
```

Then every query batch is evaluated with:

```text
query encoder embeddings + shared class means
    -> model.moe
    -> predictions
```

Despite the name of the model, this function does not call the K-means aggregator to produce K prototypes per class. For a K-means checkpoint, it therefore evaluates a class-mean/shared-support pathway rather than the model's per-query K-means `test_step` pathway.

This distinction is important when interpreting `spm_model_acc` and retrained-model metrics produced by `evaluate_retrain`.

---

## 6. Unlearning Evaluation

`evaluate_unlearn` evaluates four splits:

```text
train_unlearn  = forgotten training samples
train_rest     = retained training samples
test_unlearn   = test samples from forgotten classes/selection
test_rest      = retained test samples
```

The function first builds support embeddings from `datamodule.support_loader()`. Depending on the data module and options, this support set may exclude forgotten samples.

For each split it computes:

```text
query images
    -> query encoder embeddings
    -> shared support embeddings
    -> MoE or optional retrieval path
    -> probabilities and predictions
```

Reported metrics include:

- `train_acc_unlearn`
- `train_acc_rest`
- `test_acc_unlearn`
- `test_acc_rest`
- inference time
- membership-inference metrics when all required splits exist
- prediction agreement and KL divergence against a reference model when supplied

The support representation is shared across the split evaluation. This is different from per-query K-means aggregation in the K-means model's `test_step`.

---

## 7. Retrained-Reference Evaluation

Retrained-reference evaluation is triggered with:

```bash
python evaluate.py ... --retrain
```

A separately trained reference model is intended to represent training on the retained data without the forgotten data.

### Reference model creation

`evaluate_retrain`:

1. Loads the supplied config.
2. Determines the classes/data to retain.
3. Modifies `query_digits` and `support_digits` for retain-only training.
4. Creates a new retain-only datamodule.
5. Either:
   - loads `--retrained_path`, or
   - trains a new reference model on the retain-only data.

The target checkpoint and reference checkpoint are then evaluated through `get_predict` and `evaluate_unlearn`.

### What is compared

When `--acc_v2` is enabled, the function compares target and reference predictions using:

- target SPM accuracy
- retrained-reference SPM accuracy
- prediction agreement
- cross-entropy-style comparison
- KL divergence

The reference model is treated as the behavioral comparison target, not as a perfect ground-truth classifier.

The unlearning evaluation additionally compares:

- forget-set accuracy
- retain-set accuracy
- prediction agreement on forget and retain test splits
- KL divergence on those splits
- membership-inference metrics where available

### Paired deletion evaluation

`scripts/eval_checkpoint_paired.py` evaluates two checkpoints on identical validation support batches and identical nested support deletions. For each deletion fraction `p`, it reports:

```text
A_full: accuracy with the complete support set
A_p:    accuracy after deleting support samples
Delta:  A_full - A_p
Rel:    Delta / A_full
```

The paired evaluator is useful because both models see the same queries and support deletions. It reduces variance when comparing a target checkpoint against a retrained or reference checkpoint.

Example:

```bash
PYTHONPATH=. python scripts/eval_checkpoint_paired.py \
  --config config/classification/cifar_resnet18_deletion.yaml \
  --ckpt_a path/to/model_a.ckpt \
  --ckpt_b path/to/model_b.ckpt \
  --out_dir results/eval_paired
```

---

## 8. Which Path Should Be Used?

### Normal K-means classification accuracy

Prefer the model's Lightning `test_step` path because it implements:

```text
query -> query-specific FAISS kNN -> query-specific K-means -> MoE
```

### Legacy/shared-support classification metrics

Use `evaluate.py` without unlearning flags when compatibility with existing result files is more important than exact K-means test-time behavior. Record whether `--knn` or `--knn_c` was used.

### Unlearning metrics

Use:

```text
evaluate.py --retrain
```

with `--retrained_path` when a reference checkpoint already exists. This produces retain/forget accuracy and behavioral comparison metrics.

### Fair checkpoint-to-checkpoint comparison

Use `scripts/eval_checkpoint_paired.py` when the two checkpoints should see exactly the same validation batches and support deletions.

---

## 9. Recommended Evaluation Record

Every evaluation result should record at least:

- checkpoint path
- config path
- dataset and number of classes
- K value
- loss mode, including label smoothing status
- support size
- evaluation entry point
- `--knn` / `--knn_c` flags
- query and support class filters
- random seed
- whether a retrained reference was used
- whether support data was fixed or randomly resampled

A result containing only `test_acc` is not sufficient to identify the evaluation pathway.

---

## 10. Key Caveats

1. **Validation support sampling is random** in the standard validation collate function unless a fixed support set is introduced.
2. **K-means Lightning test evaluation and `get_predict` are different.** The former uses per-query K-means; the latter uses shared class means.
3. **`--knn` is not learned MoE evaluation.** It counts retrieved support labels.
4. **Retrained-reference comparison depends on the reference data selection.** It is only meaningful when the retained/forgotten split is configured correctly.
5. **Checkpoint filenames identify training settings, not evaluation settings.** The evaluation command and flags must still be recorded.
6. **Different model classes have different `test_step` implementations.** Do not assume the original SPM and K-means SPM checkpoints use the same test-time path.

The safest practice is to report the metric together with the exact command, checkpoint, config, support size, seed, and evaluation pathway.
