Deletion-aware training & evaluation
=================================

Overview
--------
This project contains an alternate deletion-aware semi-parametric model (SPM) training and evaluation pipeline. The deletion-aware model includes an auxiliary loss term computed on support sets with a fraction of samples removed, encouraging robustness to support deletion.

Files of interest
-----------------
- `models/spm_classifier_deletion_alt.py`: Alternate `SemiParametricClassifierAlt` that adds `loss_deleted` and logs `train_acc_deleted`.
- `dataset/cifar_datamodule_deletion_alt.py`: DataModule that returns 6-tuple training batches including deleted supports.
- `utils/dataset_deletion.py`: Collate functions that construct support episodes and deleted-support variants.
- `main_deletion_run.py`: Training+test runner that logs metrics and runs a post-training robustness evaluation.
- `scripts/eval_checkpoint_paired.py`: Paired evaluation script comparing two checkpoints across deletion fractions.
- `scripts/eval_checkpoint.py`: Single-checkpoint evaluation (kept for convenience).
- `evaluation/deletion_robustness_analysis.ipynb`: Notebook to run paired evaluation and visualize results.

How evaluation works (paired)
----------------------------
1. Use the same validation/query batches and support episodes for both models.
2. For each support episode and deletion fraction `p`, sample `k = floor(p * N)` support samples to delete, excluding the last instance of each class to ensure at least one representative remains.
3. Apply the same deletion indices (same seed) to both models and compute accuracy:
   - `A_full`: accuracy with the full support
   - `A_deleted(p)`: accuracy after deletion
4. Aggregate across validation batches to compute average `A_full`, `A_deleted(p)`, `ΔA(p)=A_full−A_deleted(p)`, and `R(p)=ΔA(p)/A_full`.

Unlearning evaluation with forget/retain metrics
-------------------------------------------------
The new forget/retain evaluation is implemented by `evaluate_unlearn()` in
`evaluation/comp_classification_metrics.py`. It uses the support loader after
the forget samples have been removed and evaluates four disjoint splits for
class-based deletion:

- `train_unlearn`: forget samples that were present in the original training set.
- `train_rest`: retained training samples used to build the support set.
- `test_unlearn`: held-out test samples from the forgotten class or classes.
- `test_rest`: held-out test samples from the retained classes.

For the original classifier, run one forgotten class (replace `0` with the
desired class list):

```bash
PYTHONPATH=. python evaluate.py \
  --model_type classifier \
  --config config/classification/cifar_resnet18.yaml \
  --ckpt_path results/classification/cifar_resnet18/last.ckpt \
  --dataset_name cifar10 \
  --query_digits 0 \
  --support_digits 1,2,3,4,5,6,7,8,9 \
  --support_size_eval 256 \
  --batch_size 256 \
  --out_file results/classification/cifar_resnet18/unlearn_class_0 \
  --seed 0
```

To also train or load the retain-only reference model and report prediction
agreement/KL divergence, add `--retrain --acc_v2`. The retrain path writes the
same original metrics plus the new metrics to the JSON output:

```bash
PYTHONPATH=. python evaluate.py \
  --model_type classifier \
  --config config/classification/cifar_resnet18.yaml \
  --ckpt_path results/classification/cifar_resnet18/last.ckpt \
  --dataset_name cifar10 \
  --query_digits 0 \
  --support_digits 1,2,3,4,5,6,7,8,9 \
  --support_size_eval 256 \
  --batch_size 256 \
  --out_file results/classification/cifar_resnet18/retrain_class_0 \
  --retrain --acc_v2 --seed 0
```

The retrain path reuses a previously saved reference checkpoint for the same
forget-set output directory when one exists. Add `--force_retrain` to the
command above to ignore that cached checkpoint and train a fresh reference model.

The output JSON contains the following new metrics:

- `train_acc_unlearn`, `test_acc_unlearn`: classification accuracy on the
  forgotten training and held-out test samples. Higher test accuracy on the
  forgotten class is not evidence of forgetting by itself; compare it with
  the retrained reference and the other metrics.
- `confidence_svc`, `prob_svc`: SVC membership-inference accuracy using,
  respectively, the true-label confidence and the complete output
  probability vector. The retain train/test splits are the shadow population;
  forget train/test splits are the target population for class deletion, while
  sample-level deletion has only a forgotten training population.
- `test_unlearn_prediction_agreement`, `test_rest_prediction_agreement`:
  fraction of test predictions matching the retain-only retrained reference,
  separately for forget and retain classes.
- `test_unlearn_kl`, `test_rest_kl`: KL divergence from the retrained
  reference probabilities, separately for forget and retain test samples.
  Lower KL and higher agreement indicate closer behavior to retraining.

For sample-level deletion (`--unlearn_portion` or `--indexes_to_replace`),
there is no held-out test set corresponding to the deleted training examples.
Accordingly, `test_acc_unlearn`, `test_unlearn_prediction_agreement`, and
`test_unlearn_kl` are reported as null; the test-rest metrics cover the full
test set.

The deletion-aware model is selectable by `evaluate.py` when its deletion
configuration is supplied. It now exposes the same raw support and split
loaders as the base datamodule, so it can produce the same four-split metrics:

```bash
PYTHONPATH=. python evaluate.py \
  --model_type classifier \
  --config config/classification/cifar_resnet18_deletion.yaml \
  --ckpt_path results/classification/cifar_resnet18_deletion/last.ckpt \
  --dataset_name cifar10 \
  --query_digits 0 \
  --support_digits 1,2,3,4,5,6,7,8,9 \
  --support_size_eval 256 \
  --batch_size 256 \
  --out_file results/classification/cifar_resnet18_deletion/unlearn_class_0 \
  --seed 0
```

Add `--retrain --acc_v2` to create a retain-only reference, or add
`--retrained_path PATH --retrain --acc_v2` to use an existing compatible
deletion-aware retrained checkpoint without training again. For support-
deletion robustness, use the separate procedure below:

```bash
PYTHONPATH=. python scripts/eval_checkpoint.py \
  --config config/classification/cifar_resnet18_deletion.yaml \
  --ckpt results/classification/cifar_resnet18_deletion/last.ckpt \
  --deletion_fraction 0.2 \
  --max_batches 200 \
  --seed 0 \
  --out_dir results/classification/cifar_resnet18_deletion
```

For a direct comparison with the original model under identical support
deletions, use `scripts/eval_checkpoint_paired.py` as shown below. Its
`A_full`, `A_deleted(p)`, `Delta`, and `Delta_robust` metrics measure support
deletion robustness, which is different from forget-set unlearning accuracy
and SVC membership inference.

Quick commands
--------------
Activate conda env:
```bash
conda activate spm
```

Train deletion-aware model (example):
```bash
python main_deletion_run.py --config config/classification/cifar_resnet18_deletion.yaml --seed 0
```

Evaluate paired checkpoints:
```bash
PYTHONPATH=. python scripts/eval_checkpoint_paired.py \
  --config config/classification/cifar_resnet18_deletion.yaml \
  --ckpt_a results/classification/cifar_resnet18_deletion/last.ckpt \
  --ckpt_b results/classification/cifar_resnet18/last.ckpt \
  --out_dir results/eval_paired
```

Open the notebook for visualization:
```bash
jupyter notebook evaluation/deletion_robustness_analysis.ipynb
```

Notes
-----
- Ensure checkpoints and configs match (data normalization, support sizes). The scripts try to load `state_dict` flexibly but prefer matching architectures.
- Use `PYTHONPATH=.` when running scripts if you encounter `ModuleNotFoundError: No module named 'utils'`.
