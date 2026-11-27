"""Paired evaluation of two checkpoints under identical support deletions.

Usage:
  PYTHONPATH=. python scripts/eval_checkpoint_paired.py \
      --config config/classification/cifar_resnet18_deletion.yaml \
      --ckpt_a results/classification/cifar_resnet18_deletion/last.ckpt \
      --ckpt_b results/classification/cifar_resnet18/last.ckpt \
      --out_dir results/eval_paired

Produces JSON with per-p deletion fractions and metrics for both models.
"""
import os
import json
import argparse
import torch
import numpy as np
from utils.config import load_config, instantiate_from_config


def build_deletion_candidates(support_labels):
    # support_labels: 1D tensor of labels
    deletion_candidates = []
    unique = torch.unique(support_labels)
    for lab in unique:
        idxs = (support_labels == lab).nonzero(as_tuple=True)[0].tolist()
        if len(idxs) > 1:
            deletion_candidates.extend(idxs[:-1])
    return np.array(deletion_candidates, dtype=int)


def evaluate_paired(config_path, ckpt_a, ckpt_b, deletion_ps=None, max_batches=200, seed=0, out_dir=None):
    if deletion_ps is None:
        deletion_ps = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

    cfg = load_config(config_path)
    model_a = instantiate_from_config(cfg['model'])
    model_b = instantiate_from_config(cfg['model'])

    # load checkpoints
    def load_ckpt(model, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            sd = {k.replace('model.', ''): v for k, v in state.items()}
            model.load_state_dict(sd, strict=False)

    load_ckpt(model_a, ckpt_a)
    load_ckpt(model_b, ckpt_b)

    datamodule = instantiate_from_config(cfg['data'])
    datamodule.setup('fit')
    val_loader = datamodule.val_dataloader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_a.to(device).eval()
    model_b.to(device).eval()

    # accumulators
    # stats will store per-batch records for later aggregation and error bars
    stats = {p: [] for p in deletion_ps}

    rng_master = np.random.RandomState(seed)
    batch_idx = 0
    for batch in val_loader:
        q_imgs, supp_imgs, q_labels, supp_labels = batch
        q_imgs = q_imgs.to(device)
        supp_imgs = supp_imgs.to(device)
        q_labels = q_labels.to(device)
        supp_labels = supp_labels.to(device)

        with torch.no_grad():
            logits_a_full = model_a(q_imgs, supp_imgs, supp_labels)
            logits_b_full = model_b(q_imgs, supp_imgs, supp_labels)
            preds_a_full = torch.argmax(logits_a_full, dim=-1)
            preds_b_full = torch.argmax(logits_b_full, dim=-1)
            Afull_a = (preds_a_full == q_labels).float().mean().item()
            Afull_b = (preds_b_full == q_labels).float().mean().item()

        # For reproducibility and nested deletions, create a RNG seeded by master seed + batch_idx
        rng = np.random.RandomState(seed + batch_idx)

        # Build eligible deletion candidates and a single permutation for nesting
        deletion_candidates = build_deletion_candidates(supp_labels.cpu())
        if deletion_candidates.size > 0:
            perm = rng.permutation(deletion_candidates)
        else:
            perm = np.array([], dtype=int)

        N = supp_labels.size(0)
        for p in deletion_ps:
            k = int(np.floor(p * N))
            k = max(0, min(k, N - 1))
            if k == 0 or perm.size == 0:
                del_idx = np.array([], dtype=int)
            else:
                # nested selection: first k items from perm (cap at perm.size)
                take = min(k, perm.size)
                del_idx = perm[:take]

            # build mask: remove del_idx positions
            mask = torch.ones(N, dtype=torch.bool)
            if del_idx.size > 0:
                mask[list(del_idx)] = False
            del_imgs = supp_imgs[mask].to(device)
            del_labels = supp_labels[mask].to(device)

            with torch.no_grad():
                la = model_a(q_imgs, del_imgs, del_labels)
                lb = model_b(q_imgs, del_imgs, del_labels)
                pa = torch.argmax(la, dim=-1)
                pb = torch.argmax(lb, dim=-1)
                A_a = (pa == q_labels).float().mean().item()
                A_b = (pb == q_labels).float().mean().item()

            # per-model baselines are their own full-support accuracies
            delta_a = Afull_a - A_a
            delta_b = Afull_b - A_b
            rel_a = (delta_a / Afull_a) if Afull_a > 0 else None
            rel_b = (delta_b / Afull_b) if Afull_b > 0 else None
            delta_robust = delta_b - delta_a

            # record per-batch result for this p
            stats[p].append({
                'batch_idx': batch_idx,
                'Afull_a': Afull_a,
                'A_p_a': A_a,
                'Delta_a': delta_a,
                'Rel_a': rel_a,
                'Afull_b': Afull_b,
                'A_p_b': A_b,
                'Delta_b': delta_b,
                'Rel_b': rel_b,
                'Delta_robust': delta_robust,
                'n_support': int(N),
                'n_deleted': int(del_idx.size),
                'del_idx': del_idx.tolist(),
            })

        batch_idx += 1
        if batch_idx >= max_batches:
            break

    # Aggregate per-p into summary statistics
    results = {'ckpt_a': ckpt_a, 'ckpt_b': ckpt_b, 'deletion_ps': deletion_ps, 'n_batches': batch_idx}
    results['per_batch'] = {str(p): stats[p] for p in deletion_ps}
    summary = []
    for p in deletion_ps:
        records = stats[p]
        if len(records) == 0:
            continue
        AfullA_vals = [r['Afull_a'] for r in records]
        A_p_A_vals = [r['A_p_a'] for r in records]
        DeltaA_vals = [r['Delta_a'] for r in records]

        AfullB_vals = [r['Afull_b'] for r in records]
        A_p_B_vals = [r['A_p_b'] for r in records]
        DeltaB_vals = [r['Delta_b'] for r in records]

        rec = {
            'p': p,
            'AfullA_mean': float(np.mean(AfullA_vals)),
            'AfullA_std': float(np.std(AfullA_vals, ddof=1)) if len(AfullA_vals) > 1 else 0.0,
            'A_p_A_mean': float(np.mean(A_p_A_vals)),
            'A_p_A_std': float(np.std(A_p_A_vals, ddof=1)) if len(A_p_A_vals) > 1 else 0.0,
            'DeltaA_mean': float(np.mean(DeltaA_vals)),
            'DeltaA_std': float(np.std(DeltaA_vals, ddof=1)) if len(DeltaA_vals) > 1 else 0.0,
            'AfullB_mean': float(np.mean(AfullB_vals)),
            'AfullB_std': float(np.std(AfullB_vals, ddof=1)) if len(AfullB_vals) > 1 else 0.0,
            'A_p_B_mean': float(np.mean(A_p_B_vals)),
            'A_p_B_std': float(np.std(A_p_B_vals, ddof=1)) if len(A_p_B_vals) > 1 else 0.0,
            'DeltaB_mean': float(np.mean(DeltaB_vals)),
            'DeltaB_std': float(np.std(DeltaB_vals, ddof=1)) if len(DeltaB_vals) > 1 else 0.0,
            'Delta_robust_mean': float(np.mean([r['Delta_robust'] for r in records])),
            'Delta_robust_std': float(np.std([r['Delta_robust'] for r in records], ddof=1)) if len(records) > 1 else 0.0,
        }
        summary.append(rec)

    results['summary'] = summary

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'paired_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print('Saved paired evaluation to', out_path)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt_a', required=True)
    parser.add_argument('--ckpt_b', required=True)
    parser.add_argument('--out_dir', default='results/eval_paired')
    parser.add_argument('--max_batches', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    evaluate_paired(args.config, args.ckpt_a, args.ckpt_b, max_batches=args.max_batches, seed=args.seed, out_dir=args.out_dir)
