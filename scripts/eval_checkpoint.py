import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # add repo root
import argparse
import json
import os
import torch
from utils.config import load_config, instantiate_from_config



def delete_support_episode(support_imgs, support_labels, deletion_fraction=0.2):
    N = support_labels.size(0)
    num_delete = int(N * deletion_fraction)
    num_delete = max(1, num_delete)
    num_delete = min(num_delete, N - 1)

    deletion_candidates = []
    for lab in torch.unique(support_labels):
        class_indices = (support_labels == lab).nonzero(as_tuple=True)[0].tolist()
        if len(class_indices) > 1:
            deletion_candidates.extend(class_indices[:-1])
    if len(deletion_candidates) == 0:
        mask = torch.ones(N, dtype=torch.bool)
        return support_imgs[mask], support_labels[mask]

    import numpy as np
    deletion_candidates = np.array(deletion_candidates)
    if len(deletion_candidates) >= num_delete:
        delete_indices = np.random.choice(deletion_candidates, num_delete, replace=False)
    else:
        delete_indices = deletion_candidates
    mask = torch.ones(N, dtype=torch.bool)
    mask[list(delete_indices)] = False
    return support_imgs[mask], support_labels[mask]


def evaluate(config_path, ckpt_path, seed=0, max_batches=200, deletion_fraction=None, out_dir=None):
    torch.manual_seed(seed)
    cfg = load_config(config_path)
    model = instantiate_from_config(cfg['model'])
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            sd = {k.replace('model.', ''): v for k, v in state.items()}
            model.load_state_dict(sd, strict=False)
    datamodule = instantiate_from_config(cfg['data'])
    datamodule.setup('fit')
    val_loader = datamodule.val_dataloader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    acc_full_total = 0.0
    acc_del_total = 0.0
    n = 0
    for batch in val_loader:
        q_imgs, supp_imgs, q_labels, supp_labels = batch
        q_imgs = q_imgs.to(device)
        supp_imgs = supp_imgs.to(device)
        q_labels = q_labels.to(device)
        supp_labels = supp_labels.to(device)

        with torch.no_grad():
            logits_full = model(q_imgs, supp_imgs, supp_labels)
            preds_full = torch.argmax(logits_full, dim=-1)
            acc_full = (preds_full == q_labels).float().mean().item()

            frac = deletion_fraction if deletion_fraction is not None else getattr(datamodule, 'deletion_fraction', 0.2)
            del_imgs, del_labels = delete_support_episode(supp_imgs, supp_labels, deletion_fraction=frac)
            if del_imgs.size(0) == 0:
                continue
            logits_del = model(q_imgs, del_imgs, del_labels)
            preds_del = torch.argmax(logits_del, dim=-1)
            acc_del = (preds_del == q_labels).float().mean().item()

        acc_full_total += acc_full
        acc_del_total += acc_del
        n += 1
        if n >= max_batches:
            break

    results = {
        'ckpt': ckpt_path,
        'config': config_path,
        'val_acc_full_avg': acc_full_total / n if n>0 else None,
        'val_acc_deleted_avg': acc_del_total / n if n>0 else None,
        'val_deleted_gap': (acc_full_total - acc_del_total) / n if n>0 else None,
        'n_batches': n,
    }

    if out_dir is None:
        out_dir = os.path.dirname(ckpt_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'eval_{os.path.basename(ckpt_path)}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print('Saved evaluation to', out_path)
    print(results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_batches', type=int, default=200)
    parser.add_argument('--deletion_fraction', type=float, default=None)
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()
    evaluate(args.config, args.ckpt, seed=args.seed, max_batches=args.max_batches, deletion_fraction=args.deletion_fraction, out_dir=args.out_dir)
