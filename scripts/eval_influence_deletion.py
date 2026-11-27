"""Influence-ranked, paired support deletion evaluation.

The deletion-aware checkpoint supplies the query-specific influence ranking.
That ranking is then reused unchanged for the normally trained checkpoint so
the comparison is paired. Existing evaluation scripts are intentionally not
used or modified.
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.config import instantiate_from_config, load_config


DEFAULT_DELETION_PS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]


LOGGER = logging.getLogger("spm.influence_deletion")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def load_checkpoint(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError:
        stripped = {key.replace("model.", "", 1): value for key, value in state.items()}
        model.load_state_dict(stripped, strict=False)


def build_deletion_candidates(support_labels):
    """Return support positions that can be deleted without losing a class."""
    candidates = []
    for label in torch.unique(support_labels, sorted=True):
        positions = (support_labels == label).nonzero(as_tuple=True)[0].tolist()
        candidates.extend(positions[:-1])
    return np.asarray(candidates, dtype=np.int64)


def fused_from_embeddings(model, query_embeddings, support_embeddings, support_labels):
    """Return log predictions and final pre-log fusion from cached embeddings.

    This mirrors ``SemiParametricClassifier.forward`` after encoding and
    retains the output of attention experts after MoE gating, immediately
    before ``log``.
    """
    if support_embeddings.shape[0] != model.num_classes:
        support_embeddings, support_labels = model.feature_aggregation(
            support_embeddings, support_labels
        )

    expert_outputs = [
        expert(query_embeddings, support_embeddings, support_labels)
        for expert in model.moe.experts
    ]
    expert_outputs = torch.stack(expert_outputs, dim=1)
    gate = model.moe.gating(query_embeddings, support_embeddings).unsqueeze(-1)
    fused = torch.sum(gate * expert_outputs, dim=1)
    return torch.log(fused.clamp(min=1e-8)), fused


def fused_forward(model, query_imgs, support_imgs, support_labels):
    """Return log predictions and the final pre-log SPM fusion output."""
    return fused_from_embeddings(
        model,
        model.encoder(query_imgs),
        model.encoder(support_imgs),
        support_labels,
    )


def loo_fusions_from_embeddings(model, query_embedding, support_embeddings, support_labels, candidates):
    """Return exact leave-one-out fusions for all candidate support positions.

    The classifier first averages supports by label, then applies each expert
    and the gate to those class means. All candidates retain every class, so
    the class means can be computed together by subtracting the held-out
    embedding from the full per-class sums.
    """
    unique_labels, inverse = torch.unique(support_labels, sorted=True, return_inverse=True)
    n_candidates = candidates.numel()
    n_classes = unique_labels.numel()
    candidate_labels = inverse[candidates]

    class_sums = torch.zeros(n_classes, support_embeddings.shape[1], device=support_embeddings.device)
    class_sums.index_add_(0, inverse, support_embeddings)
    class_counts = torch.bincount(inverse, minlength=n_classes).to(support_embeddings.dtype)
    loo_sums = class_sums.unsqueeze(0).expand(n_candidates, -1, -1).clone()
    loo_sums[torch.arange(n_candidates, device=support_embeddings.device), candidate_labels] -= support_embeddings[candidates]
    loo_counts = class_counts.unsqueeze(0).expand(n_candidates, -1).clone()
    loo_counts[torch.arange(n_candidates, device=support_embeddings.device), candidate_labels] -= 1
    loo_supports = loo_sums / loo_counts.unsqueeze(-1)

    query_batch = query_embedding.expand(n_candidates, -1)
    class_to_output = F.one_hot(unique_labels, num_classes=model.num_classes).to(
        support_embeddings.dtype
    )
    expert_outputs = []
    for expert in model.moe.experts:
        query_transformed = expert.linear_q(query_batch)
        support_transformed = expert.linear_s(loo_supports)
        scores = torch.matmul(query_transformed.unsqueeze(1), support_transformed.transpose(1, 2)).squeeze(1)
        weights = torch.softmax(scores / (expert.hidden_dim ** 0.5), dim=1)
        expert_outputs.append(torch.matmul(weights, class_to_output))
    expert_outputs = torch.stack(expert_outputs, dim=1)

    support_means = loo_supports.mean(dim=1)
    gate_input = torch.cat([query_batch, support_means], dim=-1)
    gate = torch.softmax(model.moe.gating.fc(gate_input), dim=-1).unsqueeze(-1)
    return (gate * expert_outputs).sum(dim=1)


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "std": None, "stderr": None}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "stderr": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
    }


def evaluate_influence_paired(
    config_path,
    ckpt_deletion_aware,
    ckpt_normal,
    deletion_ps=None,
    max_batches=200,
    seed=0,
    out_dir=None,
):
    start_time = time.perf_counter()
    deletion_ps = DEFAULT_DELETION_PS if deletion_ps is None else deletion_ps
    LOGGER.info("Starting influence evaluation: max_batches=%s, deletion_ps=%s", max_batches, deletion_ps)
    cfg = load_config(config_path)
    deletion_model = instantiate_from_config(cfg["model"])
    normal_model = instantiate_from_config(cfg["model"])
    load_checkpoint(deletion_model, ckpt_deletion_aware)
    load_checkpoint(normal_model, ckpt_normal)
    LOGGER.info("Loaded checkpoints: deletion-aware=%s; normal=%s", ckpt_deletion_aware, ckpt_normal)

    datamodule = instantiate_from_config(cfg["data"])
    datamodule.setup("fit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deletion_model.to(device).eval()
    normal_model.to(device).eval()
    LOGGER.info("Using device: %s", device)

    per_query = {str(p): [] for p in deletion_ps}
    influence_by_query = []
    batch_count = 0
    query_count = 0
    generator = torch.Generator().manual_seed(seed)
    val_loader = datamodule.val_dataloader()
    total_batches = len(val_loader)
    target_batches = min(max_batches, total_batches)
    LOGGER.info("Validation loader ready: %s batches available; processing up to %s", total_batches, target_batches)

    with torch.no_grad():
        for batch in val_loader:
            query_imgs, support_imgs, query_labels, support_labels = batch[:4]
            batch_start = time.perf_counter()
            LOGGER.info(
                "Batch %s/%s started: %s queries, %s supports; leave-one-out ranking in progress",
                batch_count + 1,
                target_batches,
                query_imgs.shape[0],
                support_labels.numel(),
            )
            query_imgs = query_imgs.to(device)
            support_imgs = support_imgs.to(device)
            query_labels = query_labels.to(device)
            support_labels = support_labels.to(device)
            permutation = torch.randperm(support_labels.numel(), generator=generator)
            # Make support order deterministic without changing the sampled batch.
            support_imgs = support_imgs[permutation]
            support_labels = support_labels[permutation]
            deletion_query_embeddings = deletion_model.encoder(query_imgs)
            deletion_support_embeddings = deletion_model.encoder(support_imgs)
            normal_query_embeddings = normal_model.encoder(query_imgs)
            normal_support_embeddings = normal_model.encoder(support_imgs)
            candidates = build_deletion_candidates(support_labels.cpu())
            n_support = support_labels.numel()

            for query_index in range(query_imgs.shape[0]):
                query_del_embedding = deletion_query_embeddings[query_index : query_index + 1]
                query_norm_embedding = normal_query_embeddings[query_index : query_index + 1]
                full_del_logits, full_del_fusion = fused_from_embeddings(
                    deletion_model, query_del_embedding, deletion_support_embeddings, support_labels
                )
                full_norm_logits, full_norm_fusion = fused_from_embeddings(
                    normal_model, query_norm_embedding, normal_support_embeddings, support_labels
                )

                # Influence is defined from the deletion-aware model, making
                # one ranking available for the paired comparison.
                candidate_tensor = torch.as_tensor(candidates, device=device)
                loo_fusions = loo_fusions_from_embeddings(
                    deletion_model,
                    query_del_embedding,
                    deletion_support_embeddings,
                    support_labels,
                    candidate_tensor,
                )
                influence_values = torch.linalg.vector_norm(
                    full_del_fusion - loo_fusions, dim=1
                ).tolist()
                influences = list(zip(influence_values, candidates.tolist()))
                ranking = [index for _, index in sorted(influences, key=lambda item: (-item[0], item[1]))]
                influence_by_query.append({
                    "batch_index": batch_count,
                    "query_index": query_index,
                    "influence_ranking": ranking,
                    "influence_values": influence_values,
                })

                full_del_accuracy = int(full_del_logits.argmax(-1).item() == query_labels[query_index].item())
                full_norm_accuracy = int(full_norm_logits.argmax(-1).item() == query_labels[query_index].item())
                for p in deletion_ps:
                    n_delete = min(max(0, int(np.floor(p * n_support))), len(ranking))
                    deleted_indices = ranking[:n_delete]
                    keep = torch.ones(n_support, dtype=torch.bool, device=device)
                    if deleted_indices:
                        keep[torch.tensor(deleted_indices, device=device)] = False
                    del_support = support_imgs[keep]
                    del_labels = support_labels[keep]
                    del_del_logits, del_del_fusion = fused_from_embeddings(
                        deletion_model,
                        query_del_embedding,
                        deletion_support_embeddings[keep],
                        del_labels,
                    )
                    del_norm_logits, del_norm_fusion = fused_from_embeddings(
                        normal_model,
                        query_norm_embedding,
                        normal_support_embeddings[keep],
                        del_labels,
                    )
                    del_del_accuracy = int(del_del_logits.argmax(-1).item() == query_labels[query_index].item())
                    del_norm_accuracy = int(del_norm_logits.argmax(-1).item() == query_labels[query_index].item())
                    delta_del = full_del_accuracy - del_del_accuracy
                    delta_norm = full_norm_accuracy - del_norm_accuracy
                    per_query[str(p)].append({
                        "batch_index": batch_count,
                        "query_index": query_index,
                        "Afull_deletion_aware": full_del_accuracy,
                        "A_p_deletion_aware": del_del_accuracy,
                        "Delta_deletion_aware": delta_del,
                        "Delta_rel_deletion_aware": delta_del / full_del_accuracy if full_del_accuracy else None,
                        "Afull_normal": full_norm_accuracy,
                        "A_p_normal": del_norm_accuracy,
                        "Delta_normal": delta_norm,
                        "Delta_rel_normal": delta_norm / full_norm_accuracy if full_norm_accuracy else None,
                        "Delta_robust": delta_norm - delta_del,
                        "fusion_l2_deletion_aware": torch.linalg.vector_norm(full_del_fusion - del_del_fusion).item(),
                        "fusion_l2_normal": torch.linalg.vector_norm(full_norm_fusion - del_norm_fusion).item(),
                        "n_support": int(n_support),
                        "n_deleted": int(n_delete),
                        "deleted_indices": deleted_indices,
                    })
                query_count += 1

            batch_count += 1
            LOGGER.info(
                "Batch %s/%s complete in %.1fs; elapsed=%s",
                batch_count,
                target_batches,
                time.perf_counter() - batch_start,
                _format_duration(time.perf_counter() - start_time),
            )
            if batch_count >= max_batches:
                break

    summary = []
    for p in deletion_ps:
        records = per_query[str(p)]
        summary.append({
            "p": p,
            "n_queries": len(records),
            "deletion_aware": {
                "Afull": _summary([r["Afull_deletion_aware"] for r in records]),
                "A_p": _summary([r["A_p_deletion_aware"] for r in records]),
                "Delta": _summary([r["Delta_deletion_aware"] for r in records]),
                "Delta_rel": _summary([r["Delta_rel_deletion_aware"] for r in records if r["Delta_rel_deletion_aware"] is not None]),
                "fusion_l2": _summary([r["fusion_l2_deletion_aware"] for r in records]),
            },
            "normal": {
                "Afull": _summary([r["Afull_normal"] for r in records]),
                "A_p": _summary([r["A_p_normal"] for r in records]),
                "Delta": _summary([r["Delta_normal"] for r in records]),
                "Delta_rel": _summary([r["Delta_rel_normal"] for r in records if r["Delta_rel_normal"] is not None]),
                "fusion_l2": _summary([r["fusion_l2_normal"] for r in records]),
            },
            "Delta_robust": _summary([r["Delta_robust"] for r in records]),
        })

    results = {
        "method": "leave-one-out fusion influence ranking",
        "ranking_checkpoint": ckpt_deletion_aware,
        "ckpt_deletion_aware": ckpt_deletion_aware,
        "ckpt_normal": ckpt_normal,
        "deletion_ps": deletion_ps,
        "n_batches": batch_count,
        "summary": summary,
        "per_query": per_query,
        "influence_by_query": influence_by_query,
    }
    out_dir = os.getcwd() if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "influence_deletion_results.json")
    LOGGER.info("All batches complete; writing results to %s", output_path)
    with open(output_path, "w") as output_file:
        json.dump(results, output_file, indent=2)
    LOGGER.info(
        "Evaluation complete: %s queries across %s batches in %s; saved %s",
        query_count,
        batch_count,
        _format_duration(time.perf_counter() - start_time),
        output_path,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_deletion_aware", required=True)
    parser.add_argument("--ckpt_normal", required=True)
    parser.add_argument("--out_dir", default="results/eval_influence")
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    evaluate_influence_paired(
        args.config,
        args.ckpt_deletion_aware,
        args.ckpt_normal,
        max_batches=args.max_batches,
        seed=args.seed,
        out_dir=args.out_dir,
    )