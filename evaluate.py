import faiss
import argparse
import os
import torch
import numpy as np
import json
import random

from evaluation.comp_classification_metrics import evaluate_unlearn, evaluate_all_test_data, evaluate_retrain
from evaluation.fid_diffusion import generate_and_save_samples_ddpm, compute_fid
from utils.config import get_obj_from_str, instantiate_from_config, load_config
from utils.gpu_limit import apply_gpu_limit

torch.set_float32_matmul_precision('medium')


def get_datamodule(args):
    if args.config is not None:
        config = load_config(args.config)
        data_config = config['data'].copy()
        params = data_config.get('params', {}).copy()
        params.update({
            'batch_size': args.batch_size,
            'support_size_train': args.support_size_eval,
            'support_size_eval': args.support_size_eval,
            'num_classes': args.num_classes,
            'dataset_name': args.dataset_name,
            'model_type': args.model_type,
        })
        if args.query_digits is not None:
            params['query_digits'] = list(map(int, args.query_digits.split(',')))
        if args.support_digits is not None:
            params['support_digits'] = list(map(int, args.support_digits.split(',')))
        data_config['params'] = params
        return instantiate_from_config(data_config)

    if args.dataset_name.lower() == "cifar100":
        from dataset.cifar100_datamodule import CIFAR100DataModule
        datamodule = CIFAR100DataModule
    elif args.dataset_name.lower() == "tinyimagenet" or args.dataset_name.lower() == "tiny-imagenet":
        from cross_dataset_support.tiny_imagenet_datamodule import TinyImageNetDataModule
        datamodule = TinyImageNetDataModule
    elif "cifar" in args.dataset_name:
        from dataset.cifar_datamodule import CIFARDataModule
        datamodule = CIFARDataModule
    elif "imagenet" in args.dataset_name:
        from dataset.imagenet_datamodule import ImageNetDataModule
        datamodule = ImageNetDataModule
    else:
        raise ValueError("Unsupported dataset_name: " + args.dataset_name)

    if args.query_digits is not None:
        query_digits = list(map(int, args.query_digits.split(',')))
    else:
        query_digits = None
    if args.support_digits is not None:
        support_digits = list(map(int, args.support_digits.split(',')))
    else:
        support_digits = None
    datadir = args.datadir
    if args.dataset_name.lower() in ("tinyimagenet", "tiny-imagenet") and datadir == "data":
        datadir = "./data/tiny-imagenet-200"
    return datamodule(
        data_dir=datadir,
        support_size_train=args.support_size_eval,
        support_size_eval=args.support_size_eval,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        dataset_name=args.dataset_name,
        query_digits=query_digits,
        support_digits=support_digits,
        model_type=args.model_type
    )


def get_model_class(args):
    if args.config is not None:
        return get_obj_from_str(load_config(args.config)['model']['target'])
    if args.model_type == "classifier":
        from models.spm_classifier import SemiParametricClassifier
        return SemiParametricClassifier
    elif args.model_type == "ddpm":
        from models.spm_ddpm import SemiParametricDDPM
        return SemiParametricDDPM
    else:
        raise ValueError("Unsupported model_type: " + args.model_type)


def _reduction_metrics(model):
    hparams = getattr(model, "hparams", {})
    metadata = hparams.get("dataset_reduction", {}) if hasattr(hparams, "get") else {}
    if not metadata or not metadata.get("enabled", True):
        return {
            "dataset_reduction": {
                "enabled": False,
            }
        }
    selection = metadata.get("selection", {})
    original = metadata.get("original_training_size")
    retained = metadata.get("retained_count")
    excluded = metadata.get("excluded_count")
    return {
        "dataset_reduction": {
            "enabled": True,
            "original_training_size": original,
            "retained_count": retained,
            "excluded_count": excluded,
            "overall_retention_pct": metadata.get("overall_retention_pct"),
            "overall_exclusion_pct": metadata.get("overall_exclusion_pct"),
            "el2n_low_exclusion_pct": selection.get("el2n_low_exclusion_pct"),
            "el2n_high_exclusion_pct": selection.get("el2n_high_exclusion_pct"),
            "el2n_scope": selection.get("el2n_scope"),
            "el2n_scores_path": selection.get("el2n_scores_path"),
            "el2n_scores_checksum": selection.get("el2n_scores_checksum"),
            "kcenter_retention_pct": selection.get("kcenter_retention_pct"),
            "kcenter_exclusion_pct": selection.get("kcenter_exclusion_pct"),
            "per_class_retained": metadata.get("per_class_retained", {}),
            "artifact_identifier": metadata.get("identifier"),
            "artifact_path": metadata.get("_artifact_path"),
            "pilot_checkpoint": selection.get("pilot_checkpoint"),
            "seed": selection.get("seed"),
        }
    }


def main(args):
    apply_gpu_limit(getattr(args, "gpu_limit_pct", 45.0))
    if torch.cuda.device_count() > 1 and not torch.distributed.is_initialized():
        import torch.distributed as dist
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        print(f"Distributed inference initialized on {torch.cuda.device_count()} GPUs. Local rank: {local_rank}")

    ModelClass = get_model_class(args)
    model = ModelClass.load_from_checkpoint(args.ckpt_path, strict=True)
    model.embedding_file = args.embedding_file
    model.support_size_eval = args.support_size_eval
    print(f"Loaded model from {args.ckpt_path}")

    # retrained model if provided
    if args.retrained_path is not None:
        retrained_model = ModelClass.load_from_checkpoint(
            args.retrained_path,
            strict=True,
        )
        retrained_model.embedding_file = args.embedding_file
        retrained_model.support_size_eval = args.support_size_eval
        print(f"Loaded model from {args.retrained_path}")
    else:
        retrained_model = None

    if args.model_type == "classifier":
        datamodule = get_datamodule(args)
        datamodule.setup("test")

        if args.unlearn_portion is not None and len(args.indexes_to_replace) > 0:
            raise ValueError("Use either --unlearn_portion or --indexes_to_replace, not both.")
        if args.unlearn_portion is not None and not 0.0 <= args.unlearn_portion <= 1.0:
            raise ValueError("--unlearn_portion must be between 0 and 1.")
        invalid_indexes = [
            index for index in args.indexes_to_replace
            if index < 0 or index >= len(datamodule.train_dataset)
        ]
        if invalid_indexes:
            raise ValueError(
                f"--indexes_to_replace contains out-of-range training indices: {invalid_indexes}"
            )

        if args.unlearn_portion is not None:
            datamodule.set_random_unlearn_portion(args.unlearn_portion)
        if len(args.indexes_to_replace) > 0:
            datamodule.set_indexes_to_replace(args.indexes_to_replace)

        if args.use_pretrain:
            if args.dataset_name == "imagenet":
                import torchvision.models as models
                import torch.nn as nn
                resnet = models.resnet18(pretrained=True)
                resnet.fc = nn.Identity(resnet.fc.in_features, resnet.fc.in_features)
                model.encoder = resnet
            else:
                model.encoder.load_state_dict(torch.load(args.pretrained_ckpt, map_location=model.device))
        if args.retrain:
            metrics = evaluate_retrain(model, datamodule, k=args.support_size_eval, knn=args.knn, save_dir=os.path.dirname(args.out_file), config_path=args.config, use_random_sampling=args.use_random_sampling, unlearn_portion=args.unlearn_portion, indexes_to_replace=args.indexes_to_replace, acc_v2=args.acc_v2, retrained_model=retrained_model, force_retrain=args.force_retrain)
        elif (args.query_digits is not None) or (args.unlearn_portion is not None) or (len(args.indexes_to_replace) > 0):
            metrics = evaluate_unlearn(model, datamodule, k=args.support_size_eval, knn=args.knn, save_dir=os.path.dirname(args.out_file), use_random_sampling=args.use_random_sampling)
        else:
            metrics = evaluate_all_test_data(model, datamodule, save_dir=os.path.dirname(args.out_file), cluster_portion=args.cluster_portion, knn=args.knn, knn_c=args.knn_c, k=args.support_size_eval)

        metrics.update(_reduction_metrics(model))
        print("Evaluation metrics:", metrics)

        if args.out_file:
            save_dir = os.path.join(os.path.dirname(args.out_file), "knn") if args.knn else os.path.dirname(args.out_file)
            os.makedirs(save_dir, exist_ok=True)

            if args.unlearn_portion is not None or len(args.indexes_to_replace) > 0:
                out_path = os.path.join(save_dir, os.path.basename(args.out_file) + f"_{args.seed}.json")
            else:
                out_path = os.path.join(save_dir, os.path.basename(args.out_file) + f"_{args.query_digits}.json")
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Saved evaluation results to {out_path}")

    elif args.model_type == "ddpm":
        datamodule = get_datamodule(args)
        generate_and_save_samples_ddpm(model, datamodule, args)
        compute_fid(args.support_digits, model.device, args.real_path, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for testing and generating samples")
    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--datadir", type=str, default="data", help="Directory where the dataset is stored")
    parser.add_argument("--model_type", type=str, choices=["classifier", "ddpm"], required=True,
                        help="Type of the model (classifier or ddpm)")
    parser.add_argument("--real_path", type=str, default="results/images/real_images/cifar10", help="Path to the GT imgs")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint to load")
    parser.add_argument("--save_path", type=str, default="results/images/ddpm/samples", help="Path to store imgs")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--support_size_eval", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--gpu-limit-pct", type=float, default=45.0,
                        help="Per-process CUDA memory limit percentage (1-100)")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false", help="Do not normalize the dataset")
    parser.set_defaults(normalize=True)
    # For evaluation
    parser.add_argument("--use_pretrain", action="store_true", help="Use precomputed embeddings for evaluation")
    parser.add_argument("--knn", action="store_true", help="Use KNN for evaluation")
    parser.add_argument("--knn_c", action="store_true", help="Use clustering for KNN evaluation")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--expert_name", choices=["pr", "lca"], default="lca")
    parser.add_argument("--total_samples_per_class", type=int, default=5000)
    parser.add_argument("--no_compute_fid", action="store_true", help="Do not compute FID for ddpm model")
    parser.add_argument("--use_random_sampling", action="store_true", help="Use random sampling instead of retrieval")
    # Data-specific parameters
    parser.add_argument("--query_digits", type=str, default=None, help="Comma-separated query digits, e.g. '0,1,2'")
    parser.add_argument("--support_digits", type=str, default=None, help="Comma-separated support digits, e.g. '0,1,2'")
    parser.add_argument("--embedding-file", type=str, default=None, help="Path to precomputed train embeddings file")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to pretrained encoder checkpoint")
    parser.add_argument("--out_file", type=str, default=None, help="Path to save evaluation metrics as JSON")
    parser.add_argument("--retrain", action="store_true", help="Use retrain for evaluation")
    parser.add_argument("--cluster_portion", type=float, default=1.0, help="Portion of training data to use for clustering")
    parser.add_argument("--acc_v2", action="store_true", help="Use new accuracy calculation method")
    parser.add_argument("--unlearn_portion", type=float, default=None, help="Portion of data to unlearn")
    parser.add_argument("--indexes_to_replace", type=int, nargs="+", default=[], help="Specific index data to forget")
    parser.add_argument("--retrained_path", type=str, default=None, help="Path to the retrained model checkpoint")
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Ignore cached retrained checkpoints and train a fresh reference model",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)
