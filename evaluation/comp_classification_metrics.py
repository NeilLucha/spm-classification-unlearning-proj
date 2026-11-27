import torch
import faiss
import time
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)

def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0] if target_train is not None else 0
    n_target_test = target_test.shape[0] if target_test is not None else 0

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

    return np.mean(accs) if accs else 0.0

def SVC_MIA(shadow_train, shadow_test, target_train, target_test):
    # shadow_train, shadow_test, target_train, target_test are tuples (probs, labels)
    shadow_train_prob, shadow_train_labels = shadow_train
    shadow_test_prob, shadow_test_labels   = shadow_test
    target_train_prob, target_train_labels = target_train if target_train is not None else (None, None)
    target_test_prob, target_test_labels = target_test if target_test is not None else (None, None)

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None]) if target_train_prob is not None else None
    target_test_conf = (
        torch.gather(target_test_prob, 1, target_test_labels[:, None])
        if target_test_prob is not None else None
    )

    m_conf = SVC_fit_predict(shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf)
    m_prob = SVC_fit_predict(shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob)

    return {
        "confidence_svc": m_conf,
        "prob_svc": m_prob,
    }


class UnlearnEvaluator:
    def __init__(self):
        self.metrics = {
            'shadow_train_rest_correct': 0, 'shadow_train_rest_total': 0,
            'train_unlearn_correct': 0, 'train_unlearn_total': 0,
            'train_rest_correct': 0,    'train_rest_total': 0,
            'test_unlearn_correct': 0,  'test_unlearn_total': 0,
            'test_rest_correct': 0,     'test_rest_total': 0, 'test_all_rest_correct': 0, 'test_all_rest_total': 0,
        }
        self.start_time = None

    def start_timer(self):
        self.start_time = time.perf_counter()

    def stop_timer(self):
        self.metrics['inference_time'] = time.perf_counter() - self.start_time

    def update(self, preds, labels, split, is_unlearn):
        key = f"{split}_{'unlearn' if is_unlearn else 'rest'}"
        self.metrics[f"{key}_correct"] += (preds == labels).sum().item()
        self.metrics[f"{key}_total"] += labels.size(0)
        print(f"Updated {key}: {self.metrics[f'{key}_correct']}/{self.metrics[f'{key}_total']} correct")

    def compute(self):
        def accuracy(correct, total):
            return correct / total if total else None

        return {
            'train_acc_unlearn': accuracy(self.metrics['train_unlearn_correct'], self.metrics['train_unlearn_total']),
            'train_acc_rest': accuracy(self.metrics['train_rest_correct'], self.metrics['train_rest_total']),
            'test_acc_unlearn': accuracy(self.metrics['test_unlearn_correct'], self.metrics['test_unlearn_total']),
            'test_acc_rest': accuracy(self.metrics['test_rest_correct'], self.metrics['test_rest_total']),
            'inference_time':       self.metrics.get('inference_time', 0.0)
        }


# ---- FAISS-based evaluation ----

def build_faiss_index(model, support_loader, device=None, eval_mode=True):
    """
    Build a FAISS index from model embeddings of the support set.
    support_loader should yield (query_imgs, support_imgs, query_labels, support_labels).
    """
    device = device or next(model.parameters()).device
    model.to(device)
    if eval_mode:
        model.eval()
    all_emb = []
    all_labels = []
    with torch.no_grad():
        for support_imgs, support_labels in tqdm(support_loader, desc="Building FAISS index", leave=False):
            support_imgs = support_imgs.to(device) #, dtype=model.dtype)
            try:
                emb = model.encoder(support_imgs)  # (B, D) on GPU
            except:
                emb = model(support_imgs)  # (B, D) on GPU
                emb = F.normalize(emb, p=2, dim=1)
            all_emb.append(emb)
            all_labels.append(support_labels.view(-1).to(device))
    embeddings = torch.cat(all_emb, dim=0)       # still on GPU
    labels = torch.cat(all_labels, dim=0)        # on GPU
    emb_np = embeddings.cpu().numpy().astype('float32')  # one GPU→CPU copy
    index = faiss.IndexFlatL2(emb_np.shape[1])
    index.add(emb_np)
    return index, embeddings, labels

def evaluate_all_test_data(model, datamodule, save_dir, cluster_portion, knn=False, knn_c=False, k=None):
    """
    Evaluate the model on all test data using the datamodule.
    Returns a dict with accuracy and inference time.
    """
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    test_loader = datamodule.test_dataloader()
    print(f"Evaluating {len(test_loader.dataset)} test samples...")

    if cluster_portion < 1.0:
        # Random smaple a portion of support data for clustering
        total_support = len(datamodule.support_loader().dataset)
        sample_size = int(total_support * cluster_portion)
        indices = np.random.choice(total_support, sample_size, replace=False)
        sampled_support = torch.utils.data.Subset(datamodule.support_loader().dataset, indices)
        sampled_loader = torch.utils.data.DataLoader(sampled_support, batch_size=64, shuffle=False)
        datamodule.support_loader = lambda: sampled_loader
        print(f'Building aggregated features for support data ({len(datamodule.support_loader().dataset)} samples)...')

        # Build and save
        start_index_time = time.perf_counter()
        mean_vector = {}
        each_class_count = {}
        with torch.no_grad():
            for support_imgs, support_labels in tqdm(datamodule.support_loader(), desc="Building aggregated features", leave=False):
                support_imgs = support_imgs.to(device, dtype=model.dtype)
                support_emb = model.encoder(support_imgs)  # (B, D)
                for i in range(support_labels.size(0)):
                    label = int(support_labels[i])
                    emb = support_emb[i].detach().cpu()
                    if label not in mean_vector:
                        each_class_count[label] = 1
                        mean_vector[label] = emb.clone()
                    else:
                        each_class_count[label] += 1
                        mean_vector[label] += (emb - mean_vector[label]) / each_class_count[label]
        support_embeddings = torch.stack(list(mean_vector.values()), dim=0).to(device)  # (N, D)
        support_labels = torch.tensor(list(mean_vector.keys()), device=device)  # (N,)
        build_index_time = time.perf_counter() - start_index_time
        print(f"Built support data with {len(support_embeddings)} unique classes in {build_index_time:.2f} seconds.")
    correct = 0
    total = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for query_imgs, query_labels in tqdm(test_loader, desc="Evaluating test data", leave=False):
            query_imgs = query_imgs.to(device)
            q_emb = model.encoder(query_imgs)  # (B, D)
            if knn:
                # search k nearest
                q_np = q_emb.cpu().numpy().astype('float32')
                if knn_c: # use clustering, so just pick the one closest in support embeddings
                    q_tensor = torch.from_numpy(q_np).to(device)
                    dists_all = torch.cdist(q_tensor, support_embeddings)  # (B, N)
                    dists, idxs_tensor = torch.topk(dists_all, 1, largest=False)  # (B, k)  
                else:                        
                    dists, idxs = index.search(q_np, k)
                    idxs_tensor = torch.from_numpy(idxs).to(device)
                supp_embs = support_embeddings[idxs_tensor]           # (B, k, D)
                supp_labels_batch = support_labels[idxs_tensor]       # (B, k)
                n_classes = datamodule.num_classes
                B = q_emb.size(0)
                one_hot = torch.zeros(B, n_classes, device=device)
                ones = torch.ones(B, k, device=device)
                one_hot.scatter_add_(1, supp_labels_batch, ones)
                moe_out = one_hot / k

            else:
                moe_out = model.moe(q_emb, support_embeddings, support_labels)

            logits = torch.log(moe_out.clamp(min=1e-8))
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == query_labels.to(device)).sum().item()
            total += query_labels.size(0)
    inference_time = time.perf_counter() - start_time
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        'test_acc': accuracy,
        'inference_time': inference_time,
        'build_index_time': build_index_time,
    }

    return metrics

def evaluate_unlearn(model, datamodule, k, device=None, knn=False, ablation=False, save_dir=None, use_random_sampling=False, reference_model=None):
    """
    Evaluate unlearn and rest splits on train and test, using k-NN over support embeddings.
    Returns a dict with metrics and inference_time.
    
    Args:
        model: The model to evaluate
        datamodule: The datamodule containing the data
        k: Number of nearest neighbors to use
        device: Device to use for evaluation
        knn: Whether to use k-NN instead of MoE
        ablation: Whether to use ablation mode (random sampling)
        save_dir: Directory to save FAISS index and support data
        use_random_sampling: Whether to use random sampling instead of retrieval
    """

    device = device or next(model.parameters()).device
    model.to(device)
    model.eval()
    split_results = {}
    emb_time_unlearn_train = 0.0
    search_time_unlearn_train = 0.0
    moe_time_unlearn_train = 0.0
    support_loader = datamodule.support_loader()
    # prepare shadow training loaders
    test_len = len(datamodule.rest_test_dataloader().dataset)
    full_dataset = datamodule.rest_train_dataloader().dataset
    num_samples = len(full_dataset)
    random_indices = torch.randperm(num_samples)[:test_len].tolist()
    print('Dataset sizes:')
    print(f'  Rest train: {len(datamodule.rest_train_dataloader().dataset)}')
    print(f'  Unlearn train: {len(datamodule.unlearn_train_dataloader().dataset)}')
    print(f'  Rest test: {len(datamodule.rest_test_dataloader().dataset)}')
    print(f'  Unlearn test: {len(datamodule.unlearn_test_dataloader().dataset)}')

    save_dir = os.path.join(save_dir, f"cluster_{datamodule.query_digits}") if save_dir else None
    # os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir or os.getcwd(), "support_data.pt")
    has_sample_deletion = (
        hasattr(datamodule, "random_unlearn_portion")
        or hasattr(datamodule, "indexes_to_replace")
    )
    if os.path.exists(data_path) and not has_sample_deletion:
        # Load from disk
        data = torch.load(data_path, map_location=device)
        support_embeddings = data["embeddings"]
        support_labels = data["labels"]
        build_index_time = 0.0
        print(f"Loaded support data from {save_dir}")
    elif len(datamodule.support_loader().dataset) < 100000:
        # Build and save (smaller dataset, aggregation in one pass)
        start_index_time = time.perf_counter()
        all_S_emb = []
        all_S_labels = []
        print(f'Building aggregated features for support data ({len(support_loader.dataset)} samples)...')
        with torch.no_grad():
            for support_imgs, support_labels in tqdm(support_loader, desc="Building aggregated features", leave=False):
                support_imgs = support_imgs.to(device, dtype=model.dtype)
                support_emb = model.encoder(support_imgs)  # (B, D)
                all_S_emb.append(support_emb)
                all_S_labels.append(support_labels.view(-1).to(device))
        support_embeddings = torch.cat(all_S_emb, dim=0)  # (N, D)
        support_labels = torch.cat(all_S_labels, dim=0)  # (N,)
        support_embeddings, support_labels = model.feature_aggregation(support_embeddings, support_labels)
        build_index_time = time.perf_counter() - start_index_time
        print(f"Built support data with {len(support_embeddings)} unique classes in {build_index_time:.2f} seconds.")
    else:
        # Build and save (large dataset, aggregation in multiple passes)
        start_index_time = time.perf_counter()
        print(f'Building aggregated features for support data ({len(support_loader.dataset)} samples)...')
        mean_vector = {}
        each_class_count = {}
        with torch.no_grad():
            for support_imgs, support_labels in tqdm(support_loader, desc="Building aggregated features", leave=False):
                support_imgs = support_imgs.to(device, dtype=model.dtype)
                support_emb = model.encoder(support_imgs)  # (B, D)
                for i in range(support_labels.size(0)):
                    label = int(support_labels[i])
                    emb = support_emb[i].detach().cpu()
                    if label not in mean_vector:
                        each_class_count[label] = 1
                        mean_vector[label] = emb.clone()
                    else:
                        each_class_count[label] += 1
                        mean_vector[label] += (emb - mean_vector[label]) / each_class_count[label]
        support_embeddings = torch.stack(list(mean_vector.values()), dim=0).to(device)  # (N, D)
        support_labels = torch.tensor(list(mean_vector.keys()), device=device).to(device)  # (N,)
        build_index_time = time.perf_counter() - start_index_time
        print(f"Built support data with {len(support_embeddings)} unique classes in {build_index_time:.2f} seconds.")
        # save to disk
        save_base = save_dir or os.getcwd()
        os.makedirs(save_base, exist_ok=True)

    evaluator = UnlearnEvaluator()
    evaluator.start_timer()
    # Dictionary to hold collected probabilities and labels per split

    if reference_model is not None:
        model_k = getattr(getattr(model, "feature_aggregation", None), "K", None)
        reference_k = getattr(getattr(reference_model, "feature_aggregation", None), "K", None)
        if model_k is not None and reference_k is not None and model_k != reference_k:
            raise ValueError(
                "Target and reference models must use the same K for comparison; "
                f"got K={model_k} and K={reference_k}."
            )

    for split, loader, is_unlearn in [
        ('train', datamodule.unlearn_train_dataloader(), True),
        ('train', datamodule.rest_train_dataloader(), False),
        ('test', datamodule.unlearn_test_dataloader(), True),
        ('test', datamodule.rest_test_dataloader(), False),
    ]:
        if split == 'train' and is_unlearn:
            infer_start_time_unlearn_train = time.perf_counter()
        model.eval()
        # initialize probability collector
        probs = []
        labels_all = []
        with torch.no_grad():
            for query_imgs, query_labels in tqdm(loader, desc=f"{split}_{'unlearn' if is_unlearn else 'rest'}", leave=False):
                query_imgs = query_imgs.to(device)
                # embed query
                t0 = time.perf_counter()
                q_emb = model.encoder(query_imgs)  # (B, D)
                B = q_emb.size(0)
                t1 = time.perf_counter()
                if split == 'train' and is_unlearn:
                    emb_time_unlearn_train += (t1 - t0)

                t2 = time.perf_counter()
                try: # fusion is only used for ablation !!!
                    moe_out = model.moe(q_emb, support_embeddings, support_labels)
                except:
                    moe_out = model.fusion(q_emb, support_embeddings, support_labels)
                t3 = time.perf_counter()
                if split == 'train' and is_unlearn:
                    moe_time_unlearn_train += (t3 - t2)
                logits = torch.log(moe_out.clamp(min=1e-8))

                # collect probabilities and true labels
                probs.append(moe_out.cpu())
                labels_all.append(query_labels)
                preds = torch.argmax(logits, dim=-1).cpu()
                evaluator.update(preds, query_labels, split, is_unlearn)
        if split == 'train' and is_unlearn:
            infer_time_unlearn_train = time.perf_counter() - infer_start_time_unlearn_train

        # aggregate collected probabilities and labels for this split
        if probs:
            probs = torch.cat(probs, dim=0)
            labels_all = torch.cat(labels_all, dim=0)
        else:
            probs = torch.empty((0, datamodule.num_classes))
            labels_all = torch.empty(0, dtype=torch.long)
        # store results for this split
        split_results[f"{split}_{'unlearn' if is_unlearn else 'rest'}"] = (probs, labels_all)
                
    evaluator.stop_timer()
    # Compute primary metrics
    metrics = evaluator.compute()
    metrics['build_index_time'] = build_index_time
    metrics['emb_time_unlearn_train'] = emb_time_unlearn_train
    metrics['moe_time_unlearn_train'] = moe_time_unlearn_train
    metrics['emb_moe_time_unlearn_train'] = emb_time_unlearn_train + moe_time_unlearn_train
    metrics['infer_time_unlearn_train'] = infer_time_unlearn_train

    # Use retain samples as the shadow population and measure membership on
    # the forget samples. These populations are disjoint for class deletion.
    if all(split_results[key][0].size(0) > 0 for key in ('train_rest', 'test_rest', 'train_unlearn')):
        mia_metrics = SVC_MIA(
            split_results['train_rest'],
            split_results['test_rest'],
            split_results['train_unlearn'],
            split_results['test_unlearn']
            if split_results['test_unlearn'][0].size(0) > 0
            else None,
        )
        metrics.update({key: float(value) for key, value in mia_metrics.items()})

    if reference_model is not None:
        reference_model = reference_model.to(device)
        reference_model.eval()
        ref_support_embeddings = []
        ref_support_labels = []
        with torch.no_grad():
            for support_imgs, support_batch_labels in datamodule.support_loader():
                support_imgs = support_imgs.to(device, dtype=reference_model.dtype)
                ref_support_embeddings.append(reference_model.encoder(support_imgs))
                ref_support_labels.append(support_batch_labels.view(-1).to(device))
        ref_support_embeddings = torch.cat(ref_support_embeddings, dim=0)
        ref_support_labels = torch.cat(ref_support_labels, dim=0)
        ref_support_embeddings, ref_support_labels = reference_model.feature_aggregation(
            ref_support_embeddings, ref_support_labels
        )

        reference_results = {}
        with torch.no_grad():
            for key in ('test_unlearn', 'test_rest'):
                ref_probs = []
                ref_loader = datamodule.unlearn_test_dataloader() if key == 'test_unlearn' else datamodule.rest_test_dataloader()
                for query_imgs, _ in ref_loader:
                    query_imgs = query_imgs.to(device)
                    query_emb = reference_model.encoder(query_imgs)
                    try:
                        ref_out = reference_model.moe(query_emb, ref_support_embeddings, ref_support_labels)
                    except:
                        ref_out = reference_model.fusion(query_emb, ref_support_embeddings, ref_support_labels)
                    ref_probs.append(ref_out.cpu())
                reference_results[key] = (
                    torch.cat(ref_probs, dim=0)
                    if ref_probs
                    else torch.empty((0, datamodule.num_classes))
                )

        for key in ('test_unlearn', 'test_rest'):
            model_probs = split_results[key][0]
            reference_probs = reference_results[key]
            if model_probs.size(0) > 0:
                model_predictions = model_probs.argmax(dim=1)
                reference_predictions = reference_probs.argmax(dim=1)
                metrics[f'{key}_prediction_agreement'] = (
                    model_predictions == reference_predictions
                ).float().mean().item()
                metrics[f'{key}_kl'] = F.kl_div(
                    model_probs.clamp_min(1e-8).log(),
                    reference_probs.clamp_min(1e-8),
                    reduction='batchmean',
                ).item()
            else:
                metrics[f'{key}_prediction_agreement'] = None
                metrics[f'{key}_kl'] = None


    return metrics

def evaluate_retrain(model, datamodule, k=0, knn=False, device=None, save_dir=None, config_path=None, ablation=False,  unlearn_portion=None, indexes_to_replace=[], use_random_sampling=False, acc_v2=False, retrained_model=None, force_retrain=False):
    # Retrain reference model on rest data only
    print("="*60)
    print("RETRAINING REFERENCE MODEL ON REST DATA")
    print("="*60)
    if retrained_model is not None and force_retrain:
        raise ValueError("Cannot use --force_retrain together with --retrained_path.")
    
    from utils.config import load_config, instantiate_from_config
    import pytorch_lightning as pl
    
    # Load config and modify it for rest-only training
    config = load_config(config_path)
    # Get rest classes (all classes except query_digits)
    all_classes = set(range(datamodule.num_classes))
    if datamodule.query_digits:
        if isinstance(datamodule.query_digits, (list, tuple)):
            unlearn_classes = set(datamodule.query_digits)
        else:
            unlearn_classes = {datamodule.query_digits}
    else:
        unlearn_classes = set()
    rest_classes = list(all_classes - unlearn_classes)
    
    # Modify data config to use only rest classes
    config['data']['params']['query_digits'] = rest_classes
    config['data']['params']['support_digits'] = rest_classes

    # Materialize the target's random forget set once, then reuse those exact
    # raw training indices for the retain-only reference model.
    random_unlearn_indices = None
    if unlearn_portion is not None:
        if not hasattr(datamodule, "exclude_idx"):
            datamodule.support_loader()
        random_unlearn_indices = sorted(datamodule.exclude_idx)

    # Create the retain-only datamodule used when training a new reference.
    rest_datamodule = instantiate_from_config(config['data'])
    if datamodule.query_digits: # class-specific unlearning
        rest_datamodule.set_no_train_digits(list(unlearn_classes))
    if random_unlearn_indices is not None:
        rest_datamodule.set_indexes_to_replace(random_unlearn_indices)
    elif len(indexes_to_replace) > 0: # unlearning specific indexes
        rest_datamodule.set_indexes_to_replace(indexes_to_replace)
    
    
    if retrained_model is not None:
        reference_model = retrained_model
        training_time = 0.0
        print("Using supplied retrained reference model checkpoint.")
    else:
        # Setup training callbacks (simplified)
        reference_model = instantiate_from_config(config['model'])
        lightning_config = config.get('lightning', {})
        callbacks = []
        if 'callbacks' in lightning_config and lightning_config['callbacks'] is not None:
            if 'checkpoint_callback' in lightning_config['callbacks']:
                cb_config = lightning_config['callbacks']['checkpoint_callback'].copy()
                if unlearn_portion is not None:
                    cb_config['params']['dirpath'] = os.path.join(save_dir, f"retrained_model/unl_{int(unlearn_portion*100)}")
                elif len(indexes_to_replace) > 0:
                    cb_config['params']['dirpath'] = os.path.join(save_dir, f"retrained_model/unl_{len(indexes_to_replace)}")
                else:
                    cb_config['params']['dirpath'] = os.path.join(save_dir, f"retrained_model/unl_{''.join(map(str, sorted(unlearn_classes)))}")
                callbacks.append(instantiate_from_config(cb_config))
        logger = instantiate_from_config(lightning_config['logger']) if 'logger' in lightning_config else None
        from pytorch_lightning.loggers import WandbLogger
        if logger is not None and isinstance(logger, WandbLogger):
            logger.log_hyperparams(config)

        trainer_config = lightning_config.get('trainer', {})
        trainer_config['logger'] = logger
        trainer_config['callbacks'] = callbacks
        trainer_config['enable_model_summary'] = False

        available_gpus = torch.cuda.device_count()
        trainer_config['devices'] = min(available_gpus, trainer_config.get('devices', 1))
        print(f"Using {trainer_config['devices']} GPUs for retraining (available: {available_gpus}, config requested: {lightning_config.get('trainer', {}).get('devices', 'unknown')})")

        trainer = pl.Trainer(**trainer_config)
        t0 = time.perf_counter()
        ckpt_path = os.path.join(cb_config['params']['dirpath'], 'last-v1.ckpt')
        if os.path.exists(ckpt_path) and not force_retrain:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            print(f"Reusing existing retrained checkpoint from {ckpt_path}")
            reference_model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif os.path.exists(ckpt_path.replace('last-v1.ckpt', 'last.ckpt')) and not force_retrain:
            checkpoint = torch.load(ckpt_path.replace('last-v1.ckpt', 'last.ckpt'), map_location='cpu')
            print(f"Reusing existing retrained checkpoint from {ckpt_path.replace('last-v1.ckpt', 'last.ckpt')}")
            reference_model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            if force_retrain:
                print("Force retrain enabled; training a fresh reference model.")
            else:
                print(f"No existing checkpoint found in {ckpt_path}, retraining from scratch...")
            trainer.fit(reference_model, datamodule=rest_datamodule)
        t1 = time.perf_counter()
        training_time = t1 - t0

    model_k = getattr(getattr(model, "feature_aggregation", None), "K", None)
    reference_k = getattr(getattr(reference_model, "feature_aggregation", None), "K", None)
    if model_k is not None and reference_k is not None and model_k != reference_k:
        raise ValueError(
            "Target and retrained reference models must use the same K for "
            f"comparison; got K={model_k} and K={reference_k}."
        )

    print(f"Training completed in {training_time:.2f} seconds.")
    spm_metrics = get_predict(model, datamodule, device, knn=knn, k=k, cluster_portion=1.0, ours=True)
    if acc_v2: # Show our proposed metrics
        # Evaluate reference model on rest test data
        metrics = get_predict(reference_model, datamodule, device, knn=knn, k=k, cluster_portion=1.0, ours=True)
        ModelClass = type(reference_model)
        # Calculate cross entorpy of predictions
        predictions = metrics['preds']
        spm_predictions = spm_metrics['preds']

        ce = F.cross_entropy(
            F.one_hot(spm_predictions.argmax(dim=1), num_classes=datamodule.num_classes).float(),
            F.one_hot(predictions.argmax(dim=1), num_classes=datamodule.num_classes).float(),
        ).item()
        kl = F.kl_div(
            spm_predictions.log_softmax(dim=1),
            predictions.softmax(dim=1),
            reduction="batchmean",
        ).item()
        # view retrained model as the ground truth
        agreement_rate = (spm_predictions.argmax(dim=1) == predictions.argmax(dim=1)).float().mean().item() * 100

        print(f"CE: {ce:.4f}")
        print(f"KL: {kl:.4f}")
        print(f"Agreement rate: {agreement_rate:.4f}")

        
    unlearn_metric = evaluate_unlearn(
        model, datamodule, k, device=device, knn=knn, ablation=ablation, save_dir=save_dir,
        use_random_sampling=use_random_sampling, reference_model=reference_model
    ) 
    all_merics = {
        'spm_model_acc': spm_metrics['test_acc'] if acc_v2 else None,
        'spm_inference_time': spm_metrics['inference_time'] if acc_v2 else None,
        'retrain_spm_model_acc': metrics['test_acc'] if acc_v2 else None,
        'retrain_spm_inference_time': metrics['inference_time'] if acc_v2 else None,
        'ce': ce if acc_v2 else None,
        'kl': kl if acc_v2 else None,
        'agreement_rate': agreement_rate if acc_v2 else None,
        'retrain_training_time': training_time,
        'retain_spm_unl_train_acc_unlearn': unlearn_metric['train_acc_unlearn'],
        'retain_spm_unl_train_acc_rest': unlearn_metric['train_acc_rest'],
        'retain_spm_unl_test_acc_unlearn': unlearn_metric['test_acc_unlearn'],
        'retain_spm_unl_test_acc_rest': unlearn_metric['test_acc_rest'],
        'retain_spm_unl_inference_time': unlearn_metric['inference_time'],
    }
    for metric_name in (
        'confidence_svc', 'prob_svc',
        'test_unlearn_prediction_agreement', 'test_unlearn_kl',
        'test_rest_prediction_agreement', 'test_rest_kl',
    ):
        if metric_name in unlearn_metric:
            all_merics[metric_name] = unlearn_metric[metric_name]

    return all_merics


def get_predict(model, datamodule, device, knn=False, k=None, cluster_portion=1.0, ours=False):
    """Evaluate using the model's configured support aggregation.

    ``k`` remains the explicit evaluation support/neighbor-size argument used by
    the evaluation pipeline. The prototype count is determined by the model's
    ``feature_aggregation.K`` configuration.
    """
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    

    if cluster_portion < 1.0:
        # Random smaple a portion of support data for clustering
        total_support = len(datamodule.support_loader().dataset)
        sample_size = int(total_support * cluster_portion)
        indices = np.random.choice(total_support, sample_size, replace=False)
        sampled_support = torch.utils.data.Subset(datamodule.support_loader().dataset, indices)
        sampled_loader = torch.utils.data.DataLoader(sampled_support, batch_size=64, shuffle=False)
        datamodule.support_loader = lambda: sampled_loader
        print(f'Building aggregated features for support data ({len(datamodule.support_loader().dataset)} samples)...')

    
    print(f'Building aggregated features for support data ({len(datamodule.support_loader().dataset)} samples)...')
    start_index_time = time.perf_counter()
    support_embeddings_all = []
    support_labels_all = []
    with torch.no_grad():
        for support_imgs, support_labels in tqdm(datamodule.support_loader(), desc="Building aggregated features", leave=False):
            support_imgs = support_imgs.to(device, dtype=model.dtype)
            support_emb = model.encoder(support_imgs)  # (B, D)
            support_embeddings_all.append(support_emb)
            support_labels_all.append(support_labels.view(-1).to(device))

    support_embeddings = torch.cat(support_embeddings_all, dim=0)
    support_labels = torch.cat(support_labels_all, dim=0)
    if not hasattr(model, "feature_aggregation"):
        raise AttributeError(
            f"{type(model).__name__} must define feature_aggregation for checkpoint evaluation."
        )
    support_embeddings, support_labels = model.feature_aggregation(
        support_embeddings,
        support_labels,
    )
    build_index_time = time.perf_counter() - start_index_time
    prototype_k = getattr(model.feature_aggregation, "K", "configured")
    print(
        f"Built aggregated support data with {len(support_embeddings)} prototypes "
        f"(K={prototype_k}) in {build_index_time:.2f} seconds."
    )

    correct = 0
    total = 0
    start_time = time.perf_counter()
    all_preds = []
    test_loader = datamodule.test_dataloader()
    print(f"Evaluating {len(test_loader.dataset)} test samples...")
    with torch.no_grad():
        for query_imgs, query_labels in tqdm(test_loader, desc="Evaluating test data", leave=False):
            query_imgs = query_imgs.to(device)
            q_emb = model.encoder(query_imgs)  # (B, D)
            moe_out = model.moe(q_emb, support_embeddings, support_labels)

            logits = torch.log(moe_out.clamp(min=1e-8))
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(logits.cpu())
            correct += (preds == query_labels.to(device)).sum().item()
            total += query_labels.size(0)
    inference_time = time.perf_counter() - start_time
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        'test_acc': accuracy,
        'inference_time': inference_time,
        'build_index_time': build_index_time,
        'preds': torch.cat(all_preds, dim=0)
    }
    return metrics
