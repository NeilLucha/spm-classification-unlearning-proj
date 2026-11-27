import numpy as np
import torch
from torch.utils.data import Dataset
import os


def filter_dataset(dataset, num_classes):
    filtered = [(img, lab) for img, lab in dataset if lab < num_classes]
    return filtered



class QuerySupportTrainDataset(Dataset):
    def __init__(self, dataset, support_size, num_classes):
        # Store original dataset and parameters
        self.dataset = dataset
        self.support_size = support_size
        self.num_classes = num_classes

        # Build a list of valid indices using labels without loading images
        if hasattr(dataset, 'targets'):
            all_targets = list(dataset.targets)
            self.indices = [i for i, lab in enumerate(all_targets) if lab < num_classes]
        elif hasattr(dataset, 'samples'):
            self.indices = [i for i, (_, lab) in enumerate(dataset.samples) if lab < num_classes]
        else:
            # Fallback: may load images, but only labels
            self.indices = [i for i, (_, lab) in enumerate(dataset) if lab < num_classes]

        self.num_samples = len(self.indices)
        # Precompute label-to-indices mapping for efficient support sampling
        self.label_to_indices = {}
        if hasattr(dataset, 'targets'):
            for i in self.indices:
                lab = dataset.targets[i]
                if isinstance(lab, torch.Tensor):
                    lab = lab.item()
                self.label_to_indices.setdefault(lab, []).append(i)
        elif hasattr(dataset, 'samples'):
            for i in self.indices:
                lab = dataset.samples[i][1]
                self.label_to_indices.setdefault(lab, []).append(i)
        else:
            for i in self.indices:
                _, lab = dataset[i]
                self.label_to_indices.setdefault(lab, []).append(i)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        query_img, query_label = self.dataset[real_idx]
        if query_img.dim() == 2:
            query_img = query_img.unsqueeze(0)
        return query_img, query_label, idx


def query_support_train_collate_fn(batch, support_dataset, support_size, num_classes):
    query_imgs, query_labels, query_indices = zip(*batch)
    query_imgs = torch.stack(query_imgs, dim=0)
    query_labels = torch.tensor(query_labels)
    
    # Evenly sample support_size among query labels
    unique_labels = np.unique(query_labels.numpy())
    n_labels = len(unique_labels)
    support_indices = []
    if n_labels > 0:
        base = support_size // n_labels
        extra = support_size % n_labels
        for idx, lab in enumerate(unique_labels):
            candidates = support_dataset.label_to_indices.get(int(lab), [])
            # filtering: map batch-local query indices to raw dataset indices
            real_query_indices = [support_dataset.indices[i] for i in query_indices]
            candidates = np.setdiff1d(candidates, real_query_indices).tolist()
            k = base + (1 if idx < extra else 0)
            if candidates:
                if len(candidates) >= k:
                    chosen = np.random.choice(candidates, k, replace=False)
                else:
                    chosen = np.random.choice(candidates, k, replace=True)
                support_indices.extend(int(i) for i in chosen)

    # Index into the underlying raw dataset (gives (img, label))
    support_data = [support_dataset.dataset[i] for i in support_indices]
    support_imgs = torch.stack([img if img.dim() == 3 else img.unsqueeze(0) for img, lab in support_data], dim=0) if len(support_data) > 0 else torch.empty(0)
    support_labels = torch.tensor([lab for img, lab in support_data]) if len(support_data) > 0 else torch.empty(0, dtype=torch.long)
    
    perm = torch.randperm(num_classes)
    query_labels = perm[query_labels]
    support_labels = perm[support_labels]
    
    return query_imgs, support_imgs, query_labels, support_labels


class QuerySupportTestDataset(Dataset):
    def __init__(self, query_dataset, support_dataset, support_size, query_digits, support_digits, num_classes):
        # Initialize label sets
        if query_digits is None:
            query_digits = set(range(num_classes))
        if support_digits is None:
            support_digits = set(range(num_classes))
        self.support_size = support_size
        self.num_classes = num_classes
        self.query_dataset = query_dataset

        # Vectorized computation of query indices
        if hasattr(query_dataset, 'targets'):
            targets_arr = np.array(query_dataset.targets)
            mask = np.isin(targets_arr, list(query_digits))
            self.query_indices = mask.nonzero()[0].tolist()
        elif hasattr(query_dataset, 'samples'):
            labs_arr = np.array([lab for _, lab in query_dataset.samples])
            mask = np.isin(labs_arr, list(query_digits))
            self.query_indices = mask.nonzero()[0].tolist()
        else:
            # Fallback to Python loop if necessary
            self.query_indices = [i for i, (_, lab) in enumerate(query_dataset) if lab in query_digits]
        self.num_queries = len(self.query_indices)

        # Build support indices without loading images
        if hasattr(support_dataset, 'targets'):
            targets_arr = np.array(support_dataset.targets)
            mask = np.isin(targets_arr, list(support_digits))
            self.support_indices = mask.nonzero()[0].tolist()
        elif hasattr(support_dataset, 'samples'):
            self.support_indices = [i for i, (_, lab) in enumerate(support_dataset.samples) if lab in support_digits]
        else:
            self.support_indices = [i for i, (_, lab) in enumerate(support_dataset) if lab in support_digits]

        # Keep reference to raw support dataset for on-the-fly sampling
        self.support_dataset = support_dataset


    def __len__(self):
        return self.num_queries

    def __getitem__(self, idx):
        real_idx = self.query_indices[idx]
        query_img, query_label = self.query_dataset[real_idx]
        if query_img.dim() == 2:
            query_img = query_img.unsqueeze(0)
        return query_img, query_label, idx


def query_support_test_collate_fn(batch, support_dataset, support_size):
    query_imgs, query_labels, _ = zip(*batch)
    query_imgs = torch.stack(query_imgs, dim=0)
    query_labels = torch.tensor(query_labels)
    
    available_indices = np.array(support_dataset.support_indices)
    if not support_size:
        support_size = len(available_indices)
    indices = np.random.choice(available_indices, support_size, replace=False)
    support_data = [support_dataset.support_dataset[i] for i in indices]
    support_imgs = torch.stack([img if img.dim() == 3 else img.unsqueeze(0) for img, _ in support_data], dim=0)
    support_labels = torch.tensor([lab for _, lab in support_data])
    
    return query_imgs, support_imgs, query_labels, support_labels

def query_support_train_collate_fn_deletion(
    batch,
    support_dataset,
    support_size,
    num_classes,
    deletion_fraction=0.2,
):
    query_imgs, query_labels, query_indices = zip(*batch)

    query_imgs = torch.stack(query_imgs, dim=0)
    query_labels = torch.tensor(query_labels)

    # ---------------------------------------------------------
    # 1. Construct the support set exactly as in the baseline
    # ---------------------------------------------------------
    unique_labels = np.unique(query_labels.numpy())
    n_labels = len(unique_labels)

    support_indices = []

    if n_labels > 0:
        base = support_size // n_labels
        extra = support_size % n_labels

        for idx, lab in enumerate(unique_labels):

            candidates = support_dataset.label_to_indices.get(
                int(lab), []
            )

            # Don't use query images as support images. Map batch-local
            # query indices to raw dataset indices before filtering.
            real_query_indices = [support_dataset.indices[i] for i in query_indices]
            candidates = np.setdiff1d(
                candidates,
                real_query_indices
            ).tolist()

            k = base + (1 if idx < extra else 0)

            if candidates:
                if len(candidates) >= k:
                    chosen = np.random.choice(
                        candidates,
                        k,
                        replace=False
                    )
                else:
                    chosen = np.random.choice(
                        candidates,
                        k,
                        replace=True
                    )

                support_indices.extend(
                    int(i) for i in chosen
                )

    # Index underlying raw dataset using raw indices
    support_data = [support_dataset.dataset[i] for i in support_indices]
    support_imgs = torch.stack([
        img if img.dim() == 3 else img.unsqueeze(0)
        for img, lab in support_data
    ], dim=0) if len(support_data) > 0 else torch.empty(0)

    support_labels = torch.tensor([
        lab for img, lab in support_data
    ]) if len(support_data) > 0 else torch.empty(0, dtype=torch.long)

    # ---------------------------------------------------------
    # 2. Construct deletion subset M
    # ---------------------------------------------------------
    num_support = len(support_indices)

    num_delete = int(
        num_support * deletion_fraction
    )

    # We specifically want M > 1
    num_delete = max(2, num_delete)

    # Never delete the entire support set
    num_delete = min(
        num_delete,
        num_support - 1
    )

    # ---------------------------------------------------------
    # 3. Delete samples while retaining at least one sample
    #    from every class.
    # ---------------------------------------------------------
    keep_indices = set(range(num_support))

    for label in torch.unique(support_labels):

        class_indices = torch.where(
            support_labels == label
        )[0].tolist()

        # We must retain at least one representative
        # of every class.
        removable = class_indices[:-1]

        keep_to_delete = min(
            len(removable),
            num_delete
        )

    # Randomly select deletion candidates globally, but exclude
    # the last sample of each class from the candidate pool.
    deletion_candidates = []

    for label in torch.unique(support_labels):

        class_indices = torch.where(
            support_labels == label
        )[0].tolist()

        if len(class_indices) > 1:
            deletion_candidates.extend(
                class_indices[:-1]
            )

    deletion_candidates = np.array(
        deletion_candidates
    )

    if len(deletion_candidates) >= num_delete:

        delete_indices = np.random.choice(
            deletion_candidates,
            num_delete,
            replace=False
        )

    else:
        delete_indices = deletion_candidates

    delete_mask = np.ones(
        num_support,
        dtype=bool
    )

    delete_mask[delete_indices] = False

    deleted_support_imgs = support_imgs[
        delete_mask
    ]

    deleted_support_labels = support_labels[
        delete_mask
    ]

    # ---------------------------------------------------------
    # 4. Apply identical label permutation to both support sets
    # ---------------------------------------------------------
    perm = torch.randperm(num_classes)

    query_labels = perm[query_labels]

    support_labels = perm[support_labels]

    deleted_support_labels = perm[
        deleted_support_labels
    ]

    # ---------------------------------------------------------
    # 5. Return both versions
    # ---------------------------------------------------------
    return (
        query_imgs,
        support_imgs,
        query_labels,
        support_labels,
        deleted_support_imgs,
        deleted_support_labels,
    )