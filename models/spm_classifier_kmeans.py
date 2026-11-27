"""
SPM Classifier with K-means based prototype aggregation.

This module extends the base SPM classifier to use K-means clustering for generating
multiple prototypes per class instead of a single mean prototype.

Key changes:
1. Feature_Aggregation_KMeans: Generates K prototypes per class using K-means
2. Handles variable K per class (< K samples means fewer clusters)
3. Returns prototypes with corresponding class labels
4. Fully compatible with existing MoE/PairwiseRelationExpert downstream code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import random

import faiss
import math
import numpy as np


class PairwiseRelationExpert(nn.Module):
    def __init__(self, embed_dim_q, embed_dim_s, num_classes, hidden_dim=None):
        super(PairwiseRelationExpert, self).__init__()
        self.num_classes = num_classes
        if hidden_dim is None:
            hidden_dim = embed_dim_q
        self.linear_q = nn.Linear(embed_dim_q, hidden_dim, bias=False)
        self.linear_s = nn.Linear(embed_dim_s, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
    
    def forward(self, q, S, support_labels):
        # Compute one-hot for current batch classes
        unique_labels, inv_indices = torch.unique(support_labels, sorted=True, return_inverse=True)
        V_small = F.one_hot(inv_indices, num_classes=unique_labels.size(0)).float()
        # Transform query and support embeddings
        q_trans = self.linear_q(q)    # (B, hidden_dim)
        S_trans = self.linear_s(S)    # (C, hidden_dim)
        # Compute pairwise relation scores
        # h = F.relu(q_trans.unsqueeze(1) + S_trans.unsqueeze(0) + self.bias)
        h = torch.matmul(q_trans, S_trans.transpose(0, 1)) # (B, C)
        h = h / math.sqrt(self.hidden_dim)
        
        # Normalize over individual prototypes, then aggregate their probabilities by class
        weights = F.softmax(h, dim=1)  # (B, num_prototypes)
        out_small = torch.matmul(weights, V_small)  # (B, num_unique_classes)
        
        # Map back to full class dimension
        B = q.size(0)
        full_out = torch.zeros(B, self.num_classes, device=weights.device)
        # Map class logits back to full dimension, zero elsewhere
        for i, lab in enumerate(unique_labels):
            if lab < self.num_classes: # To ignore random unlearning labels
                full_out[:, lab] = out_small[:, i]
        return full_out


class Feature_Aggregation_KMeans(nn.Module):
    """
    Aggregate support embeddings using PyTorch-based K-means clustering per class.
    
    Instead of computing a single mean prototype per class, this module:
    1. Groups embeddings by class label
    2. Performs K-means clustering within each class (PyTorch-based, preserves gradients)
    3. Returns K cluster centroids per class (or fewer if < K samples in that class)
    4. Returns a corresponding class label for each prototype
    
    The PyTorch K-means implementation preserves the autograd path from centroids
    back to the original embeddings, enabling end-to-end training of the encoder.
    
    Args:
        num_classes (int): Total number of classes
        K (int): Target number of prototypes per class (default: 3)
            If a class has fewer than K samples, the actual number of prototypes
            equals the number of available samples.
        kmeans_iterations (int): Number of K-means iterations (default: 10)
    """
    
    def __init__(self, num_classes, K=3, kmeans_iterations=10):
        super(Feature_Aggregation_KMeans, self).__init__()
        self.num_classes = num_classes
        self.K = K
        self.kmeans_iterations = kmeans_iterations
    
    @staticmethod
    def pytorch_kmeans_plusplus_init(embeddings, K):
        """
        K-means++ initialization for stable, deterministic clustering.
        
        Args:
            embeddings: Tensor of shape (N, D)
            K: int - number of clusters
        
        Returns:
            centroids: Tensor of shape (K, D) - initial centroids
        """
        N, D = embeddings.shape
        device = embeddings.device
        
        # First center: choose the first point deterministically
        first_idx = 0
        centroids = [embeddings[first_idx:first_idx+1]]  # (1, D)
        
        # Remaining centers: choose with probability proportional to distance squared
        for _ in range(K - 1):
            # Compute distance to nearest centroid for each point
            distances = torch.cdist(embeddings, torch.cat(centroids, dim=0))  # (N, num_centroids)
            min_distances = distances.min(dim=1)[0]  # (N,)
            
            # Deterministic farthest-point equivalent of K-means++ selection
            next_idx = torch.argmax(min_distances ** 2).item()
            centroids.append(embeddings[next_idx:next_idx+1])
        
        return torch.cat(centroids, dim=0)  # (K, D)
    
    @staticmethod
    def pytorch_kmeans(embeddings, K, num_iterations=10):
        """
        PyTorch-based K-means clustering that preserves gradients.
        Uses K-means++ initialization for stability.
        
        Args:
            embeddings: Tensor of shape (N, D) - data points
            K: int - number of clusters
            num_iterations: int - number of iterations
        
        Returns:
            centroids: Tensor of shape (K, D) - cluster centroids
        
        Note: Hard cluster assignments are not differentiable, but the centroid
        computation (mean of assigned points) is fully differentiable with respect
        to the input embeddings.
        """
        N, D = embeddings.shape
        device = embeddings.device
        
        # Initialize centroids using K-means++ (deterministic, stable)
        centroids = Feature_Aggregation_KMeans.pytorch_kmeans_plusplus_init(embeddings, K)
        
        for iteration in range(num_iterations):
            # Compute L2 distances: (N, K)
            distances = torch.cdist(embeddings, centroids)  # (N, K)
            
            # Assign each point to nearest centroid (hard assignment)
            assignments = torch.argmin(distances, dim=1)  # (N,)
            
            # Recompute centroids as mean of assigned points
            # This operation is differentiable w.r.t. embeddings
            new_centroids = []
            for k in range(K):
                mask = (assignments == k)
                if mask.sum() > 0:
                    # Mean of points assigned to cluster k
                    new_centroids.append(embeddings[mask].mean(dim=0))
                else:
                    # If cluster is empty, keep old centroid
                    new_centroids.append(centroids[k])
            
            centroids = torch.stack(new_centroids)  # (K, D)
        
        return centroids
    
    def forward(self, S, support_labels):
        """
        Args:
            S: Tensor of shape (N, embed_dim) - support embeddings
            support_labels: Tensor of shape (N,) - class labels for each embedding
        
        Returns:
            prototypes: Tensor of shape (total_prototypes, embed_dim)
                where total_prototypes = sum of min(K, count_per_class) for each class
            prototype_labels: Tensor of shape (total_prototypes,)
                class label for each prototype
        
        Data flow:
            1. Group embeddings by class label
            2. For each class:
               - If num_samples >= K: run PyTorch K-means with K clusters
               - If num_samples < K: use all samples as prototypes
            3. Concatenate all prototypes and labels across classes
            
        Gradient flow:
            Prototypes are computed as means of assigned embeddings, so gradients
            flow back through the centroids to the original embeddings S.
        """
        unique_labels = torch.unique(support_labels, sorted=True)
        
        all_prototypes = []
        all_prototype_labels = []
        
        for label in unique_labels:
            # Get embeddings for this class
            class_mask = (support_labels == label)
            class_embeddings = S[class_mask]  # (n_samples_for_class, embed_dim)
            n_samples = class_embeddings.shape[0]
            
            # Determine actual number of clusters for this class
            actual_k = min(self.K, n_samples)
            
            if actual_k == 1:
                # If only 1 cluster, use mean (preserves gradients)
                prototype = class_embeddings.mean(dim=0, keepdim=True)
            else:
                # Run PyTorch K-means (preserves gradients)
                prototype = self.pytorch_kmeans(class_embeddings, actual_k, self.kmeans_iterations)
            
            # Add prototypes and their labels
            all_prototypes.append(prototype)
            all_prototype_labels.append(torch.full((prototype.shape[0],), label, dtype=support_labels.dtype, device=S.device))
        
        # Concatenate all prototypes and labels
        prototypes = torch.cat(all_prototypes, dim=0)  # (total_prototypes, embed_dim)
        prototype_labels = torch.cat(all_prototype_labels, dim=0)  # (total_prototypes,)
        
        return prototypes, prototype_labels


class GatingNetwork(nn.Module):
    def __init__(self, embed_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(2 * embed_dim, num_experts)
        
    def forward(self, q, S):
        # q: (B, embed_dim), S: (N, embed_dim)
        S_mean = S.mean(dim=0)                     # (embed_dim)
        S_mean_exp = S_mean.unsqueeze(0).expand(q.size(0), -1)  # (B, embed_dim)
        combined = torch.cat([q, S_mean_exp], dim=-1)           # (B, 2 * embed_dim)
        gating_logits = self.fc(combined)            # (B, num_experts)
        gating_weights = F.softmax(gating_logits, dim=-1)
        return gating_weights
    

class MoEModel(nn.Module):
    def __init__(self, embed_dim, num_experts, num_heads=1, num_classes=10):
        super().__init__()
        expert_class = PairwiseRelationExpert
        self.experts = nn.ModuleList([
            expert_class(embed_dim, embed_dim, num_classes)
            for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(embed_dim, num_experts)
        
    def forward(self, q, S, support_labels):
        # S can now have variable number of prototypes per class
        # compute each expert's full-class output
        expert_outputs = [expert(q, S, support_labels) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, C)
        gate = self.gating(q, S).unsqueeze(-1)               # (B, num_experts, 1)
        out = torch.sum(gate * expert_outputs, dim=1)       # (B, C)
        return out


class LabelSmoothingNLLLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, log_probs, targets):
        num_classes = log_probs.size(-1)

        # Standard NLL component
        nll_loss = -log_probs.gather(
            dim=-1,
            index=targets.unsqueeze(-1)
        ).squeeze(-1)

        # Uniform smoothing component
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (
            (1.0 - self.smoothing) * nll_loss
            + self.smoothing * smooth_loss
        )

        return loss.mean()

class SemiParametricClassifier_KMeans(pl.LightningModule):
    """
    SPM Classifier with K-means based multi-prototype aggregation.
    
    The key difference from the base SPM classifier is the use of Feature_Aggregation_KMeans
    instead of Feature_Aggregation. This generates K prototypes per class during training/validation,
    which are then fed to the MoE model.
    
    Args:
        embed_dim (int): Embedding dimension
        num_experts (int): Number of experts in MoE (default: 1)
        num_heads (int): Number of attention heads (default: 1, currently unused)
        num_classes (int): Total number of classes
        K (int): Number of prototypes per class (default: 3)
        encoder_type (str): Type of encoder (default: "resnet18")
        learning_rate (float): Learning rate for optimization
        pretrained_imagenet (bool): Whether to use ImageNet pretrained weights
        pretrained_cifar (bool): Whether to use CIFAR pretrained weights
        ckpt_path (str): Path to checkpoint for pretrained weights
        use_label_smoothing (bool): Use label-smoothed NLL loss when enabled
    """
    
    def __init__(self, embed_dim=None, num_experts=1, num_heads=1, num_classes=10, K=3,
                 encoder_type="cnn", learning_rate=1e-3, pretrained_imagenet=True,
                 pretrained_cifar=False, ckpt_path=None, use_label_smoothing=False):
        super(SemiParametricClassifier_KMeans, self).__init__()
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.encoder_type = encoder_type
        self.pretrained_imagenet = pretrained_imagenet
        self.pretrained_cifar = pretrained_cifar
        self.K = K  # Number of prototypes per class
        self.use_label_smoothing = use_label_smoothing
        
        if encoder_type.lower() == "resnet18":
            if pretrained_imagenet:
                import torchvision.models as models
                resnet = models.resnet18(pretrained=True)
                embed_dim = resnet.fc.in_features
                resnet.fc = nn.Identity(resnet.fc.in_features, resnet.fc.in_features)
                self.encoder = resnet
            else:
                from nets.resnet import ResNet18
                self.encoder = ResNet18(embed_dim=embed_dim)
                if pretrained_cifar:
                    assert ckpt_path is not None, "Checkpoint path must be provided for pretrained CIFAR ResNet"
                    self.encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
                    print(f"Loaded pretrained ResNet18 from {ckpt_path}")
        else:
            raise ValueError("encoder_type is not supported")
        
        self.moe = MoEModel(embed_dim, num_experts, num_heads, num_classes)
        self.feature_aggregation = Feature_Aggregation_KMeans(num_classes, K=K)
        if self.use_label_smoothing:
            self.loss_fn = LabelSmoothingNLLLoss(smoothing=0.1)
        else:
            self.loss_fn = nn.NLLLoss()
        self.encoder_params = list(self.encoder.parameters())
        self.other_params = list(self.moe.parameters()) + list(self.loss_fn.parameters())
        self.num_classes = num_classes

    def forward(self, query_img, support_imgs, support_labels):
        """
        Forward pass with K-means prototype aggregation.
        
        Data flow during training/validation:
        1. Encode query_img and support_imgs using ResNet18 encoder
        2. Always aggregate support embeddings using Feature_Aggregation_KMeans:
           - For each class, run K-means clustering to get K prototypes per class
           - Return (total_prototypes, embed_dim) and prototype_labels
        3. Pass query embedding and aggregated prototypes to MoE model
        4. Return log probabilities
        
        The aggregation happens for every forward pass regardless of support set size,
        because we want to test whether multiple prototypes per class improve SPM.
        """
        q_emb = self.encoder(query_img)  # [B, D]
        support_emb = self.encoder(support_imgs)
        
        # Always apply K-means aggregation (key change from original)
        support_emb, support_labels = self.feature_aggregation(support_emb, support_labels)
        
        moe_output = self.moe(q_emb, support_emb, support_labels)
        return torch.log(moe_output.clamp(min=1e-8))
    
    # ORIGINAL TRAINING STEP. 
    # def training_step(self, batch, batch_idx):
    #     query_img, support_imgs, query_label, support_labels = batch
    #     logits = self(query_img, support_imgs, support_labels)
    #     loss = self.loss_fn(logits, query_label)
    #     self.log("train_loss", loss, prog_bar=True)
    #     preds = torch.argmax(logits, dim=-1)
    #     acc = (preds == query_label).float().mean()
    #     self.log("train_acc", acc, prog_bar=True)
    #     return loss
    
    # EXPERIMENTAL TRAINING StEP FOR NLLLoss
    def on_train_epoch_start(self):
        self._train_correct = 0
        self._train_examples = 0
        self._train_loss_sum = 0.0

    def training_step(self, batch, batch_idx):
        query_img, support_imgs, query_label, support_labels = batch

        log_probs = self(
            query_img,
            support_imgs,
            support_labels
        )

        loss = self.loss_fn(log_probs, query_label)

        preds = torch.argmax(log_probs, dim=-1)
        batch_size = query_label.size(0)
        self._train_correct += int((preds == query_label).sum().item())
        self._train_examples += batch_size
        self._train_loss_sum += float(loss.detach().item()) * batch_size

        # These are running epoch aggregates, not per-batch metrics. On the
        # final batch they equal the aggregate values for the whole epoch.
        self.log(
            "train_loss",
            self._train_loss_sum / self._train_examples,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train_acc",
            self._train_correct / self._train_examples,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss
    
    # ORIGINAL VALIDATION STEP
    # def validation_step(self, batch, batch_idx):
    #     query_img, support_imgs, query_label, support_labels = batch
    #     logits = self(query_img, support_imgs, support_labels)
    #     loss = self.loss_fn(logits, query_label)
    #     self.log("val_loss", loss, prog_bar=True)
    #     preds = torch.argmax(logits, dim=-1)
    #     acc = (preds == query_label).float().mean()
    #     self.log("val_acc", acc, prog_bar=True)
    
    
    # EXPERIMENTAL VALIDATION STEP FOR NLLLoss
    def validation_step(self, batch, batch_idx):
        query_img, support_imgs, query_label, support_labels = batch

        log_probs = self(
            query_img,
            support_imgs,
            support_labels
        )

        loss = self.loss_fn(log_probs, query_label)

        self.log("val_loss", loss, prog_bar=True)

        preds = torch.argmax(log_probs, dim=-1)
        acc = (preds == query_label).float().mean()

        self.log("val_acc", acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        """
        Evaluation on test set using aggregation + MoE (main evaluation pathway).
        
        CRITICAL FIX: Each query's k-NN support set is aggregated independently.
        Do NOT mix support sets from different queries together.
        
        Data flow during evaluation (PER QUERY):
        1. Compute query embedding q_i
        2. Retrieve k nearest neighbors of q_i from training set (FAISS)
        3. Apply Feature_Aggregation_KMeans to q_i's k-NN neighbors
           - K-means per class within q_i's support set only
           - Returns K prototypes per class for q_i
        4. Pass q_i and its aggregated prototypes to MoE
        5. Predict and compute accuracy
        
        This ensures:
        - Each query uses its own independent support set clustering
        - Consistent with training: both use K-means aggregation
        - Fair hypothesis testing: multiple prototypes per class
        """
        query_img, query_label = batch
        B = query_img.size(0)
        k = getattr(self, "support_size_eval", 1024)  # default k for evaluation

        # compute query embeddings
        q_emb = self.encoder(query_img)  # (B, D)

        # build FAISS index and load precomputed embeddings once
        if not hasattr(self, "_faiss_index"):
            # Load embeddings and labels from file
            if hasattr(self, "embedding_file"):
                emb_file = getattr(self, "embedding_file", "train_embeddings.pt")
                print(f"Loading embeddings from {emb_file}")
                data = torch.load(emb_file, map_location="cpu")
                embeddings = data["embeddings"]  # Tensor [N, D]
                labels = data["labels"]          # Tensor [N]
            else:
                # compute embeddings from training data
                train_loader = self.trainer.datamodule.train_dataloader()
                emb_list, label_list = [], []
                for imgs, lbls in train_loader:
                    emb = self.encoder(imgs.to(self.device))
                    emb_list.append(emb.detach().cpu())
                    label_list.append(lbls)
                embeddings = torch.cat(emb_list, dim=0)
                labels = torch.cat(label_list, dim=0)

            # Create FAISS index
            emb_np = embeddings.numpy().astype("float32")
            index = faiss.IndexFlatL2(emb_np.shape[1])
            index.add(emb_np)
            self._faiss_index = index
            self._support_embeddings = embeddings.to(self.device)
            self._support_labels = labels.to(self.device)

        # retrieve k nearest neighbors with FAISS
        q_np = q_emb.detach().cpu().numpy().astype("float32")
        dists, idxs = self._faiss_index.search(q_np, k)
        idxs_tensor = torch.from_numpy(idxs).to(self.device)
        
        # gather support embeddings and labels
        supp_embs = self._support_embeddings[idxs_tensor]  # (B, k, D)
        supp_labels = self._support_labels[idxs_tensor]    # (B, k)

        # Process each query's support set independently (CRITICAL FIX)
        # Do NOT flatten all queries' supports together!
        moe_outputs = []
        for b in range(B):
            # Get this query's support set
            query_support_embs = supp_embs[b]  # (k, D)
            query_support_labels = supp_labels[b]  # (k,)
            
            # Apply K-means aggregation to THIS QUERY'S support set only
            aggregated_embs_b, aggregated_labels_b = self.feature_aggregation(
                query_support_embs, query_support_labels
            )
            
            # Forward through MoE for this query
            moe_out_b = self.moe(q_emb[b:b+1], aggregated_embs_b, aggregated_labels_b)
            moe_outputs.append(moe_out_b)
        
        # Concatenate outputs from all queries
        moe_out = torch.cat(moe_outputs, dim=0)  # (B, num_classes)
        logits = torch.log(moe_out.clamp(min=1e-8))

        # compute accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == query_label).float().mean()
        loss = self.loss_fn(logits, query_label)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def load_pretrained_encoder(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        encoder_state = {k.replace("encoder.", ""): v for k, v in state["state_dict"].items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Pretrained encoder loaded from {checkpoint_path}")

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen.")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder parameters unfrozen.")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.encoder_type.lower() == "resnet18" and self.pretrained_imagenet is True:
            encoder_params = self.encoder_params
            other_params = self.other_params
            optimizer = optim.SGD([
                {"params": encoder_params, "lr": self.learning_rate / 100},
                {"params": other_params, "lr": self.learning_rate}
                ], 
                momentum=0.9, 
                weight_decay=5e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            return optimizer
        elif (self.encoder_type.lower() == 'resnet18' and not self.pretrained_imagenet) or self.encoder_type.lower() == "dla":
            optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=5e-4) # Original was 5e-4. In between i changed to 1e-3 for experimenting, trying again with 5e-4. Keeping everything same except for LRReduction
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=5,
                min_lr=1e-5,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.encoder_type.lower() != "cnn":
            optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)# Original was 5e-4. Changed for experimenting.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            return [optimizer], [scheduler]
        else:
            return optim.Adam(params, lr=self.learning_rate)
