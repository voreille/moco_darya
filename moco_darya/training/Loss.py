import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseContrastiveLoss(nn.Module):
    def __init__(self):
        """
        Base class for contrastive loss functions.
        """
        super(BaseContrastiveLoss, self).__init__()

    def loss_orginal(self, q, k, queue):
        """
        Compute the original contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings.
        - k (torch.Tensor): Positive key embeddings.
        - queue (torch.Tensor): Negative key embeddings.

        Returns:
        - loss (torch.Tensor): Contrastive loss.
        """
        # Positive logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach().T])

        # Combine logits and apply temperature scaling
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # Ground truth labels (positive key is at index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss


class ContrastiveLoss(BaseContrastiveLoss):
    def __init__(self, temperature=0.07):
        """
        Initialize the contrastive loss.

        Parameters:
        - temperature (float): Temperature scaling for the logits.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, queue):
        """
        Compute the contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings.
        - k (torch.Tensor): Positive key embeddings.
        - queue (torch.Tensor): Negative key embeddings.

        Returns:
        - loss (torch.Tensor): Computed contrastive loss.
        """
        return self.loss_orginal(q, k, queue)


class MultiHeadContrastiveLoss(BaseContrastiveLoss):
    def __init__(self, temperature=0.07, alpha=0.5):
        """
        Multi-head contrastive loss for MoCo with neighbor augmentation.

        Parameters:
        - temperature (float): Temperature scaling for the logits.
        - alpha (float): Weight for the central tile loss in the combined loss.
        """
        super(MultiHeadContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def loss_neighbor(self, q, k_neighbors, queue, valid_neighbors_count):
        """
        Compute the neighbor contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings.
        - k_neighbors (torch.Tensor): Neighbor key embeddings.
        - queue (torch.Tensor): Negative key embeddings.
        - valid_neighbors_count (torch.Tensor): Number of valid neighbors.

        Returns:
        - loss (torch.Tensor): Neighbor contrastive loss.
        """
        # Positive logits for neighbors
        l_pos_neighbors = torch.einsum("nc,nkc->nk", [q, k_neighbors])

        # Average over valid neighbors
        l_pos_neighbors = l_pos_neighbors.sum(dim=1, keepdim=True) / valid_neighbors_count.unsqueeze(1).clamp(min=1)

        # Negative logits
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach().T])

        # Combine logits and apply temperature scaling
        logits_neighbors = torch.cat([l_pos_neighbors, l_neg], dim=1)
        logits_neighbors /= self.temperature

        # Ground truth labels (positive key is at index 0)
        labels = torch.zeros(logits_neighbors.shape[0], dtype=torch.long, device=logits_neighbors.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits_neighbors, labels)
        return loss

    def forward(self, q, k_center, k_neighbors, queue, valid_neighbors_count):
        """
        Compute the multi-head contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings.
        - k_center (torch.Tensor): Center key embeddings.
        - k_neighbors (torch.Tensor): Neighbor key embeddings.
        - queue (torch.Tensor): Negative key embeddings.
        - valid_neighbors_count (torch.Tensor): Number of valid neighbors.

        Returns:
        - loss (torch.Tensor): Combined multi-head contrastive loss.
        """
        # Center loss
        loss_center = self.loss_orginal(q, k_center, queue)

        # Neighbor loss
        loss_neighbors = self.loss_neighbor(q, k_neighbors, queue, valid_neighbors_count)

        # Combine losses
        loss = self.alpha * loss_center + (1 - self.alpha) * loss_neighbors
        return loss


class MocoConvnetLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Contrastive loss for MoCo using ConvNeXt as the encoder.

        Parameters:
        - temperature (float): Temperature scaling for the logits.
        """
        super(MocoConvnetLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, queue):
        """
        Compute the contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings of shape (batch_size, output_dim).
        - k (torch.Tensor): Key embeddings of shape (batch_size, output_dim).
        - queue (torch.Tensor): Negative key embeddings of shape (queue_size, output_dim).

        Returns:
        - loss (torch.Tensor): Contrastive loss value.
        """
        # Ensure embeddings are normalized
        q = F.normalize(q, dim=1)  # (batch_size, output_dim)
        k = F.normalize(k, dim=1)  # (batch_size, output_dim)
        queue = F.normalize(queue, dim=1)  # (queue_size, output_dim)

        # Positive logits: similarity between q and k
        pos_logits = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)  # (batch_size, 1)

        # Negative logits: similarity between q and queue
        neg_logits = torch.einsum("nc,mc->nm", [q, queue])  # (batch_size, queue_size)

        # Concatenate positive and negative logits
        logits = torch.cat([pos_logits, neg_logits], dim=1)  # (batch_size, 1 + queue_size)

        # Apply temperature scaling
        logits /= self.temperature

        # Labels: positive samples are at index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss     


class ContrastiveLossWithNeighbors(BaseContrastiveLoss):
    def __init__(self, temperature=0.07, alpha=0.5):
        """
        Contrastive loss function that combines both augmentation-based and neighbor-based contrastive loss.

        Parameters:
        - temperature (float): Temperature scaling for contrastive loss.
        - alpha (float): Weighting factor for the augmentation-based loss (0 ≤ alpha ≤ 1).
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, q, k1, k2, queue):
        """
        Compute the contrastive loss using both augmentation-based and neighbor-based positives.

        Parameters:
        - q (torch.Tensor): Query embeddings.
        - k1 (torch.Tensor): Positive embeddings (augmented version of q).
        - k2 (torch.Tensor): Positive embeddings (neighbor tile).
        - queue (torch.Tensor): Negative key embeddings.

        Returns:
        - loss (torch.Tensor): Weighted contrastive loss.
        """
        # Compute the original contrastive loss separately for k1 and k2
        loss_tile = self.loss_orginal(q, k1, queue)  # Augmentation-based contrastive loss
        loss_neighbor = self.loss_orginal(q, k2, queue)  # Neighbor-based contrastive loss

        # Weighted combination
        loss = self.alpha * loss_tile + (1 - self.alpha) * loss_neighbor

        return loss           