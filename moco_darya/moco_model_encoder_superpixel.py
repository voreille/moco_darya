import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_resnet_weights(base_encoder):
    resnet_weights_map = {
        "resnet18": ResNet18_Weights.DEFAULT,
        "resnet34": ResNet34_Weights.DEFAULT,
        "resnet50": ResNet50_Weights.DEFAULT,
        "resnet101": ResNet101_Weights.DEFAULT,
        "resnet152": ResNet152_Weights.DEFAULT,
    }
    base_encoder = base_encoder.lower()
    if base_encoder in resnet_weights_map:
        return resnet_weights_map[base_encoder]
    else:
        raise ValueError(f"Unsupported base_encoder: {base_encoder}. Supported values: {list(resnet_weights_map.keys())}")

class MoCoV2Encoder(nn.Module):
    def __init__(self, base_encoder='resnet50', output_dim=128, queue_size=65536, momentum=0.999, temperature=0.07):
        super(MoCoV2Encoder, self).__init__()

        # Load encoders
        self.encoder_q = self._load_resnet(base_encoder, output_dim)
        self.encoder_k = self._load_resnet(base_encoder, output_dim)

        self.temperature = temperature
        self.momentum = momentum

        # Initialize a single queue for both augmented and neighbor features
        self.register_buffer("queue", torch.randn(queue_size, output_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Initialize key encoder parameters
        self._initialize_momentum_encoder()
        logger.info(f"MoCoV2Encoder initialized with base encoder: {base_encoder}, output_dim: {output_dim}, queue_size: {queue_size}, momentum: {momentum}")

    def _load_resnet(self, base_encoder, output_dim, pretrained=False):
        try:
            weight_type = "not default"
            weights = None
            if pretrained:
                weights = get_resnet_weights(base_encoder)
                weight_type = "default"

            encoder = getattr(models, base_encoder)(weights=weights)
            logger.info(f"Using {weight_type} weights of ResNet")

        except AttributeError:
            raise ValueError(f"Unsupported ResNet variant '{base_encoder}'.")

        # Projection head
        hidden_dim = encoder.fc.in_features
        encoder.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        logger.info(f"Loaded {base_encoder} with output dimension: {output_dim}")
        return encoder

    def _initialize_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, neighbor_keys):
        """
        Update the queue by adding both keys and neighbor_keys in a single queue.
        
        Parameters:
        - keys: Tensor of shape (batch_size, output_dim), embeddings of augmented patches.
        - neighbor_keys: Tensor of shape (batch_size, output_dim), embeddings of neighboring patches.
        """
        batch_size = keys.shape[0]
        combined_keys = torch.cat([keys, neighbor_keys], dim=0)  # Concatenate keys and neighbor keys

        ptr = int(self.queue_ptr)
        total_batch_size = combined_keys.shape[0]  # Now 2 * batch_size

        # Update queue with new embeddings
        end_ptr = ptr + total_batch_size
        if end_ptr <= self.queue.size(0):
            self.queue[ptr:end_ptr, :] = combined_keys
        else:
            first_part = self.queue.size(0) - ptr
            self.queue[ptr:, :] = combined_keys[:first_part, :]
            self.queue[:end_ptr % self.queue.size(0), :] = combined_keys[first_part:, :]

        # Update queue pointer
        ptr = (ptr + batch_size) % self.queue.size(0)
        self.queue_ptr[0] = ptr

    def forward(self, x_q, x_k1, x_k2):
        """
        Forward pass for query and key inputs.
        - x_q: original patch
        - x_k1: augmented version of x_q (same patch, different augmentations)
        - x_k2: neighboring patch (spatially close but different)
        """
        q = self.encoder_q(x_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k1 = self.encoder_k(x_k1)
            k1 = F.normalize(k1, dim=1)

            k2 = self.encoder_k(x_k2)
            k2 = F.normalize(k2, dim=1)

        return q, k1, k2

    def update_queue(self, keys, neighbor_keys):
        """
        Update the queue with keys and neighbor_keys.
        """
        self._dequeue_and_enqueue(keys, neighbor_keys)