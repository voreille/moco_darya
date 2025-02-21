import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_convnext_weights(base_encoder):
    """
    Get the appropriate ConvNeXt weights class based on the base_encoder string.

    Parameters:
    - base_encoder (str): The name of the ConvNeXt model (e.g., "convnext_tiny", "convnext_base").

    Returns:
    - weights (torchvision.models.weights): The corresponding ConvNeXt weights class.
    """
    convnext_weights_map = {
        "convnext_tiny": ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_small": ConvNeXt_Small_Weights.DEFAULT,
        "convnext_base": ConvNeXt_Base_Weights.DEFAULT,
        "convnext_large": ConvNeXt_Large_Weights.DEFAULT,
    }

    base_encoder = base_encoder.lower()

    if base_encoder in convnext_weights_map:
        return convnext_weights_map[base_encoder]
    else:
        raise ValueError(f"Unsupported base_encoder: {base_encoder}. Supported values are: {list(convnext_weights_map.keys())}")

class MoCoV2ConvNeXt(nn.Module):
    def __init__(self, base_encoder='convnext_base', output_dim=128, queue_size=65536, momentum=0.999, temperature=0.07):
        """
        Initialize MoCoV2 with ConvNeXt as the backbone.

        Parameters:
        - base_encoder: str, specifies the ConvNeXt variant to use ('convnext_tiny', 'convnext_base', etc.)
        - output_dim: int, the dimension of the output features after the projection head
        - queue_size: int, the size of the queue for negative samples
        - momentum: float, the momentum for updating the key encoder
        - temperature: float, temperature scaling for contrastive loss
        """
        super(MoCoV2ConvNeXt, self).__init__()

        self.encoder_q = self._load_convnext(base_encoder, output_dim)
        self.encoder_k = self._load_convnext(base_encoder, output_dim)

        self.temperature = temperature
        self.momentum = momentum

        # Initialize queue for negative samples
        self.register_buffer("queue", torch.randn(queue_size, output_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.queue_ptr = 0

        # Initialize the momentum encoder parameters
        self._initialize_momentum_encoder()
        logger.info(f"MoCoV2ConvNeXt initialized with base encoder: {base_encoder}, output dimension: {output_dim}, queue size: {queue_size}, momentum: {momentum}")

    def _load_convnext(self, base_encoder, output_dim, pretrained=False):
        """
        Load the specified ConvNeXt model and add a projection head.

        Parameters:
        - base_encoder: str, the name of the ConvNeXt model to load
        - output_dim: int, the dimension of the output features after projection head
        
        Returns:
        - encoder: nn.Module, the modified ConvNeXt model with a projection head
        """
        try:
            weight_type = "not default"
            weights = None
            if pretrained:
                weights = get_convnext_weights(base_encoder)  # Get weights for the specific ConvNeXt
                weight_type = "default"

            encoder = getattr(models, base_encoder)(weights=weights)
            logger.info(f"Using {weight_type} weights for ConvNeXt")
            
        except AttributeError:
            raise ValueError(f"Unsupported ConvNeXt variant '{base_encoder}'. Choose from 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'")

        # Projector MLP
        hidden_dim = encoder.classifier[2].in_features
        encoder.classifier[2] = nn.Identity()  # Remove original classifier
        encoder.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        logger.info(f"Loaded {base_encoder} with output dimension: {output_dim}")
        return encoder

    def _initialize_momentum_encoder(self):
        """
        Initialize the momentum encoder by copying parameters from the query encoder.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Update the momentum encoder parameters using momentum update.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update the queue with the given keys.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        end_ptr = ptr + batch_size
        if end_ptr <= self.queue.size(0):
            self.queue[ptr:end_ptr, :] = keys
        else:
            first_part = self.queue.size(0) - ptr
            self.queue[ptr:, :] = keys[:first_part, :]
            self.queue[:end_ptr % self.queue.size(0), :] = keys[first_part:, :]

        self.queue_ptr = (ptr + batch_size) % self.queue.size(0)

    def forward(self, x_q, x_k):
        """
        Forward pass for query and key inputs.

        Parameters:
        - x_q: torch.Tensor, input tensor for query
        - x_k: torch.Tensor, input tensor for key

        Returns:
        - q: torch.Tensor, normalized feature vector from the query encoder
        - k: torch.Tensor, normalized feature vector from the key encoder
        """
        # Query embeddings
        q = self.encoder_q(x_q)
        q = self.encoder_q.projector(q)  # Pass through projection head
        q = F.normalize(q, dim=1)

        # Key embeddings
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(x_k)
            k = self.encoder_k.projector(k)  # Pass through projection head
            k = F.normalize(k, dim=1)

        # Debugging shapes
        #print(f"q shape: {q.shape}, k shape: {k.shape}, queue shape: {self.queue.shape}")

        return q, k

    def update_queue(self, keys):
        """
        Public method to update the queue with keys after each forward pass.
        
        Parameters:
        - keys: torch.Tensor, the keys (feature vectors) to add to the queue
        """
        self._dequeue_and_enqueue(keys)