import torch
import torch.nn as nn
from model.Attention import MultiheadAttentionLayer


class Encoder(nn.Module):

    def __init__(self, model_d: int, num_features: int, hidden_d: int, n_heads: int):
        super(Encoder, self).__init__()
        self.hidden_d = hidden_d
        self.model_d = model_d
        self.num_features = num_features

        # position encoding?
        # self.position_embedding
        # attention layer
        self.attention = MultiheadAttentionLayer(model_d, hidden_d, n_heads)

        # Position-wise Feed Forward Neural Net (Deep Set)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_d, 2 * hidden_d),
            nn.GELU(),
            nn.Linear(2 * hidden_d, hidden_d)
        )
        #self.projection = nn.Linear(num_features, hidden_d)
        # Layer Norm
        self.layer_norm1 = nn.LayerNorm(hidden_d)
        #self.layer_norm2 = nn.LayerNorm(hidden_d)


    # for anomaly detection
    def compute_attention(self, x: torch.Tensor) -> torch.Tensor:
        # compute self attention
        y, A = self.attention(x)
        self.A = A
        return y
    
    # for forecasting
    def project_attention(self, y: torch.Tensor) -> torch.Tensor:
        pass

