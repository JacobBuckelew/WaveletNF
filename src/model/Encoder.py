import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, model_d: int, hidden_d, n_heads: int):
        super(Encoder, self).__init__()
        self.hidden_d = hidden_d
        self.model_d = model_d

        # position encoding?
        # self.position_embedding
        # attention layer
        self.attention = MultiheadAttentionLayer(model_d, hidden_d, n_heads)

        # Position-wise Feed Forward Neural Net (Deep Set)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_d, hidden_d)
        )
        # Layer Norm
        self.layer_norm1 = nn.LayerNorm(hidden_d)
        self.layer_norm2 = nn.LayerNorm(hidden_d)


    # for anomaly detection
    def compute_attention(self, x: torch.Tensor) -> Torch.Tensor:
        # compute self attention
        y = self.attention(x)
        # store attention for future forecasting 
        self.attention_output = y
        return y
        pass
    
    # for forecasting
    def project_attention(self, y: torch.Tensor) -> Torch.Tensor:
        pass

