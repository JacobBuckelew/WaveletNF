import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MultiheadAttentionLayer implements the vanilla multihead self attention from "Attention is all you need" (Vaswani et. al, 2017)

    :param model_d: input dimension
    :param hidden_d: dimension of the query, keys, values
    :param n_heads: number of heads for self-attention

"""
class MultiheadAttentionLayer(nn.Module):
  def __init__(self, model_d: int, hidden_d, n_heads: int):
    super(MultiheadAttentionLayer, self).__init__()
    self.model_d = model_d
    self.hidden_d = hidden_d
    self.n_heads = n_heads
    self.head_d = hidden_d // n_heads
    self.head = nn.Linear(model_d, 2 * hidden_d)
    self.W_o = nn.Linear(hidden_d, model_d)
    self.value = nn.Linear(model_d, hidden_d)


  def scaled_dot_product_att(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attention=True) -> (torch.Tensor, torch.Tensor):  
    d_k = queries.size()[-1]
    logits = torch.matmul(queries, keys.transpose(-2, -1))
    logits = logits/math.sqrt(d_k)

    #attention_matrix = logits
    attention_matrix = F.softmax(logits, dim=-1)
    values = values.unsqueeze(1)
    output = torch.matmul(attention_matrix, values)
    if attention:
        return output, attention_matrix
    else:
        return output, None

  def forward(self, x: torch.Tensor, attention=True) -> (torch.Tensor, torch.Tensor):
    if len(x.shape) == 2:
      x = x.unsqueeze(0)
    shape = x.size()
    x_shape = x
    #print(x_shape.shape)
    batch, length, dim = x_shape.size()
    #print("x shape:", x_shape.shape)
    projection = self.head(x_shape)

    projection = projection.reshape(batch, length, self.n_heads, 2 * self.head_d)
    projection = projection.permute(0,2,1,3)
    queries, keys = projection.chunk(2, dim=-1)
    values = self.value(x_shape)
    # call scaled dot production attention
    values, A = self.scaled_dot_product_att(queries, keys, values, attention)
    values = values.permute(0,2,1,3)
    values = values.reshape(batch, length, self.hidden_d)
    output = self.W_o(values)
    #print("output:", output.shape)
    if attention:
        return output, A
    else:
        return output, None
    