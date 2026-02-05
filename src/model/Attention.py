import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttentionLayer(nn.Module):
    """
    Fixed multi-head attention supporting both self-attention and cross-attention
    """
    def __init__(self, k: int, model_d: int, hidden_d: int, n_heads: int, dropout: float = 0.1):
        super(MultiheadAttentionLayer, self).__init__()
        
        assert hidden_d % n_heads == 0, "hidden_d must be divisible by n_heads"
        
        self.model_d = model_d
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.head_d = hidden_d // n_heads
        
        # FIXED: Separate projections for Q, K, V
        self.query_proj = nn.Linear(model_d, hidden_d)
        self.key_proj = nn.Linear(model_d, hidden_d)
        self.value_proj = nn.Linear(model_d, hidden_d)
        
        # Output projection
        self.W_o = nn.Linear(hidden_d, model_d)
        
        self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(model_d)
    
    def scaled_dot_product_att(self, queries, keys, values, attention=True):
        """
        Args:
            queries: [batch, n_heads, seq_len_q, head_d]
            keys: [batch, n_heads, seq_len_k, head_d]
            values: [batch, n_heads, seq_len_v, head_d]
        
        Returns:
            output: [batch, n_heads, seq_len_q, head_d]
            attention_matrix: [batch, n_heads, seq_len_q, seq_len_k]
        """
        d_k = queries.size(-1)
        
        # Compute attention scores
        logits = torch.matmul(queries, keys.transpose(-2, -1))
        logits = logits / math.sqrt(d_k)
        
        # Softmax
        attention_matrix = F.softmax(logits, dim=-1)
        attention_matrix = self.dropout(attention_matrix)
        
        # Apply attention to values
        output = torch.matmul(attention_matrix, values)
        
        if attention:
            return output, attention_matrix
        else:
            return output, None
    
    def forward(self, query, key=None, value=None, attention=True):
        """
        
        Args:
            query: [batch, seq_len_q, model_d] - queries (e.g., all sensors)
            key: [batch, seq_len_k, model_d] - keys (e.g., top-k sensors)
                 If None, defaults to query (self-attention)
            value: [batch, seq_len_v, model_d] - values (e.g., top-k sensors)
                   If None, defaults to query (self-attention)
            attention: whether to return attention weights
        
        Returns:
            output: [batch, seq_len_q, model_d]
            attention_matrix: [batch, n_heads, seq_len_q, seq_len_k] or None
        """
        # Handle 2D input (add batch dimension)
        #if len(query.shape) == 2:
            #query = query.unsqueeze(0)
        
        if key is None:
            key = query
        if value is None:
            value = query
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]
        
        Q = self.query_proj(query)   # [batch, seq_len_q, hidden_d]
        K = self.key_proj(key)       # [batch, seq_len_k, hidden_d]
        V = self.value_proj(value)   # [batch, seq_len_v, hidden_d]
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.head_d).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.head_d).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.n_heads, self.head_d).transpose(1, 2)
        # All: [batch, n_heads, seq_len, head_d]
        
        # Scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_att(Q, K, V, attention)
        # attn_output: [batch, n_heads, seq_len_q, head_d]
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch, seq_len_q, n_heads, head_d]
        
        attn_output = attn_output.view(batch_size, seq_len_q, self.hidden_d)
        # [batch, seq_len_q, hidden_d]
        # Output projection
        output = self.W_o(attn_output)
        # [batch, seq_len_q, model_d]
        
        #output = self.layer_norm(query + self.dropout(output))
        #output = output.unsqueeze(-1)
        if attention:
            return output, attn_weights
        else:
            return output, None













"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


    MultiheadAttentionLayer implements the vanilla multihead self attention from "Attention is all you need" (Vaswani et. al, 2017)

    :param model_d: input dimension
    :param hidden_d: dimension of the query, keys, values
    :param n_heads: number of heads for self-attention


class MultiheadAttentionLayer(nn.Module):
  def __init__(self, hidden_d: int, query_d: int, key_d: int, value_d: int, model_d: int, n_heads: int):
    super(MultiheadAttentionLayer, self).__init__()

    self.query_d = query_d
    self.key_d = key_d
    self.value_d = value_d
    self.n_heads = n_heads
    self.head_d = value_d // n_heads
    self.hidden_d = hidden_d
    self.model_d = model_d
    
    # Separate linear projections for query, key, and value
    self.query = nn.Linear(query_d, hidden_d)
    self.key = nn.Linear(key_d, hidden_d)
    self.value = nn.Linear(value_d, hidden_d)
    self.W_o = nn.Linear(hidden_d, model_d)
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
    #values = values.unsqueeze(1)
    output = torch.matmul(attention_matrix, values)
    if attention:
        return output, attention_matrix
    else:
        return output, None

  def forward(self, x: torch.Tensor, y: torch.Tensor, attention=True) -> (torch.Tensor, torch.Tensor):
    
    if len(x.shape) == 2:
      x = x.unsqueeze(0)

    # compute self attention
    if y is None:
        y = x
    else:
        # compute cross attention
        if len(y.shape) == 2:
          y = y.unsqueeze(0)

    batch_size, q_len, _ = x.size()
    _, k_len, _ = y.size()
    
    # Project to queries, keys, values
    queries = self.query(x)  # [batch, q_len, hidden_d]
    keys = self.key(y)       # [batch, k_len, hidden_d]
    values = self.value(y)   # [batch, k_len, hidden_d]
    
    # Split into heads: [batch, n_heads, seq_len, head_d]
    queries = queries.view(batch_size, q_len, self.n_heads, self.hidden_d).permute(0, 2, 1, 3)
    keys = keys.view(batch_size, k_len, self.n_heads, self.hidden_d).permute(0, 2, 1, 3)
    values = values.view(batch_size, k_len, self.n_heads, self.hidden_d).permute(0, 2, 1, 3)
    
    # Compute scaled dot-product attention
    output, A = self.scaled_dot_product_att(queries, keys, values, attention)
    # Concatenate heads: [batch, q_len, n_heads, head_d] -> [batch, q_len, hidden_d]
    output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, q_len, self.hidden_d)
    
    # Final linear projection
    output = self.W_o(output)

    if attention:
        return output, A
    else:
        return output, None
    


class FeatureGCN(nn.Module):
    def __init__(self, num_features, hidden_features):
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, hidden_features)
        
        # Fully connected adjacency (all features connected)
        # Initialize as learnable or use fixed fully-connected
        self.adj = torch.ones(num_features, num_features)
        
    def forward(self, x):
        Args:
            x: (batch, time_length, features)
        Returns:
            output: (batch, time_length, hidden_features)
        #print("x shape in GCN:", x.shape)
        batch, time_length, features = x.shape
        
        # Reshape to (batch * time_length, features) to process all timesteps
        x_flat = x.reshape(batch * time_length, features)
        
        # Normalize adjacency matrix
        adj = self.adj.to(x.device) + torch.eye(self.num_features, device=x.device)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        norm_adj = degree_inv_sqrt.view(-1, 1) * adj * degree_inv_sqrt.view(1, -1)
        
        # GCN operation: X @ normalized_adj^T (transpose for feature dimension)
        x_flat = torch.mm(x_flat, norm_adj.T)  # (batch*time, features)
        x_flat = self.linear(x_flat)            # (batch*time, hidden_features)
        
        # Reshape back to (batch, time_length, hidden_features)
        output = x_flat.view(batch, time_length, -1)
        #print("output shape in GCN:", output.shape)
        
        return output




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
"""