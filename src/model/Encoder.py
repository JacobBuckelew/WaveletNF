import torch
import torch.nn as nn
from model.Attention import MultiheadAttentionLayer
from model.Wavelet import DiscreteWaveletTransform
import torch.nn.functional as F

class TopKSelector(nn.Module):

    def __init__(self, k: int, window_length: int):
        super(TopKSelector, self).__init__()
        self.k = k

        self.temporal_proj = nn.Linear(window_length, 1)

    def compute_scores(self, D: torch.Tensor) -> torch.Tensor:

        #energy = (D ** 2).sum(dim=-1)
        scores = self.temporal_proj(D)
        #print("scores shape", scores.shape)

        return scores.squeeze()
    def forward(self, D: torch.Tensor) -> torch.Tensor:
        # D shape: Batch, Sensor, Seq_Len, Hidden_D
        # compute importance scores for each feature
        scores = self.compute_scores(D)
        # get top k indices
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(0)
        #print("scores shape:", scores.shape)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=1)
        #print(top_k_indices)
        batch_idx = torch.arange(D.shape[0], device=D.device).unsqueeze(1).expand(-1, self.k)
        top_k_embeds = D[batch_idx, top_k_indices]
        return top_k_embeds, top_k_indices, F.sigmoid(scores)
    

class Encoder(nn.Module):

    def __init__(self, num_sensors: int, model_d: int, window_length: int, hidden_d: int, n_heads: int, wavelet: str, k: int, dropout: float = 0.1,
                 use_wavelet=True, use_attention=True):
        super(Encoder, self).__init__()
        self.hidden_d = hidden_d
        self.model_d = model_d
        #self.num_features = num_features
        self.wavelet = wavelet
        self.num_sensors = num_sensors
        self.k = k
        self.window_length = window_length
        self.use_wavelet = use_wavelet
        self.use_attention = use_attention
        # attention layer
        if use_attention:
            self.cross_attention = MultiheadAttentionLayer(k, window_length, hidden_d, n_heads)

        #self.norm_A = nn.LayerNorm(window_length)
        #self.norm_D = nn.LayerNorm(window_length)
        #self.wavelet_bases = []
        # wavelet transform
        #for w in wavelets:
            #setattr(self, 'dwt_' + w, DiscreteWaveletTransform(w, window_length))
        #self.wavelet_bases.append(DiscreteWaveletTransform(w, window_length))
        if use_wavelet:
            self.wavelet_transform = DiscreteWaveletTransform(wavelet, window_length)
        # Wavelet Fusion Block
        #self.wavelet_fusion = WaveletFusionBlock(model_d, hidden_d, wavelets)

        # Approximation Coefficient Temporal Embedding
        self.rnn = nn.GRU(input_size=num_sensors,hidden_size=hidden_d,batch_first=True, dropout=dropout)

        # Top-K selector
        if self.use_attention:
            self.top_k_selector = TopKSelector(k, window_length)

        # Final GNN Layer
            self.cross_att_embedding = nn.Linear(num_sensors, hidden_d)
        self.temporal_embedding = nn.Linear(hidden_d, hidden_d)
        self.output_proj = nn.Linear(hidden_d, hidden_d)
        #self.gnn = GNNLayer(num_sensors, hidden_d, hidden_d)


    # for anomaly detection
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_shape = x.shape
        # compute wavelet transforms for each basis
        #D_list = []
        #A_list = []
        #x = x.squeeze()
        #for w in self.wavelet_bases:
        if self.use_wavelet:
            A, D = self.wavelet_transform(x.transpose(1,2))
            if full_shape[0] == 1:
                A = A.unsqueeze(0)

        #A = self.norm_A(A)
        #D = self.norm_D(D)
        #print("A shape after DWT:", A.shape)
        #A = A.transpose(1,2)
            #D_list.append(D.unsqueeze(-1))
            #A_list.append(A.unsqueeze(-1))
        # fuse wavelet coefficients

        #D, A = self.wavelet_fusion(D_list, A_list)
            D = D.squeeze()
            A = A.reshape((A.shape[0] * A.shape[2], A.shape[1]))
        # compute temporal embedding of approximation coefficients using RNN
            A, _ = self.rnn(A)
        else:
            #print("x.shape;", x.shape)
            A, _ = self.rnn(x.reshape((x.shape[0] * x.shape[1], x.shape[2])))
            D = x.transpose(1,2)
            #print("D shape:", D.shape)
        #print("A shape after rnn:", A.shape)

        A = A.reshape((full_shape[0], full_shape[1], A.shape[1]))

        #print("A shape after rnn:", A.shape)
        # find top k important features based on detail coeff embeddings
        if self.use_attention:
            if len(D.shape) == 2:
                D = D.unsqueeze(0)
            D_topk, top_k_idx, scores = self.top_k_selector(D)
            self.top_k_idx = top_k_idx
        #print("H shape before cross attention:", H.shape)

        # cross attention computes intra-variate correlations using the detail coeff embeddings as queries

            cross_att, att_scores = self.cross_attention(D, D_topk, D_topk, attention=True)
            self.att_scores = att_scores

            cross_att = cross_att * scores.unsqueeze(-1)
        #cross_att = c
        #print("Cross att shape:", cross_att.shape)

        #print("cross att shape:", cross_att.shape)
        # compute GNN Layer combining cross-attention with temporal embedding)
        #H = self.gnn(cross_att, A)
        
        # cocmbine cross att with temporal embedding to get spatio-temporal representation
        #H = cross_att + A
        # spatiotemporal embedding
            H = self.output_proj(F.relu(self.cross_att_embedding(cross_att.transpose(1,2)) + self.temporal_embedding(A)))
        else:
            H = self.output_proj(F.relu(self.temporal_embedding(A)))
        #print("final H shape:", H.shape)
        return H



"""
import torch
import torch.nn as nn
from model.Attention import MultiheadAttentionLayer, FeatureGCN


class Encoder(nn.Module):

    def __init__(self, model_d: int, window_length: int, num_features: int, hidden_d: int, n_heads: int):
        super(Encoder, self).__init__()
        self.hidden_d = hidden_d
        self.model_d = model_d
        self.window_length = window_length
        self.num_features = num_features

        # position encoding?
        # self.position_embedding
        # attention layer
        #self.self_attention = MultiheadAttentionLayer(hidden_d, hidden_d, hidden_d, hidden_d, window_length, n_heads)
        self.feature_gcn = FeatureGCN(num_features, num_features)
        self.cross_attention = MultiheadAttentionLayer(hidden_d, num_features, num_features, num_features, num_features, n_heads)
        self.d_embedding = nn.Linear(window_length, hidden_d)

        self.layer_norm_1 = nn.LayerNorm(num_features)
        self.layer_norm_2 = nn.LayerNorm(num_features)
        self.layer_norm_3 = nn.LayerNorm(num_features)

        self.final_projection = nn.Linear(num_features, num_features)
        self.ffn = nn.Sequential(
            nn.Linear(num_features, hidden_d),
            nn.GELU(),
            nn.Linear(hidden_d, num_features)
        )
        # Position-wise Feed Forward Neural Net (Deep Set)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_d, 2 * hidden_d),
            nn.GELU(),
            nn.Linear(2 * hidden_d, hidden_d)
        )
        #self.projection = nn.Linear(num_features, hidden_d)
        # Layer Norm
        self.layer_norm1 = nn.LayerNorm(model_d)
        #self.layer_norm2 = nn.LayerNorm(hidden_d)

    def forward(self, D: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # compute initial embedding of D and its 
        #print("D shape:", D.shape)
        #d_initial_embed = self.d_embedding(D)

        # compute inter-variate correlations using self-attention on d_embed and apply layer Norm
        #d_embed, A = self.self_attention(D, y=None)
        D = torch.transpose(D, 1, 2)
        d_embed = self.feature_gcn(D)
        self.A = None
        d_embed = self.layer_norm_1(d_embed)

        #d_embed = torch.transpose(d_embed, 1, 2)
        #print("d embed after transpose:", d_embed.shape)
        y = torch.transpose(y, 1, 2)
        # cross attention computes intra-variate temporal correlations using the detail coeff embeddings as queries
        cross_att, B = self.cross_attention(d_embed, y, attention=None)
        #self.A = B
        cross_att = cross_att + d_embed

        #cross_att = self.layer_norm_2(cross_att)

        # feed forward network

        #output = self.ffn(cross_att) + cross_att

        #output = self.layer_norm_3(output)

        # final projection
        #output = self.final_projection(cross_att)


        return cross_att


    # for anomaly detection
    def compute_attention(self, x: torch.Tensor) -> torch.Tensor:
        # compute self attention
        y, A = self.attention(x)
        #print(y.shape)
        #y = self.layer_norm1(y)
        self.A = A
        #print(y)
        return y
    
    # for forecasting
    def project_attention(self, y: torch.Tensor) -> torch.Tensor:
        pass
        





    class GNNLayer(nn.Module):
    def __init__(self, n: int, input_size: int, hidden_size: int):
        super(GNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(1, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        self.A = torch.ones(n, n)

    def forward(self, cross_att: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # cross_att shape: Batch, Sensor, Seq_Len, Hidden_D
        # A shape: Sensor, Sensor
        # H shape: Batch, Sensor, Seq_Len, Hidden_D
        A = self.A.to(H.device)
        h_n = self.activation(self.linear_1(torch.einsum('nkld,kj->njld',H,A)) + self.linear_2(cross_att))
        return self.output_proj(h_n)


class WaveletFusionBlock(nn.Module):

    def __init__(self, model_d: int, hidden_d: int, wavelets: list):
        super(WaveletFusionBlock, self).__init__()
        self.hidden_d = hidden_d
        self.model_d = model_d
        self.wavelets = wavelets
        self.n_bases = len(wavelets)

        # Detail Coefficient Embedding Layer

        self.detail_embedding = nn.Sequential(nn.Linear(model_d, hidden_d),
                                         nn.ReLU(),
                                        nn.Linear(hidden_d, model_d))

        # Approximation Coefficient Embedding Layer

        self.approximation_embedding = nn.Sequential(nn.Linear(model_d, hidden_d),
                                                nn.ReLU(),
                                                nn.Linear(hidden_d, hidden_d))
        

        # define fusion parameters for detail and approximation coefficients

        self.detail_fusion_weights = nn.Parameter(torch.ones(len(wavelets)))
        self.approximation_fusion_weights = nn.Parameter(torch.ones(len(wavelets)))

    
    def forward(self, D_list: list, A_list: list) -> torch.Tensor:

        # for each basis, compute embeddings for D and A
        D_embeds = []
        A_embeds = []
        for i in range(len(self.wavelets)):
            D_embed = self.detail_embedding(D_list[i])
            A_embed = self.approximation_embedding(A_list[i])
            D_embeds.append(D_embed)
            A_embeds.append(A_embed)

        # stack 
        D_embeds = torch.stack(D_embeds)
        A_embeds = torch.stack(A_embeds)

        # normalize fusion weights
        detail_weights = F.softmax(self.detail_fusion_weights, dim=0)
        approximation_weights = F.softmax(self.approximation_fusion_weights, dim=0)
        # compute weighted sum
        
        detail_weights = detail_weights.view(self.n_bases, 1, 1, 1, 1)
        approx_weights = approximation_weights.view(self.n_bases, 1, 1, 1, 1)
        D_embeds = (D_embeds * detail_weights).sum(dim=0)
        A_embeds = (A_embeds * approx_weights).sum(dim=0)


        return D_embeds, A_embeds

"""