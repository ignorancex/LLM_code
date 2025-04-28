import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import maxscore, zscore, BasicConv1D, Inception1DResidualBlock, SwapLengthDepth

########################################################################################################################
################################################# Deep Set fot IID data ################################################
########################################################################################################################
class InstanceEmbedder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InstanceEmbedder, self).__init__()
        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(output_dim, output_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.embedder(x)


class DeepSetSummary(nn.Module):
    def __init__(self, input_dim, embedding_dim, rho=None, phi=None):
        super(DeepSetSummary, self).__init__()
        if phi == None:
            self.instance_embedder = InstanceEmbedder(input_dim, embedding_dim)
        else:
            self.instance_embedder = phi(input_dim, embedding_dim)
        if rho == None:
            self.rho = nn.Sequential(
                nn.Linear(embedding_dim + 2 * input_dim + 1, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            self.rho = rho(embedding_dim + 2 * input_dim + 1, embedding_dim)

        # Learnable log scaling factors for mean/std deviation of each feature, sqrt(sample_size)
        self.log_mean_scale = nn.Parameter(torch.zeros(input_dim) - 2.0)
        self.log_std_scale = nn.Parameter(torch.zeros(input_dim) - 2.0)
        self.log_n_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, n_samples, n_features = x.shape

        # Apply z-score normalization
        normalized_x, (mean, std_dev) = zscore(x, dim=1)
        # it helps for normal gamma
        normalized_x, (_, _) = maxscore(x, dim=1)

        # Flatten mean and std_dev if input_dim is 1
        if n_features == 1:
            mean = mean.view(batch_size, -1)
            std_dev = std_dev.view(batch_size, -1)

        # Apply learnable scaling to mean, log std deviation, log sample_size
        scaled_mean = torch.exp(self.log_mean_scale) * mean  # Shape: [batch_size, n_features]
        scaled_std = torch.exp(self.log_std_scale) * torch.log(std_dev)  # Shape: [batch_size, n_features]

        # Handle n_samples (size of each point cloud)
        # Expand the scaled_logn to have the shape [batch_size, 1]
        scaled_n = torch.exp(self.log_n_scale) * torch.sqrt(torch.tensor(n_samples, dtype=torch.float32)).to(x.device)
        scaled_n_expanded = scaled_n.expand(batch_size, 1)

        # Reshape the normalized input tensor to [batch_size * n_samples, n_features]
        reshaped_input = normalized_x.view(-1, n_features)

        # Efficiently calculate embeddings for each z-scored instance across all items in batch
        embeddings = self.instance_embedder(reshaped_input)

        # Reshape embeddings back to [batch_size, n_samples, embedding_dim]
        embeddings = embeddings.view(batch_size, n_samples, -1)

        # Average the embedding vectors across all instances in each dataset
        summed_embeddings = embeddings.mean(dim=1)

        # Append the mean, std, and scaled n_samples to the summed embeddings
        summed_embeddings = torch.cat((summed_embeddings, scaled_mean, scaled_std, scaled_n_expanded), dim=1)
        summary_embeddings = self.rho(summed_embeddings)

        return summary_embeddings


########################################################################################################################
############################################ Bayes Flow for time series data ###########################################
########################################################################################################################
class BiLSTMEncoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 embedding_dim: int,  # these dimensions are split over forward/backward directions
                 n_layers_LSTM: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim

        assert embedding_dim % 2 == 0, "Embedding dimension must be an even number in a bidirectional LSTM."

        # the LSTM layer encodes the sequence using a forward and backward LSTM,
        # each having dimension embedding_dim/2.
        # The reducer is just a linear projection back down to embedding_dim dimensions
        self.bilstm = nn.LSTM(input_size=in_dim, hidden_size=embedding_dim // 2,
                              num_layers=n_layers_LSTM, batch_first=True, bidirectional=True)

    def forward(self, X):
        # input:    X = (batch_size, seq_length, num_features)
        # output:   out = (batch_size, seq_length, embedding_dim)
        out, (_, _) = self.bilstm(X)
        return out

class SummaryLastBiLSTMState(nn.Module):
    # dim_in: the dimension of the embedding, includes BOTH directions of the biLSTM
    def __init__(self, dim_in):
        super().__init__()
        assert dim_in % 2 == 0, "Embedding dimension must be an even number in a bidirectional LSTM."
        self.dim_in = dim_in
        self.half_dim = dim_in // 2

    def forward(self, X):
        # assumes LSTM encoding has batch_first = True
        if len(X.shape) == 3:
            # batched input, typical case
            last_forward_state = X[:, -1, :self.half_dim]
            last_reverse_state = X[:, 0, self.half_dim:]
        elif len(X.shape) == 2:
            last_forward_state = X[-1, :self.half_dim]
            last_reverse_state = X[0, self.half_dim:]
        else:
            raise Exception("Shape of input tensor must be 2 (unbatched input) or 3 (batched input).")

        out = torch.cat((last_forward_state, last_reverse_state), dim=1)
        return out

class BayesFlowEncoder(torch.nn.Module):
    def __init__(self, y_dim, n_summaries, n_conv_filters=16, n_layers_LSTM=2):
        super().__init__()
        self.encoder = torch.nn.Sequential(SwapLengthDepth(),
                                BasicConv1D(y_dim, 4 * y_dim, kernel_size=1),
                                Inception1DResidualBlock(in_dim=4 * y_dim, out_dim=n_summaries,
                                                         n_filters=n_conv_filters),
                                SwapLengthDepth(),
                                BiLSTMEncoder(n_summaries, n_summaries, n_layers_LSTM),
                                SummaryLastBiLSTMState(dim_in=n_summaries)
                                )
    def forward(self,y):
        return self.encoder(y)





########################################################################################################################
############################################ Set Transformer for IID data ##############################################
########################################################################################################################

class SetEmbedderClean(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_seeds, max_len=None, num_inducing=0,
                 num_attention_blocks=4, ff=None):
        super().__init__()

        # encoding the raw data set: an instance embedding + two set-attention blocks
        # the instance embedder produces nonlinear features for each observation in a data set
        # the two SAB modules produce higher-order interactions of these instance-level features
        if num_inducing <= 0:
            attention_blocks = [SABClean(embedding_dim, num_heads) for _ in range(num_attention_blocks)]
        else:
            attention_blocks = [ISABClean(embedding_dim, num_heads, num_inducing) for _ in range(num_attention_blocks)]

        self.pos = max_len if max_len == None else PositionalEncoding(input_dim, max_len)

        self.encoder = nn.Sequential(
            InstanceEmbedder(input_dim, embedding_dim),
            *attention_blocks
        )

        # decoder layer:  rFF(SAB(PMAk(X)))
        self.pma = PMAClean(embedding_dim, num_heads, num_seeds)
        self.sab = nn.Sequential(
            SABClean(embedding_dim, num_heads, norm="none"),
            SABClean(embedding_dim, num_heads, norm="none"),
            SABClean(embedding_dim, num_heads, norm="none")
        )

        # this layer mixes over the fixed summary stats (mean, std_dev, n) and the learned stats
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim * num_seeds + 2 * input_dim + 1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Learnable log scaling factors for mean/std deviation of each feature, log(sample_size)
        self.log_mean_scale = nn.Parameter(torch.zeros(input_dim) - 2.0)
        self.log_std_scale = nn.Parameter(torch.zeros(input_dim) - 2.0)
        self.log_n_scale = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        batch_size, n_samples, n_features = X.shape

        # Apply z-score normalization
        normalized_x, (mean, std_dev) = zscore(X, dim=1)

        # add positional embeddings if max_len != None
        # if self.pos!=None:
        #     normalized_x = self.pos(normalized_x)

        # Flatten mean and std_dev if input_dim is 1
        if n_features == 1:
            mean = mean.view(batch_size, -1)
            std_dev = std_dev.view(batch_size, -1)

        # Apply learnable scaling to mean, log std deviation, log sample_size
        scaled_mean = torch.exp(self.log_mean_scale) * mean  # Shape: [batch_size, n_features]
        scaled_std = torch.exp(self.log_std_scale) * torch.log(std_dev)  # Shape: [batch_size, n_features]

        # Handle n_samples (size of each data set)
        # Expand the scaled_n to have the shape [batch_size, 1]
        scaled_n = torch.exp(self.log_n_scale) * torch.sqrt(torch.tensor(n_samples, dtype=torch.float32)).to(X.device)
        scaled_n_expanded = scaled_n.expand(batch_size, 1)

        # first an encoder layer on the raw data
        Z = self.encoder(normalized_x)

        # now the summary layer, which sees the learned PMA statistics as well as some fixed stats:
        #   - the mean and std_dev that we normalized by
        #   - the sample size of that data set
        outputs_k = self.sab(self.pma(Z))
        outputs_k = outputs_k.view(batch_size, -1)
        outputs_k = torch.cat((outputs_k, scaled_mean, scaled_std, scaled_n_expanded), dim=1)

        # now we mix over the "fixed" stats and the "learned" stats to produced final stats
        adequate_stats = self.ff(outputs_k)
        return (adequate_stats)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension to match the batch size
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register buffer to avoid updating during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input
        return x + self.pe[:x.size(0), :]


class SetNorm(nn.Module):
    def __init__(self, feature_dim):
        super(SetNorm, self).__init__()
        self.eps = 1e-5
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x):
        # standardization
        batch, n_samples, n_features = x.shape
        x = x.reshape(batch, n_samples, -1)
        means = x.sum(dim=[1, 2]) / (n_samples * n_features)
        means = means.reshape(batch, 1)
        std = torch.sqrt(((x - means[:, None]).square()).sum(dim=[1, 2]) / (n_samples * n_features) + self.eps)
        std = std.reshape(batch, 1)
        out = (x - means[:, None]) / std[:, None]

        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out

def get_norm(norm, feature_dim=None):
    if norm=="setnorm":
        return SetNorm(feature_dim)
    else:
        return nn.Identity()

class MABClean(nn.Module):
    # here X (queries) attends to Y (keys, values)
    def __init__(self, dim_X, dim_Y, num_heads, normQ=True, norm="setnorm"):
        super().__init__()
        self.dim_Y = dim_Y
        self.num_heads = num_heads

        assert dim_Y % num_heads == 0, "dim_Y must be divisible by num_heads"

        # linear layers to get multiple sets of queries, keys, vals
        self.fc_q = nn.Linear(dim_X, dim_Y)
        self.fc_k = nn.Linear(dim_Y, dim_Y)
        self.fc_v = nn.Linear(dim_Y, dim_Y)

        # output linear layer to apply to concatenated attention heads
        # output is dim_X to make sure X and residual can be added directly
        self.fc_o = nn.Linear(dim_Y, dim_X)

        self.normQ = get_norm(norm, dim_X) if normQ else nn.Idenitity()
        self.normK = get_norm(norm, dim_Y)
        self.norm0 = get_norm(norm, dim_X)

        # the final residual layer, note that RelU is first per paper
        self.fc_res = nn.Linear(dim_X, dim_X)

    def forward(self, X, Y):
        # MABClean(X,Y) = H + fcc(relu(H))
        # where H = X + Attn(X, Y, Y)

        # Normalize accordingily X, Y, Y
        # Calculate Attn(X,Y,Y)
        # linear transformations of (X,Y) to get Q, K, V
        Q = self.fc_q(self.normQ(X))
        K, V = self.fc_k(self.normK(Y)), self.fc_v(Y)

        # splitting to get multiple attention heads
        dim_split = self.dim_Y // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # attention weights
        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_Y), 2)

        # Apply attention weights to the value vectors to get attention outputs
        O = A.bmm(V_)

        # Concatenate the attention heads' outputs along the feature dimension
        O = torch.cat(O.split(Q.size(0), 0), 2)

        # Final linear transformation on attention output (brings back to dim_X if needed)
        O = self.fc_o(O)

        # Now the final residual connection applied to relu(X+O)
        H = X + O
        H = H + self.fc_res(F.relu(self.norm0(H)))
        return H


class SABClean(nn.Module):
    def __init__(self, dim_in, num_heads, norm="setnorm"):
        super().__init__()
        self.mab = MABClean(dim_in, dim_in, num_heads, norm=norm)

    def forward(self, X):
        return self.mab(X, X)


class ISABClean(nn.Module):
    def __init__(self, dim_in, num_heads, num_inducing, norm="setnorm"):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inducing, dim_in))
        nn.init.xavier_uniform_(self.I)

        self.mab1 = MABClean(dim_in, dim_in, num_heads, normQ=False, norm=norm)
        self.mab2 = MABClean(dim_in, dim_in, num_heads, norm=norm)

    def forward(self, X):
        H = self.mab1(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab2(X, H)


class PMAClean(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim

        # a feedforward network that gets applied to each row of the input independently
        # before pooling-by-attention is applied
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # seed vectors
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, hidden_dim))
        nn.init.xavier_uniform_(self.S, gain=0.1)

        # the attention mechanism
        self.mab = MABClean(hidden_dim, hidden_dim, num_heads, norm="none")

    def forward(self, X):
        # MAB(S, rFF(X)) where rFF(X) is our rowwise feedforward network
        return self.mab(self.S.repeat(X.size(0), 1, 1), self.ff(X)).squeeze()