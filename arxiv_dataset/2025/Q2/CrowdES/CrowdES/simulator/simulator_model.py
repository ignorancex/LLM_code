
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
from typing import Optional, Tuple, Union
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from dataclasses import dataclass

from CrowdES.layers import ConcatSquashLinear, MLP, social_transformer, ConcatScaleLinear, GroupNormLinear
from CrowdES.simulator.simulator_config import CrowdESSimulatorConfig


@dataclass
class CrowdESSimulatorModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    preds: torch.FloatTensor = None
    states: torch.LongTensor = None


class CrowdESSimulatorModel(PreTrainedModel):
    
    config_class = CrowdESSimulatorConfig
    base_model_prefix = 'crowdes_simulator'
    main_input_name = 'meter_values'

    def __init__(self, config):
        super().__init__(config)

        # Environment Encoder
        # Encode 8*64*64 map data to 256 dim vector (64*64 -> 32*32 -> 16*16 -> 8*8 -> flatten -> 256)
        env_encoder = []
        env_hidden_dims = [config.env_dim] + config.env_hidden_dims
        env_flatten_size = (config.env_size[0] * config.env_size[1]) // (4 ** len(config.env_hidden_dims)) * env_hidden_dims[-1]
        for i_dim, o_dim in zip(env_hidden_dims[:-1], env_hidden_dims[1:]):
            env_encoder.append(nn.Conv2d(i_dim, o_dim, kernel_size=3, stride=2, padding=1))
            env_encoder.append(nn.ReLU())
            env_encoder.append(nn.GroupNorm(num_groups=4, num_channels=o_dim))
        env_encoder.append(nn.Flatten())
        env_encoder.append(nn.Linear(env_flatten_size, config.env_embedding_dim))
        self.env_encoder = nn.Sequential(*env_encoder)
        
        self.cross_attentions = nn.ModuleList()
        for _ in range(config.env_num_attention_layers):
            self.cross_attentions.append(ConcatSquashLinear(config.hidden_dim, config.hidden_dim, config.env_embedding_dim))
        
        # Neighbor Encoder
        self.neighbor_encoders = nn.ModuleList()
        neighbor_dim = config.history_length * 2
        neighbor_hidden_dims = [neighbor_dim] + config.neighbor_hidden_dims + [config.neighbor_embedding_dim]
        for i_dim, o_dim in zip(neighbor_hidden_dims[:-1], neighbor_hidden_dims[1:]):
            self.neighbor_encoders.append(social_transformer(i_dim, o_dim, n_head=4, n_layers=2))

        # Trajectory Encoder
        _traj_dim = config.history_length * 2
        _goal_dim = 2
        _attr_dim = 2
        _control_dim = 2
        _neighbor_dim = config.neighbor_embedding_dim
        input_dim = _traj_dim + _goal_dim + _attr_dim + _control_dim + _neighbor_dim + config.env_embedding_dim
        traj_encoder_hidden_dims = [input_dim] + config.traj_encoder_hidden_dims + [config.hidden_dim]
        self.traj_encoders = nn.ModuleList()
        for i_dim, o_dim in zip(traj_encoder_hidden_dims[:-1], traj_encoder_hidden_dims[1:]):
            self.traj_encoders.append(ConcatSquashLinear(i_dim, o_dim, i_dim))
        
        # Trajectory Decoder
        traj_decoder_hidden_dims = [config.hidden_dim + config.latent_dim] + config.traj_decoder_hidden_dims + [config.future_length * 2]
        self.traj_decoders = nn.ModuleList()
        for i_dim, o_dim in zip(traj_decoder_hidden_dims[:-1], traj_decoder_hidden_dims[1:]):
            self.traj_decoders.append(ConcatSquashLinear(i_dim, o_dim, i_dim))

        # Latent Distribution
        self.lantent_predictor = MLP(config.hidden_dim, config.latent_dim, config.latent_hidden_dims, activation=nn.ReLU(), dropout=-1)
        self.endpoint_cluster_centers = None  # shape=(num_clusters, latent_dim)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        traj_hist: torch.FloatTensor,
        traj_fut: Optional[torch.FloatTensor] = None,
        goal: Optional[torch.FloatTensor] = None,
        attr: Optional[torch.FloatTensor] = None,
        control: Optional[torch.FloatTensor] = None,
        neighbor: Optional[torch.FloatTensor] = None,
        env_data: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        sampling: bool = True,
        previous_state: Optional[int] = None,
        alpha: Optional[float] = 1.0,
        tmp: Optional[float] = 1.0,
    ) -> Union[Tuple, CrowdESSimulatorModelOutput]:
        
        # traj_hist.shape=(B, T_hist, 2)
        # traj_fut.shape =(B, T_fut, 2)
        # goal.shape     =(B, 2)
        # attr.shape     =(B, 2)
        # control.shape  =(B, 2)
        # neighbor.shape =(B, N, T_hist, 2)
        # env_data.shape =(B, C, H, W (C=8))

        is_training = traj_fut is not None

        batch_size = traj_hist.shape[0]
        traj_hist = traj_hist.flatten(1)  # shape=(B, T_hist*2)
        if goal is None:
            goal = torch.zeros(batch_size, 2, device=traj_hist.device)
        if attr is None:
            attr = torch.zeros(batch_size, 2, device=traj_hist.device)
        if control is None:
            control = torch.zeros(batch_size, 2, device=traj_hist.device)
        
        # Encode environment
        if env_data is not None:
            env_embedding = self.env_encoder(env_data)
        else:
            env_embedding = torch.zeros(batch_size, self.config.env_embedding_dim, device=traj_hist.device)
        
        # Encode neighborhood
        if neighbor is not None:
            neighbor = neighbor.flatten(2)  # shape=(B, N, T_hist*2)
            # concat with traj_hist
            neighbor_embedding = torch.cat([traj_hist.unsqueeze(1), neighbor], dim=1)  # shape=(B, (N+1), T_hist*2)
            for neighbor_encoder in self.neighbor_encoders:
                neighbor_embedding = neighbor_encoder(neighbor_embedding)  # shape=(B, (N+1), neighbor_embedding_dim)
            neighbor_embedding = neighbor_embedding[:, 0]  # shape=(B, neighbor_embedding_dim)
        else:
            neighbor_embedding = torch.zeros(batch_size, self.config.neighbor_embedding_dim, device=traj_hist.device)
        
        # Concat all embeddings
        embedding = torch.cat([traj_hist, goal, attr, control, neighbor_embedding, env_embedding], dim=1)  # shape=(B, input_dim)

        # Trajectory Encoder
        for traj_encoder in self.traj_encoders:
            embedding = traj_encoder(embedding, embedding)

        # Estimate behavior state transition probability of markov chain
        latent_log_prob = self.lantent_predictor(embedding)  # shape=(B, latent_dim)
        latent_log_prob = latent_log_prob - latent_log_prob.mean(dim=1, keepdim=True)  # normalize
        latent_log_prob = latent_log_prob.clamp(min=-5, max=5)  # prevent overflow

        if is_training:
            # During training, use ground-truth behavior state
            endpoint = traj_fut[:, -1]  # shape=(B, 2)
            endpoint_norm, *_ = self.normalize_rotation_scale(endpoint, control)

            dist = endpoint_norm.unsqueeze(1).detach() - self.endpoint_cluster_centers.detach()  # shape=(B, num_clusters, 2)
            dist = torch.linalg.norm(dist, ord=2, dim=2)  # shape=(B, num_clusters)
            
            gt_behavior = torch.argmin(dist, dim=1).detach()  # shape=(B)
            latent = torch.nn.functional.one_hot(gt_behavior, num_classes=self.config.latent_dim).float()  # shape=(B, latent_dim)

            # Calculate loss to make latent_log_prob to be close to the ground-truth behavior
            loss_l_fct = CrossEntropyLoss()
            loss_l = loss_l_fct(latent_log_prob, gt_behavior)

        else:
            # During inference, sample behavior from the transition probability of markov chain
            if sampling:
                # Sample
                if previous_state is not None:
                    probs = (latent_log_prob / tmp).softmax(dim=1)  # shape=(B, latent_dim)
                    probs = probs + previous_state * alpha  # add previous behavior

                    probs = probs / probs.sum(dim=1, keepdim=True)  # normalize
                    latent = torch.distributions.OneHotCategorical(probs=probs).sample()  # shape=(B, latent_dim)
                else:
                    latent = torch.distributions.OneHotCategorical(logits=latent_log_prob).sample()  # shape=(B, latent_dim)
            else:
                # Most likely
                latent = torch.argmax(latent_log_prob, dim=1)  # shape=(B,)
                latent = torch.nn.functional.one_hot(latent, num_classes=self.config.latent_dim).float()  # shape=(B, latent_dim)
            
        # Decode trajectory
        embedding = torch.cat([embedding, latent], dim=1)  # shape=(B, hidden_dim + latent_dim)
        for traj_decoder in self.traj_decoders:
            embedding = traj_decoder(embedding, embedding)
        traj_fut_pred = embedding.view(batch_size, self.config.future_length, 2)  # shape=(B, T_fut, 2)

        # Calculate loss
        loss = None
        if is_training:
            # loss_t = torch.linalg.norm(traj_fut_pred - traj_fut, ord=2, dim=2).mean()  # MSE(L2)
            loss_t = torch.linalg.norm(traj_fut_pred - traj_fut, ord=1, dim=2).mean()  # MAE(L1)
            loss = loss_t + loss_l * self.config.loss_l_scaling

        if not return_dict:
            output = (traj_fut_pred, latent_log_prob, latent)
            return ((loss,) + output) if loss is not None else output

        return CrowdESSimulatorModelOutput(
            loss=loss,
            logits=latent_log_prob,
            preds=traj_fut_pred,
            states=latent,
        )
    
    def normalize_rotation_scale(self, endpoint, control):
        dir = control
        mask = control.norm(dim=1) > 1e-4  # prevent zero division
        dir[~mask, 0], dir[~mask, 1] = 1, 0  # if the control is zero, set the base direction to x-axis 
        sca = torch.linalg.norm(dir, ord=2, dim=1)  # shape=(B)
        rot = torch.atan2(dir[:, 1], dir[:, 0])  # shape=(B)
        rot_mat = torch.stack([torch.stack([rot.cos(), -rot.sin()], dim=1),
                                torch.stack([rot.sin(), rot.cos()], dim=1)], dim=1)
        endpoint_norm = (endpoint.unsqueeze(dim=1) @ rot_mat).squeeze(dim=1) / sca.unsqueeze(1)  # shape=(B, 2)
        return endpoint_norm, rot_mat, sca
    
    def endpoint_cluster_center_generation(self, endpoint, control):
        num_clusters = self.config.latent_dim

        endpoint_norm, _, _ = self.normalize_rotation_scale(endpoint, control)
        endpoint_norm = endpoint_norm.detach().cpu().numpy()

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, init='k-means++', n_init='auto').fit(endpoint_norm)
        cluster_centers = kmeans.cluster_centers_
        self.endpoint_cluster_centers = torch.tensor(cluster_centers, device=self.device, dtype=endpoint.dtype)
