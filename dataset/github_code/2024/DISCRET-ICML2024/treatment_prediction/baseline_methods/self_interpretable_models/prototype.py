
# ProtoVAE: A Trustworthy Self-Explainable Prototypical Variational Model
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

    
class ProtoVAE(nn.Module):

    def __init__(self, img_size, input_size, latent, num_prototypes, num_treatments):

        super(ProtoVAE, self).__init__()
        self.img_size = img_size
        self.prototype_shape = (num_prototypes,latent)
        self.num_prototypes = num_prototypes
        self.num_classes = num_treatments
        self.epsilon = 1e-4

        self.latent = latent
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        
        self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, 1,
                                    bias=False) # do not use bias


        self.features = nn.Sequential(  ###### CIFAR-10
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2, stride=2), ##16x16
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(2, stride=2), ##8x8
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2, stride=2), ##4x4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(2, stride=2), ##2x2

            nn.Flatten(),
        )
        
        self.feature_outnet = nn.Sequential(
            nn.Linear(112896  + input_size, latent * 2),
            nn.ReLU(),
            nn.Linear(latent * 2, latent * 2)
            )
        
        self.decoder_layers = nn.Sequential(
                nn.Linear(latent, latent * 11 * 11),  # 64x7x7 is chosen for the initial spatial dimensions
                nn.ReLU(),
                View((-1, latent,11,11)),
                nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1)
            )

        # self.decoder_layers = nn.Sequential(  ###### CIFAR-10
        #         nn.Linear(latent, 256 * 2 * 2),
        #         nn.ReLU(inplace=True),
        #         nn.BatchNorm1d(256 * 2 * 2),
        #         View((-1, 256,2,2)),

        #         nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(128),
        #         nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(128),    ##8x8

        #         nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(64),
        #         nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(64),  ##16x16

        #         nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),
        #         nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),  ##32x32

        #         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),
        #         nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),

        #         nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),

        #     )
        self.decoder_layer1 = nn.Sequential(  ###### CIFAR-10
                nn.Linear(latent, 256 * 2 * 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256 * 2 * 2),
                View((-1, 256,2,2)))
        self.decoder_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128))
        self.decoder_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64)
        )
        self.decoder_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),  ##32x32
        )

        self.decoder_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(in_channels=self.img_size, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        
        self._initialize_weights()


    def decoder(self, z):
        x = self.decoder_layers(z)
        x = torch.tanh(x)
        return x



    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps


    def distance_2_similarity(self, distances):
        return torch.log((distances + 1) / (distances + self.epsilon))


    def forward(self, x, x_tab, y=None, is_train=True):
        conv_features = self.features(x)
        conv_features = self.feature_outnet(torch.cat((conv_features, x_tab), dim=1))

        mu = conv_features[:,:self.latent]
        logVar = conv_features[:,self.latent:].clamp(np.log(1e-8), -np.log(1e-8))
        z = self.reparameterize(mu, logVar)
        if(~is_train):
            z = mu

        prototypes_of_correct_class = torch.t(self.prototype_class_identity.to(device)[:, y.view(-1).long()])
        index_prototypes_of_correct_class = (prototypes_of_correct_class == 1).nonzero(as_tuple=True)[1]
        index_prototypes_of_correct_class = index_prototypes_of_correct_class.view(x.shape[0],self.num_prototypes_per_class)
        prototypes_per_sample = self.prototype_vectors[index_prototypes_of_correct_class]
        sim_scores = self.calc_sim_scores(z, prototypes_per_sample)

        full_sim_scores = torch.zeros((sim_scores.shape[0], self.num_prototypes)).to(device)

        for k in range(self.num_prototypes_per_class):
            full_sim_scores[torch.arange(sim_scores.shape[0]),index_prototypes_of_correct_class[:,k]] = sim_scores[:,k]
        

        kl_loss = self.kl_divergence_nearest(mu, logVar, index_prototypes_of_correct_class, full_sim_scores)

        out = self.last_layer(full_sim_scores)

        decoded = self.decoder(z)

        ortho_loss = self.ortho_loss()

        return out, decoded, kl_loss, ortho_loss



    def ortho_loss(self):
        s_loss = 0
        for k in range(self.num_classes):
            p_k = self.prototype_vectors[k*self.num_prototypes_per_class:(k+1)*self.num_prototypes_per_class,:]
            p_k_mean = torch.mean(p_k, dim=0)
            p_k_2 = p_k - p_k_mean
            p_k_dot = p_k_2 @ p_k_2.T
            s_matrix = p_k_dot - (torch.eye(p_k.shape[0]).to(device))
            s_loss+= torch.norm(s_matrix,p=2)
        return s_loss/self.num_classes


    def calc_sim_scores(self, z, prototypes_per_sample):
        d = torch.cdist(z.unsqueeze(1), prototypes_per_sample, p=2)  ## Batch size x prototypes
        d = d.squeeze(1)
        sim_scores = self.distance_2_similarity(d)
        return sim_scores


    def kl_divergence_nearest(self, mu, logVar, nearest_pt, sim_scores):
        kl_loss = torch.zeros(sim_scores.shape).to(device)
        for i in range(self.num_prototypes_per_class):
            p = torch.distributions.Normal(mu, torch.exp(logVar / 2))
            p_v = self.prototype_vectors[nearest_pt[:,i],:]
            q = torch.distributions.Normal(p_v, torch.ones(p_v.shape).to(device))
            kl = torch.mean(torch.distributions.kl.kl_divergence(p, q), dim=1)
            kl_loss[np.arange(sim_scores.shape[0]),nearest_pt[:,i]] = kl
        kl_loss = kl_loss*sim_scores
        mask = kl_loss > 0
        kl_loss = torch.sum(kl_loss, dim=1) / (torch.sum(sim_scores * mask, dim=1))
        kl_loss = torch.mean(kl_loss)
        return kl_loss


    def get_prototype_images(self):
        p_decoded = self.decoder(self.prototype_vectors)
        return p_decoded


    def pred_class(self, x):
        conv_features = self.features(x)

        mu = conv_features[:, :latent]
        z = mu

        sim_scores = self.calc_sim_scores(z)

        out = self.last_layer(sim_scores)

        return out, sim_scores


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.decoder_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
                    
class ProtoVAE_tab(nn.Module):

    def __init__(self, input_size, latent, num_prototypes, num_treatments):

        super(ProtoVAE_tab, self).__init__()
        self.prototype_shape = (num_prototypes,latent)
        self.num_prototypes = num_prototypes
        self.num_classes = num_treatments
        self.epsilon = 1e-4

        self.latent = latent
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        
        self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, 1,
                                    bias=False) # do not use bias


        # self.features = nn.Sequential(  ###### CIFAR-10
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32),
        #     nn.AvgPool2d(2, stride=2), ##16x16
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.AvgPool2d(2, stride=2), ##8x8
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        #     nn.AvgPool2d(2, stride=2), ##4x4
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.AvgPool2d(2, stride=2), ##2x2

        #     nn.Flatten(),
        # )
        
        self.feature_outnet = nn.Sequential(
            nn.Linear(input_size, latent * 2),
            nn.ReLU(),
            nn.Linear(latent * 2, latent * 2)
            )
        
        # self.decoder_layers = nn.Sequential(
        #         nn.Linear(latent, latent * 11 * 11),  # 64x7x7 is chosen for the initial spatial dimensions
        #         nn.ReLU(),
        #         View((-1, latent,11,11)),
        #         nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1)
        #     )
        self.decoder_layers = nn.Sequential(nn.Linear(latent, latent), nn.ReLU(), nn.Linear(latent, input_size))

        # self.decoder_layers = nn.Sequential(  ###### CIFAR-10
        #         nn.Linear(latent, 256 * 2 * 2),
        #         nn.ReLU(inplace=True),
        #         nn.BatchNorm1d(256 * 2 * 2),
        #         View((-1, 256,2,2)),

        #         nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(128),
        #         nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(128),    ##8x8

        #         nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(64),
        #         nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(64),  ##16x16

        #         nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),
        #         nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),  ##32x32

        #         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),
        #         nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),

        #         nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),

        #     )
        # self.decoder_layer1 = nn.Sequential(  ###### CIFAR-10
        #         nn.Linear(latent, 256 * 2 * 2),
        #         nn.ReLU(inplace=True),
        #         nn.BatchNorm1d(256 * 2 * 2),
        #         View((-1, 256,2,2)))
        # self.decoder_layer2 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(128),
        #         nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(128))
        # self.decoder_layer3 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(64),
        #         nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(64)
        # )
        # self.decoder_layer4 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),
        #         nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),  ##32x32
        # )

        # self.decoder_layer5 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),
        #         nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(32),

        #         nn.Conv2d(in_channels=self.img_size, out_channels=3, kernel_size=3, stride=1, padding=1),
        # )
        
        # self._initialize_weights()


    def decoder(self, z):
        x = self.decoder_layers(z)
        x = torch.tanh(x)
        return x



    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps


    def distance_2_similarity(self, distances):
        return torch.log((distances + 1) / (distances + self.epsilon))


    def forward(self, x_tab, t, d=None, test=True):
        # conv_features = self.features(x)
        conv_features = self.feature_outnet(torch.cat((conv_features, x_tab), dim=1))

        mu = conv_features[:,:self.latent]
        logVar = conv_features[:,self.latent:].clamp(np.log(1e-8), -np.log(1e-8))
        z = self.reparameterize(mu, logVar)
        if(~is_train):
            z = mu

        prototypes_of_correct_class = torch.t(self.prototype_class_identity.to(device)[:, y.view(-1).long()])
        index_prototypes_of_correct_class = (prototypes_of_correct_class == 1).nonzero(as_tuple=True)[1]
        index_prototypes_of_correct_class = index_prototypes_of_correct_class.view(x.shape[0],self.num_prototypes_per_class)
        prototypes_per_sample = self.prototype_vectors[index_prototypes_of_correct_class]
        sim_scores = self.calc_sim_scores(z, prototypes_per_sample)

        full_sim_scores = torch.zeros((sim_scores.shape[0], self.num_prototypes)).to(device)

        for k in range(self.num_prototypes_per_class):
            full_sim_scores[torch.arange(sim_scores.shape[0]),index_prototypes_of_correct_class[:,k]] = sim_scores[:,k]
        

        kl_loss = self.kl_divergence_nearest(mu, logVar, index_prototypes_of_correct_class, full_sim_scores)

        out = self.last_layer(full_sim_scores)

        decoded = self.decoder(z)

        ortho_loss = self.ortho_loss()

        return out, decoded, kl_loss, ortho_loss



    def ortho_loss(self):
        s_loss = 0
        for k in range(self.num_classes):
            p_k = self.prototype_vectors[k*self.num_prototypes_per_class:(k+1)*self.num_prototypes_per_class,:]
            p_k_mean = torch.mean(p_k, dim=0)
            p_k_2 = p_k - p_k_mean
            p_k_dot = p_k_2 @ p_k_2.T
            s_matrix = p_k_dot - (torch.eye(p_k.shape[0]).to(device))
            s_loss+= torch.norm(s_matrix,p=2)
        return s_loss/self.num_classes


    def calc_sim_scores(self, z, prototypes_per_sample):
        d = torch.cdist(z.unsqueeze(1), prototypes_per_sample, p=2)  ## Batch size x prototypes
        d = d.squeeze(1)
        sim_scores = self.distance_2_similarity(d)
        return sim_scores


    def kl_divergence_nearest(self, mu, logVar, nearest_pt, sim_scores):
        kl_loss = torch.zeros(sim_scores.shape).to(device)
        for i in range(self.num_prototypes_per_class):
            p = torch.distributions.Normal(mu, torch.exp(logVar / 2))
            p_v = self.prototype_vectors[nearest_pt[:,i],:]
            q = torch.distributions.Normal(p_v, torch.ones(p_v.shape).to(device))
            kl = torch.mean(torch.distributions.kl.kl_divergence(p, q), dim=1)
            kl_loss[np.arange(sim_scores.shape[0]),nearest_pt[:,i]] = kl
        kl_loss = kl_loss*sim_scores
        mask = kl_loss > 0
        kl_loss = torch.sum(kl_loss, dim=1) / (torch.sum(sim_scores * mask, dim=1))
        kl_loss = torch.mean(kl_loss)
        return kl_loss


    def get_prototype_images(self):
        p_decoded = self.decoder(self.prototype_vectors)
        return p_decoded


    def pred_class(self, x):
        conv_features = self.features(x)

        mu = conv_features[:, :latent]
        z = mu

        sim_scores = self.calc_sim_scores(z)

        out = self.last_layer(sim_scores)

        return out, sim_scores


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.decoder_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.08, 0.08)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)





