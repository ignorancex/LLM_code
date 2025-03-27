import numpy as np
import torch
import torch.distributions as td
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from ae.decoder import DecoderMMNIST
from ae.encoder import EncoderMMNIST
from macaw.flows import Flow, NormalizingFlowModel


class mmnist:

    def __init__(self, encoded_dim):
        self.encoded_dim = encoded_dim

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        self.encoder = EncoderMMNIST(encoded_dim=encoded_dim).to(self.device)
        self.decoder = DecoderMMNIST(encoded_dim=encoded_dim).to(self.device)
        self.macaw = self._init_macaw()

        self.feature_mean = None
        self.feature_std = None

    def _init_macaw(self):
        n_layers = 4
        hidden = [4, 6, 4]

        labels_to_latents = [(l, i) for l in range(10) for i in range(13, self.encoded_dim + 13)]
        labels_to_area = [(l, 11) for l in range(10)]
        thickness_to_area = [(10, 11)]
        area_to_latents = [(11, i) for i in range(13, self.encoded_dim + 13)]
        slant_to_latents = [(12, i) for i in range(13, self.encoded_dim + 13)]
        autoregressive_latents = [(i, j) for i in range(13, self.encoded_dim + 13) for j in
                                  range(i + 1, self.encoded_dim + 13)]

        edges = labels_to_latents + labels_to_area + thickness_to_area + area_to_latents + slant_to_latents + autoregressive_latents

        priors = [(slice(0, 10), td.OneHotCategorical(0.1 * torch.ones(10).to(self.device))),
                  (slice(10, self.encoded_dim + 13),
                   td.Normal(torch.zeros(self.encoded_dim + 3).to(self.device),
                             torch.ones(self.encoded_dim + 3).to(self.device))),
                  ]

        flow_list = [Flow(self.encoded_dim + 13, edges, self.device, hm=hidden) for _ in range(n_layers)]
        return NormalizingFlowModel(priors, flow_list).to(self.device)

    def _batch_to_x(self, batch):
        image_batch, features_batch, labels_batch = batch[0], batch[1], batch[2]
        latents = self.encoder(image_batch.to(self.device))
        flow_batch = latents[:, :self.encoded_dim]
        labels_batch = one_hot(labels_batch).to(self.device)
        features_batch = (features_batch.to(self.device) - self.feature_mean) / self.feature_std
        x = torch.hstack([labels_batch, features_batch, flow_batch]).type(torch.float32)
        return x, latents

    def _x_to_batch(self, x, latents):
        latents[:, :self.encoded_dim] = x[:, 13:]
        cfs = np.squeeze(self.decoder(latents).detach().cpu().numpy())
        labels = inverse_one_hot(x[:, :10]).detach().cpu().numpy()
        features = (x[:, 10:13] * self.feature_std + self.feature_mean).detach().cpu().numpy()
        return [cfs, features, labels]

    def _compute_feature_mean(self, train_loader):
        features = []
        for b in train_loader:
            features.append(b[1].numpy())
        features = np.vstack(features)
        scaler_f = StandardScaler()
        _ = scaler_f.fit_transform(features)
        self.feature_mean = torch.tensor(scaler_f.mean_).to(self.device)
        self.feature_std = torch.tensor(np.sqrt(scaler_f.var_)).to(self.device)

    def train_ae(self, train_loader, lr=0.0005, weight_decay=1e-06):
        self.encoder.train()
        self.decoder.train()

        params = [{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        loss_func = torch.nn.MSELoss()

        train_loss = []
        for batch in train_loader:
            image_batch = batch[0]
            image_batch = image_batch.to(self.device)
            encoded_data = self.encoder(image_batch)
            decoded_data = self.decoder(encoded_data)

            loss = loss_func(decoded_data, image_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test_MSE(self, dataloader):
        self.encoder.eval()
        self.decoder.eval()

        loss_func = torch.nn.MSELoss()

        with torch.no_grad():
            conc_out = []
            conc_label = []
            for batch in dataloader:
                image_batch = batch[0]
                image_batch = image_batch.to(self.device)
                encoded_data = self.encoder(image_batch)
                decoded_data = self.decoder(encoded_data)

                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())

            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            val_loss = loss_func(conc_out, conc_label)
        return val_loss.data.numpy()

    def train_macaw(self, train_loader, lr=0.005, weight_decay=1e-6):

        optimizer = torch.optim.Adam(self.macaw.parameters(), lr=lr, weight_decay=weight_decay)

        self.macaw.train()
        self.encoder.eval()

        if self.feature_mean is None:
            self._compute_feature_mean(train_loader)

        loss_val = []
        for batch in train_loader:
            with torch.no_grad():
                x, _ = self._batch_to_x(batch)

            _, prior_logprob, log_det = self.macaw(x)
            loss = - torch.sum(prior_logprob + log_det)
            loss /= train_loader.batch_size
            loss_val.append(loss.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return np.mean(loss_val)

    def test_likelihood(self, dataloader):

        self.macaw.eval()
        self.encoder.eval()

        loss_val = []
        for batch in dataloader:
            with torch.no_grad():
                x, _ = self._batch_to_x(batch)

                _, prior_logprob, log_det = self.macaw(x)
                loss = - torch.sum(prior_logprob + log_det)
                loss /= dataloader.batch_size
                loss_val.append(loss.detach().cpu().numpy())

        return np.mean(loss_val)

    def _cf(self, x_obs, cf_vals):
        self.macaw.eval()

        with torch.no_grad():
            z_obs = self.macaw(x_obs)[0][-1]
            x_cf = x_obs.detach().clone()
            for key in cf_vals:
                x_cf[:, key] = cf_vals[key]

            z_cf_val = self.macaw(x_cf)[0][-1]
            for key in cf_vals:
                z_obs[:, key] = z_cf_val[:, key]

        return self.macaw.backward(z_obs)[0][-1]

    def counterfactual(self, batch, cf_vals):
        self.encoder.eval()
        self.macaw.eval()
        self.decoder.eval()

        cf_vals = self._cfvals_transform(cf_vals)

        with torch.no_grad():
            x_obs, latents = self._batch_to_x(batch)
            x_cf = self._cf(x_obs, cf_vals)
            return self._x_to_batch(x_cf, latents)

    def _cfvals_transform(self, cf_vals):
        cfv = {}

        if cf_vals is not None:
            for key in cf_vals:
                print(key, cf_vals[key])
                if key == 'label':
                    cfv.update({i: int(x) for i, x in enumerate(one_hot(np.array([cf_vals[key]]))[0])})

                if key == 'thickness':
                    tf = (torch.tensor([cf_vals[key], 0, 0]).to(self.device) - self.feature_mean) / self.feature_std
                    cfv.update({10: tf[0]})
                if key == 'area':
                    tf = (torch.tensor([0, cf_vals[key], 0]).to(self.device) - self.feature_mean) / self.feature_std
                    cfv.update({11: tf[1]})
                if key == 'slant':
                    tf = (torch.tensor([0, 0, cf_vals[key]]).to(self.device) - self.feature_mean) / self.feature_std
                    cfv.update({12: tf[2]})

            return cfv

    def classify_labels(self, val_loader):
        labels = []
        predictions = []

        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].shape[0]
                x, latents = self._batch_to_x(batch)
                labels += batch[2].detach().cpu().numpy().tolist()

                x[:, :10] = 0
                x = x.repeat((10, 1))

                for i in range(10):
                    x[i * bs:i * bs + bs, i] = 1

                ll = self.macaw.log_likelihood(x).reshape(10, bs)
                predictions += (np.argmax(ll, axis=0).tolist())

        return sum([(l == p) * 1 for l, p in zip(labels, predictions)]) / len(predictions)

    def plot_ae_outputs(self, dataloader, n=10):
        self.encoder.eval()
        self.decoder.eval()

        fig = plt.figure(figsize=(20, 4))
        targets = next(iter(dataloader))[0]
        with torch.no_grad():
            for i in range(n):
                ax = fig.add_subplot(2, n, i + 1)
                img = targets[i].unsqueeze(0).to(self.device)

                rec_img = self.decoder(self.encoder(img))

                ax.imshow(img.cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                if i == n // 2:
                    ax.set_title('Original images')

                ax = fig.add_subplot(2, n, i + 1 + n)
                ax.imshow(rec_img.cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                if i == n // 2:
                    ax.set_title('Reconstructed images')

        plt.close()
        return fig

    def cf_test(model, val_loader, nsamples=12):

        batch = next(iter(val_loader))
        obs = batch[0].detach().cpu().numpy()[:nsamples]

        plt.rcParams['figure.figsize'] = (20, 3)
        fig = plt.figure()

        subfigs = fig.subfigures(nrows=4, ncols=1)

        cf_vals = {}
        cfs = model.counterfactual(batch, cf_vals)[0][:nsamples]
        axs0 = subfigs[0].subplots(nrows=1, ncols=nsamples)
        for i, img in enumerate(obs):
            axs0[i].imshow(np.squeeze(img), cmap='cividis')
            axs0[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)

        axs1 = subfigs[1].subplots(nrows=1, ncols=12)
        for i, img in enumerate(cfs):
            axs1[i].imshow(np.squeeze(img), cmap='cividis')
            axs1[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)

        cf_vals = {'label': 0}
        cfs = model.counterfactual(batch, cf_vals)[0][:nsamples]
        axs2 = subfigs[2].subplots(nrows=1, ncols=12)
        for i, img in enumerate(cfs):
            axs2[i].imshow(np.squeeze(img), cmap='cividis')
            axs2[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)

        cf_vals = {'thickness': 2, 'slant': 0.3}
        cfs = model.counterfactual(batch, cf_vals)[0][:nsamples]
        axs3 = subfigs[3].subplots(nrows=1, ncols=nsamples)
        for i, img in enumerate(cfs):
            axs3[i].imshow(np.asarray(img), cmap='cividis')
            axs3[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], xlabel=None)

        plt.subplots_adjust(top=1, bottom=0, left=0, right=0.5, wspace=0, hspace=0)
        plt.close()
        return fig


def one_hot(a, veclen=10):
    b = torch.zeros((a.shape[0], veclen))
    b[torch.arange(a.shape[0]), a] = 1
    return b


def inverse_one_hot(b):
    return torch.argmax(b, axis=1)
