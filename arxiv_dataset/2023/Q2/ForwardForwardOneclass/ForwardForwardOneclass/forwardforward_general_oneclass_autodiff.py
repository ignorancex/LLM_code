import random
import numpy as np
import torch
from torch.optim import Adam, SGD
import os
import copy
import sys
sys.path.append(os.path.join('scripts', 'loss-landscape-anims'))
from loss_landscape_anim.loss_landscape import LossGrid, DimReduction
from loss_landscape_anim._plot import animate_contour, sample_frames


class EarlyStopper:
    def __init__(self, 
    patience=10,
    min_delta_pct=0.05 # 0.1%
    ):
        self.patience = patience
        self.min_delta_pct = min_delta_pct
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        delta_pct = ( self.min_validation_loss - validation_loss ) / validation_loss
        # print( self.min_validation_loss, validation_loss, delta_pct, self.counter)
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif delta_pct < self.min_delta_pct:
            self.counter += 1
            if self.counter >= self.patience:
                print("Final new validation loss: ", validation_loss, "and previous loss: ", self.min_validation_loss, ". Had delta of ", delta_pct, " which was less than ", self.min_delta_pct, f" for {self.counter} iterations")
                return True
        return False


def norm(x, p=2, dim=-1, keepdims=True):
    return (x ** p).sum(axis=dim, keepdims=keepdims)

def normalize(X, epsilon=1e-9):
    """
    normalize the inputs from previous layers
    Math: X / ||X||_2
    Ref: ``To prevent this, FF normalizes the length of the
            hidden vector before using it as input to the
            next layer.``
    """
    return X / (X.norm(p=2, dim=1, keepdim=True) + epsilon)


class FFLayer(torch.nn.Linear):
    
    def __init__(self, 
                 in_dim, out_dim, 
                 loss_type='goodness',
                 layerno=0, save_folder='',
                 bias=True, device=None, dtype=None,
                 do_norm=True, constant=1.0, seed=1,
                 lr=5e-4):
        torch.manual_seed(seed)
        super().__init__(in_dim, out_dim, bias=bias)#, bias, device, dtype)
        self._c = constant
        self._do_normalize = do_norm
        # self.lr_is_positive = False
        self.loss_type = loss_type
        self.epoch = 0

        self.relu = torch.nn.ReLU()
        self.opt = SGD(self.parameters(), lr=lr)#, momentum=0.)

        # for visualization
        self.optim_path = []
        self.layer_no_string = f"Layer_{layerno}"
        self.folder = os.path.join(save_folder, self.layer_no_string)
        os.makedirs(self.folder, exist_ok=True)

        print(self.layer_no_string)
        print(self.weight.mean(), self.bias.mean())

        # self.not_done = True

    def loss_fn(self, h, y=None, update_hyperparams=False):
        if self.loss_type == 'origgoodness':
            loss = torch.sigmoid( norm(h) - self._c )
            return loss
        elif self.loss_type == 'goodness':
            loss = torch.log( 1 + torch.exp( ( norm(h) - self._c) ) )
            return loss
        elif self.loss_type == 'hbsvdd':
            if update_hyperparams or (self.epoch == 0):
                self.a = torch.Tensor( h.mean(0).detach().numpy() )
            loss = (1/2)*norm(h-self.a)
            return loss
        elif self.loss_type in ['svdd','lssvdd']:
            if update_hyperparams or (self.epoch == 0):
                self.a = torch.Tensor( h.mean(0).detach().numpy() )
            dist = norm(h-self.a)
            if update_hyperparams or (self.epoch == 0):
                self.r = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu)
                # print(self.layer_no_string)
                # print(self.a)
                # print(self.r)
                if np.isnan(self.r):
                    sys.exit()
            scores = dist - self.r**2
            if self.loss_type == 'svdd':
                loss = self.r**2 + (1 / self.nu) * torch.max(torch.zeros_like(scores), scores)
            elif self.loss_type == 'lssvdd':
                loss = self.r**2 + (1 / self.nu) * scores**2
            return loss

    def forward(self, X, notate_opt=False):
        # print(X.shape)
        # print(self.weight.mean(), self.bias.mean())
        # print(self.weight.shape)
        if self._do_normalize:
            X = normalize(X)
        h = self.relu(torch.mm(X, self.weight.T) + self.bias.unsqueeze(0))
        if self.training:
            loss = self.loss_fn(h).view(-1).mean()
            # print("l", loss)
            # print(self.layer_no_string)
            # print(self.weight.mean(), self.bias.mean())
            
            self.opt.zero_grad()
            loss.backward(retain_graph=True)
            # if ( sum([is_positive, self.lr_is_positive]) == 1 ):
            #     print("Resett", is_positive, self.opt.param_groups[0]['lr'])
            #     self.opt.param_groups[0]['lr'] = -1 * self.opt.param_groups[0]['lr']
            #     self.lr_is_positive = self.lr_is_positive * -1
            self.opt.step()

            # if self.layer_no_string == "Layer_2":
            #     sys.exit()

            if notate_opt:
                # Track optimization steps
                flat_w = self.get_flat_params()
                self.optim_path.append({'loss': loss, "flat_w":flat_w})
        
            # print(self.weight.mean(), self.weight.var(), self.bias.mean(), self.bias.var())

        # if self.not_done:
        #     self.not_done=False
        #     print(f"AT LAYER: {self.layer_no_string}")
        #     from torchinfo import summary
        #     print( summary(self, X.shape ) )
        
        return h.detach()

    ##############################################################
    # Loss landscape functions
    ##############################################################
    def vis_loss_landscape(self, X,
                           reduction_method="pca",
                           n_frames=300,
                           giffps=50,
                           sampling=False,
                           output_to_file=True,
                           ):
    
        import matplotlib.pyplot as plt

        output_filename = os.path.join(self.folder, f"sample_{self.layer_no_string}")
        sampled_optim_path = sample_frames(self.optim_path, max_frames=n_frames)
        optim_path, loss_path = zip(
            *[
                (path["flat_w"], path["loss"])
                for path in sampled_optim_path
            ]
        )

        print(f"Dimensionality reduction method specified: {reduction_method}")
        dim_reduction = DimReduction(
            params_path=optim_path,
            reduction_method=reduction_method,
            custom_directions=None,
            seed=0,
        )
        reduced_dict = dim_reduction.reduce()
        path_2d = reduced_dict["path_2d"]
        directions = reduced_dict["reduced_dirs"]
        pcvariances = reduced_dict.get("pcvariances")

        self.optim_path = None
        # a = self.a
        # self.a = None

        loss_grid = LossGrid(
            optim_path=optim_path,
            model=copy.deepcopy(self),
            data=torch.Tensor(X),
            path_2d=path_2d,
            directions=directions,
            # res=60, margin=10
        )

        animate_contour(
            param_steps=path_2d.tolist(),
            loss_steps=loss_path,
            loss_grid=loss_grid.loss_values_log_2d,
            coords=loss_grid.coords,
            true_optim_point=loss_grid.true_optim_point,
            true_optim_loss=loss_grid.loss_min,
            pcvariances=pcvariances,
            giffps=giffps,
            sampling=sampling,
            output_to_file=output_to_file,
            filename=output_filename+'.gif',
        )

        # self.a = a

        fig = plt.figure(figsize=(9,6))
        ax = plt.axes(projection='3d')
        # ax     = fig.gca(projection='3d')
        coords_x, coords_y = loss_grid.coords
        ax.contourf(coords_x, coords_y, loss_grid.loss_values_log_2d, levels=35, alpha=0.9, cmap="YlGnBu")
        ax.set_title("Optimization in Loss Landscape")
        xlabel_text = "direction 0"
        ylabel_text = "direction 1"
        if pcvariances is not None:
            xlabel_text = f"principal component 0, {pcvariances[0]:.1%}"
            ylabel_text = f"principal component 1, {pcvariances[1]:.1%}"
        ax.set_xlabel(xlabel_text)
        ax.set_ylabel(ylabel_text)
        plt.savefig(output_filename+'.png')
        plt.close()

    def get_flat_params(self):
        """Get flattened and concatenated params of the model."""
        params = self._get_params()
        flat_params = torch.Tensor()
        if torch.cuda.is_available() and self.gpus > 0:
            flat_params = flat_params.cuda()
        for _, param in params.items():
            flat_params = torch.cat((flat_params, torch.flatten(param)))
        return flat_params

    def init_from_flat_params(self, flat_params):
        """Set all model parameters from the flattened form."""
        if not isinstance(flat_params, torch.Tensor):
            raise AttributeError(
                "Argument to init_from_flat_params() must be torch.Tensor"
            )
        shapes = self._get_param_shapes()
        state_dict = self._unflatten_to_state_dict(flat_params, shapes)
        self.load_state_dict(state_dict, strict=True)

    def _get_param_shapes(self):
        shapes = []
        for name, param in self.named_parameters():
            shapes.append((name, param.shape, param.numel()))
        return shapes

    def _get_params(self):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.data
        return params

    def _unflatten_to_state_dict(self, flat_w, shapes):
        state_dict = {}
        counter = 0
        for shape in shapes:
            name, tsize, tnum = shape
            param = flat_w[counter : counter + tnum].reshape(tsize)
            state_dict[name] = torch.nn.Parameter(param)
            counter += tnum
        assert counter == len(flat_w), "counter must reach the end of weight vector"
        return state_dict






class BPLayer(torch.nn.Linear):
    
    def __init__(self, 
                 in_dim, out_dim, 
                 loss_type='goodness',
                 layerno=0, save_folder='',
                 bias=True, device=None, dtype=None,
                 do_norm=True, constant=1.0, seed=1,
                 lr=5e-4):
        torch.manual_seed(seed)
        super().__init__(in_dim, out_dim, bias=bias)#, bias, device, dtype)
        self._c = constant
        self._do_normalize = do_norm
        # self.lr_is_positive = False
        self.loss_type = loss_type
        self.epoch = 0

        self.relu = torch.nn.ReLU()
        self.opt = SGD(self.parameters(), lr=lr)#, momentum=0.)

        # for visualization
        self.optim_path = []
        self.layer_no_string = f"Layer_{layerno}"
        self.folder = os.path.join(save_folder, self.layer_no_string)
        os.makedirs(self.folder, exist_ok=True)


    def loss_fn(self, h, y=None, update_hyperparams=False):
        if self.loss_type == 'origgoodness':
            loss = torch.sigmoid( norm(h) - self._c )
            return loss
        elif self.loss_type == 'goodness':
            loss = torch.log( 1 + torch.exp( ( norm(h) - self._c) ) )
            return loss
        elif self.loss_type == 'hbsvdd':
            if update_hyperparams or (self.epoch == 0):
                self.a = torch.Tensor( h.mean(0).detach().numpy() )
            loss = (1/2)*norm(h-self.a)
            return loss
        elif self.loss_type in ['svdd','lssvdd']:
            if update_hyperparams or (self.epoch == 0):
                self.a = torch.Tensor( h.mean(0).detach().numpy() )
            dist = norm(h-self.a)
            if update_hyperparams or (self.epoch == 0):
                self.r = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu)
                if np.isnan(self.r):
                    sys.exit()
            scores = dist - self.r**2
            if self.loss_type == 'svdd':
                loss = self.r**2 + (1 / self.nu) * torch.max(torch.zeros_like(scores), scores)
            elif self.loss_type == 'lssvdd':
                loss = self.r**2 + (1 / self.nu) * scores**2
            return loss

    def forward(self, X):
        if self._do_normalize:
            X = normalize(X)
        h = self.relu(torch.mm(X, self.weight.T) + self.bias.unsqueeze(0))
        # print(f"AT LAYER: {self.layer_no_string}")
        # from torchinfo import summary
        # print( summary(self, X.shape) )
        return h


class GeneralOneclass(torch.nn.Module):
    
    name = "GeneralOneclass"
    
    def __init__(self, dims, save_folder, nntype='ff', **kwargs):
        # nntype = 'ff' or 'bp'
        super().__init__()
        self._dims = dims
        self.nntype = nntype
        
        exp_nm = "_".join(np.array(dims).astype(str))

        loss_type = kwargs.get('loss_type')
        self.save_folder = os.path.join(save_folder, loss_type, exp_nm)

        seed = kwargs.pop('seed')

        if nntype == 'ff':
            Layer = FFLayer
        elif nntype == 'bp':
            Layer = BPLayer
        else:
            raise Exception("")

        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], layerno=d, seed=seed+d, save_folder=self.save_folder, **kwargs)]

        print(self.layers)

    def forward(self, X, return_loss=True):
        # print("Forward at nn", X.shape)
        # TODO: clean
        X = torch.Tensor(X)

        # should always be used in inference
        # so setting to eval mode
        self.eval()
        for layer in self.layers:
            layer.eval()
            X = layer(X)
        if return_loss:
            return self.layers[-1].loss_fn(X)
        return X

    def train_step(self, positive, epoch, opt_viz_epoch_freq=2, n_iters=5):
        # TODO: clean 
        positive = torch.Tensor(positive)
        # negative = torch.Tensor(negative)

        self.train()
        X = positive
        for layer in self.layers:
            #for niter in range(n_iters):
                # print(layer)
            layer.train()
            # layer(X)#, is_positive=True)

                # if self.nntype == 'bp':
                #     # just using optimizer in the last layer 
                #     loss = layer.loss_fn(layer(X)).view(-1).mean()
                #     # print(loss)
                #     layer.opt.zero_grad()
                #     loss.backward(retain_graph=True)
                #     layer.opt.step()

                # print(layer.weight.mean(), layer.weight.var(), layer.bias.mean(), layer.bias.var())
                # sys.exit()

            if self.nntype == 'ff':
                # print(layer)
                # print("Before", [ f"Avg(W{i}) = {l.weight.mean()}, {l.weight.shape}" for i,l in enumerate(self.layers) ])
                X = layer(X, notate_opt= (epoch%opt_viz_epoch_freq)==0)
                # print("After", [ f"Avg(W{i}) = {l.weight.mean()}, {l.weight.shape}" for i,l in enumerate(self.layers) ])
            if self.nntype == 'bp':
                X = layer(X)


        if self.nntype == 'bp':
            # just using optimizer in the last layer 
            loss = layer.loss_fn(X).view(-1).mean()
            layer.opt.zero_grad()
            loss.backward(retain_graph=True)
            layer.opt.step()

        # self.eval()
        # from torchinfo import summary
        # print("AT NN")
        # print( summary(self, positive.shape) )
        # sys.exit()


    def fit(self, X, Y, valX, valY, nu=0.05, batch_size=32, epochs=1000, log_freq=100, viz_opt_landscape=True):
        self.nu = nu
        for l in self.layers:
            l.nu = nu

        from time import time
        from sklearn.metrics import accuracy_score
        Y = Y.astype(np.int32)
        self._labels = ylist = set(Y.tolist())
        assert len(ylist) <= self._dims[-1]
        assert all((isinstance(y, int) for y in ylist)), "labels should be integers"
        assert all((0 <= y < self._dims[-1] for y in ylist)), "labels should fall between 0 and %d" % (self._dims[-1],)

        early_stopper = EarlyStopper()

        begin = time()
        for epoch in range(epochs):

            for l in self.layers:
                l.epoch = epoch

            # for y in ylist:
            #     subY = list(ylist - {y})
            #     subX = X[Y == y]
            for batch in self._generate_batches(X[Y == 0], batch_size):
                # real_y = np.zeros((len(batch), len(ylist)))
                # real_y[:, y] = 1
                # fake_y = np.zeros((len(batch), len(ylist)))
                # fake_y[range(len(batch)), random.choices(subY, k=len(batch))] = 1
                self.train_step(positive=batch, epoch=epoch)

            Yhat = self.predict(X, Y, batch_size, calc_thresh=True)

            # check for convergence
            Yhatprob_val = self.predict_proba(valX, batch_size)
            
            def BinaryCrossEntropy(y_true, y_pred):
                y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
                term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
                term_1 = y_true * np.log(y_pred + 1e-7)
                return -np.mean(term_0+term_1, axis=0)
            val_loss = BinaryCrossEntropy(valY.reshape(-1, 1), Yhatprob_val.reshape(-1, 1))[0]

            if early_stopper.early_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

            if np.isnan(val_loss):
                print("Detected nan")
                print([l.r for l in self.layers])
                raise Exception("")

            if epoch % log_freq == 0:
                # print(Yhat)
                acc = accuracy_score(Y, Yhat)
                print("Epoch-%d | Spent=%.4f | Train Accuracy=%.4f | Validation Loss=%.4f | Val Avg(Prob[Y==0])=%.4f | Thresh=%.4f | Val Avg(Prob[Y==1])==%.4f" % (epoch, time() - begin, acc, val_loss, Yhatprob_val[valY==0].mean(), self.threshold, Yhatprob_val[valY==1].mean()))
                begin = time()

                print([ f"Avg(W{i}) = {l.weight.mean()}, {l.weight.shape}" for i,l in enumerate(self.layers) ])
                print([ f"Avg(b{i}) = {l.bias.mean()}, {l.bias.shape}" for i,l in enumerate(self.layers) ])

            # Update hyperparams
            if self.nntype == 'ff':
                self.eval()
                h = torch.Tensor(X)
                for layer in self.layers:
                    layer.eval()
                    h = layer(h)
                    layer.loss_fn(h, update_hyperparams=True)
            elif self.nntype == 'bp':
                self.eval()
                h = self.forward(X, return_loss=False)
                self.layers[-1].loss_fn(h, update_hyperparams=True)

            # print("$$$$$$$$$$$$$$$$")
            # print(self.layers[0].weight.mean(), self.layers[0].weight.var(), self.layers[0].bias.mean(), self.layers[0].bias.var())
            # print(Yhat.mean())
            # print(self.layers[-1](torch.Tensor(X)).mean())
            # print(h.mean(), h.var())


        proba = self.predict_proba(X, batch_size)
        self.threshold = self.calc_threshold(proba, Y, nu)


        if viz_opt_landscape:
            assert self.nntype == 'ff'
            # visualize opt for each layer
            X_for_vis = torch.Tensor( copy.deepcopy(X) )
            self.eval()
            for l in self.layers:
                l.eval()
                l.vis_loss_landscape(X_for_vis)
                X_for_vis = l(X_for_vis)

    @staticmethod
    def calc_threshold(proba, Y, nu):
        return np.quantile(proba[Y==0], 1 - nu)

    def predict_proba(self, X, batch_size=32):
        Yhat = []
        for batch in self._generate_batches(X, batch_size):
            pred_y = self.forward(batch)
            Yhat.extend( list(pred_y.detach().numpy().flatten()) )
        Yhat = np.array(Yhat)
        p = Yhat/Yhat.max()
        return p
                
    def predict(self, X, Y=None, batch_size=32, calc_thresh=False, return_probs=False):
        proba = self.predict_proba(X, batch_size)
        # torch.sum((proba - a)**2, dim=1)
        if calc_thresh:
            self.threshold = self.calc_threshold(proba, Y, self.nu)
        
        preds = (proba > self.threshold).astype(int)
        if return_probs:
            return preds, proba
        return preds

    def _generate_batches(self, X, batch_size=32):
        batch = []
        for sample in X:
            batch.append(sample)
            if len(batch) == batch_size:
                yield np.vstack(batch)
                batch.clear()
        if len(batch) > 0:
            yield np.vstack(batch)
