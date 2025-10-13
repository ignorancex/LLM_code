import os
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import torch
import torchio as tio
import torch.nn.functional as F
import torch.optim as optim
import torch.fft
import torchvision
import wandb
import seaborn as sns
from lightning import LightningModule
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from torch.distributions import MultivariateNormal, kl_divergence

from src.models.modules.branching import BranchingModel
from src.utils.utils_eval import _test_step, get_eval_dictionary, _test_end, gpu_filter
from src.utils.utils import clip_to_percentiles, extract_volume_patches

class Patch2Loc(LightningModule):
    def __init__(self, imageDim, rescaleFactor, patch_percentage, rejection_rate, num_patches_per_batch, cfg,
                 latent_dim, compile, lr, beta, loss_type, median_filter_size=7, abnormality_analysis=False, prefix=None):
        super().__init__()
        image_size = (int(imageDim[0] / rescaleFactor), int(imageDim[1] / rescaleFactor),)
        self.patch_percentage = patch_percentage
        self.image_size = image_size
        self.lr = lr
        self.cfg = cfg
        self.rejection_rate = rejection_rate
        self.num_patches = num_patches_per_batch
        self.beta = beta
        self.prefix = prefix
        self.abnormality_analysis = abnormality_analysis
        self.model = BranchingModel(int(image_size[0] * patch_percentage) * int(image_size[1] * patch_percentage),
                                    latent_dim)
        self.loss_type = loss_type
        if self.loss_type == 'stirn':
            self.model.set_detach_variance()
        if self.loss_type != 'stirn' and self.loss_type != 'beta_nll' and self.loss_type != 'mse':
            raise RuntimeError(f'{self.loss_type} is not supported as a loss type')
        self.patch_size = (int(self.image_size[0] * self.patch_percentage),int(self.image_size[1] * self.patch_percentage))
        self.median_filter_size = median_filter_size
        self.save_hyperparameters(logger=False)


        # Parameters for histograms
        self.bins = 100
        self.range_2d = (0,100)  # Range for the 2D histogram
        self.range_1d = (0,100)  # Range for the 1D histogram

        # Initialize histograms
        self.hist_2d = np.zeros((self.bins, self.bins))  # 2D histogram for dimensions 1 and 2
        self.hist_1d = np.zeros(self.bins)
        self.quantile_thresholds = []
        self.threshold_prefixes = ['Supervised', 'Unsupervised']

    def setup(self, stage):
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)
    

    def update_histograms(self, locations):
        """
        Update running histograms for densities.
        """
        if locations.ndim != 2 or locations.shape[1] != 3:
            raise ValueError("locations should be a batch of 3D coordinates with shape [batch_size, 3].")

        # Convert to numpy for easier handling
        locations_np = locations.detach().cpu().numpy()

        # Update 2D histogram for dimensions 1 and 2
        hist_2d, _, _ = np.histogram2d(
            locations_np[:, 0], locations_np[:, 1],
            bins=self.bins, range=[self.range_2d, self.range_2d]
        )
        self.hist_2d += hist_2d

        # Update 1D histogram for dimension 3
        hist_1d, _ = np.histogram(
            locations_np[:, 2],
            bins=self.bins, range=self.range_1d
        )
        self.hist_1d += hist_1d

        
    def extract_random_patches(self, images, masks,  rejection_rate, patch_size, num_patches):
        """
        Extracts random patches from a batch of images, rejecting patches with too many zeros.
        Fully vectorized implementation with no loops.

        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            rejection_rate (float): Maximum allowed percentage of zeros (0.0 to 1.0)
            patch_size (int,int): Size of patches to extract (assumes square patches)
            num_patches (int): Number of candidate patches to sample initially

        Returns:
            tuple: (
                patches [N, C, patch_size, patch_size], 
                locations [N, 2] (x, y coordinates),
                batch_indices [N] (which batch each patch came from)
            )
            where N is the number of valid patches after rejection
        """
        B, C, H, W = images.shape
        device = images.device

        patch_h, patch_w = patch_size[0], patch_size[1]
        
        # Calculate padding sizes
        pad_h = patch_h // 2
        pad_w = patch_w // 2
        
        # Pad the volume
        images = torch.nn.functional.pad(
            images,
            (pad_w, pad_w, pad_h, pad_h),  # left, right, top, bottom
            mode='constant',
            value=0
        )

        # Generate random locations for all batches
        batch_indices = torch.randint(0, B, (num_patches,), device=device)
        y_coords = torch.randint(0, H, (num_patches,), device=device)
        x_coords = torch.randint(0, W, (num_patches,), device=device)

        # Create offset grid for patch extraction
        y_offsets = torch.arange(patch_size[0], device=device)
        x_offsets = torch.arange(patch_size[1], device=device)
        y_grid, x_grid = torch.meshgrid(y_offsets, x_offsets, indexing='ij')

        # Broadcasting to get all coordinates: [num_patches, patch_size, patch_size]
        y_indices = y_coords[:, None, None] + y_grid[None, :, :]
        x_indices = x_coords[:, None, None] + x_grid[None, :, :]

        # Extract all patches at once using advanced indexing
        patches = images[batch_indices[:, None, None, None],  # [num_patches, 1, 1, 1]
        torch.arange(C, device=device)[None, :, None, None],  # [1, C, 1, 1]
        y_indices[:, None, :, :],  # [num_patches, 1, patch_size, patch_size]
        x_indices[:, None, :, :]]  # [num_patches, 1, patch_size, patch_size]

        masks = torch.nn.functional.pad(
            masks,
            (pad_w, pad_w, pad_h, pad_h),  # left, right, top, bottom
            mode='constant',
            value=0
        )
        patches_masks = masks[batch_indices[:, None, None, None],  # [num_patches, 1, 1, 1]
        torch.arange(C, device=device)[None, :, None, None],  # [1, C, 1, 1]
        y_indices[:, None, :, :],  # [num_patches, 1, patch_size, patch_size]
        x_indices[:, None, :, :]]  # [num_patches, 1, patch_size, patch_size]
        
        
        percentages = (patches_masks==1).float().mean(dim=(1, 2, 3))

        # Create mask for valid patches
        valid_mask = percentages >= rejection_rate

        # Filter patches and locations
        valid_patches = patches[valid_mask]
        valid_locations = torch.stack([y_coords[valid_mask],
                                       x_coords[valid_mask]], dim=1)
        valid_batch_indices = batch_indices[valid_mask]

        return valid_patches, valid_locations, valid_batch_indices


    def visualize_patches(
            self, patch_size, valid_locations, valid_batch_indices, tensors, patch_metrics=None, N=10
    ):
        """
        Visualizes patches on batch images using matplotlib and colors patches based on metrics.

        Args:
            valid_locations (torch.Tensor): Tensor of shape [num_patches, 2], 
                                            specifying the (y, x) locations of top-left corners of patches.
            valid_batch_indices (torch.Tensor): Tensor of shape [num_patches], 
                                                specifying the batch index corresponding to each patch.
            tensors (torch.Tensor): Tensor of shape [batch_size, channels, H, W], 
                                    the batch of images.
            patch_metrics (torch.Tensor or None): Tensor of shape [num_patches], containing
                                                scalar values (e.g., prediction errors or log variances)
                                                for each patch. If None, patches are outlined in red.
            N (int): Number of images to plot. Defaults to 5.

        Returns:
            matplotlib.figure.Figure: Figure with subplots showing images and colored patches.
        """
        # Ensure tensors are in CPU and numpy format for visualization
        tensors = tensors.detach().cpu().numpy()
        valid_locations = valid_locations.detach().cpu().numpy()
        valid_batch_indices = valid_batch_indices.detach().cpu().numpy()

        # If patch_metrics is provided, convert it to numpy
        if patch_metrics is not None:
            patch_metrics = patch_metrics.detach().cpu().numpy()
            # Normalize patch_metrics for colormap mapping
            norm = mcolors.Normalize(vmin=patch_metrics.min(), vmax=patch_metrics.max())
            colormap = matplotlib.colormaps['jet']

        # Initialize figure
        fig, axes = plt.subplots(1, N, figsize=(15, 5))

        # If N is 1, ensure axes is a list for uniform handling
        if N == 1:
            axes = [axes]

        # Determine the height and width of each patch
        patch_h, patch_w = patch_size

        # Loop over the first N batch indices
        for i, ax in enumerate(axes[:N]):
            if i >= len(tensors):
                break

            # Get the i-th image
            img = tensors[i]

            # If the image has multiple channels, transpose to HxWxC for plotting
            if img.shape[0] > 1:
                img = np.transpose(img, (1, 2, 0))
            else:
                img = img.squeeze()

            # Plot the original image
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.axis('off')
            # Extract patches corresponding to the current image
            image_patches = valid_locations[valid_batch_indices == i]
            patch_values = patch_metrics[valid_batch_indices == i] if patch_metrics is not None else None

            for j, loc in enumerate(image_patches):
                y, x = loc
                # Determine color for the patch
                if patch_values is not None:
                    color = colormap(norm(patch_values[j]))
                    facecolor = color
                    edgecolor = color
                else:
                    facecolor = "none"
                    edgecolor = "red"

                # Draw a rectangle to represent the patch with the computed color
                rect = plt.Rectangle(
                    (x, y),
                    patch_w,
                    patch_h,
                    edgecolor=edgecolor,
                    facecolor=facecolor,
                    linewidth=2,
                    alpha=0.6,
                )
                ax.add_patch(rect)

        plt.tight_layout()
        return fig



    def extract_patches_and_locations(self, batch):
        input = batch['vol'][tio.DATA].squeeze(-1)
        brain_masks = batch['mask'][tio.DATA].squeeze(-1)
        slices_num = batch['ind']
        total_slices_num = torch.tensor(batch['total_slices'],device=self.device)
        # input = self.high_pass_filter(input)
        input = brain_masks * input
        patches, locations, batch_indices = self.extract_random_patches(input, brain_masks, self.rejection_rate,
                                                                        (self.patch_percentage * torch.tensor(input.shape[2:])).int(),
                                                                        self.num_patches * 4)
        extra_output = (locations,batch_indices,input,)
        locations = torch.cat([locations, slices_num[batch_indices]], dim=1)
        max_locations = torch.tensor([[input.shape[-2], input.shape[-1]]] * len(locations)).to(self.device)
        max_locations = torch.cat([max_locations, total_slices_num[batch_indices].unsqueeze(1)], dim=1)

        locations = locations / max_locations
        return patches, 100*locations,extra_output


    def beta_nll_loss(self, mean, logvariance, target):
        """Compute beta-NLL loss

        :param mean: Predicted mean of shape B x D
        :param logvariance: Predicted variance of shape B x D
        :param target: Target of shape B x D
        :param beta: Parameter from range [0, 1] controlling relative 
            weighting between data points, where `0` corresponds to 
            high weight on low error points and `1` to an equal weighting.
        :returns: Loss per batch element of shape B
        """
        loss = 0.5 * ((target - mean) ** 2 / logvariance.exp() + logvariance)
        if self.beta > 0:
            loss = loss * (logvariance.exp().detach() ** self.beta)
        return loss.sum(axis=-1).mean()

    def stirn_loss(self, mean, logvariance, target):
        loglikelihood = 0.5 * ((target - mean.detach()) ** 2 / logvariance.exp() + logvariance)
        return (.5*(target - mean) ** 2 + loglikelihood).sum(axis=-1).mean()

    def on_train_epoch_end(self):
        if self.current_epoch%1000==0:
            self.log_location_density()

    def model_step(self, batch, stage):

        patches, locations, extra_output = self.extract_patches_and_locations(batch)
        self.update_histograms(locations)
        predicted_locations, predicted_logvariance = self.model(patches,locations[:,-1])
        locations = locations[:,:-1]

        if self.loss_type == 'stirn':
            loss = self.stirn_loss(predicted_locations, predicted_logvariance, locations)
        elif self.loss_type == 'beta_nll':
            loss = self.beta_nll_loss(predicted_locations, predicted_logvariance, locations)
        elif self.loss_type == 'mse':
            loss = ((locations - predicted_locations)**2).sum(axis=-1).mean()
        if self.current_epoch%500==0:
           self.logger.experiment.log({f'{self.prefix}{stage}/patches':wandb.Image(torchvision.utils.make_grid(patches[:64],nrows=8))})
        self.log(f'{self.prefix}{stage}/Loss_comb', loss, prog_bar=False, on_step=False,
                 on_epoch=True, sync_dist=True)
        self.log(f'{self.prefix}{stage}/mse', abs(predicted_locations - locations).mean(), prog_bar=False,
                 on_step=False,on_epoch=True, sync_dist=True)
        self.log(f'{self.prefix}{stage}/logvariance', predicted_logvariance.mean(), prog_bar=False, on_step=False,
                 on_epoch=True, sync_dist=True)
    

        if self.current_epoch%1000==0:
            pixel_locations, batch_indices, input = extra_output
            metric = ((locations - predicted_locations)**2).sum(axis=-1)
            fig = self.visualize_patches(self.patch_size, pixel_locations, batch_indices, input, metric)
            self.logger.experiment.log({f'{self.prefix}{stage}/patches_locations':wandb.Image(fig)})
        return loss

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        if self.current_epoch%1000==0:
            self.plot_slice_error(batch)
        return self.model_step(batch, 'val')

    def plot_slice_error(self, batch):
        input = batch['vol'][tio.DATA].squeeze(-1)
        data_mask = batch['mask'][tio.DATA].squeeze(-1)
        ID = batch['ID']
        total_slices_num = torch.tensor(batch['total_slices'],device=self.device)
        index = 0
        if input.ndim == 5:
            input = input.squeeze(0).permute(3,0,1,2) 
            data_mask = data_mask.squeeze(0).permute(3,0,1,2) 

        patches, locations = extract_volume_patches(input[index,...].unsqueeze(0),self.patch_size,slice_start=batch['ind'][index], 
                                                         total_slices_num=total_slices_num[index],rejection_rate=self.rejection_rate)
        if patches.ndim < 4:
            patches = patches.unsqueeze(1)
        predicted_locations, predicted_logvariance = self.model(patches,locations[:,-1])
        locations = locations[:,:-1]
        error = abs(predicted_locations - locations).mean(-1)
        error = error.reshape(input[index].shape) * data_mask[index]
        error = error.detach().cpu().numpy().squeeze()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a 1x2 subplot

        # Plot the input image
        axes[0].imshow(np.rot90(input[index].squeeze().cpu().numpy(),3), cmap='gray')
        axes[0].axis('off')  # Hide x and y labels
        axes[0].set_title("Input Slice")

        # Plot the error map
        error_map = axes[1].imshow(np.rot90(error,3), vmin=0, vmax=100, cmap=plt.cm.jet)
        axes[1].axis('off')  # Hide x and y labels
        axes[1].set_title("Error Map")

        # Add a colorbar for the error map
        cbar = fig.colorbar(error_map, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label("Error Magnitude", rotation=270, labelpad=15)

        # Log the figure to WandB
        self.logger.experiment.log({f'val/slice_error/{ID[0]}': wandb.Image(fig)})
        plt.close(fig)

        error = error[error > 0]
        fig = plt.figure()
        plt.hist(error,bins=100,density=True,alpha=.5,label='normal')
        plt.legend()
        self.logger.experiment.log({'val/histograms': wandb.Image(fig)})
        plt.clf()
        plt.cla()
        plt.close()


    def log_location_density(self):
        """
        Logs accumulated density visualizations of the 3D locations.
        """
        # Create a single figure with two subplots
        fig, (ax_2d, ax_1d) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the 2D histogram for the first two dimensions
        ax_2d.imshow(
            self.hist_2d.T, origin='lower', extent=(*self.range_2d, *self.range_2d), cmap='Blues', aspect='auto'
        )
        ax_2d.set_title("2D Density: Dimensions 1 vs 2")
        ax_2d.set_xlabel("Dimension 1")
        ax_2d.set_ylabel("Dimension 2")

        # Plot the 1D histogram for the third dimension
        ax_1d.bar(
            np.linspace(self.range_1d[0], self.range_1d[1], self.bins),
            self.hist_1d,
            width=(self.range_1d[1] - self.range_1d[0]) / self.bins,
            color='blue', alpha=0.7
        )
        ax_1d.set_title("1D Density: Dimension 3")
        ax_1d.set_xlabel("Dimension 3")
        ax_1d.set_ylabel("Frequency")

        plt.tight_layout()
        plt.close(fig)  # Avoid duplicate display in notebooks

        # Log the figure using logger
        self.logger.experiment.log({f"training_location_density": wandb.Image(fig)})

    
    def on_test_start(self):
        self.eval_dict = get_eval_dictionary(self.threshold_prefixes)
        self.inds = []
        self.latentSpace_slice = []
        self.new_size = [160,190,160]
        self.diffs_list = []
        self.seg_list = []
        self.logerror = []
        self.logvariance = []
        self.seg = []
        self.valid_indices = []
        self.model = self.model.half()
        if not hasattr(self,'threshold'):
            self.threshold = {}
        self.patch2loc_score_values = []

    
    def do_abnormality_analysis(self, batch, batch_idx):
        if not batch['seg_available'][0]:
            raise RuntimeError('cannot run abnormality analysis for dataset without abnormals')
        input = batch['vol'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA]
        data_mask = batch['mask_orig'][tio.DATA]
        input = input.squeeze(0).permute(3,0,1,2) 
        brain_masks = data_mask.squeeze(0).permute(3,0,1,2) 
        data_seg = data_seg.squeeze(0).permute(3,0,1,2) 

        patches, locations = extract_volume_patches(input, self.patch_size,slice_start=0, total_slices_num=input.shape[0],
                                                             stride=(self.patch_size[0],self.patch_size[1],1),rejection_rate=self.rejection_rate,brain_masks=brain_masks)
        
        patches_seg, _ = extract_volume_patches(data_seg, self.patch_size,slice_start=0, total_slices_num=input.shape[0],
                                                             stride=(self.patch_size[0],self.patch_size[1],1),rejection_rate=self.rejection_rate,brain_masks=brain_masks)
        
        predicted_locations, predicted_logvariance = self.model(patches.half(), locations[:, -1])
        logerror = (torch.linalg.norm(predicted_locations - locations[:, :-1],dim=1).cpu()**2 + 5e-1).log()

        patches_seg = patches_seg.mean((1,2,3)).cpu().numpy()
        self.logerror.append(logerror.numpy())
        self.logvariance.append(predicted_logvariance.cpu().numpy().mean(-1))
        self.seg.append(patches_seg)

    def do_abnormality_analysis_test_end(self):
        self.logerror = np.concatenate(self.logerror)
        self.logvariance = np.concatenate(self.logvariance)
        self.seg = np.concatenate(self.seg)

        spearman_logerror_seg = spearmanr(self.logerror, self.seg, alternative='greater')
        spearman_logvariance_seg = spearmanr(self.logvariance, self.seg, alternative='greater')
        spearman_combined_seg = spearmanr(self.logerror + self.logvariance, self.seg, alternative='greater')

        metrics = {
            "Spearman logerror vs seg (correlation)": spearman_logerror_seg.correlation,
            "Spearman logerror vs seg (p-value)": spearman_logerror_seg.pvalue,
            "Spearman logvariance vs seg (correlation)": spearman_logvariance_seg.correlation,
            "Spearman logvariance vs seg (p-value)": spearman_logvariance_seg.pvalue,
            "Spearman logerror  + logvariance vs seg (correlation)": spearman_combined_seg.correlation,
            "Spearman logerror  + logvariance vs seg (p-value)": spearman_combined_seg.pvalue,
        }

        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True)

        

        valid_indices = np.logical_or(self.seg  < .1, self.seg  > .9)
        self.seg[self.seg > .9] = 1
        self.seg[self.seg < .1] = 0

        self.logerror = self.logerror[valid_indices]
        self.logvariance = self.logvariance[valid_indices]
        self.seg = self.seg[valid_indices]

        self.plot_kde_log_mean_variance(self.logerror, self.logvariance,  self.seg.astype(bool), self.dataset, '')
        
        self.plot_hist(self.logerror * self.logvariance,
                       np.ones_like(self.logerror),self.seg.astype(bool), self.dataset, '')
        self.logerror = torch.from_numpy(self.logerror).unsqueeze(1)
        self.logvariance = torch.from_numpy(self.logvariance).unsqueeze(1)
        if self.seg.sum() > 0:
            kl_divergence_score = self.kl_divergence_gaussians_torch(torch.cat([self.logerror[self.seg == 0], self.logvariance[self.seg == 0]],dim=1), 
                                                                    torch.cat([self.logerror[self.seg == 1], self.logvariance[self.seg == 1]],dim=1))
            print(f'KL Divergene between normal and abnormal distribution is:{kl_divergence_score}')


    def test_step(self, batch, batch_idx):
        self.dataset = batch['Dataset']
        if self.abnormality_analysis:
            self.do_abnormality_analysis(batch, batch_idx)
            return 
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'][0] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID'][0]
        age = batch['age'][0]
        self.stage = batch['stage'][0]
        label = batch['label'][0]
        # self.plot_slice_error(batch)
        input = input.squeeze(0).permute(3,0,1,2) 
        brain_masks = data_mask.squeeze(0).permute(3,0,1,2) 
        # Determine the number of chunks
        num_chunks = 15 # Adjust this based on memory constraints
        chunk_size = input.size(0) // num_chunks
        results_error = []
        results_logvariance = []
        stride = (1,1,1)
        for start in range(0, input.size(0), chunk_size):
            # Slice the input tensor for the current chunk
            chunk = input[start: start + chunk_size]

            # Process the chunk
            patches, locations = extract_volume_patches(chunk, self.patch_size,slice_start=start, total_slices_num=input.shape[0],
                                                             stride=stride,rejection_rate=self.rejection_rate)
            predicted_locations, predicted_logvariance = self.model(patches.half(), locations[:, -1])
            
            # Calculate error and reshape
            error = torch.linalg.norm(predicted_locations - locations[:, :-1],dim=1)**2
            error = self.reshape_and_upsample(error, chunk.shape, stride)
            predicted_logvariance = self.reshape_and_upsample(predicted_logvariance.mean(-1),chunk.shape, stride)

            # Move results to CPU to free GPU memory
            results_error.append(error.cpu())
            results_logvariance.append(predicted_logvariance.cpu())

        # Concatenate results from all chunks
        error = torch.log(torch.cat(results_error, dim=0).squeeze().permute(1, 2, 0)+1e-10)
        predicted_logvariance = torch.cat(results_logvariance, dim=0).squeeze().permute(1, 2, 0)

        # Final computation
        heatmap = error + predicted_logvariance
        if self.dataset[0] == 'IXI':
            self.patch2loc_score_values.append(heatmap[brain_masks.squeeze().permute(1,2,0).cpu().bool()])
            _test_step(self, heatmap.float().cpu() , data_orig.float().cpu(), data_seg.float().cpu(), data_mask.float().cpu(), batch_idx, ID, label) # everything that is independent of the model choice
            return
        heatmap = gpu_filter(heatmap, type_param=self.median_filter_size, type='median')
        _test_step(self, heatmap.float().cpu() , data_orig.float().cpu(), data_seg.float().cpu(), data_mask.float().cpu(), batch_idx, ID, label) # everything that is independent of the model choice
    def calculate_quantiles(self):
        """
        Calculate the 85%, 90%, 95%, and 99% quantiles of the stored data.
        
        Returns:
            dict: Quantiles as keys and their corresponding values.
        """
        if not self.patch2loc_score_values:
            raise ValueError("No data to calculate quantiles.")

        # Combine all values into a single tensor
        all_values = torch.cat(self.patch2loc_score_values)

        # Define quantiles
        quantile_levels = np.array([0.85, 0.90, 0.95,.98 ,0.99])

        # Calculate quantiles
        quantile_values = np.quantile(all_values.numpy(), quantile_levels)

        # Return results as a dictionary
        return {f"{int(q*100)}": v.item() for q, v in zip(quantile_levels, quantile_values)}

    def on_test_epoch_end(self):
        #stupid lightning doesn't let to log in on_test_end
        if self.abnormality_analysis:
            self.do_abnormality_analysis_test_end()
        return super().on_test_epoch_end()
    
    def on_test_end(self) :
        if self.abnormality_analysis:
            return 
        # calculate metrics
        # if self.dataset[0] == 'IXI':
        #     self.healthy_sets = ['IXI']
        #     quantiles = self.calculate_quantiles()
        #     self.quantile_thresholds = quantiles
        #     self.cfg.unsupervised_threshold = quantiles[str(self.cfg.unsupervised_quantile)]
        #     self.patch2loc_score_values = None
        _test_end(self) # everything that is independent of the model choice 



    def kl_divergence_gaussians_torch(self, sample1, sample2, reg_eps=1e-5):
        """
        Computes the KL divergence between two Gaussian distributions using PyTorch's built-in API.
        Handles NaNs and singular covariance matrices.

        Parameters:
            sample1 (torch.Tensor): Samples from the first Gaussian (shape: [N, D])
            sample2 (torch.Tensor): Samples from the second Gaussian (shape: [N, D])
            reg_eps (float): Small value added to the diagonal to prevent singular matrices.

        Returns:
            torch.Tensor: KL divergence scalar value.
        """
        # Handle NaN values by replacing with column means
        sample1 = torch.where(torch.isnan(sample1), torch.nanmean(sample1, dim=0, keepdim=True), sample1)
        sample2 = torch.where(torch.isnan(sample2), torch.nanmean(sample2, dim=0, keepdim=True), sample2)

        # Estimate means
        mu1, mu2 = sample1.mean(dim=0), sample2.mean(dim=0)

        # Compute empirical covariance matrices
        cov1, cov2 = torch.cov(sample1.T), torch.cov(sample2.T)

        # Ensure covariance matrices are non-singular by adding a small diagonal term
        identity = torch.eye(cov1.shape[0], device=sample1.device)
        cov1 += reg_eps * identity
        cov2 += reg_eps * identity

        # Define Gaussian distributions
        dist1 = MultivariateNormal(mu1, covariance_matrix=cov1)
        dist2 = MultivariateNormal(mu2, covariance_matrix=cov2)

        # Compute KL divergence using PyTorch's built-in function
        kl_div = kl_divergence(dist1, dist2)
        
        return kl_div


    def plot_kde_log_mean_variance(self, log_mean, log_variance, seg, ID, plot_type):
        """
        Plots KDE for log-mean, log-variance, and their 2D KDE for normal and abnormal regions.

        :param log_mean: Tensor of log-means for each region.
        :param log_variance: Tensor of log-variances for each region.
        :param seg: Segmentation tensor indicating normal and abnormal regions.
        :param ID: Unique identifier for the sample.
        :param plot_type: Type of the plot (used for labeling and saving).
        """
        # Prepare the path for saving images
        ImagePathList = {
            'imagesGrid': os.path.join(os.getcwd(), 'kde_plots')
        }
        for key in ImagePathList:
            if not os.path.isdir(ImagePathList[key]):
                os.mkdir(ImagePathList[key])

        # Separate normal and abnormal regions
        normal_log_mean = log_mean[~seg]
        abnormal_log_mean = log_mean[seg]
        normal_log_variance = log_variance[~seg]
        abnormal_log_variance = log_variance[seg]

        normal_log_mean = clip_to_percentiles(normal_log_mean)
        abnormal_log_variance = clip_to_percentiles(abnormal_log_variance)
        abnormal_log_mean = clip_to_percentiles(abnormal_log_mean)
        normal_log_variance = clip_to_percentiles(normal_log_variance)
        # Create KDE plots for log-mean and log-variance
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot KDE for log-mean
        if len(normal_log_mean) > 0:
            sns.kdeplot(normal_log_mean, ax=axes[0], label='Normal', fill=True, color='blue', alpha=0.5)
        if len(abnormal_log_mean) > 0:
            sns.kdeplot(abnormal_log_mean, ax=axes[0], label='Abnormal', fill=True, color='red', alpha=0.5)
        axes[0].set_title(r'$\log(\mathrm{Error}^2)$ KDE')
        axes[0].set_xlabel(r'$\log(\mathrm{Error}^2)$')
        axes[0].set_ylabel('Density')
        axes[0].legend()

        # Plot KDE for log-variance
        if len(normal_log_variance) > 0:
            sns.kdeplot(normal_log_variance, ax=axes[1], label='Normal', fill=True, color='blue', alpha=0.5)
        if len(abnormal_log_variance) > 0:
            sns.kdeplot(abnormal_log_variance, ax=axes[1], label='Abnormal', fill=True, color='red', alpha=0.5)
        axes[1].set_title(r'$\log(\mathrm{Variance})$ KDE')
        axes[1].set_xlabel(r'$\log(\mathrm{Variance})$')
        axes[1].set_ylabel('Density')
        axes[1].legend()

        # Plot 2D KDE for log-mean and log-variance
        if len(normal_log_mean) > 0 and len(normal_log_variance) > 0:
            sns.kdeplot(
                x=normal_log_mean, y=normal_log_variance, 
                ax=axes[2], cmap='Blues', fill=True, alpha=0.5, label='Normal'
            )
        if len(abnormal_log_mean) > 0 and len(abnormal_log_variance) > 0:
            sns.kdeplot(
                x=abnormal_log_mean, y=abnormal_log_variance, 
                ax=axes[2], cmap='Reds', fill=True, alpha=0.5, label='Abnormal'
            )
        axes[2].set_title(r'2D KDE $\log(\mathrm{Error}^2)$ vs $\log(\mathrm{Variance})$')
        axes[2].set_xlabel(r'$\log(\mathrm{Error}^2)$')
        axes[2].set_ylabel(r'$\log(\mathrm{Variance})$')
        axes[2].legend()

        # Save and log the plots
        if self.cfg.get('save_to_disc', True):
            plt.savefig(
                os.path.join(ImagePathList['imagesGrid'], f'{ID[0]}_{plot_type}_kde.png'), 
                bbox_inches='tight'
            )
        self.logger.experiment.log({
            f'kde_plots/{self.dataset[0] + plot_type}/{ID[0]}.png': wandb.Image(plt)
        })

        # Clear the figure
        plt.clf()
        plt.cla()
        plt.close()


    def plot_hist(self,score,mask,seg, ID, type):
        ImagePathList = {
                    'imagesGrid': os.path.join(os.getcwd(),'histograms')}
        for key in ImagePathList :
            if not os.path.isdir(ImagePathList[key]):
                os.mkdir(ImagePathList[key])
        score = (score*mask)
        abnormal_scores = score[np.logical_and(seg > 1e-3, score > 0)]
        normal_scores = score[np.logical_and(seg <= 1e-3, score > 0)]
        fig = plt.figure()
        plt.hist(normal_scores,bins=100,density=True,alpha=.5,label='normal')
        plt.hist(abnormal_scores,bins=100,density=True,alpha=.8,label='abnormal')
        plt.legend()
        if self.cfg.get('save_to_disc',True):
            plt.savefig(os.path.join(ImagePathList['imagesGrid'], '{}_histogram.png'.format(ID[0])),bbox_inches='tight')
        self.logger.experiment.log({f'histograms/{self.dataset[0] + type}/{ID[0]}.png' : wandb.Image(fig)})
        plt.clf()
        plt.cla()
        plt.close()



    def reshape_and_upsample(self, tensor, input_shape, stride):
        """
        Reshapes tensor to its corresponding 3D shape based on the input dimensions and stride,
        and upsamples it to match the input shape.

        Args:
            tensor (torch.Tensor)
            input_shape (tuple): Original input shape (num_slices, H, W).
            stride (tuple): Strides (stride_z, stride_h, stride_w).

        Returns:
            torch.Tensor: Reshaped and upsampled tensor matching the input shape [num_slices, H, W].
        """
        num_slices, C, H, W = input_shape
        stride_h, stride_w, stride_z = stride

        # Compute the reduced dimensions based on the stride
        H_out = (H + stride_h - 1) // stride_h
        W_out = (W + stride_w - 1) // stride_w
        num_slices_out = (num_slices + stride_z - 1) // stride_z

        # Reshape locations to the reduced 3D grid
        reshaped = tensor.reshape((H_out, W_out, C, num_slices)).permute(3, 2, 0, 1)
        if reshaped.shape[-1] == W and reshaped.shape[-2] == H:
            return reshaped
        # Upsample to match the input shape
        reshaped_upsampled = F.interpolate(
            reshaped, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=True
        )

        return reshaped_upsampled

    def on_train_epoch_start(self):
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log(f'{self.prefix}train/lr', current_lr, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def on_train_start(self):
        # Update optimizer learning rate
        new_lr = self.lr
        
        # Access the first optimizer
        optimizer = self.optimizers()
        
        # Now modify the param groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Reset scheduler if needed
        scheduler = self.lr_schedulers()  # Also access first scheduler if multiple
        if scheduler:
            scheduler.base_lrs = [new_lr]
            scheduler.last_epoch = -1

    def update_prefix(self, prefix):
        self.prefix = prefix