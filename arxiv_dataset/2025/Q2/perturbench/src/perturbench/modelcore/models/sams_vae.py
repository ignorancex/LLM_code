from typing import Tuple
import torch
import torch.distributions as dist
import lightning as L
import numpy as np

from perturbench.data.types import Batch
from ..nn.vae import BaseEncoder
from ..nn.mlp import gumbel_softmax_bernoulli
from .base import PerturbationModel
from ..nn.decoders import (
    DeepIsotropicGaussian,
)


class SparseAdditiveVAE(PerturbationModel):
    """
    Sparse Additive Variational Autoencoder (VAE) model, following the model proposed in the paper:

    Bereket, Michael, and Theofanis Karaletsos.
    "Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder."
    Advances in Neural Information Processing Systems 36 (2024).

    Attributes:
        n_genes (int): Number of genes.
        n_perts (int): Number of perturbations.
        lr (int): Learning rate.
        wd (int): Weight decay.
        lr_scheduler_freq (int): Frequency of the learning rate scheduler.
        lr_scheduler_patience (int): Patience of the learning rate scheduler.
        lr_scheduler_factor (int): Factor of the learning rate scheduler.
        latent_dim (int): Latent dimension.
        sparse_additive_mechanism (bool): Whether to use sparse additive mechanism.
        mean_field_encoding (bool): Whether to use mean field encoding.
        inject_covariates_encoder (bool): Whether to inject covariates in the encoder.
        inject_covariates_decoder (bool): Whether to inject covariates in the decoder.
        mask_prior_probability (float): The target probability for the masks.
        datamodule (L.LightningDataModule | None): LightningDataModule for data loading.

    Methods:
        reparameterize(mu, log_var): Reparametrizes the Gaussian distribution.
        training_step(batch, batch_idx): Performs a training step.
        validation_step(batch, batch_idx): Performs a validation step.
        configure_optimizers(): Configures the optimizers.

    """

    def __init__(
        self,
        n_genes: int | None = None,
        n_perts: int | None = None,
        n_layers_encoder_x: int = 2,
        n_layers_encoder_e: int = 2,
        n_layers_decoder: int = 3,
        hidden_dim_x: int = 850,
        hidden_dim_cond: int = 128,
        latent_dim: int = 40,
        dropout: float = 0.2,
        inject_covariates_encoder: bool = False,
        inject_covariates_decoder: bool = False,
        mask_prior_probability: float = 0.01,
        lr: int | None = None,
        wd: int | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        softplus_output: bool = True,
        generative_counterfactual: bool = False,
        embedding_width: int | None = None,
        disable_sparsity: bool = False,
        disable_e_dist: bool = False,
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        Initializes the SparseAdditiveVAE model.

        Args:
            n_genes (int): Number of genes.
            n_perts (int): Number of perturbations.
            transform (Dispatch): Transform for the data.
            context (dict): Context for the data.
            evaluation (DictConfig): Evaluation configuration.
            n_layers_encoder_x (int): Number of layers in the encoder for x.
            n_layers_encoder_e (int): Number of layers in the encoder for e.
            n_layers_decoder (int): Number of layers in the decoder.
            hidden_dim_x (int): Hidden dimension for x.
            hidden_dim_cond (int): Hidden dimension for the conditional input.
            latent_dim (int): Latent dimension.
            lr (int): Learning rate.
            wd (int): Weight decay.
            lr_scheduler_freq (int): Frequency of the learning rate scheduler.
            lr_scheduler_patience (int): Patience of the learning rate scheduler.
            lr_scheduler_factor (int): Factor of the learning rate scheduler.
            inject_covariates_encoder (bool): Whether to inject covariates in the encoder.
            inject_covariates_decoder (bool): Whether to inject covariates in the decoder.
            mask_prior_probability (float): The target probability for the masks.
            softplus_output (bool): Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            generative_counterfactual (bool): Whether to use the generative mode, i.e. sample from the prior distribution. Only used for inference.

        Returns:
            None
        """
        if datamodule is not None:
            n_genes = datamodule.num_genes
            n_perts = datamodule.num_perturbations
        
        super(SparseAdditiveVAE, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
        )
        self.save_hyperparameters(ignore=["datamodule"])

        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        self.latent_dim = latent_dim
        self.latent_dim_pert = latent_dim * self.n_perts
        self.inject_covariates_encoder = inject_covariates_encoder
        self.inject_covariates_decoder = inject_covariates_decoder
        self.mask_prior_probability = mask_prior_probability
        self.softplus_output = softplus_output
        self.generative_counterfactual = generative_counterfactual
        
        perturbations_all = datamodule.train_dataset.transform['perturbations'](list(datamodule.train_dataset.perturbations))
        self.perturbations_all_sum = perturbations_all.sum(axis=0)

        if self.inject_covariates_encoder or self.inject_covariates_decoder:
            if datamodule is None or datamodule.train_context is None:
                raise ValueError(
                    "If inject_covariates is True, datamodule must be provided"
                )
            self.n_total_covariates = np.sum(
                [
                    len(unique_covs)
                    for unique_covs in datamodule.train_context[
                        "covariate_uniques"
                    ].values()
                ]
            )

        encoder_input_dim = (
            self.n_genes + self.n_total_covariates
            if self.inject_covariates_encoder
            else self.n_genes
        )
        decoder_input_dim = (
            latent_dim + self.n_total_covariates
            if self.inject_covariates_decoder
            else latent_dim
        )

        self.encoder_x = BaseEncoder(
            input_dim=encoder_input_dim + self.latent_dim,
            hidden_dim=hidden_dim_x,
            latent_dim=latent_dim,
            n_layers=n_layers_encoder_x,
        )

        self.disable_sparsity = disable_sparsity
        self.disable_e_dist = disable_e_dist
        self.encoder_e = BaseEncoder(
            input_dim=latent_dim + self.n_perts
            if not self.disable_sparsity
            else self.n_perts,
            hidden_dim=hidden_dim_x,
            latent_dim=latent_dim,
            n_layers=n_layers_encoder_e,
        )

        self.m_logits = torch.nn.Parameter(-torch.ones((self.n_perts, self.latent_dim)))

        self.decoder = DeepIsotropicGaussian(
            decoder_input_dim,
            hidden_dim_x,
            self.n_genes,
            n_layers_decoder,
            dropout,
            softplus_output,
        )

    def forward(
        self,
        observed_perturbed_expression: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict,
        inference: bool = False,
    ) -> Tuple:
        batch_size = observed_perturbed_expression.shape[0]
        perturbations_per_cell = perturbation.sum(axis=1)

        if self.inject_covariates_encoder or self.inject_covariates_decoder:
            merged_covariates = torch.cat(
                [cov.squeeze() for cov in covariates.values()], dim=1
            )

        if self.inject_covariates_encoder:
            observed_expression_with_covariates = torch.cat(
                [observed_perturbed_expression, merged_covariates.to(self.device)],
                dim=1,
            )
        else:
            observed_expression_with_covariates = observed_perturbed_expression

        if self.disable_sparsity:
            m = torch.ones_like(self.m_logits)
        else:
            m_probs = torch.sigmoid(self.m_logits)
            m = gumbel_softmax_bernoulli(m_probs)

        # Get indices where perturbations are active (1s)
        z_p_index_batch, z_p_index_pert = torch.where(perturbation.bool())

        # Initialize z_p with zeros early
        z_p = torch.zeros((batch_size, self.latent_dim), device=self.device)
        # Initialize e_mu and e_log_var as None
        e_mu, e_log_var = None, None

        # Only process perturbations if there are any in the batch
        if z_p_index_batch.nelement() > 0:
            m_t = torch.cat(
                [
                    m[perturbation[i].bool()]
                    for i in range(batch_size)
                    if perturbation[i].bool().any()
                ]
            )
            perturbation_expanded = perturbation.repeat_interleave(
                perturbations_per_cell.int(), dim=0
            )

            if self.disable_sparsity:
                mask_and_perturbation = perturbation_expanded
            else:
                mask_and_perturbation = torch.cat([m_t, perturbation_expanded], dim=-1)
            e_mu, e_log_var = self.encoder_e(mask_and_perturbation)

            if self.disable_e_dist:
                e_t = e_mu
            else:
                # Sample from q(e|x,p)
                e_dist = dist.Normal(e_mu, torch.exp(0.5 * e_log_var).clip(min=1e-8))
                e_t = e_dist.rsample()

            # Calculate element-wise product
            combined_effect = m_t * e_t

            # Use scatter_add_ to sum the effects for each batch sample
            z_p.index_add_(0, z_p_index_batch, combined_effect)

        observed_expression_with_covariates_and_z_p = torch.cat(
            [observed_expression_with_covariates, z_p], dim=-1
        )  # use torch.zeros_like(z_p) to mimic posterior inference
        z_mu_x, z_log_var_x = self.encoder_x(
            observed_expression_with_covariates_and_z_p
        )

        # Sample from q(z|x)
        q_z = dist.Normal(z_mu_x, torch.exp(0.5 * z_log_var_x).clip(min=1e-8))
        # only z_basal is sampled from the prior at inference time
        z_basal = q_z.rsample() if not inference else torch.randn_like(z_mu_x)

        z = z_basal + z_p

        if self.inject_covariates_decoder:
            z = torch.cat([z, merged_covariates], dim=1)

        predictions = self.decoder(z, library_size=None)

        # Compute log probabilities
        p_z = dist.Normal(torch.zeros_like(z_mu_x), torch.ones_like(z_mu_x))
        log_qz = q_z.log_prob(z_basal).sum(axis=-1)
        log_pz = p_z.log_prob(z_basal).sum(axis=-1)

        # Initialize log_qe and log_pe
        log_qe_pe = torch.zeros(batch_size, device=self.device)

        if not self.disable_e_dist:
            # Calculate log probabilities for perturbation effects if there are any
            if e_mu is not None:
                q_e = dist.Normal(e_mu, torch.exp(0.5 * e_log_var).clip(min=1e-8))
                p_e = dist.Normal(torch.zeros_like(e_mu), torch.ones_like(e_mu))
                log_qe = q_e.log_prob(e_t).sum(axis=-1)
                log_pe = p_e.log_prob(e_t).sum(axis=-1)

                # Add log prob terms to the correct batch samples
                log_qe_pe.index_add_(0, z_p_index_batch, log_qe - log_pe)

            # Apply adjustment factor
            adjustment_factor = 1 / (
                perturbation @ self.perturbations_all_sum.to(self.device)
            )

            # Set adjustment_factor to 0 if it is 0 to avoid division by 0 for control values
            adjustment_factor[adjustment_factor.isinf()] = 0
            log_qe_pe = log_qe_pe * adjustment_factor

        # Compute reconstruction loss
        reconstruction_loss = self.decoder.reconstruction_loss(
            predictions, observed_perturbed_expression
        )

        if self.disable_sparsity:
            log_qm_pm = torch.zeros(
                perturbation.shape[1],
                device=reconstruction_loss.device,
                dtype=reconstruction_loss.dtype,
            )
        else:
            # Compute mask prior log probabilities
            q_m = dist.Bernoulli(probs=torch.sigmoid(self.m_logits))
            p_m = dist.Bernoulli(
                probs=self.mask_prior_probability * torch.ones_like(self.m_logits)
            )
            log_qm_pm = (q_m.log_prob(m) - p_m.log_prob(m)).sum(axis=-1)
            log_qm_pm = (
                log_qm_pm
                * perturbation.sum(axis=0)
                / self.perturbations_all_sum.to(self.device)
            )

        # Final ELBO computation
        kld = (log_qz - log_pz).mean() + log_qe_pe.mean() + log_qm_pm.sum() / batch_size
        elbo = -(reconstruction_loss + kld)

        return predictions, reconstruction_loss, kld, elbo

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        observed_perturbed_expression = batch.gene_expression.squeeze()
        perturbation = batch.perturbations.squeeze()
        covariates = batch.covariates

        _, mse, kld, elbo = self(
            observed_perturbed_expression, perturbation, covariates
        )
        loss = -elbo  # Minimize negative ELBO
        self.log(
            "kld",
            kld,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        self.log(
            "recon_loss",
            mse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        self.log(
            "elbo",
            elbo,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        observed_perturbed_expression = batch.gene_expression.squeeze()
        perturbation = batch.perturbations.squeeze()
        covariates = batch.covariates

        _, mse, kld, elbo = self(
            observed_perturbed_expression, perturbation, covariates
        )
        val_loss = -elbo  # Minimize negative ELBO
        self.log(
            "val_kld",
            kld,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        return val_loss

    def predict(self, batch: Batch) -> torch.Tensor:
        observed_perturbed_expression = batch.gene_expression.squeeze().to(self.device)
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = batch.covariates

        if self.generative_counterfactual:
            x_sample, _, _, _ = self(
                observed_perturbed_expression, perturbation, covariates, inference=True
            )
        else:
            x_sample, _, _, _ = self(
                observed_perturbed_expression, perturbation, covariates, inference=False
            )
        return x_sample

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparametrizes the Gaussian distribution so (stochastic) backpropagation can be applied.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std
