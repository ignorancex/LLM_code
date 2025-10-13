from typing import Literal, Optional
import math
from functools import partial

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.tree_util import tree_map
import optax

from src.models.transformer import EncoderTransformer, DecoderTransformer
from src.models.utils import EncoderTransformerConfig, DecoderTransformerConfig
from src.data_utils import make_leave_one_out


class LPN(nn.Module):
    encoder: EncoderTransformer
    decoder: DecoderTransformer

    def __call__(
        self,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
        mode: Literal["mean", "all", "random_search", "gradient_ascent"],
        prior_kl_coeff: Optional[float] = None,
        pairwise_kl_coeff: Optional[float] = None,
        **mode_kwargs,
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        """
        Forward pass of the LPN model.

        Args:
            pairs: input data as tokens. Shape (*B, N, R, C, 2).
                - N: number of (input, output) pairs per progam.
                - R: number of rows.
                - C: number of columns.
                - 2: two channels (input and output)
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, N, 2, 2). The last two dimension
                represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].
                Expects grid shapes values to be in [1, max_rows] and [1, max_cols].
            dropout_eval: if false dropout is applied otherwise it is not.
            mode: mode of the forward pass. Can be "mean" or "all".
                - "mean": decodes the output using the mean latent of all the other pairs.
                - "all": decodes the output N-1 times, each time using a different latent from the other
                    pairs.
                - "random_search": randomly search for a latent that best explains the (input, output) pairs
                    and then decodes the output using that latent.
                - "gradient_ascent": uses gradient ascent to find the latent that best explains the
                    (input, output) pairs and then decodes the output using that latent
            prior_kl_coeff: KL divergence coefficient for the variational inference. Required when using
                variational inference.
            pairwise_kl_coeff: KL divergence coefficient for the pairwise KL divergence. Optional.
            mode_kwargs: additional keyword arguments for the inference mode (e.g. 'remove_encoder_latents').

        Returns:
            loss: loss value.
            metrics: dictionary of metrics.
        """
        assert pairs.shape[-4] > 1, f"Number of pairs should be greater than 1, got {pairs.shape[-4]}."
        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, dropout_eval)

        if latents_logvar is not None:
            key = self.make_rng("latents")
            latents, prior_kl_loss, kl_metrics = self._sample_latents(latents_mu, latents_logvar, key)
            # Compute Gaussian KL between all the latents from each batch.
            pairwise_kl_loss = self._compute_pairwise_gaussian_kl(latents_mu, latents_logvar).mean()
            kl_metrics["pairwise_kl"] = pairwise_kl_loss
        else:
            latents, prior_kl_loss, pairwise_kl_loss, kl_metrics = latents_mu, None, None, {}

        if mode_kwargs.get("remove_encoder_latents", False):
            key = self.make_rng("latents_init")
            latents = jax.random.normal(key, latents.shape)
        leave_one_out_latents = make_leave_one_out(latents, axis=-2)  # (*B, N, N-1, H)
        if mode == "mean":
            # Compute the context vector by taking the mean of all but one latents.
            context = leave_one_out_latents.mean(axis=-2)  # (*B, N, H)
            # Compute the loss for each pair using the mean of all but one latents. Shape (*B, N).
            loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
        elif mode == "all":
            # Compute the loss for each pair using all but one latents. Shape (*B, N, N-1).
            loss, metrics = jax.vmap(
                self._loss_from_pair_and_context, in_axes=(-2, None, None, None), out_axes=-1
            )(leave_one_out_latents, pairs, grid_shapes, dropout_eval)
            # For logging purposes
            context = latents
            distance_context_latents = norm(latents[..., None, :] - leave_one_out_latents, axis=-1)
        elif mode == "random_search":
            for arg in ["num_samples", "scale"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'random_search' training mode."
            key = self.make_rng("random_search")
            # Repeat all the pairs and grid shapes except the one to leave out.
            leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)  # (*B, N, N-1, R, C, 2)
            leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)  # (*B, N, N-1, 2, 2)
            # Get the best context for each pair using random search.
            context, _ = self._get_random_search_context(
                leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key, **mode_kwargs
            )  # (*B, N, H)
            # Compute the loss for each pair using the context from the random search. Shape (*B, N).
            loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
        elif mode == "gradient_ascent":
            for arg in ["num_steps", "lr"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'gradient_ascent' training mode."
            if mode_kwargs.get("random_perturbation", None) is not None:
                key = self.make_rng("gradient_ascent_random_perturbation")
            else:
                key = None
            # Repeat all the pairs and grid shapes except the one to leave out.
            leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)  # (*B, N, N-1, R, C, 2)
            leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)  # (*B, N, N-1, 2, 2)
            # Get the best context for each pair using gradient ascent.
            context, _ = self._get_gradient_ascent_context(
                leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, key, **mode_kwargs
            )  # (*B, N, H)
            # Compute the loss for each pair using the context from the gradient ascent. Shape (*B, N).
            loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, dropout_eval)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        leave_one_out_contexts = make_leave_one_out(context, axis=-2)
        cosine_between_contexts = jnp.einsum("...h,...nh->...n", context, leave_one_out_contexts) / (
            norm(context, axis=-1)[..., None] * norm(leave_one_out_contexts, axis=-1) + 1e-5
        )
        cosine_between_latents = jnp.einsum("...h,...nh->...n", latents, leave_one_out_latents) / (
            norm(latents, axis=-1)[..., None] * norm(leave_one_out_latents, axis=-1) + 1e-5
        )
        if mode != "all":
            distance_context_latents = norm(context - latents, axis=-1)
        metrics.update(
            latents_norm=norm(latents, axis=-1),
            context_norm=norm(context, axis=-1),
            distance_context_latents=distance_context_latents,
            distance_between_contexts=norm(context[..., None, :] - leave_one_out_contexts, axis=-1),
            cosine_between_contexts=cosine_between_contexts,
            distance_between_latents=norm(latents[..., None, :] - leave_one_out_latents, axis=-1),
            cosine_between_latents=cosine_between_latents,
        )
        loss, metrics = tree_map(jnp.mean, (loss, metrics))
        metrics.update(kl_metrics)
        if prior_kl_loss is not None:
            if prior_kl_coeff is None:
                raise ValueError("Prior KL coefficient is required when using variational inference.")
            loss += prior_kl_coeff * prior_kl_loss
            if pairwise_kl_coeff is not None:
                loss += pairwise_kl_coeff * pairwise_kl_loss

        return loss, metrics

    @staticmethod
    def _compute_pairwise_gaussian_kl(mu: chex.Array, log_var: chex.Array, eps: float = 1e-7) -> chex.Array:
        """
        Compute pairwise KL divergence between Gaussian distributions.

        Args:
            mu: mean of shape (*B, N, H)
            log_var: log variance of shape (*B, N, H)

        Returns:
            Mean KL divergence of shape B where before averaging, KL[..., i, j] is
            KL(N(mu[..., i], exp(log_var[..., i])) || N(mu[..., j], exp(log_var[..., j])))
        """
        # Expand dimensions for broadcasting
        mu1 = mu[..., :, None, :]  # (*B, N, 1, H)
        mu2 = mu[..., None, :, :]  # (*B, 1, N, H)
        log_var1 = log_var[..., :, None, :]  # (*B, N, 1, H)
        log_var2 = log_var[..., None, :, :]  # (*B, 1, N, H)
        # KL divergence formula for Gaussians:
        # KL(N1||N2) = 0.5 * (log(var2/var1) + var1/var2 + (mu1-mu2)^2/var2 - 1)
        var1, var2 = jnp.exp(log_var1), jnp.exp(log_var2)
        log_var_ratio = log_var2 - log_var1
        var_ratio = var1 / (var2 + eps)
        mu_diff_sq = (mu1 - mu2) ** 2 / (var2 + eps)
        kl = jnp.sum(0.5 * (log_var_ratio + var_ratio + mu_diff_sq - 1), axis=-1)  # (*B, N, N)
        # Average over the pairwise matrices to return a single KL divergence measure of shape (*B,)
        # Mask the diagonal to avoid comparing the same latents.
        num_pairs = mu.shape[-2]
        kl = jnp.sum(jnp.where(jnp.eye(num_pairs) == 0, kl, 0), axis=(-1, -2)) / (num_pairs * (num_pairs - 1))
        return kl

    @staticmethod
    def _sample_latents(
        latents_mu: chex.Array, latents_logvar: chex.Array, key: chex.PRNGKey
    ) -> tuple[chex.Array, chex.Array, dict]:
        latents_std = jnp.exp(0.5 * latents_logvar)
        latents = latents_mu + latents_std * jax.random.normal(key, latents_mu.shape)
        kl_loss = jnp.mean(
            -0.5 * jnp.sum(1 + latents_logvar - latents_mu**2 - jnp.exp(latents_logvar), axis=-1)
        )
        kl_metrics = {
            "prior_kl": kl_loss,
            "latents_mu": latents_mu.mean(),
            "norm_latents_mu": norm(latents_mu, axis=-1).mean(),
            "latents_logvar": latents_logvar.mean(),
        }
        return latents, kl_loss, kl_metrics

    def _loss_from_pair_and_context(
        self,
        context: chex.Array,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        dropout_eval: bool,
    ) -> tuple[chex.Array, dict]:
        """
        Computes the loss for a single pair given a context.

        Args:
            context: context vector. Shape (*B, H).
            pairs: input data as tokens. Shape (*B, R, C, 2).
            grid_shapes: shapes of the grids. Shape (*B, 2, 2).
            dropout_eval: if false dropout is applied otherwise it is not.

        Returns:
            loss: loss value. Shape (*B,).
            metrics: dictionary of metrics of shape (*B,).
        """
        config = self.decoder.config

        # Make the input and output sequences.
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)

        # Decode the output sequence (teacher forcing).
        row_logits, col_logits, grid_logits = self.decoder(input_seq, output_seq, context, dropout_eval)

        # Compute cross entropy losses.
        grid_shapes_row, grid_shapes_col = grid_shapes[..., 0, 1], grid_shapes[..., 1, 1]
        # -1 to shift the tokens to [0, max_rows-1]
        one_hot_grid_shapes_row_labels = jax.nn.one_hot(grid_shapes_row - 1, config.max_rows)
        row_loss = -jnp.sum(jax.nn.log_softmax(row_logits) * one_hot_grid_shapes_row_labels, axis=-1)

        # -1 to shift the tokens to [0, max_cols-1]
        one_hot_grid_shapes_col_labels = jax.nn.one_hot(grid_shapes_col - 1, config.max_cols)
        col_loss = -jnp.sum(jax.nn.log_softmax(col_logits) * one_hot_grid_shapes_col_labels, axis=-1)

        # Copy the grid logits from the last non-padded column of each row to the first column of the next
        # row, skipping the padding tokens.
        last_non_padded_logits = self._get_last_non_padded_logits(
            grid_logits, grid_shapes_col[..., None, None]
        )
        grid_logits = grid_logits.at[..., config.max_cols :: config.max_cols, :].set(last_non_padded_logits)

        one_hot_grid_labels = jax.nn.one_hot(pairs[..., 1].reshape(*pairs.shape[:-3], -1), config.vocab_size)
        grid_losses = -jnp.sum(jax.nn.log_softmax(grid_logits) * one_hot_grid_labels, axis=-1)
        grid_loss = self._normalized_mean_over_sequence(grid_losses, grid_shapes_row, grid_shapes_col)

        loss = row_loss + col_loss + grid_loss
        metrics = {
            "shape_row_loss": row_loss,
            "shape_col_loss": col_loss,
            "grid_loss": grid_loss,
            "total_loss": loss,
        }
        return loss, metrics

    def _normalized_mean_over_sequence(
        self, grid_seq: chex.Array, num_rows: chex.Array, num_cols: chex.Array
    ) -> chex.Array:
        """
        Computes the mean of the sequence (e.g. losses or log probs) over the sequence length. Masks the
        losses corresponding to the padding tokens in the sequence length. Expects num_rows to be in
        [1, max_rows] and num_cols to be in [1, max_cols].

        Args:
            grid_seq: sequence of e.g. losses. Shape (*B, R*C).
            num_rows: number of rows. Shape (*B,).
            num_cols: number of columns. Shape (*B,).

        Returns:
            mean_seq: mean sequence over the sequence length. Shape (*B,).
        """
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        row_arange_broadcast = jnp.arange(max_rows).reshape(*len(num_rows.shape) * (1,), max_rows)
        col_arange_broadcast = jnp.arange(max_cols).reshape(*len(num_cols.shape) * (1,), max_cols)
        grid_row_mask = row_arange_broadcast < num_rows[..., None]
        grid_col_mask = col_arange_broadcast < num_cols[..., None]
        grid_pad_mask = grid_row_mask[..., None] & grid_col_mask[..., None, :]
        grid_pad_mask = grid_pad_mask.reshape(*grid_pad_mask.shape[:-2], -1)
        # Mask the elements corresponding to the padding tokens.
        grid_seq = jnp.where(grid_pad_mask, grid_seq, 0)
        # Mean over the sequence length, normalizing by the number of non-padded tokens.
        mean_seq = jnp.sum(grid_seq, axis=-1) / (jnp.sum(grid_pad_mask, axis=-1) + 1e-5)
        return mean_seq

    def generate_output(
        self,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        input: chex.Array,
        input_grid_shape: chex.Array,
        key: Optional[chex.PRNGKey],
        dropout_eval: bool,
        mode: Literal["mean", "first", "random_search", "gradient_ascent"],
        return_two_best: bool = False,
        **mode_kwargs,
    ) -> tuple[chex.Array, chex.Array, dict] | tuple[chex.Array, chex.Array, chex.Array, chex.Array, dict]:
        """
        Predicts the output grid given an input grid and other (input, output) pairs and shapes that follow
        the same transformation. Returns two predictions: the best prediction and the second best prediction.

        Args:
            pairs: input data as tokens. Shape (*B, N, R, C, 2).
                - N: number of (input, output) pairs per progam.
                - R: number of rows.
                - C: number of columns.
                - 2: two channels (input and output)
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, N, 2, 2). The last two dimension
                represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].
                Expects grid shapes values to be in [1, max_rows] and [1, max_cols].
            input: input grid. Shape (*B, R, C).
            input_grid_shape: shape of the input grid. Shape (*B, 2). Number of rows and columns of the
                input grid.
            key: optional random key for stochastic generating processes. Used in "random_search" mode, in
                "gradient_ascent" mode when using random_perturbations, and in variational inference. None
                or shape (*B, 2). If None, the key is not used.
            dropout_eval: if false dropout is applied otherwise it is not.
            mode: mode of the forward pass. Can be "mean", "first", or "random_search".
                - "mean": decodes the output using the mean latent from the (input, output) pairs.
                - "first": decodes the output using the first latent.
                - "random_search": randomly search for a latent that best explains the (input, output) pairs
                    and then decodes the output using that latent. Requires 'key' argument to not be None.
                    Also requires mode_kwargs to have the following arguments: 'num_samples' and 'scale'.
                    Other optional arguments are 'scan_batch_size' (default to all samples),
                    'include_mean_latent' (default to True), and 'include_all_latents' (default to False).
                    Computes maximum likelihood over (X + num_samples) latents, with X in [1, N, N+1] and
                    where N is the number of (input, output) pairs and num_samples is the number of random
                    samples.
                - "gradient_ascent": uses gradient ascent to find the latent that best explains the
                    (input, output) pairs and then decodes the output using that latent. Requires mode_kwargs
                    to have the following arguments: 'num_steps' and 'lr'. Other optional arguments are
                    'include_mean_latent' (default to True) and 'include_all_latents' (default to False).
                    Computes gradient ascent on decoder likelihood for 'num_steps' steps with learning rate
                    'lr'.
            return_two_best: if true returns the two best predictions, otherwise returns the best prediction.
            mode_kwargs: additional keyword arguments for the inference mode (e.g. 'remove_encoder_latents').

        Returns:
            first_output_grids: most likely predicted output grids. Shape (*B, R, C).
            first_output_shapes: shapes of the most likely output grids. Shape (*B, 2). Number of rows and
                columns of the output grids. Grid shapes values are in [1, max_rows] and [1, max_cols].
            if return_two_best, also returns:
                second_output_grids: second most likely predicted output grids. Shape (*B, R, C).
                second_output_shapes: shapes of the second most likely output grids. Shape (*B, 2).
            info: dictionary of additional information.
        """
        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, dropout_eval)

        if latents_logvar is not None:
            assert key is not None, "'key' argument required for variational inference."
            key, key_latents = jax.random.split(key)
            latents, *_ = self._sample_latents(latents_mu, latents_logvar, key_latents)
        else:
            latents = latents_mu

        if mode_kwargs.get("remove_encoder_latents", False):
            assert key is not None, "'key' argument required when 'remove_encoder_latents' is True."
            latents = jax.random.normal(key, latents.shape)
        if mode == "mean":
            context = latents.mean(axis=-2)
            first_context, second_context = context, context
        elif mode == "first":
            context = latents[..., 0, :]
            first_context, second_context = context, context
        elif mode == "random_search":
            assert key is not None, "'key' argument required for 'random_search' inference mode."
            for arg in ["num_samples", "scale"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'random_search' inference mode."

            first_context, second_context = self._get_random_search_context(
                latents, pairs, grid_shapes, key, **mode_kwargs
            )
        elif mode == "gradient_ascent":
            for arg in ["num_steps", "lr"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'gradient_ascent' inference mode."

            first_context, second_context = self._get_gradient_ascent_context(
                latents, pairs, grid_shapes, key, **mode_kwargs
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        info = {"context": first_context}

        if return_two_best:
            output_grids, output_shapes = jax.vmap(
                partial(
                    self._generate_output_from_context,
                    input=input,
                    input_grid_shape=input_grid_shape,
                    dropout_eval=dropout_eval,
                )
            )(jnp.stack([first_context, second_context], axis=0))
            first_output_grids, second_output_grids = output_grids[0], output_grids[1]
            first_output_shapes, second_output_shapes = output_shapes[0], output_shapes[1]
            return first_output_grids, first_output_shapes, second_output_grids, second_output_shapes, info
        else:
            output_grids, output_shapes = self._generate_output_from_context(
                first_context, input, input_grid_shape, dropout_eval
            )
            return output_grids, output_shapes, info

    def _generate_output_from_context(
        self, context: chex.Array, input: chex.Array, input_grid_shape: chex.Array, dropout_eval: bool
    ) -> tuple[chex.Array, chex.Array]:
        flattened_input = jnp.reshape(input, (*input.shape[:-2], -1))
        input_seq = jnp.concatenate([input_grid_shape, flattened_input], axis=-1)
        output_seq = jnp.zeros_like(input_seq).at[..., :2].set(1)  # Initialize the grid shape tokens to 1.

        def grid_shape_step(output_seq: chex.Array, row: bool) -> chex.Array:
            row_logits, col_logits, _ = self.decoder(input_seq, output_seq, context, dropout_eval)
            if row:
                logits = row_logits
            else:
                logits = col_logits
            # +1 to shift the tokens to [1, max_rows] or [1, max_cols]
            new_token = jnp.argmax(logits, axis=-1).astype(output_seq.dtype) + 1
            output_seq = output_seq.at[..., int(not row)].set(new_token)
            return output_seq

        # First predict the number of rows and then the number of columns.
        output_seq = grid_shape_step(output_seq, row=True)
        output_seq = grid_shape_step(output_seq, row=False)
        output_shapes = output_seq[..., :2]
        max_cols = self.decoder.config.max_cols

        def one_step(decoder: DecoderTransformer, output_seq: chex.Array, i: int) -> tuple[chex.Array, None]:
            *_, grid_logits = decoder(input_seq, output_seq, context, dropout_eval)
            # If we are at the beginning of a new row, the index of the logits to predict the next token is
            # the index of the last non-padded token of the previous row.
            logits_index = jnp.where(
                (i % max_cols == 0) & (i > 0),
                (i // max_cols - 1) * max_cols + output_shapes[..., 1].astype(jnp.int32),
                i,
            )
            logits = jnp.take_along_axis(grid_logits, logits_index[..., None, None], axis=-2).squeeze(axis=-2)
            new_token = jnp.argmax(logits, axis=-1).astype(output_seq.dtype)
            output_seq = output_seq.at[..., 2 + i].set(new_token)  # +2 to skip the grid shapes
            return output_seq, None

        # Then predict the grid values.
        output_seq, _ = nn.scan(
            one_step,
            variable_broadcast="params",
            variable_carry="output_seq",
            split_rngs={"params": False},
        )(self.decoder, output_seq, jnp.arange(self.decoder.config.max_len))
        output_grids = jnp.reshape(output_seq[..., 2:], (*input.shape[:-2], *input.shape[-2:]))

        return output_grids, output_shapes

    def _get_random_search_context(
        self,
        latents: chex.Array,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        key: chex.PRNGKey,
        num_samples: int,
        scale: float,
        scan_batch_size: Optional[int] = None,
        include_mean_latent: bool = True,
        include_all_latents: bool = False,
        **kwargs,
    ) -> tuple[chex.Array, chex.Array]:
        """Returns the best two contexts using a batched random search.

        Args:
            latents: latents from the encoder. Shape (*B, N, H).
            pairs: input data as tokens. Shape (*B, N, R, C, 2).
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, N, 2, 2). Expects grid shapes values
                to be in [1, max_rows] and [1, max_cols].
            key: random key to generate the random search. Shape (2,).
            num_samples: number of random samples to generate.
            scale: Gaussian scale of the random samples.
            scan_batch_size: batch size for the scan function. If None, the batch size is the same as the
                number of latents.
            include_mean_latent: if true (default to true), includes the mean latent in the latents from which
                to start the random search.
            include_all_latents: if true (default to false), includes all the pair latents in the latents from
                which to start the random search.

        Returns:
            best_context: best context. Shape (*B, H).
            second_best_context: second best context. Shape (*B, H).
        """
        latents = self._prepare_latents_before_search(include_mean_latent, include_all_latents, latents)

        if num_samples > 0:
            # Sample some random latents around the given latents.
            num_latents = latents.shape[-2]
            num_padded_samples = math.ceil(num_samples / num_latents) * num_latents
            random_vectors = jax.random.normal(
                key,
                (*latents.shape[:-2], num_latents, num_padded_samples // num_latents, latents.shape[-1]),
            )
            random_latents = latents[..., None, :] + scale * random_vectors
            random_latents = random_latents.reshape(*random_latents.shape[:-3], -1, random_latents.shape[-1])[
                ..., :num_samples, :
            ]
            latents = jnp.concatenate([latents, random_latents], axis=-2)

        # Flatten input/output for decoding likelihood
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)

        # Use the same latent for all pairs of the same task.
        latents = latents[..., None, :, :].repeat(output_seq.shape[-2], axis=-3)

        # Decode the output sequence in batches.
        batch_size = scan_batch_size or latents.shape[-2]
        num_batches = latents.shape[-2] // batch_size
        batched_latents = jnp.reshape(
            latents[..., : num_batches * batch_size, :],
            (*latents.shape[:-2], num_batches, batch_size, latents.shape[-1]),
        )
        dropout_eval = True  # no dropout during decoder inference
        _, (row_logits, col_logits, grid_logits) = nn.scan(
            lambda decoder, _, latents: (
                None,
                jax.vmap(decoder, in_axes=(None, None, -2, None), out_axes=-2)(
                    input_seq, output_seq, latents, dropout_eval
                ),
            ),
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=-3,
            out_axes=-3,
        )(self.decoder, None, batched_latents)
        row_logits, col_logits, grid_logits = jax.tree_util.tree_map(
            lambda x: x.reshape(*x.shape[:-3], -1, x.shape[-1]), (row_logits, col_logits, grid_logits)
        )
        if num_batches * batch_size < latents.shape[-2]:
            remaining_latents = latents[..., num_batches * batch_size :, :]
            row_logits_remainder, col_logits_remainder, grid_logits_remainder = jax.vmap(
                self.decoder, in_axes=(None, None, -2, None), out_axes=-2
            )(input_seq, output_seq, remaining_latents, dropout_eval)
            row_logits, col_logits, grid_logits = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x, y], axis=-2),
                (row_logits, col_logits, grid_logits),
                (row_logits_remainder, col_logits_remainder, grid_logits_remainder),
            )

        # Compute the the output sequence log probabilities for the different latents.
        log_probs = jax.vmap(self._compute_log_probs, in_axes=(-2, -2, -2, None), out_axes=-1)(
            row_logits, col_logits, grid_logits, output_seq
        )

        # Remove the duplication of the latents over the pairs.
        latents = latents[..., 0, :, :]
        best_context, second_best_context = self._select_best_and_second_best_latents(log_probs, latents)

        return best_context, second_best_context

    def _get_gradient_ascent_context(
        self,
        latents: chex.Array,
        pairs: chex.Array,
        grid_shapes: chex.Array,
        key: Optional[chex.PRNGKey],
        num_steps: int,
        lr: float,
        lr_schedule: bool = False,
        lr_schedule_exponent: float = 0.5,
        accumulate_gradients_decoder_pairs: bool = False,
        scan_gradients_latents: bool = False,
        optimizer: Literal["sgd", "adam"] = "sgd",
        optimizer_kwargs: Optional[dict] = None,
        include_mean_latent: bool = True,
        include_all_latents: bool = False,
        random_perturbation: Optional[dict] = None,
        stop_gradient_latent_move: bool = True,
        **kwargs,
    ) -> tuple[chex.Array, chex.Array]:
        """Returns the best two contexts using a gradient ascent algorithm.

        Args:
            latents: latents from the encoder. Shape (*B, N, H).
            pairs: input data as tokens. Shape (*B, N, R, C, 2).
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, N, 2, 2). Expects grid shapes values
                to be in [1, max_rows] and [1, max_cols].
            num_steps: number of gradient ascent steps.
            lr: learning rate for the gradient ascent.
            lr_schedule: if true, uses a cosine learning rate schedule, default to false.
            lr_schedule_exponent: exponent for the cosine learning rate schedule, default to 0.5.
            accumulate_gradients_decoder_pairs: if true, accumulates the gradients over the pairs, default to
                false.
            scan_gradients_latents: if true, scans the gradients over the latents, otherwise, use vmap,
                default to false.
            optimizer: optimizer to use for the gradient ascent. Can be "sgd" or "adam", default to "sgd".
            optimizer_kwargs: additional keyword arguments for the optimizer (e.g. b1, b2, eps for adam).
            include_mean_latent: if true (default to true), includes the mean latent in the latents from which
                to start the gradient ascent.
            include_all_latents: if true (default to false), includes all the pair latents in the latents from
                which to start the gradient ascent.
            random_perturbation: dictionary of random perturbation arguments. If not None, the following
                arguments are required:
                - num_samples: number of random samples to generate around the mean latent.
                - scale: Gaussian scale of the random perturbations.
            stop_gradient_latent_move: if true (default to true), do not propagate the loss gradient through
                the latent modification from the gradient ascent.

        Returns:
            best_context: best context. Shape (*B, H).
            second_best_context: second best context. Shape (*B, H).
        """
        latents = self._prepare_latents_before_search(
            include_mean_latent, include_all_latents, latents, random_perturbation, key
        )

        # Flatten input/output for decoding likelihood
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)

        def log_probs_fn(
            latents: chex.Array, input_seq: chex.Array, output_seq: chex.Array, decoder: DecoderTransformer
        ) -> chex.Array:
            # Use the same latent for all pairs of the same task.
            latents = latents[..., None, :].repeat(output_seq.shape[-2], axis=-2)
            row_logits, col_logits, grid_logits = decoder(input_seq, output_seq, latents, dropout_eval=True)
            log_probs = self._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)
            return log_probs

        value_and_grad_log_probs_fn = jax.vmap(
            jax.value_and_grad(log_probs_fn), in_axes=(-2, None, None, None), out_axes=(-1, -2)
        )
        # Add vmaps for batch dimensions
        for batch_dim in range(input_seq[..., 0, 0].ndim):
            value_and_grad_log_probs_fn = jax.vmap(value_and_grad_log_probs_fn, in_axes=(0, 0, 0, None))

        vmap_log_probs_fn = jax.vmap(log_probs_fn, in_axes=(-2, None, None, None), out_axes=-1)

        if accumulate_gradients_decoder_pairs:

            def wrap_value_and_grad(value_and_grad_log_probs):
                def wrapped(latents, input_seq, output_seq, decoder):
                    def body_fn(decoder, carry, seqs):
                        log_probs, grads = carry
                        log_probs_i, grads_i = value_and_grad_log_probs(
                            latents, seqs[0][..., None, :], seqs[1][..., None, :], decoder
                        )
                        return (log_probs + log_probs_i, grads + grads_i), None

                    init_carry = (jnp.zeros_like(latents[..., 0]), jnp.zeros_like(latents))
                    (log_probs, grads), _ = nn.scan(
                        body_fn,
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=-2,
                    )(decoder, init_carry, (input_seq, output_seq))

                    return log_probs, grads

                return wrapped

            def wrap_log_prob(log_probs_fn):
                def wrapped(latents, input_seq, output_seq, decoder):
                    log_probs, _ = nn.scan(
                        lambda decoder, log_prob, seqs: (
                            log_prob
                            + log_probs_fn(latents, seqs[0][..., None, :], seqs[1][..., None, :], decoder),
                            None,
                        ),
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=-2,
                    )(decoder, jnp.zeros_like(latents[..., 0]), (input_seq, output_seq))
                    return log_probs

                return wrapped

            value_and_grad_log_probs_fn = wrap_value_and_grad(value_and_grad_log_probs_fn)
            vmap_log_probs_fn = wrap_log_prob(vmap_log_probs_fn)

        if scan_gradients_latents:

            def wrap_value_and_grad(value_and_grad_log_probs):
                def wrapped(latents, input_seq, output_seq, decoder):
                    _, (log_probs, grads) = nn.scan(
                        lambda decoder, _, latent: (
                            _,
                            value_and_grad_log_probs(latent[..., None, :], input_seq, output_seq, decoder),
                        ),
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=-2,
                        out_axes=(-1, -2),
                    )(decoder, None, latents)
                    return jnp.squeeze(log_probs, axis=-2), jnp.squeeze(grads, axis=-3)

                return wrapped

            def wrap_log_prob(log_probs_fn):
                def wrapped(latents, input_seq, output_seq, decoder):
                    _, log_probs = nn.scan(
                        lambda decoder, _, latent: (
                            _,
                            log_probs_fn(latent[..., None, :], input_seq, output_seq, decoder),
                        ),
                        variable_broadcast="params",
                        split_rngs={"params": False},
                        in_axes=-2,
                        out_axes=-1,
                    )(decoder, None, latents)
                    return jnp.squeeze(log_probs, axis=-2)

                return wrapped

            value_and_grad_log_probs_fn = wrap_value_and_grad(value_and_grad_log_probs_fn)
            vmap_log_probs_fn = wrap_log_prob(vmap_log_probs_fn)

        if lr_schedule:
            lr = optax.cosine_decay_schedule(lr, num_steps, exponent=lr_schedule_exponent)
        if optimizer == "sgd":
            optimizer: optax.GradientTransformation = optax.chain(
                optax.clip_by_global_norm(1.0), optax.sgd(learning_rate=lr, **(optimizer_kwargs or {}))
            )
        elif optimizer == "adam":
            optimizer: optax.GradientTransformation = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=lr, eps_root=1e-8, **(optimizer_kwargs or {})),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        opt_state = optimizer.init(latents)

        def update_latents(decoder, carry, _):
            latents, opt_state = carry
            log_probs, grads = value_and_grad_log_probs_fn(latents, input_seq, output_seq, decoder)
            assert grads.shape == latents.shape
            if stop_gradient_latent_move:
                grads = jax.lax.stop_gradient(grads)
            updates, opt_state = optimizer.update(-grads, opt_state)
            latents += updates
            return (latents, opt_state), (latents, log_probs)

        (last_latents, _), (all_latents, all_log_probs) = nn.scan(
            update_latents,
            variable_broadcast="params",
            split_rngs={"params": False},
            length=num_steps,
            out_axes=(-2, -1),
        )(self.decoder, (latents, opt_state), None)

        # Concatenate original latents to all_latents and flatten all the latents.
        latents = jnp.concatenate([latents[..., None, :], all_latents], axis=-2).reshape(
            *latents.shape[:-2], -1, latents.shape[-1]
        )
        # Get all log_probs
        last_log_probs = vmap_log_probs_fn(last_latents, input_seq, output_seq, self.decoder)

        log_probs = jnp.concatenate([all_log_probs, last_log_probs[..., None]], axis=-1).reshape(
            *last_log_probs.shape[:-1], -1
        )

        best_context, second_best_context = self._select_best_and_second_best_latents(log_probs, latents)

        return best_context, second_best_context

    @classmethod
    def _prepare_latents_before_search(
        cls,
        include_mean_latent: bool,
        include_all_latents: bool,
        latents: chex.Array,
        random_perturbation: Optional[dict] = None,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        """
        Selects the latents from which to start the search. If include_mean_latent is True, the mean latent
        is included in the latents from which to start the search. If include_all_latents is True, all the pair
        latents are included in the latents from which to start the search. If both are True, the mean latent
        is concatenated to the latents from which to start the search. If both are False, an error is raised.

        Args:
            include_mean_latent: if true, includes the mean latent in the latents from which to start the search.
            include_all_latents: if true, includes all the pair latents in the latents from which to start the
                search.
            latents: latents from the encoder. Shape (*B, N, H).
            random_perturbation: dictionary of random perturbation arguments. If not None, the following
                arguments are required:
                - num_samples: number of random samples to generate around the mean latent.
                - scale: Gaussian scale of the random perturbations.
            key: random key to generate the random perturbation. Shape (2,).

        Returns:
            latents: latents from which to start the search. Shape (*B, 1, H), (*B, N, H), or (*B, N+1, H).
        """
        if include_mean_latent:
            mean_latent = latents.mean(axis=-2, keepdims=True)
            if include_all_latents:
                # Include the mean latent in the latents from which to start the search.
                prep_latents = jnp.concatenate([mean_latent, latents], axis=-2)
            else:
                # Only start the search from the mean latent.
                prep_latents = mean_latent
        else:
            # Start the search from all the pair latents.
            if not include_all_latents:
                raise ValueError(
                    "At least one of 'include_mean_latent' or 'include_all_latents' should be True."
                )
            prep_latents = latents
        if random_perturbation is not None:
            assert key is not None, "'key' argument required for random perturbation."
            for arg in ["num_samples", "scale"]:
                assert arg in random_perturbation, f"'{arg}' argument required for random perturbation."
            num_samples = random_perturbation["num_samples"]
            scale = random_perturbation["scale"]
            random_vectors = jax.random.normal(key, (*latents.shape[:-2], num_samples, latents.shape[-1]))
            random_latents = latents.mean(axis=-2, keepdims=True) + scale * random_vectors
            prep_latents = jnp.concatenate([prep_latents, random_latents], axis=-2)
        return prep_latents

    @classmethod
    def _flatten_input_output_for_decoding(
        cls, pairs: chex.Array, grid_shapes: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        flattened_pairs = jnp.reshape(pairs, (*pairs.shape[:-3], -1, 2))
        input_seq = jnp.concatenate([grid_shapes[..., 0], flattened_pairs[..., 0]], axis=-1)
        output_seq = jnp.concatenate([grid_shapes[..., 1], flattened_pairs[..., 1]], axis=-1)
        return input_seq, output_seq

    @classmethod
    def _select_best_and_second_best_latents(
        cls, log_probs: chex.Array, latents: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        sorted_log_probs_indices = jnp.argsort(log_probs, axis=-1, descending=True)
        best_context = jnp.take_along_axis(
            latents, sorted_log_probs_indices[..., 0:1, None], axis=-2
        ).squeeze(axis=-2)
        try:
            second_best_context = jnp.take_along_axis(
                latents, sorted_log_probs_indices[..., 1:2, None], axis=-2
            ).squeeze(axis=-2)
        except ValueError:
            # If there is only one latent, the second best context is the same as the best context.
            second_best_context = best_context
        return best_context, second_best_context

    def _compute_log_probs(
        self,
        row_logits: chex.Array,
        col_logits: chex.Array,
        grid_logits: chex.Array,
        output_seq: chex.Array,
        grid_log_prob_weight: float = 1.0,
    ) -> chex.Array:
        """
        Computes the log probabilities of the given output sequence given the row, column and grid logits.
        Sums over the pairs and the sequence length.

        Args:
            row_logits: row logits. Shape (*B, N, R).
            col_logits: column logits. Shape (*B, N, C).
            grid_logits: grid logits. Shape (*B, N, R*C, V).
            output_seq: output sequence. Shape (*B, N, R*C+2).
            grid_log_prob_weight: weight for the grid log probabilities.

        Returns:
            log_probs: log probabilities of the output sequence. Shape (*B,).
        """
        max_cols = self.decoder.config.max_cols
        num_rows, num_cols = output_seq[..., 0], output_seq[..., 1]
        row_all_log_probs = jax.nn.log_softmax(row_logits)
        row_log_probs = jnp.take_along_axis(row_all_log_probs, num_rows[..., None] - 1, axis=-1).squeeze(
            axis=-1
        )
        col_all_log_probs = jax.nn.log_softmax(col_logits)
        col_log_probs = jnp.take_along_axis(col_all_log_probs, num_cols[..., None] - 1, axis=-1).squeeze(
            axis=-1
        )
        # Copy the grid logits from the last non-padded column of each row to the first column of the
        # next row, skipping the padding tokens.
        last_non_padded_logits = self._get_last_non_padded_logits(grid_logits, num_cols[..., None, None])
        grid_logits = grid_logits.at[..., max_cols::max_cols, :].set(last_non_padded_logits)

        grid_all_log_probs = jax.nn.log_softmax(grid_logits)
        grid_log_probs = jnp.take_along_axis(grid_all_log_probs, output_seq[..., 2:, None], axis=-1).squeeze(
            axis=-1
        )
        grid_log_probs = self._normalized_mean_over_sequence(grid_log_probs, num_rows, num_cols)

        log_probs = row_log_probs + col_log_probs + grid_log_prob_weight * grid_log_probs
        log_probs = jnp.sum(log_probs, axis=-1)  # sum log_probs over the pairs
        return log_probs

    def _get_last_non_padded_logits(self, grid_logits: chex.Array, num_cols: chex.Array) -> chex.Array:
        """Selects the grid logits from the last non-padded column of each row."""
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        last_non_padded_logits = []
        for i in range(1, max_rows):
            end_of_row_logits = jnp.take_along_axis(
                grid_logits, max_cols * i - (max_cols - num_cols.astype(jnp.int32)), axis=-2
            )
            last_non_padded_logits.append(end_of_row_logits)
        return jnp.concatenate(last_non_padded_logits, axis=-2)


if __name__ == "__main__":
    from src.models.utils import TransformerLayerConfig

    batch_size = 4
    mini_batch_size = 3
    max_rows = 5
    max_cols = 5
    vocab_size = 10

    encoder_config = EncoderTransformerConfig(
        vocab_size=vocab_size,
        max_rows=max_rows,
        max_cols=max_cols,
        transformer_layer=TransformerLayerConfig(dropout_rate=0.05),
        variational=True,
    )
    decoder_config = DecoderTransformerConfig(
        vocab_size=vocab_size,
        max_rows=max_rows,
        max_cols=max_cols,
        transformer_layer=TransformerLayerConfig(dropout_rate=0.05),
    )

    encoder = EncoderTransformer(encoder_config)
    decoder = DecoderTransformer(decoder_config)
    lpn = LPN(encoder=encoder, decoder=decoder)

    key = jax.random.PRNGKey(0)
    pairs = jax.random.randint(
        key,
        (batch_size, mini_batch_size, max_rows, max_cols, 2),
        minval=0,
        maxval=vocab_size,
    )
    grid_shapes = jax.random.randint(
        key,
        (batch_size, mini_batch_size, 2, 2),
        minval=1,
        maxval=min(max_rows, max_cols) + 1,
    )
    variables = lpn.init(
        key, pairs, grid_shapes, dropout_eval=False, mode="mean", prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4
    )
    num_parameters = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
    print(f"Number of parameters: {num_parameters:,}")

    rngs = {"dropout": key}
    loss, metrics = jax.jit(lpn.apply, static_argnames=["dropout_eval", "mode"])(
        variables,
        pairs,
        grid_shapes,
        dropout_eval=False,
        mode="mean",
        rngs=key,
        prior_kl_coeff=1e-4,
        pairwise_kl_coeff=1e-4,
    )
    print("Mean Loss:", loss)
    loss, metrics = jax.jit(lpn.apply, static_argnames=["dropout_eval", "mode"])(
        variables,
        pairs,
        grid_shapes,
        dropout_eval=False,
        mode="all",
        rngs=key,
        prior_kl_coeff=1e-4,
        pairwise_kl_coeff=1e-4,
    )
    print("All Loss:", loss)
    loss, metrics = jax.jit(lpn.apply, static_argnames=["dropout_eval", "mode", "num_samples", "scale"])(
        variables,
        pairs,
        grid_shapes,
        dropout_eval=False,
        prior_kl_coeff=1e-4,
        pairwise_kl_coeff=1e-4,
        mode="random_search",
        rngs=key,
        num_samples=10,
        scale=1.0,
    )
    print("Random Search Loss:", loss)
    loss, metrics = jax.jit(
        partial(
            lpn.apply,
            optimizer_kwargs={"b1": 0.5},
            accumulate_gradients_decoder_pairs=True,
            scan_gradients_latents=True,
            include_all_latents=True,
            random_perturbation={"num_samples": 3, "scale": 0.5},
        ),
        static_argnames=["dropout_eval", "mode", "num_steps", "lr", "optimizer"],
    )(
        variables,
        pairs,
        grid_shapes,
        dropout_eval=False,
        prior_kl_coeff=1e-4,
        pairwise_kl_coeff=1e-4,
        mode="gradient_ascent",
        rngs=key,
        num_steps=2,
        lr=5e-2,
        optimizer="adam",
    )
    print("Gradient Ascent Loss:", loss)

    output_grids, output_shapes, _ = jax.jit(
        lpn.apply,
        static_argnames=[
            "dropout_eval",
            "mode",
            "method",
            "num_samples",
            "scale",
            "scan_batch_size",
            "include_mean_latent",
            "include_all_latents",
        ],
    )(
        variables,
        pairs,
        grid_shapes,
        pairs[:, 0, ..., 0],
        grid_shapes[:, 0, ..., 0],
        dropout_eval=False,
        mode="random_search",
        rngs=key,
        method=lpn.generate_output,
        num_samples=10,
        scale=2.0,
        scan_batch_size=None,
        include_mean_latent=True,
        include_all_latents=False,
        key=key,
    )
    print("Random search")
    print("Output grids of shape:", output_grids.shape)
    print("Output shapes of shape:", output_shapes.shape)

    output_grids, output_shapes, _ = jax.jit(
        partial(
            lpn.apply, optimizer_kwargs={"b1": 0.5}, random_perturbation={"num_samples": 3, "scale": 0.5}
        ),
        static_argnames=[
            "dropout_eval",
            "mode",
            "method",
            "num_steps",
            "lr",
            "optimizer",
            "include_mean_latent",
            "include_all_latents",
            "stop_gradient_latent_move",
            "remove_encoder_latents",
        ],
    )(
        variables,
        pairs,
        grid_shapes,
        pairs[:, 0, ..., 0],
        grid_shapes[:, 0, ..., 0],
        dropout_eval=False,
        mode="gradient_ascent",
        rngs=key,
        method=lpn.generate_output,
        num_steps=10,
        lr=5e-3,
        optimizer="adam",
        include_mean_latent=True,
        include_all_latents=False,
        stop_gradient_latent_move=True,
        remove_encoder_latents=True,
        key=key,
    )
    print("Gradient ascent")
    print("Output grids of shape:", output_grids.shape)
    print("Output shapes of shape:", output_shapes.shape)
