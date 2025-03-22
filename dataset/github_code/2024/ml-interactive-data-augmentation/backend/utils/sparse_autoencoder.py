"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Sparse autoencoder model.
"""

import json
import math

import safetensors.torch
import torch
from torch.nn.functional import mse_loss, relu


class SparseAutoencoder(torch.nn.Module):
    """The baseline sparse autoencoder"""

    def __init__(self, n_inputs: int, n_features: int):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_features = n_features
        self.prescaler = 1.0

        scaler = 1.0 / math.sqrt(n_features / n_inputs * 0.5 + 1.0)
        w0 = torch.randn((n_features, n_inputs), dtype=torch.float32)
        w0 = w0 / w0.norm(dim=1, keepdim=True) * scaler

        self.W_enc = torch.nn.Parameter(w0.T.clone().contiguous())
        self.b_enc = torch.nn.Parameter(
            torch.zeros(n_features, dtype=torch.float32))

        self.W_dec = torch.nn.Parameter(w0)
        self.b_dec = torch.nn.Parameter(
            torch.zeros(n_inputs, dtype=torch.float32))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return relu(torch.matmul(x, self.W_enc) + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.matmul(f, self.W_dec) + self.b_dec

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        count = x.shape[0]

        features = self.encode(x)
        dec = self.decode(features)

        l_recon = mse_loss(dec, x, reduction="sum") / count
        l1_sparsity = (features * torch.norm(self.W_dec, dim=1)).sum() / count

        return features, {
            "l_recon": l_recon,
            "l1_sparsity": l1_sparsity,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad
    def normalize_decoder_weights(self):
        W_dec_norm = self.W_dec.norm(dim=1)
        W_enc = self.W_enc * W_dec_norm.unsqueeze(0)
        b_enc = self.b_enc * W_dec_norm
        W_dec = self.W_dec / W_dec_norm.unsqueeze(1)
        b_dec = self.b_dec
        self.W_enc.copy_(W_enc)
        self.b_enc.copy_(b_enc)
        self.W_dec.copy_(W_dec)
        self.b_dec.copy_(b_dec)

    def export(self) -> tuple[dict, dict[str, torch.Tensor]]:
        prescaler = self.prescaler

        W_dec_norm = self.W_dec.norm(dim=1)
        W_enc = self.W_enc * W_dec_norm.unsqueeze(0)
        b_enc = self.b_enc * W_dec_norm / prescaler
        W_dec = self.W_dec / W_dec_norm.unsqueeze(1)
        b_dec = self.b_dec / prescaler

        config = {
            "type": "SparseAutoencoder",
            "n_inputs": self.n_inputs,
            "n_features": self.n_features,
        }

        tensors = {
            "W_enc": W_enc.contiguous(),
            "b_enc": b_enc.contiguous(),
            "W_dec": W_dec.contiguous(),
            "b_dec": b_dec.contiguous(),
        }

        return (config, tensors)

    @staticmethod
    def from_exported(config: dict, tensors: dict[str, torch.Tensor]):
        sae = SparseAutoencoder(
            n_inputs=config["n_inputs"], n_features=config["n_features"]
        )
        sae.prescaler = 1.0
        with torch.no_grad():
            sae.W_enc.copy_(tensors["W_enc"])
            sae.b_enc.copy_(tensors["b_enc"])
            sae.W_dec.copy_(tensors["W_dec"])
            sae.b_dec.copy_(tensors["b_dec"])
        return sae

    @classmethod
    def _from_normalized_state_dict(cls, state_dict: dict):
        (n_inputs, n_features) = state_dict["W_enc"].shape
        model = SparseAutoencoder(n_inputs=n_inputs, n_features=n_features)
        model.prescaler = 1.0
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_safetensors(cls, data: bytes):
        return cls._from_normalized_state_dict(safetensors.torch.load(data))

    @classmethod
    def from_safetensors_file(cls, filename: str):
        return cls._from_normalized_state_dict(safetensors.torch.load_file(filename))


class GatedSparseAutoencoder(torch.nn.Module):
    """Gated sparse autoencoder (http://arxiv.org/abs/2404.16014)"""

    def __init__(self, n_inputs: int, n_features: int):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_features = n_features
        self.prescaler = 1.0

        scaler = 1.0 / math.sqrt(n_features / n_inputs * 0.5 + 1.0)
        w0 = torch.randn((n_features, n_inputs), dtype=torch.float32)
        w0 = w0 / w0.norm(dim=1, keepdim=True) * scaler

        self.W_enc = torch.nn.Parameter(w0.T.clone().contiguous())
        self.b_enc = torch.nn.Parameter(
            torch.zeros(n_features, dtype=torch.float32))
        self.b_gate = torch.nn.Parameter(
            torch.zeros(n_features, dtype=torch.float32))
        self.r_gate = torch.nn.Parameter(
            torch.zeros(n_features, dtype=torch.float32))

        self.W_dec = torch.nn.Parameter(w0)
        self.b_dec = torch.nn.Parameter(
            torch.zeros(n_inputs, dtype=torch.float32))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.b_dec
        x_W_enc = torch.matmul(x, self.W_enc)
        mag = relu(x_W_enc + self.b_enc)
        gate = x_W_enc * torch.exp(self.r_gate).unsqueeze(0) + self.b_gate
        result = mag * (gate > 0)
        return result

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.matmul(f, self.W_dec) + self.b_dec

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        count = x.shape[0]

        x = x - self.b_dec
        x_W_enc = torch.matmul(x, self.W_enc)
        mag = relu(x_W_enc + self.b_enc)
        gate = x_W_enc * torch.exp(self.r_gate).unsqueeze(0) + self.b_gate
        features = mag * (gate > 0)
        dec = torch.matmul(features, self.W_dec) + self.b_dec
        l_recon = mse_loss(dec, x, reduction="sum") / count
        relu_gate = relu(gate)
        l1_sparsity = relu_gate.sum() / count
        l_aux = (
            mse_loss(
                torch.matmul(relu_gate, self.W_dec.detach()) +
                self.b_dec.detach(),
                x,
                reduction="sum",
            )
            / count
        )
        return features, {
            "l_recon": l_recon,
            "l1_sparsity": l1_sparsity,
            "l_aux": l_aux,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad
    def normalize_decoder_weights(self):
        W_dec_norm = self.W_dec.norm(dim=1)
        W_enc = self.W_enc * W_dec_norm.unsqueeze(0)
        b_enc = self.b_enc * W_dec_norm
        b_gate = self.b_gate * W_dec_norm
        W_dec = self.W_dec / W_dec_norm.unsqueeze(1)
        b_dec = self.b_dec
        self.W_enc.copy_(W_enc)
        self.b_enc.copy_(b_enc)
        self.b_gate.copy_(b_gate)
        self.W_dec.copy_(W_dec)
        self.b_dec.copy_(b_dec)

    def export(self) -> tuple[dict, dict[str, torch.Tensor]]:
        prescaler = self.prescaler

        W_dec_norm = self.W_dec.norm(dim=1)
        W_enc = self.W_enc * W_dec_norm.unsqueeze(0)
        b_enc = self.b_enc * W_dec_norm / prescaler
        b_gate = self.b_gate * W_dec_norm / prescaler
        r_gate = self.r_gate
        W_dec = self.W_dec / W_dec_norm.unsqueeze(1)
        b_dec = self.b_dec / prescaler

        config = {
            "type": "GatedSparseAutoencoder",
            "n_inputs": self.n_inputs,
            "n_features": self.n_features,
        }

        tensors = {
            "W_enc": W_enc.contiguous(),
            "b_enc": b_enc.contiguous(),
            "b_gate": b_gate.contiguous(),
            "r_gate": r_gate.contiguous(),
            "W_dec": W_dec.contiguous(),
            "b_dec": b_dec.contiguous(),
        }

        return (config, tensors)

    @staticmethod
    def from_exported(config: dict, tensors: dict[str, torch.Tensor]):
        sae = GatedSparseAutoencoder(
            n_inputs=config["n_inputs"], n_features=config["n_features"]
        )
        sae.prescaler = 1.0
        with torch.no_grad():
            sae.W_enc.copy_(tensors["W_enc"])
            sae.b_enc.copy_(tensors["b_enc"])
            sae.b_gate.copy_(tensors["b_gate"])
            sae.r_gate.copy_(tensors["r_gate"])
            sae.W_dec.copy_(tensors["W_dec"])
            sae.b_dec.copy_(tensors["b_dec"])
        return sae


def save_sae(
    sae: SparseAutoencoder | GatedSparseAutoencoder, parameters: dict = {}
) -> bytes:
    config, tensors = sae.export()
    metadata = {"config": json.dumps(
        config), "parameters": json.dumps(parameters)}
    return safetensors.torch.save(tensors, metadata=metadata)


def save_sae_file(
    sae: SparseAutoencoder | GatedSparseAutoencoder,
    filename: str,
    parameters: dict = {},
):
    config, tensors = sae.export()
    metadata = {"config": json.dumps(
        config), "parameters": json.dumps(parameters)}
    return safetensors.torch.save_file(tensors, filename=filename, metadata=metadata)


def _load_safetensors_metadata(data: bytes):
    header_length = int.from_bytes(data[:8], "little")
    header = json.loads(data[8: 8 + header_length].decode("utf-8"))
    metadata = header.get("__metadata__", {})
    return metadata


def _load_safetensors_metadata_file(filename: str):
    with open(filename, "rb") as f:
        header_length = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_length).decode("utf-8"))
    metadata = header.get("__metadata__", {})
    return metadata


def _load_sae_from_config_and_tensors(
    config: dict, tensors: dict[str, torch.Tensor]
) -> SparseAutoencoder | GatedSparseAutoencoder:
    if config["type"] == "SparseAutoencoder":
        return SparseAutoencoder.from_exported(config, tensors)
    elif config["type"] == "GatedSparseAutoencoder":
        return GatedSparseAutoencoder.from_exported(config, tensors)
    else:
        raise ValueError("invalid SAE type")


def load_sae(data: bytes):
    metadata = _load_safetensors_metadata(data)
    config = json.loads(metadata["config"])
    tensors = safetensors.torch.load(data)
    return _load_sae_from_config_and_tensors(config, tensors)


def load_sae_file(filename: str) -> SparseAutoencoder | GatedSparseAutoencoder:
    metadata = _load_safetensors_metadata_file(filename)
    config = json.loads(metadata["config"])
    tensors = safetensors.torch.load_file(filename)
    return _load_sae_from_config_and_tensors(config, tensors)
