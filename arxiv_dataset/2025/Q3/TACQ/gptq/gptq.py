import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer: torch.nn.Module, important_mask: torch.Tensor = None, add_until_fail = True):  
        """
        layer being usually a nn.Linear for LLMs
        important_mask being a binary mask where 1s denote parameters to keep at 16 bits.
        """
        self.layer = layer  # torch.nn.Module
        self.add_until_fail = add_until_fail
        self.dev = self.layer.weight.device  # cuda/cpu
        W = layer.weight.data.clone()  # torch.tensor, no_grad, copy. .data is a vestiage that returns a new tensor with requires_grad = False. Should replace with detach in newer versions.
        # W is not saved as self.W
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)  # Hessian (initialized to zero)
        self.nsamples = 0

        # Support for dynamic masking
        self.final_mask = torch.zeros_like(W, device=self.dev, dtype=torch.bool)

        if important_mask == None:
            self.important_mask = torch.zeros_like(W, dtype=torch.bool)
        else:
            self.important_mask = important_mask.clone()  
            self.final_mask = self.important_mask.clone() # both are floats with only binary values

        self.save_quant_dict = dict()

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if tmp > 1:
            raise Exception("VERIFY: batch size should be 1, current input shape: ", inp.shape)
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1])) 
            inp = inp.t()
            # print(f"VERIFY: inp.shape {inp.shape} == (#cols) x (batch_size x seq_len)")
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # print(f"VERIFY: self.H.shape {self.H.shape} == col x col")

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, save_quantization = False
    ):
        W = self.layer.weight.data.clone()
        ORIGINAL_W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # INIT SPQR FORMAT SAVING UTILITIES
        save_quant_dict = dict()
        if save_quantization:
            save_quant_dict["quant_weights"] = []
            save_quant_dict["quant_layer_scale"] = []
            save_quant_dict["quant_layer_zeros"] = []
            save_quant_dict["quant_layer_scale_qq_scale"] = []
            save_quant_dict["quant_layer_scale_qq_zero"] = []
            save_quant_dict["quant_layer_zero_qq_scale"] = []
            save_quant_dict["quant_layer_zero_qq_zero"] = []
            save_quant_dict["save_float_dtype"] = self.layer.weight.dtype
            save_quant_dict["outliers_matrix"] = torch.zeros(
                W.shape, dtype=save_quant_dict["save_float_dtype"]
            ).to(
                W.device
            )  # shape = [out_features, in_features]

        tick = time.time()

        if not self.quantizer.ready():
            """
            Masking here is done since we need the quantization scales to be calculated without considering outliers.
            """
            if False:  # Turn this on if we want to ignore the impact of outliers when calculating params for quantization.
                self.quantizer.find_params(W, weight=True)
            else:
                non_important_mask = ~self.important_mask
                # Channelwise Mean
                mean_over_non_important = torch.sum(
                    W * non_important_mask, dim=1, keepdim=True
                ) / torch.sum(non_important_mask, dim=1, keepdim=True).clamp_min(1)
                # print(f"VERIFY: mean_over_non_important.shape {mean_over_non_important.shape} == channels x 1: this will broadcast when multiplied with channels x cols to channels x cols")
                group_weight_without_important = W * non_important_mask + mean_over_non_important * (
                    ~non_important_mask
                )
                self.quantizer.find_params(group_weight_without_important, weight=True)
                # print("VERIFY: Find params finished running; quantizer scales and zeros configured for channel wise quantization.")
                # self.quantizer.find_params(W, weight=True)
        else:
            raise Exception("VERIFY: This should NOT be ran")

        H = self.H  # This is the hessian after add_batch is called [given our setup of running gptq with default params]
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0  # Dead weights with no signal from Hessian: pruned

        # Create a new per channel quantizer instance with different scales and zeros for each group of columns [:, i:(i + groupsize)], 16 columns when groupsize = 16
        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)
                raise Exception("VERIFY: This should NOT be ran unless groupwise quantization")

        # If actorder is True, reorder columns by descending Hessian diagonal (importance)
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm) # To restore original order later

        # Losses: track quantization error per element. Same shape as W: [out_features, columns].
        Losses = torch.zeros_like(W)
        # Q: will hold quantized weights after processing
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)

        # Added to make sure we are able to do the cholesky decomposition.
        if self.add_until_fail:
            multiplier = 1
            while multiplier < 150:
                try:
                    H[diag, diag] += damp  # This should only effect diagonal elements of H
                    # Compute upper-triangular factor of inverse: 
                    # Steps: 
                    #   - Invert H (Cholesky-based inversion)
                    #   - Re-factor to upper-triangular form for stable updates
                    H = torch.linalg.cholesky(H)
                    H = torch.cholesky_inverse(H)
                    H = torch.linalg.cholesky(H, upper=True)
                    break
                except Exception as e:
                    print(H, H[diag, diag])
                    diag_values = H[diag, diag]
                    print("Smallest 5 diagonal elements causing Cholesky fail:", torch.sort(diag_values)[0][:5])
                    multiplier += 1
                    print(f"WARNING: DAMPENING MULTIPLIER INCREASED TO {multiplier}; Error due to {e}")
            else:
                print("Error: Dampening multiplier reached 150 without proper cholesky decomposition. [User specified error]")
        else:
            H[diag, diag] += damp  
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # Now we process columns in blocks to reduce memory usage; this is separete from groupsize. Purely computational optimization.
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)  # Handles if last block reaches past the shape of the matrix X
            count = i2 - i1

            # Below initializations have the shape [out_features, blocksize]
            W1 = W[:, i1:i2].clone()  # Weights
            Q1 = torch.zeros_like(W1)  # Quantized results
            Err1 = torch.zeros_like(W1)  # Adjustments
            Losses1 = torch.zeros_like(W1)  # Per element loss

            # Shape: [blocksize x blocksize] hessian relevant to these columns. Hinv was [col x col]
            Hinv1 = Hinv[i1:i2, i1:i2]

            # Process each column individually
            for i in range(count):
                w = W1[:, i]  # shape [channels] vector; this is the 16 bit unquantized weight that have been adjusted by GPTQ
                d = Hinv1[i, i]  # Diagonal element used for normalizing errors

                # Updates the quantizer params for groups, if we are using them.
                if groupsize != -1:
                    if not static_groups:
                        # Every instance of find_params is important.
                        # If not static groups, dynamically recompute scale/zero for each column group start (call find_params)
                        # Edit here to enable 16 groupsize
                        if (i1 + i) % groupsize == 0:
                            if False:
                                self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                            else:
                                non_important_mask = ~self.important_mask[:, (i1 + i):(i1 + i + groupsize)]
                                # print(f"VERIFY: {non_important_mask.shape=} channels x groupsize")
                                # Channelwise Mean
                                mean_over_non_important = torch.sum(
                                    W[:, (i1 + i):(i1 + i + groupsize)] * non_important_mask, dim=1, keepdim=True
                                ) / torch.sum(non_important_mask, dim=1, keepdim=True).clamp_min(1)
                                # print(f"VERIFY: mean_over_non_important.shape {mean_over_non_important.shape} == channels x 1: this will broadcast when multiplied with channels x cols to channels x cols")
                                group_weight_without_important = W[:, (i1 + i):(i1 + i + groupsize)] * non_important_mask + mean_over_non_important * (
                                    ~non_important_mask
                                )
                                # print(f"VERIFY: {group_weight_without_important.shape=} == channel x groupsize")
                                self.quantizer.find_params(group_weight_without_important, weight=True)
                    else:
                        # Use map previously duplicated and configured (with find_params already called) quantizers 
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                # SPQR Saving Utility, save result of quantizer.find_params
                if save_quantization:
                    if quantizer.qq_scale_bits is not None:
                        save_quant_dict["quant_layer_scale"].append(quantizer.quant_scale.to(torch.int8))
                        save_quant_dict["quant_layer_scale_qq_scale"].append(
                            quantizer.qq_scale.scale.to(save_quant_dict["save_float_dtype"])
                        )
                        save_quant_dict["quant_layer_scale_qq_zero"].append(
                            quantizer.qq_scale.zero.to(save_quant_dict["save_float_dtype"])
                        )
                    else:
                        save_quant_dict["quant_layer_scale"].append(
                            quantizer.scale.to(save_quant_dict["save_float_dtype"])
                        )

                    if quantizer.qq_zero_bits is not None and (
                        (not quantizer.round_zero) or quantizer.qq_zero_bits < quantizer.bits
                    ):
                        save_quant_dict["quant_layer_zeros"].append(quantizer.quant_zero.to(torch.int8))
                        save_quant_dict["quant_layer_zero_qq_scale"].append(
                            quantizer.qq_zero.scale.to(save_quant_dict["save_float_dtype"])
                        )
                        save_quant_dict["quant_layer_zero_qq_zero"].append(
                            quantizer.qq_zero.zero.to(save_quant_dict["save_float_dtype"])
                        )
                    else:
                        save_quant_dict["quant_layer_zeros"].append(
                            quantizer.zero.to(save_quant_dict["save_float_dtype"])
                        )

                w_preserved = w*self.important_mask[:, i1 + i]  # At the first iteration of the first block. i1 == 0 and i == 0 -> we correctly select the 0th column of the mask.
                w_without_important = w*(~self.important_mask[:, i1 + i])  # zero the important weights we don't need to quantize revent overflow / errors
                q = self.quantizer.quantize(w_without_important.unsqueeze(1)).flatten() # .quantize quantizes only the weights passed in as arguments, and uses the state of the Quantizer object, determined by .configure and .find_params

                # print(f"VERIFY: q == q*(~self.important_mask[:, i1 + i]) {q == q*(~self.important_mask[:, i1 + i])} == True; {torch.sum(q - q*(~self.important_mask[:, i1 + i]))}")
                # print(f"VERIFY: Verified torch.all(q == q*(~self.important_mask[:, i1 + i])) == True")
                assert torch.all(q == q*(~self.important_mask[:, i1 + i])) == True  # Verifies all important weights have been quantized to 0 prior to adding them back in

                if save_quantization:
                    save_quant_dict["outliers_matrix"][:, i1 + i] = w_preserved

                if save_quantization:
                    from quant import spqr_real_dequantize, spqr_real_quantize
                    spqr_quantized_weights = spqr_real_quantize(w_without_important.unsqueeze(1)).flatten()
                    save_quant_dict["quant_weights"].append(spqr_quantized_weights.to(torch.int8))

                q = q + w_preserved  # Add back the preserved weights
                # print(f"VERIFY: torch.sum(w_preserved) == 0 {torch.sum(w_preserved) == 0} should not be True always; torch.sum(w_preserved) {torch.sum(w_preserved)}")
                
                # q has same shape as w: [out_features]
                # Update the quantized weights Q1 and losses in the block
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                # This error is used to update the remaining weights to minimize global loss.
                err1 = (w - q) / d  # d is a diagonal element of Hinv for this block
                # Update W1 for columns after i to compensate for quantization error using Hinv for the block and err1 for this column:
                # This adjusts future columns to minimize accumulated error.
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            # After processing this block, save quantized block Q1 into Q and Losses1 into Losses
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            # Update W for columns after the current block to account for error propagation now that we have all the losses from blocks processed:
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('quantization error for whole layer', torch.sum(Losses).item())  # Quantization error for whole layer

        # Now Q contains the quantized weights for the layer, shape matches original weight shape.
        # assert torch.all(Q == Q * (~self.important_mask) + W * self.important_mask) == True
        # print(f"VERIFY: The number of weights unquantized: {torch.sum(self.important_mask)}")

        # If actorder was True, we need to restore original column ordering:
        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if save_quantization:
            save_quant_dict["perm"] = perm.to(torch.int32)
            save_quant_dict["keep_last_columns"] = 0
            save_quant_dict["blocksize"] = 128
            save_quant_dict["weight_shape"] = W.shape
            save_quant_dict["groupsize"] = groupsize if groupsize != -1 else W.shape[1]
            save_quant_dict["quant_weights"] = torch.cat(save_quant_dict["quant_weights"], dim=1)
            save_quant_dict["outliers_matrix"] = save_quant_dict["outliers_matrix"].to_sparse()
            self.save_quant_dict = save_quant_dict

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()