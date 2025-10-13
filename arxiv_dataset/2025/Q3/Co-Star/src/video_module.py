import torch
import clip
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import functional as F
from models.components.solver import WarmupCosineAnnealingLR
from src.utils.utils import get_classes, create_logits, gen_label, compute_entropy
from src.models.components.text_prompt import (
    text_prompt,
    manually_enriched_text_prompt,
    gpt_text_prompt,
    merged_text_prompt,
    hierarchical_text_prompt,
)
from tqdm import tqdm
from collections import defaultdict


def linear_weight_function(r, beta, t, T, max_weight, min_weight):

    beta = torch.tensor(beta, device=r.device, dtype=r.dtype)
    t = torch.tensor(t, device=r.device, dtype=r.dtype)
    T = torch.tensor(T, device=r.device, dtype=r.dtype)
    max_weight = torch.tensor(max_weight, device=r.device, dtype=r.dtype)
    min_weight = torch.tensor(min_weight, device=r.device, dtype=r.dtype)
    
    y = r * torch.exp(beta * t / T)
    w = torch.clamp(y, min_weight, max_weight)
    
    return w





def step_weight_function(r, num_steps, t, T, min_weight=0, max_weight=0.6):

    num_steps = torch.tensor(num_steps, device=r.device, dtype=torch.long)
    t = torch.tensor(t, device=r.device, dtype=r.dtype)
    T = torch.tensor(T, device=r.device, dtype=r.dtype)
    min_weight = torch.tensor(min_weight, device=r.device, dtype=r.dtype)
    max_weight = torch.tensor(max_weight, device=r.device, dtype=r.dtype)
    
    current_step = torch.floor(num_steps * t / T).long()
    
    weight_increment = (max_weight - r) / num_steps
    
    w = r + current_step * weight_increment
    
    w = torch.clamp(w, min=min_weight, max=max_weight)
    
    return w

class FixedSeedRNG:
    def __init__(self, seed=42):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
    
    def rand(self, *size):
        return torch.rand(*size, generator=self.generator)
class VideoModel(pl.LightningModule):
    def __init__(self, student_model, teacher_model, clip_model, num_classes, learning_rate, 
                 temperature, temperature_clip_reliability, temperature_clip_collaborative, scale, prob_invert, ema_decay, scr, 
                 min_weight, max_weight, mu_temp, beta_temp, imp_fac, extra_args=None, 
                 num_steps=4, reliability_alpha=0.5, beta=0.6, confidence_threshold=0.8, 
                 kl_threshold=0.04, memory_size=10):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.clip_model = clip_model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.extra_args = extra_args or {}
        self.ema_decay = ema_decay
        self.temperature = temperature
        self.temperature_clip_reliability = temperature_clip_reliability
        self.temperature_clip_collaborative = temperature_clip_collaborative
        self.num_steps = num_steps
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.reliability_alpha = reliability_alpha
        self.prob_invert = prob_invert
        self.scr = scr
        self.fixed_rng = FixedSeedRNG(seed=42)
        self.imp_fac = imp_fac
        self.mu_temp = mu_temp
        self.beta_temp = beta_temp
        self.beta = beta
        self.scale = scale

        # New attributes for SCR logic
        self.confidence_threshold = confidence_threshold
        self.kl_threshold = kl_threshold
        self.memory_size = memory_size
        self.prediction_memory = {}
        self.prediction_memory = defaultdict(list)


        self.classes_names = get_classes(extra_args)

        self.classes, self.num_text_aug, self.text_dict = text_prompt(
            classes_names=self.classes_names,
            dataset=self.extra_args.get("dataset", None),
            n_templates=-1,  # Use all templates
            image_templates="none"  # Or "simple" or "clip" based on your preference
        )

        # Move text_dict to the same device as the model
        self.text_dict = {k: v.to(self.device) for k, v in self.text_dict.items()}

       # self.temporal_consistency_weight = temporal_consistency_weight

        # Freeze the teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Freeze CLIP's visual and text models
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.student_model(x)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim > 2:
            features = torch.mean(features, dim=1)  # Global average pooling
        outputs = self.student_model.classifier(features)
        return outputs

    def teacher_forward(self, x):
        features = self.teacher_model(x)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim > 2:
            features = torch.mean(features, dim=1)  # Global average pooling
        outputs = self.teacher_model.classifier(features)
        return outputs

    def compute_temporal_consistency_weight(self, current_epoch):
        """Compute the exponentially decaying weight for temporal consistency loss."""
        current_epoch_tensor = torch.tensor(current_epoch, device=self.device, dtype=torch.float32)
        
        # Convert mu_temp and beta_temp to tensors
        mu_temp = torch.tensor(float(self.mu_temp), device=self.device, dtype=torch.float32)
        beta_temp = torch.tensor(float(self.beta_temp), device=self.device, dtype=torch.float32)
        
        return mu_temp * torch.exp(-beta_temp * current_epoch_tensor)

    def precompute_clip_probabilities(self):
        train_dataloader = self.trainer.datamodule.train_dataloader()
        self.precomputed_clip_probs_reliability = {}
        self.precomputed_clip_probs_collaborative = {}
        self.clip_model.eval()
        
        with torch.no_grad():
            for batch in tqdm(train_dataloader, desc="Precomputing CLIP probabilities"):
                if len(batch) == 4:  # Training data
                    global_inputs, _, _, indices = batch
                elif len(batch) == 3:  # Validation data
                    global_inputs, _, indices = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")
                
                global_inputs = global_inputs.to(self.device)
                clip_probs_reliability, clip_probs_collaborative = self.get_clip_score_distribution(
                    global_inputs, 
                    self.temperature_clip_reliability, 
                    self.temperature_clip_collaborative
                )
                
                for idx, prob_rel, prob_col in zip(indices, clip_probs_reliability, clip_probs_collaborative):
                    self.precomputed_clip_probs_reliability[idx.item()] = prob_rel.cpu()
                    self.precomputed_clip_probs_collaborative[idx.item()] = prob_col.cpu()


    def compute_temporal_consistency_loss(self, global_outputs, alt_global_outputs):
        global_probs = F.softmax(global_outputs / 2.0, dim=1)
        alt_global_probs = F.softmax(alt_global_outputs / 2.0, dim=1)
        return F.kl_div(alt_global_probs.log(), global_probs, reduction='batchmean')

    def compute_kl_divergence(self, p, q):
        return F.kl_div(p.log(), q, reduction='batchmean')
    def precompute_clip_scores(self, dataloader):
        precomputed_clip_scores = {}
        self.clip_model.eval()

        total_batches = len(dataloader)
        
        with tqdm(total=total_batches, desc="Computing CLIP pseudo-labels") as pbar:
            for batch in dataloader:
                inputs, _, indices = batch  # Unpack the batch
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    clip_probs = self.get_clip_score_distribution(inputs, self.temperature_clip)
                for idx, prob in zip(indices, clip_probs):
                    precomputed_clip_scores[idx.item()] = prob.to(self.device)  # Store on GPU if possible


                
                pbar.update(1)
        
        return precomputed_clip_scores
    def get_clip_score_distribution(self, x, tmp_reliability, tmp_collaborative):
        with torch.no_grad():
            batch_size, num_frames, c, h, w = x.shape
            x = x.view(batch_size * num_frames, c, h, w)

            # Encode all frames
            video_embedding = self.clip_model.encode_image(x)
            video_embedding /= video_embedding.norm(dim=-1, keepdim=True)

            # Reshape to [batch_size, num_frames, embedding_dim]
            video_embedding = video_embedding.view(batch_size, num_frames, -1)

            # Average pooling across frames
            video_embedding = video_embedding.mean(dim=1)

            all_similarities = []

            for template_id in range(self.num_text_aug):
                text_inputs = self.text_dict[template_id].to(self.device)
                
                # Encode text inputs
                text_embedding = self.clip_model.encode_text(text_inputs)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

                # Calculate similarity
                logit_scale = self.clip_model.logit_scale.exp()
                similarity = logit_scale * video_embedding @ text_embedding.T
                all_similarities.append(similarity)

            all_similarities = torch.stack(all_similarities, dim=1)  # [batch_size, num_prompts, num_classes]

            # Average across prompts
            averaged_similarity = all_similarities.mean(dim=1)  # [batch_size, num_classes]

            probabilities_reliability = F.softmax(averaged_similarity / tmp_reliability, dim=-1)
            probabilities_collaborative = F.softmax(averaged_similarity / tmp_collaborative, dim=-1)

            return probabilities_reliability, probabilities_collaborative


    # def compute_reliability_score(self, teacher_probs, clip_probs):
    #     kl_div = F.kl_div(teacher_probs.log(), clip_probs, reduction="none").sum(dim=1)
    #     reliability_score = torch.exp(-self.reliability_alpha * kl_div)
    #     return reliability_score


    def compute_reliability_score(self, teacher_probs, clip_probs):
        # KL(teacher || clip)
        kl_teacher_clip = F.kl_div(clip_probs.log(), teacher_probs, reduction="none").sum(dim=1)
        
        kl_clip_teacher = F.kl_div(teacher_probs.log(), clip_probs, reduction="none").sum(dim=1)
        
        avg_kl_div = (kl_teacher_clip + kl_clip_teacher) / 2
        
        reliability_score = torch.exp(-self.reliability_alpha * avg_kl_div)
        
        return reliability_score





    def get_inversion_probability(self, current_epoch, max_epochs, scale=4):
        normalized_epoch = current_epoch / max_epochs
        return 1 - torch.exp(-scale * torch.tensor(normalized_epoch, device=self.device))




    def get_collaborative_label(self, teacher_probs, clip_probs, teacher_conf_threshold=0.1, clip_conf_threshold=0.1):
        teacher_preds = torch.argmax(teacher_probs, dim=1)
        clip_preds = torch.argmax(clip_probs, dim=1)
        
        teacher_conf = torch.max(teacher_probs, dim=1).values
        clip_conf = torch.max(clip_probs, dim=1).values
        
        collaborative_labels = torch.full_like(teacher_preds, -1)
        
        # Case 1: Both models agree
        agree_mask = teacher_preds == clip_preds
        collaborative_labels[agree_mask] = teacher_preds[agree_mask]
        
        # Case 2: Models disagree, but both are confident
        both_confident_mask = (teacher_preds != clip_preds) & (clip_conf >= clip_conf_threshold) & (teacher_conf >= teacher_conf_threshold)
        collaborative_labels[both_confident_mask] = torch.where(
            teacher_conf[both_confident_mask] > clip_conf[both_confident_mask],
            teacher_preds[both_confident_mask],
            clip_preds[both_confident_mask]
        )
        
        # Case 3: Models disagree, CLIP is confident but teacher is not
        clip_confident_mask = (teacher_preds != clip_preds) & (clip_conf >= clip_conf_threshold) & (teacher_conf < teacher_conf_threshold)
        collaborative_labels[clip_confident_mask] = clip_preds[clip_confident_mask]
        
        # Case 4: Models disagree, teacher is confident but CLIP is not
        teacher_confident_mask = (teacher_preds != clip_preds) & (teacher_conf >= teacher_conf_threshold) & (clip_conf < clip_conf_threshold)
        collaborative_labels[teacher_confident_mask] = teacher_preds[teacher_confident_mask]
        
        # Case 5: Neither model is confident - these samples remain labeled as -1 (to be discarded)
        
        return collaborative_labels


   # def training_step(self, batch, batch_idx):
    def training_step(self, batch, batch_idx):
        if len(batch) == 4:  # Training data
            global_inputs, alt_global_inputs, labels, indices = batch
        elif len(batch) == 3:  # Validation data
            global_inputs, labels, indices = batch
            alt_global_inputs = global_inputs  # Use the same inputs for both global and alt
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

        original_batch_size = global_inputs.size(0)

        with torch.no_grad():
            teacher_outputs = self.teacher_forward(global_inputs)
            teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)

        # Use precomputed CLIP probabilities
        # clip_probs = torch.stack([self.precomputed_clip_probs[idx.item()].to(self.device) for idx in indices])
        clip_probs_reliability = torch.stack([self.precomputed_clip_probs_reliability[idx.item()].to(self.device) for idx in indices])
        clip_probs_collaborative = torch.stack([self.precomputed_clip_probs_collaborative[idx.item()].to(self.device) for idx in indices])
    
        
        collaborative_labels = self.get_collaborative_label(teacher_probs, clip_probs_collaborative)
        
        valid_mask = collaborative_labels != -1
        valid_global_inputs = global_inputs[valid_mask]
        valid_alt_global_inputs = alt_global_inputs[valid_mask]
        valid_labels = collaborative_labels[valid_mask]
        valid_teacher_probs = teacher_probs[valid_mask]
        valid_clip_probs = clip_probs_collaborative[valid_mask]
        valid_indices = indices[valid_mask]
        valid_batch_size = valid_global_inputs.size(0)

        if valid_batch_size == 0:
            return None  # Skip this batch if all samples were discarded
        
        student_global_outputs = self(valid_global_inputs)
        student_alt_global_outputs = self(valid_alt_global_inputs)
        student_probs = F.softmax(student_global_outputs / self.temperature, dim=1)

        reliability_score = self.compute_reliability_score(valid_teacher_probs, clip_probs_reliability[valid_mask])
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        weights = linear_weight_function(reliability_score, self.beta, current_epoch, max_epochs, self.max_weight, self.min_weight)


        if self.scr:
            inversion_prob = self.get_inversion_probability(current_epoch, max_epochs, self.scale)
            self.log('inversion_prob', inversion_prob, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            random_numbers = self.fixed_rng.rand(valid_batch_size).to(self.device)
            
            inversion_mask = random_numbers < inversion_prob
            num_to_invert = int(self.prob_invert * valid_batch_size)
            
            indices_to_invert = torch.nonzero(inversion_mask).squeeze()
            
            if indices_to_invert.numel() > 0:
                if indices_to_invert.dim() == 0:
                    indices_to_invert = indices_to_invert.unsqueeze(0)
                
                indices_to_invert = indices_to_invert[:num_to_invert]
                
                inverted_weights = (1 - weights) * self.imp_fac + weights * (1 - self.imp_fac)
                
                if not hasattr(self, 'prediction_memory'):
                    self.prediction_memory = {idx.item(): [] for idx in valid_indices}
                
                # Update prediction memory
                for i in range(valid_batch_size):
                    idx = valid_indices[i].item()
                    current_prob = student_probs[i].detach()
                    
                    # Append to memory and pop oldest if memory size is exceeded
                    self.prediction_memory[idx].append(current_prob)
                    if len(self.prediction_memory[idx]) > self.memory_size:
                        self.prediction_memory[idx].pop(0)
                
                # Initialize lists to store KL divergences and moving average maxima for logging
                kl_divs = []
                moving_avg_maxima = []
                
                # Invert weights based on conditions
                for i in indices_to_invert:
                    idx = valid_indices[i].item()
                    current_prob = student_probs[i]
                    max_confidence = current_prob.max().item()
                    
                    # Invert weights if the confidence is below threshold or based on KL divergence
                    if max_confidence < self.confidence_threshold:
                        weights[i] = inverted_weights[i]
                    elif len(self.prediction_memory[idx]) > 1:
                        # Compute moving average and KL divergence
                        moving_avg = torch.stack(self.prediction_memory[idx]).mean(dim=0)
                        kl_div = F.kl_div(current_prob.log(), moving_avg, reduction='batchmean')
                        moving_avg_max = moving_avg.max().item()
                        
                        # Store KL divergence and moving average maximum for logging
                        kl_divs.append(kl_div.item())
                        moving_avg_maxima.append(moving_avg_max)
                        
                        if kl_div < self.kl_threshold and moving_avg_max > self.confidence_threshold:
                            weights[i] = inverted_weights[i]
                
                if kl_divs:
                    avg_kl_div = sum(kl_divs) / len(kl_divs)
                    avg_moving_max = sum(moving_avg_maxima) / len(moving_avg_maxima)
                    self.log('avg_kl_div_inverted', avg_kl_div, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                    self.log('avg_moving_max_inverted', avg_moving_max, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                
                # Log the number of inverted samples
                num_inverted = (weights != inverted_weights).sum().item()
               # self.log('num_inverted_samples', num_inverted, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            
            else:
                print(f"Warning: No samples selected for inversion in batch {batch_idx}. Inversion probability: {inversion_prob}")
               # self.log('num_inverted_samples', 0, on_step=True, on_epoch=True, prog_bar=False, logger=True)






        student_log_probs = F.log_softmax(student_global_outputs / self.temperature, dim=1)
        kl_loss = self.kl_div_loss(student_log_probs, valid_teacher_probs)
        ce_loss = self.criterion(student_global_outputs, valid_labels)

        loss = (1 - weights) * kl_loss + weights * ce_loss  

        # Calculate accuracy using pseudo labels (for evaluation only, not used in training)
        preds = torch.argmax(student_global_outputs, dim=1)
        accuracy = torch.mean((preds == valid_labels).float())

        if self.trainer.is_global_zero:
            self.log('train_kl_loss', kl_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('reliability_score', reliability_score.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            if self.scr:
                self.log('inversion_probability', inversion_prob, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 50 == 0:
            print(f"Epoch {current_epoch}, Batch {batch_idx}:")
            print(f"Reliability Score: Mean = {reliability_score.mean().item():.4f}, "
                  f"Min = {reliability_score.min().item():.4f}, "
                  f"Max = {reliability_score.max().item():.4f}")
            print(f"Weights: Mean = {weights.mean().item():.4f}, "
                  f"Min = {weights.min().item():.4f}, "
                  f"Max = {weights.max().item():.4f}")
            if self.scr:
                print(f"Inversion Probability: {inversion_prob:.4f}")

        self.update_teacher_model()

        return {"loss": loss.mean(), "accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.mean((preds == labels).float())

        if self.trainer.is_global_zero:
            self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": accuracy}

    def update_teacher_model(self):
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
                teacher_param.data = self.ema_decay * teacher_param.data + (1.0 - self.ema_decay) * student_param.data




    def on_fit_start(self):

    
        print("Precomputing CLIP probabilities...")
        self.precompute_clip_probabilities()
        
        print("Computing initial accuracies...")
        val_dataloader = self.trainer.datamodule.val_dataloader()
        clip_correct_reliability = 0
        clip_correct_collaborative = 0
        teacher_correct = 0
        collab_correct = 0
        total = 0
        
        self.clip_model.eval()
        self.teacher_model.eval()
        device = self.device

        for batch in val_dataloader:
            inputs, labels, indices = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                # CLIP predictions for both temperatures
                clip_probs_reliability, clip_probs_collaborative = self.get_clip_score_distribution(
                    inputs, 
                    self.temperature_clip_reliability, 
                    self.temperature_clip_collaborative
                )
                clip_preds_reliability = torch.argmax(clip_probs_reliability, dim=1)
                clip_preds_collaborative = torch.argmax(clip_probs_collaborative, dim=1)
                
                # Teacher predictions
                teacher_outputs = self.teacher_forward(inputs)
                teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
                teacher_preds = torch.argmax(teacher_probs, dim=1)
                
                # Collaborative pseudo-labels
                collab_labels = self.get_collaborative_label(teacher_probs, clip_probs_collaborative)
                
                # Update correct predictions
                clip_correct_reliability += (clip_preds_reliability == labels).sum().item()
                clip_correct_collaborative += (clip_preds_collaborative == labels).sum().item()
                teacher_correct += (teacher_preds == labels).sum().item()
                collab_correct += (collab_labels == labels).sum().item()
                total += labels.size(0)

        # Calculate accuracies
        clip_accuracy_reliability = clip_correct_reliability / total * 100
        clip_accuracy_collaborative = clip_correct_collaborative / total * 100
        teacher_accuracy = teacher_correct / total * 100
        collab_accuracy = collab_correct / total * 100

        print(f"CLIP Validation Accuracy (reliability temp) before training: {clip_accuracy_reliability:.2f}%")
        print(f"CLIP Validation Accuracy (collaborative temp) before training: {clip_accuracy_collaborative:.2f}%")
        print(f"Teacher Model Validation Accuracy before training: {teacher_accuracy:.2f}%")
        print(f"Collaborative Pseudo-label Validation Accuracy before training: {collab_accuracy:.2f}%")

    def configure_optimizers(self):
        lr = self.hparams.get('solver', {}).get('lr', self.learning_rate)
        weight_decay = self.hparams.get('solver', {}).get('weight_decay', 0.0)
        warmup_steps = self.hparams.get('solver', {}).get('lr_warmup_steps', 0)
        epochs = self.trainer.max_epochs

        optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=weight_decay
        )

        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            epochs,
            warmup_epochs=warmup_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }