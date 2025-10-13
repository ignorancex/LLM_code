"""
Main VLM modeling function.
"""
import copy

import torch
import torchvision
import numpy as np
import os
import pandas as pd

from .constants import CATEGORIES
from .prompts import generate_prompt_ensemble, generate_name_prompt, DESCRIPTIONS_PROMPTS

from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, logging
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VLMModel(torch.nn.Module):
    def __init__(self, vision_type='resnet_v1', bert_type='emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None,
                 out_path=None, image_size=224, caption="A radiology image of [CLS]", projection=True,
                 norm_features=True, learning_criteria="unicl", apply_t_scaling=True, learn_t_scaling=True, patience=6,
                 lambda_dlilp=0.1):

        super().__init__()

        # Set attributes
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.logit_scale_init_value = logit_scale_init_value
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.out_path = out_path
        if self.out_path is not None:
            if not os.path.exists(self.out_path):
                os.makedirs(self.out_path)
        self.image_size = image_size
        self.caption = caption
        self.projection = projection
        self.norm_features = norm_features
        self.learning_criteria = learning_criteria  # "unicl", "clip", "unimodal", "dlilp"
        self.learn_t_scaling = learn_t_scaling
        self.apply_t_scaling = apply_t_scaling
        self.history_train = []
        self.history_val = []
        self.patience = patience
        self.lambda_dlilp = lambda_dlilp

        # Set vision and text encoder
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features)

        if self.learning_criteria != "unimodal":
            self.text_model = TextModel(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                        projection=self.projection, norm=self.norm_features)
        else:
            self.text_model = None

            if self.projection: prototypes_dim = self.proj_dim
            else: prototypes_dim = self.vision_model.vision_dim

            self.class_prototypes = torch.nn.Parameter(torch.nn.init.kaiming_normal_(
                torch.empty((len(CATEGORIES), prototypes_dim))))

        # learnable temperature for contrastive loss
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))
        if not learn_t_scaling:
            self.logit_scale.requires_grad = False

        # Additional projection and temperature scaling for DLILP optimization
        if self.learning_criteria == "dlilp":
            # New temperature scaling trainable
            self.logit_scale_head2 = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))
            # New projection head
            self.vision_model.projection_head2 = ProjectionLayer(layer=torch.nn.Linear(
                self.vision_model.vision_dim, self.vision_model.proj_dim, bias=False),
                projection=self.projection, norm=self.norm_features)
            # Class prototypes
            self.class_prototypes = torch.nn.Parameter(torch.nn.init.kaiming_normal_(
                torch.empty((len(CATEGORIES), self.vision_model.proj_dim))))

        # Load pretrained weights
        if from_checkpoint:
            self.load_from_pretrained(self.weights_path)

        # Set model to device
        self.to(device)

    def load_from_pretrained(self, weights_path=None):

        if weights_path is None:
            print("Pre-trained weights path not specified")
            return

        if "medclip" in weights_path:
            state_dict = torch.load(weights_path)
            state_dict["text_model.projection_head_text.projection.weight"] =\
                state_dict.pop("text_model.projection_head.weight")
            state_dict["vision_model.projection_head_vision.projection.weight"] =\
                state_dict.pop("vision_model.model.fc.weight")
            strict = True
        elif "cxr-clip" in weights_path:
            import omegaconf
            ckpt = torch.load(weights_path, map_location="cpu")
            state_dict = ckpt["model"]
            for key in list(state_dict.keys()):
                if "image_encoder" in key:
                    state_dict[key.replace("image_encoder.resnet.", "vision_model.model.")] = state_dict.pop(key)
            strict = False
        elif "gloria" in weights_path:
            import pytorch_lightning
            import omegaconf
            state_dict = torch.load(weights_path, map_location=device)
            state_dict = state_dict["state_dict"]
            for key in list(state_dict.keys()):
                state_dict[key.replace("gloria.img_encoder", "vision_model")] = state_dict.pop(key)
            strict = False
        elif "medklip" in weights_path:
            state_dict = torch.load(weights_path, map_location=device)["model"]
            resnet = torchvision.models.resnet50(pretrained=False)
            self.vision_model = torch.nn.Sequential(*list(resnet.children())[:-3],
                                                    torch.nn.AdaptiveAvgPool2d(1),
                                                    torch.nn.Flatten())
            self.vision_model.vision_dim = int(resnet.fc.in_features / 2)
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.res_features", "vision_model")] = state_dict.pop(key)
            self.load_state_dict(state_dict, strict=False)
            return
        elif "biovil" in weights_path:
            state_dict = torch.load(weights_path, map_location="cpu")
            for key in list(state_dict.keys()):
                if "encoder.encoder" in key:
                    state_dict[key.replace("encoder.encoder.", "vision_model.model.")] = state_dict.pop(key)
            strict = False
        elif "KAD" in weights_path:
            state_dict = torch.load(weights_path, map_location="cpu")
            state_dict = state_dict["image_encoder"]
            for key in list(state_dict.keys()):
                if "resnet." in key:
                    state_dict[key.replace("resnet.", "vision_model.model.")] = state_dict.pop(key)
            strict = False
        else:
            # Load state dict weights
            state_dict = torch.load(weights_path)
            # Fix for version change in bioclinicalbert
            flag_version_removed = "text_model.model.embeddings.position_ids"
            if flag_version_removed in list(state_dict.keys()):
                state_dict.pop(flag_version_removed)
            strict = True

        if self.learning_criteria == "unimodal":
            if state_dict["class_prototypes"].shape[-1] != self.class_prototypes.shape[-1]:
                state_dict["class_prototypes"] = self.class_prototypes.clone()

        self.load_state_dict(state_dict, strict=strict)
        print('load model weight from:', weights_path)

    def softce_clip_loss(self, logits_per_text, target_pseudo):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        return (caption_loss + image_loss) / 2.0

    def softce_clip_loss_assimetrical(self, logits_per_text, target_image2text, target_text2image):
        caption_loss = self.ce_loss(logits_per_text, target_text2image)
        image_loss = self.ce_loss(logits_per_text.T, target_image2text)
        return (caption_loss + image_loss) / 2.0

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t())
        if self.apply_t_scaling:
            logits_per_text *= logit_scale
        return logits_per_text.t()

    def fit(self, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
            transforms=None):

        # Early stopping val loss tracker
        best_val_loss, counter_patience = 100, 0

        # Set optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Set scheduler
        if scheduler:
            from cxrvlms.pretraining.utils import get_scheduler_per_iteration
            scheduler = get_scheduler_per_iteration(optimizer, lr, warmup_epoch, len(datalaoders["train"]))
        else:
            scheduler = None

        # Training along epochs
        epoch = 1
        while epoch <= epochs:

            # Train epoch
            if self.learning_criteria != "unimodal":
                hist_t, loss_epoch_train = self.train_epoch(datalaoders["train"], optimizer, scheduler,
                                                            transforms, epoch)
            else:
                hist_t, loss_epoch_train = self.train_epoch_unimodal(datalaoders["train"], optimizer, scheduler,
                                                                     transforms, epoch)

            # Eval epoch
            with torch.no_grad():
                if self.learning_criteria != "unimodal":
                    hist_v, loss_epoch_val = self.train_epoch(datalaoders["val"], optimizer, scheduler,
                                                              transforms, epoch, train=False)
                else:
                    hist_v, loss_epoch_val = self.train_epoch_unimodal(datalaoders["val"], optimizer, scheduler,
                                                                       transforms, epoch, train=False)

            # Display epoch-wise loss
            print('Epoch=%d: ave_loss_train=%2.4f / ave_loss_val=%2.4f' % (epoch, loss_epoch_train, loss_epoch_val))

            # Save model
            if epoch % store_num == 0:
                if self.out_path is not None:
                    if not os.path.isdir(self.out_path):
                        os.makedirs(self.out_path)
                    torch.save(self.state_dict(), self.out_path + self.vision_type + '_epoch' + str(epoch) + '.pth')

            # Update epoch
            epoch += 1

            # Store kpi
            self.history_train.extend(hist_t)
            self.history_val.extend(hist_v)

            # Early stopping based on validation loss
            if best_val_loss > loss_epoch_val:
                print("Validation loss improvement!", end="\n")
                torch.save(self.state_dict(), self.out_path + self.vision_type + '_best' + '.pth')
                best_val_loss = loss_epoch_val
                counter_patience = 0
            else:
                counter_patience += 1

            # Check patience for stop training
            if counter_patience >= self.patience:
                print("Validation loss did not improve during " + str(self.patience) +
                      " epochs... stopping training!", end="\n")
                torch.save(self.state_dict(), self.out_path + self.vision_type + '_last' + '.pth')
                break

        # Save last model
        torch.save(self.state_dict(), self.out_path + self.vision_type + '_last' + '.pth')

        # Save learning curves
        df = pd.DataFrame.from_dict(self.history_train)
        df.to_csv(self.out_path + 'history_train' + '.csv')
        df = pd.DataFrame.from_dict(self.history_val)
        df.to_csv(self.out_path + 'history_val' + '.csv')

    def train_epoch(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, train=True):
        if train:
            self.train()
            mode = "Training"
        else:
            self.eval()
            mode = "Validating"
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()
        loss_ave = 0.0

        # Set iterator
        epoch_iterator = tqdm(
            loader, desc=mode + " (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        # Iterate trough training batches
        hist_epoch = []
        for step, batch in enumerate(epoch_iterator):
            # Retrieve documents
            images = batch["image"].to(device).to(torch.float32)

            # Create text tokens
            text_tokens = self.text_model.tokenize(list(batch["prompt_selected"][0]))
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            if self.learning_criteria == "unicl":
                # Create similarity matrix with soft labels as ground truth
                similarity_refs = torch.matmul(batch["study_categories"],
                                               batch["category_prompt_selected"].transpose(0, 1)).to(torch.float32)
                similarity_refs += torch.eye(batch["study_categories"].shape[0])
                similarity_refs = similarity_refs.clip(0, 1)

                # Get image2text target matrix (one image might present different categories)
                target_image2text = (similarity_refs / np.expand_dims(similarity_refs.sum(-1), 1)).detach().to(device).to(
                    torch.float32)
                # Get text2image target matrix (one text is associated only to a subset of the image cateogires)
                target_text2image = (
                        similarity_refs.transpose(0, 1) / similarity_refs.transpose(0, 1).sum(-1).unsqueeze(1)).detach().to(
                    device).to(torch.float32)

            # Forward
            with autocast():

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision and text encoder
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                # Compute cross-entropy loss
                if self.learning_criteria == "unicl":
                    # Compute similarity matrix and logits
                    logits_per_image = self.compute_logits(img_embeds, text_embeds)
                    logits_per_text = logits_per_image.t()
                    # Compute loss with soft similarity matrix
                    loss = self.softce_clip_loss_assimetrical(logits_per_text, target_image2text, target_text2image)
                elif self.learning_criteria == "clip":
                    # Compute similarity matrix and logits
                    logits_per_image = self.compute_logits(img_embeds, text_embeds)
                    logits_per_text = logits_per_image.t()
                    # Compute loss pairing image-texts
                    target = torch.eye(text_embeds.shape[0]).detach().to(device)
                    loss = self.softce_clip_loss(logits_per_text, target)
                elif self.learning_criteria == "dlilp":

                    # 1) CLIP loss using paired images and text reports

                    # Retrieve samples with text reports
                    idx = np.argwhere([self.caption.replace("[CLS]", "").lower() not in i.lower() for i in batch["prompt_selected"][0]]).flatten()
                    if len(idx) > 1:
                        img_embeds_h1 = img_embeds[idx, :]
                        text_embeds_h1 = text_embeds[idx, :]

                        # Compute similarity matrix and logits
                        logits_per_image = self.compute_logits(img_embeds_h1, text_embeds_h1)
                        logits_per_text = logits_per_image.t()

                        # contrastive vision-language alignment loss
                        target = torch.eye(text_embeds_h1.shape[0]).detach().to(device)
                        loss_clip = self.softce_clip_loss(logits_per_text, target)
                    else:
                        loss_clip = torch.tensor(0.)
                        logits_per_image = torch.tensor(0.)

                    # 2) CE loss with retrieved labels from reports
                    gt = batch["study_categories"].detach().to(device).to(torch.float32)
                    mask = batch["mask_categories"].detach().to(device).to(torch.float32)

                    vision_features = self.vision_model.projection_head_vision.last_features
                    img_embeds_h2 = self.vision_model.projection_head2(vision_features)

                    # Norm class prototypes
                    class_embeds = self.class_prototypes / self.class_prototypes.norm(dim=-1, keepdim=True)

                    # Compute similarity matrix and logits
                    self.logit_scale_head2.data = torch.clamp(self.logit_scale_head2.data, 0, 4.6052)
                    logits = torch.matmul(img_embeds_h2, class_embeds.t()) * self.logit_scale_head2.exp()

                    # Compute cross-entropy loss
                    loss_ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none')
                    loss_ce = torch.mean((loss_ce * mask).sum(-1) / (mask.sum(-1) + 1e-3))

                    loss = loss_clip * self.lambda_dlilp + loss_ce

            if train:
                # Update model with scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                optimizer.zero_grad()

            # Overall losses track
            h = {"epoch": epoch, "step": step + 1,
                 "loss": np.round(loss.item(), 4),
                 "tau": np.round(self.logit_scale.exp().item(), 4),
                 "l": np.round(logits_per_image.detach().abs().mean().item() / self.logit_scale.exp().item(), 4)}
            hist_epoch.append(h)

            # Overall losses track
            loss_ave += loss.item()
            torch.cuda.empty_cache()

            # Set description
            epoch_iterator.set_description(
                "Epoch=%d: %s (%d / %d Steps) " % (epoch, mode, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # Update optimizer scheduler
            if train:
                skip_lr_sched = (scale > scaler.get_scale())
                if scheduler is not None and not skip_lr_sched:
                    scheduler.step()

        self.eval()
        return hist_epoch, loss_ave / len(loader)

    def train_epoch_unimodal(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, train=True):
        if train:
            self.train()
            mode = "Training"
        else:
            self.eval()
            mode = "Validating"
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()
        loss_ave = 0.0

        # Set iterator
        epoch_iterator = tqdm(
            loader, desc=mode + " (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        # Iterate trough training batches
        hist_epoch = []
        for step, batch in enumerate(epoch_iterator):
            # Retrieve documents
            images = batch["image"].to(device).to(torch.float32)
            gt = batch["study_categories"].detach().to(device).to(torch.float32)
            mask = batch["mask_categories"].detach().to(device).to(torch.float32)

            # Forward
            with autocast():

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision and text encoder
                img_embeds = self.vision_model(images)

                # Norm class prototypes
                text_embeds = self.class_prototypes / self.class_prototypes.norm(dim=-1, keepdim=True)

                # Compute similarity matrix and logits
                logits_per_image = self.compute_logits(img_embeds, text_embeds)

                # Compute cross-entropy loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_per_image, gt, reduction='none')
                loss = torch.mean((loss * mask).sum(-1) / (mask.sum(-1) + 1e-3))

            if train:
                # Update model with scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                optimizer.zero_grad()

            # Overall losses track
            h = {"epoch": epoch, "step": step + 1,
                 "loss": np.round(loss.item(), 4),
                 "tau": np.round(self.logit_scale.exp().item(), 4),
                 "l": np.round(logits_per_image.detach().abs().mean().item() / self.logit_scale.exp().item(), 4)}
            hist_epoch.append(h)

            loss_ave += loss.item()
            torch.cuda.empty_cache()

            # Set description
            epoch_iterator.set_description(
                "Epoch=%d: %s (%d / %d Steps) " % (epoch, mode, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # Update optimizer scheduler
            if train:
                skip_lr_sched = (scale > scaler.get_scale())
                if scheduler is not None and not skip_lr_sched:
                    scheduler.step()

        self.eval()
        return hist_epoch, loss_ave / len(loader)

    def forward(self, image, text):
        self.eval()

        # Pre-process image
        image = self.preprocess_image(image)

        # Pre-process text
        text_input_ids, text_attention_mask = self.preprocess_text(text)

        # Forward vision and text encoder
        with torch.no_grad():
            img_embeds = self.vision_model(image)
            text_embeds = self.text_model(text_input_ids, text_attention_mask)

            # Compute similarity matrix and logits
            logits = self.compute_logits(img_embeds, text_embeds)

            # Compute probabilities
            probs = logits.softmax(dim=-1)

        return probs.cpu().numpy(), logits.cpu().numpy()

    def preprocess_image(self, image):

        # Set image dtype
        if image.dtype != np.float32:
            image = np.float32(image)

        # Intensity scaling
        if image.max() > 0:
            image /= 255

        # Channel first
        if len(image.shape) > 2:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, 0)

        # Batch dimension
        image = np.expand_dims(image, 0)

        # Resize to training size using a canvas
        image = torch.tensor(image)
        sizes = image.shape[-2:]
        max_size = max(sizes)
        scale = max_size / self.image_size
        image = torchvision.transforms.Resize((int(image.shape[-2] / scale), int((image.shape[-1] / scale))))(image)
        image = torch.nn.functional.pad(image, (0, self.image_size - image.shape[-1], 0, self.image_size - image.shape[-2], 0, 0))

        # Set format and device
        image = image.to(torch.float32).to(device)

        return image

    def preprocess_text(self, text):

        # Create text prompt
        prompts = [self.caption.replace("[CLS]", category) for category in text]

        # Create text tokens
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

        return input_ids, attention_mask

    def compute_text_embeddings(self, categories, prompt_type="names"):
        # Obtain text embeddings per class
        text_embeds_dict = {}

        # Determine number of prompts for ensemble or not
        if prompt_type == "ensemble":
            prompts = generate_prompt_ensemble(100)
        elif prompt_type == "description":
            prompts = copy.deepcopy(DESCRIPTIONS_PROMPTS)
        else:
            prompts = generate_name_prompt(template=self.caption)

        for iKey in range(len(categories)):
            # Forwards prompts trough text encoder
            with torch.no_grad():
                descriptions = prompts[categories[iKey]]
                print(descriptions)
                text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)

                text_embeds = self.text_model(input_ids, attention_mask)

            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)

        text_embeds_dict = text_embeds_dict
        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds, self.logit_scale

    def select_class_prototypes(self, categories):
        categories = copy.deepcopy(categories)

        # Obtain text embeddings per class
        text_embeds_dict = {}

        embedding_categories = []
        temperature_param = []
        for iKey in range(len(categories)):
            # Correct name discrepancies
            if categories[iKey] == "Normal":
                embedding_categories.append(["No Finding"])
            elif categories[iKey] == "COVID":
                embedding_categories.append(["Lung Opacity", "Consolidation"])
            elif categories[iKey] == "No COVID":
                embedding_categories.append(["No Finding"])
            elif categories[iKey] == "Effusion":
                embedding_categories.append(["Pleural Effusion"])
            else:
                embedding_categories.append([categories[iKey]])

            class_embedding = []
            for ii in range(len(embedding_categories[iKey])):
                # Select index
                idx = np.argwhere([embedding_categories[iKey][ii].lower() == iCat for iCat in CATEGORIES]).item()
                class_embedding.append(self.class_prototypes[idx, :].clone().unsqueeze(0))

            # Average embedding per class over possible categories
            class_prototype = torch.cat(class_embedding, 0).mean(0).unsqueeze(0)
            text_embeds_dict[categories[iKey]] = class_prototype / class_prototype.norm(dim=-1, keepdim=True)

        text_embeds_dict = text_embeds_dict
        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds, self.logit_scale


class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,
                 norm=True):
        super().__init__()
        self.proj_dim = proj_dim

        # Assert vision encoders
        if vision_type not in ['resnet_v1', 'resnet_v2', 'efficientnet', "convnext"]:
            print("Vision model should be one of resnet/efficientnet... using resnet.")
            vision_type = "resnet_v1"

        # Set vision encoder architecture and pretrained weights
        if vision_type in ["resnet_v1", "resnet_v2"]:
            # Set pretrained weights from Imagenet and get model
            if vision_type == "resnet_v1":
                weights = 'IMAGENET1K_V1' if pretrained else None
            elif vision_type == "resnet_v2":
                weights = 'IMAGENET1K_V2' if pretrained else None
            else:
                weights = 'IMAGENET1K_V1' if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = torchvision.models.resnet50(weights=weights)
            # Set number of extracted features
            self.vision_dim = 2048
            # Replace classifier by Identity layer
            self.model.fc = torch.nn.Identity()
        elif vision_type == "efficientnet":
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.efficientnet_b7(weights=weights)
            self.vision_dim = 2096
        elif vision_type == "convnext":
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.convnext_tiny(weights=weights)
            # Set number of extracted features
            self.vision_dim = 768
            # Replace classifier by Identity layer
            self.model.classifier = torch.nn.Flatten()

        # Set output dimension
        if projection:
            self.out_dim = self.proj_dim
        else:
            self.out_dim = self.vision_dim

        # Set projection head
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,
                                                                            bias=proj_bias),
                                                      projection=projection, norm=norm)

    def forward(self, pixel_values):
        # Forwards trough vision encoder
        embed = self.model(pixel_values)

        # Compute projection from vision embedding to multi-modal projection
        embed = self.projection_head_vision(embed)
        return embed


class TextModel(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=True,
                 norm=True):
        super().__init__()

        # Set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 77

        # Load text encoder from pretrained
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        # Set projection head
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):

        # Forwards trough text encoder
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Combine last feature layers to compute text embedding
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],
                                          output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        # Compute projection from text embedding to multi-modal projection
        embed = self.projection_head_text(embed)
        return embed


class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()

        self.apply_projection = projection
        if projection:
            self.norm_modality = False
        else:
            self.norm_modality = norm
        self.norm_projection = norm
        self.projection = layer
        self.last_features = None

    def forward(self, x):

        # Storing last features before projection
        self.last_features = x

        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)

        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x