import torch
from transformers.models.llava_next_video.modeling_llava_next_video import logger


def _merge_input_ids_with_image_features_get_token_type(
    self,
    image_features,
    feature_lens,
    inputs_embeds,
    input_ids,
    attention_mask,
    position_ids=None,
    labels=None,
    image_token_index=None,
    ignore_index=-100,
):
    """
    Merge input_ids with with image features into final embeddings

    Args:
        image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
            All vision vectors of all images in the batch
        feature_lens (`torch.LongTensor` of shape `(num_images)`):
            The length of visual embeddings of each image as stacked in `image_features`
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
            Token embeddings before merging with visual embeddings
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Input_ids of tokens, possibly filled with image token
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            :abels need to be recalculated to support training (if provided)
        image_token_index (`int`, *optional*)
            Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
        ignore_index (`int`, *optional*)
            Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
    Returns:
        final_embedding, final_attention_mask, position_ids, final_labels

    Explanation:
        each image has variable length embeddings, with length specified by feature_lens
        image_features is concatenation of all visual embed vectors
        task: fill each <image> with the correct number of visual embeddings
        Example:
            X (5 patches), Y (3 patches), Z (8)
            X, Y are in the same sequence (in-context learning)
        if right padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                o p q r Z s t u v _ _ _ _ _ _
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
            ]
        elif left padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                _ _ _ _ _ _ o p q r Z s t u v
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
            ]
        Edge cases:
            * If tokens are same but image token sizes are different, then cannot infer left or right padding
            ```python
            cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
            chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)
            prompts = [
                "[INST] <image>\nWhat is shown in this image? [/INST]",
                "[INST] <image>\nWhat is shown in this image? [/INST]",
            ]
            inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
                chart_img has 2634 tokens, while cat_img has 2340 tokens
            ```

            input_ids: [
                a b c d X g h
                i j Y k l m n
            ]
            where X is 3 tokens while Y is 5, this mean after merge
            if left-padding (batched generation)
                input_ids should be: [
                    _ _ a b c d X X X g h
                    i j Y Y Y Y Y k l m n
                ]
            elif (right padding) (training)
                input_ids should be: [
                    a b c d X X X g h _ _
                    i j Y Y Y Y Y k l m n
                ]
    """
    image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
    ignore_index = ignore_index if ignore_index is not None else self.config.ignore_index

    if self.training and self.padding_side == "left":
        logger.warning_once(
            "Padding side is set to 'left' but the model is in training mode. For training " "it is recommended to set `model.padding_side='right' and `processor.tokenizer.padding_side='right'`. " "If that's intended, ignore this warning"
        )
    if not self.training and self.padding_side == "right":
        logger.warning_once(
            "Padding side is set to 'right' but the model is in inference mode. For correct " "generation results, please set `model.padding_side='left'` and `processor.tokenizer.padding_side='left'`. " "If that's intended, ignore this warning"
        )

    with torch.no_grad():
        # ! in llava 1.6, number of patches is variable
        num_images = feature_lens.size(0)
        num_image_features, embed_dim = image_features.shape
        if feature_lens.sum() != num_image_features:
            raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
        batch_size = input_ids.shape[0]
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = self.padding_side == "left"
        if batch_size > 1:
            if _left_padding and _right_padding:
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")
            elif _right_padding and left_padding:
                left_padding = False
            elif _left_padding and not left_padding:
                left_padding = True

        # Whether to turn off right padding
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == image_token_index
        # special_image_token_mask: [bsz, seqlen]
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # num_special_image_tokens: [bsz]
        # Reserve for padding of num_images
        total_num_special_image_tokens = torch.sum(special_image_token_mask)
        if total_num_special_image_tokens != num_images:
            raise ValueError(f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images}).")
        # Compute the maximum embed dimension
        # max_image_feature_lens is max_feature_lens per batch
        feature_lens = feature_lens.to(input_ids.device)
        feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
        feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=input_ids.device)
        embed_sequence_lengths = (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
        max_embed_dim = embed_sequence_lengths.max()

        batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        # ! instead of special_image_token_mask * (num_image_patches - 1)
        #   special_image_token_mask * (num_feature_len - 1)
        special_image_token_mask = special_image_token_mask.long()
        special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
        new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
        if left_padding:
            # shift right token positions so that they are ending at the same number
            # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
            new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    # 3. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    final_attention_mask = torch.zeros(batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device)
    final_input_ids = torch.full((batch_size, max_embed_dim), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device)
    # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
    # set the corresponding tensors into their correct target device.
    target_device = inputs_embeds.device
    batch_indices, non_image_indices, text_to_overwrite = (
        batch_indices.to(target_device),
        non_image_indices.to(target_device),
        text_to_overwrite.to(target_device),
    )
    attention_mask = attention_mask.to(target_device)
    input_ids = input_ids.to(target_device)

    # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
    # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
    final_labels = None
    if labels is not None:
        labels = labels.to(target_device)
        final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
        final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

    # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
    with torch.no_grad():
        image_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device)
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
        embed_indices = embed_indices.expand(batch_size, max_embed_dim)
        embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

        if left_padding:
            # exclude padding on the left
            max_embed_dim = max_embed_dim.to(target_device)
            val = (max_embed_dim - embed_indices) <= embed_seq_lens
        else:
            # exclude padding on the right
            val = embed_indices < embed_seq_lens
        image_to_overwrite &= val

        if image_to_overwrite.sum() != num_image_features:
            raise ValueError(
                f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. "
                f"This prevents correct indexing and breaks batch generation."
            )
    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

    token_type = torch.ones_like(final_input_ids) * -10
    token_type[batch_indices, text_to_overwrite] = -1
    token_per_frame = self.vision_tower.vision_model.embeddings.num_patches // self.vision_resampler.pool.kernel_size**2
    for n_batch in range(token_type.shape[0]):
        n_frame = image_to_overwrite[n_batch].sum() // token_per_frame
        frame_token_type = torch.arange(n_frame, dtype=token_type.dtype, device=token_type.device).reshape(-1, 1).expand(-1, token_per_frame).reshape(-1)
        token_type[n_batch, image_to_overwrite[n_batch]] = frame_token_type
    self.token_type = token_type
    self.current_embedding=final_embedding

    return final_embedding, final_attention_mask, position_ids, final_labels, final_input_ids
