import sys, os
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    LEDForConditionalGeneration,
)

from model.PartialSrcAttnModels import (
    BartPartialAttnForConditionalGeneration,
    LEDPartialAttnForConditionalGeneration,
)

class Seq2SeqModel(object):
    def __init__(self, model_args, training_args, logger):
        self.model_args = model_args
        self.training_args = training_args
        self.logger = logger

        if model_args.model_name_or_path == 'allenai/led-base-16384':
            gradient_checkpointing = True
            use_cache = False
        else:
            gradient_checkpointing = False
            use_cache = True

        if not self.model_args.partial_attn:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                gradient_checkpointing=gradient_checkpointing,
                use_cache=use_cache,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                local_files_only=model_args.local_files_only)
        else:
            self.model = self.load_partial_attn_models(
                model_args.model_name_or_path,
                gradient_checkpointing,
                use_cache,
                model_args.cache_dir,
                model_args.model_revision,
                model_args.use_auth_token,
                model_args.local_files_only)

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if self.training_args.label_smoothing_factor > 0 \
            and not hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                self.logger.warn(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory")

    def load_partial_attn_models(self, model_name,
                                    gradient_checkpointing,
                                    use_cache,
                                    cache_dir,
                                    model_revision,
                                    use_auth_token,
                                    local_files_only):
        if model_name == 'facebook/bart-large':
            model = BartPartialAttnForConditionalGeneration.from_pretrained(
                        model_name,
                        gradient_checkpointing=gradient_checkpointing,
                        use_cache=use_cache,
                        from_tf=bool(".ckpt" in model_name),
                        cache_dir=cache_dir,
                        revision=model_revision,
                        use_auth_token=True if use_auth_token else None,
                        local_files_only=local_files_only)
        elif model_name == 'allenai/led-base-16384':
            model = LEDPartialAttnForConditionalGeneration.from_pretrained(
                        model_name,
                        gradient_checkpointing=gradient_checkpointing,
                        use_cache=use_cache,
                        from_tf=bool(".ckpt" in model_name),
                        cache_dir=cache_dir,
                        revision=model_revision,
                        use_auth_token=True if use_auth_token else None,
                        local_files_only=local_files_only)
        return model
