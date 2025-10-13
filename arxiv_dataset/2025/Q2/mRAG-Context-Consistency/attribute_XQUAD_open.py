import os
#cache_dir = os.getenv("TMPDIR")
cache_dir = "/projects/prjs1335/cache/"

from tqdm import tqdm
import jsonlines
import argparse
import json
import torch

import inseq
from inseq.commands.attribute_context.attribute_context import AttributeContextArgs, attribute_context, attribute_context_with_model


langs = ['en', 'ar', 'de', 'el', 'es', 'hi', 'ro', 'ru', 'th', 'tr', 'vi', 'zh']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mname", type=str, default="CohereForAI/aya-expanse-8b", help="LLM name on HF")
    parser.add_argument("--lang", type=str, default="en", help="Language")

    args = parser.parse_args()
    

    mname = args.mname.split('/')[-1]
    
    model_mirage = inseq.load_model(
            args.mname,
            "saliency",
            model_kwargs={"device_map": 'auto', "torch_dtype": torch.bfloat16, "cache_dir": cache_dir},
            tokenizer_kwargs={"use_fast": True},
            )

    save_dir = f"mirage/XQUAD_open/{mname}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f"results/XQUAD_open/{mname}_{args.lang}.json", encoding='utf-8') as f:
        data = json.load(f)
    f.close()
    

    with open("instruction_open.json", encoding='utf-8') as f:
        instructions = json.load(f)
    f.close()

    failure_cases = []
    failure_path = f"mirage/XQUAD_open/{mname}/failures.json"

    for ins_id, ins in enumerate(tqdm(data)):
        for lang in langs:
            if "aya" in mname.lower():
                input_template=f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{instructions[args.lang]['ctx']}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"+"{context}\n{current}<|END_OF_TURN_TOKEN|>"
                input_context_text = ins[f'prompt_ctx_{lang}'].split('<|USER_TOKEN|>')[1].split('\n')[0]
                input_current_text = ins[f'prompt_ctx_{lang}'].split('<|USER_TOKEN|>')[1].split('\n')[1].split('<|END_OF_TURN_TOKEN|>')[0]
                output_template = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\n{current}"
                output_current_text = ins[f'response_ctx_{lang}']
                special_tokens_to_keep = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK_TOKEN>', '<BOS_TOKEN>', '<EOS_TOKEN>', '<EOP_TOKEN>', '<|START_OF_TURN_TOKEN|>', '<|END_OF_TURN_TOKEN|>', '<|YES_TOKEN|>', '<|NO_TOKEN|>', '<|GOOD_TOKEN|>', '<|BAD_TOKEN|>', '<|USER_TOKEN|>', '<|CHATBOT_TOKEN|>', '<|SYSTEM_TOKEN|>', '<|USER_0_TOKEN|>', '<|USER_1_TOKEN|>', '<|USER_2_TOKEN|>', '<|USER_3_TOKEN|>', '<|USER_4_TOKEN|>', '<|USER_5_TOKEN|>', '<|USER_6_TOKEN|>', '<|USER_7_TOKEN|>', '<|USER_8_TOKEN|>', '<|USER_9_TOKEN|>', '<|EXTRA_0_TOKEN|>', '<|EXTRA_1_TOKEN|>', '<|EXTRA_2_TOKEN|>', '<|EXTRA_3_TOKEN|>', '<|EXTRA_4_TOKEN|>', '<|EXTRA_5_TOKEN|>', '<|EXTRA_6_TOKEN|>', '<|EXTRA_7_TOKEN|>', '<|EXTRA_8_TOKEN|>', '<|EXTRA_9_TOKEN|>']
                decoder_input_output_separator = ''
            elif "llama" in mname.lower():
                input_template = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 04 Mar 2025\n\n{instructions[args.lang]['ctx']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"+"{context}\n{current}<|eot_id|>"
                input_context_text = ins[f'prompt_ctx_{lang}'].split("<|start_header_id|>user<|end_header_id|>\n\n")[1].split('\n')[0]
                input_current_text = ins[f'prompt_ctx_{lang}'].split("<|start_header_id|>user<|end_header_id|>\n\n")[1].split('\n')[1].split('<|eot_id|>')[0]
                output_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{current}"
                output_current_text = ins[f'response_ctx_{lang}']
                decoder_input_output_separator = ''
                special_tokens_to_keep = ['<|begin_of_text|>', '<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>', '<|finetune_right_pad_id|>', '<|reserved_special_token_2|>', '<|start_header_id|>', '<|end_header_id|>', '<|eom_id|>', '<|eot_id|>', '<|python_tag|>', '<|reserved_special_token_3|>', '<|reserved_special_token_4|>', '<|reserved_special_token_5|>', '<|reserved_special_token_6|>', '<|reserved_special_token_7|>', '<|reserved_special_token_8|>', '<|reserved_special_token_9|>', '<|reserved_special_token_10|>', '<|reserved_special_token_11|>', '<|reserved_special_token_12|>', '<|reserved_special_token_13|>', '<|reserved_special_token_14|>', '<|reserved_special_token_15|>', '<|reserved_special_token_16|>', '<|reserved_special_token_17|>', '<|reserved_special_token_18|>', '<|reserved_special_token_19|>', '<|reserved_special_token_20|>', '<|reserved_special_token_21|>', '<|reserved_special_token_22|>', '<|reserved_special_token_23|>', '<|reserved_special_token_24|>', '<|reserved_special_token_25|>', '<|reserved_special_token_26|>', '<|reserved_special_token_27|>', '<|reserved_special_token_28|>', '<|reserved_special_token_29|>', '<|reserved_special_token_30|>', '<|reserved_special_token_31|>', '<|reserved_special_token_32|>', '<|reserved_special_token_33|>', '<|reserved_special_token_34|>', '<|reserved_special_token_35|>', '<|reserved_special_token_36|>', '<|reserved_special_token_37|>', '<|reserved_special_token_38|>', '<|reserved_special_token_39|>', '<|reserved_special_token_40|>', '<|reserved_special_token_41|>', '<|reserved_special_token_42|>', '<|reserved_special_token_43|>', '<|reserved_special_token_44|>', '<|reserved_special_token_45|>', '<|reserved_special_token_46|>', '<|reserved_special_token_47|>', '<|reserved_special_token_48|>', '<|reserved_special_token_49|>', '<|reserved_special_token_50|>', '<|reserved_special_token_51|>', '<|reserved_special_token_52|>', '<|reserved_special_token_53|>', '<|reserved_special_token_54|>', '<|reserved_special_token_55|>', '<|reserved_special_token_56|>', '<|reserved_special_token_57|>', '<|reserved_special_token_58|>', '<|reserved_special_token_59|>', '<|reserved_special_token_60|>', '<|reserved_special_token_61|>', '<|reserved_special_token_62|>', '<|reserved_special_token_63|>', '<|reserved_special_token_64|>', '<|reserved_special_token_65|>', '<|reserved_special_token_66|>', '<|reserved_special_token_67|>', '<|reserved_special_token_68|>', '<|reserved_special_token_69|>', '<|reserved_special_token_70|>', '<|reserved_special_token_71|>', '<|reserved_special_token_72|>', '<|reserved_special_token_73|>', '<|reserved_special_token_74|>', '<|reserved_special_token_75|>', '<|reserved_special_token_76|>', '<|reserved_special_token_77|>', '<|reserved_special_token_78|>', '<|reserved_special_token_79|>', '<|reserved_special_token_80|>', '<|reserved_special_token_81|>', '<|reserved_special_token_82|>', '<|reserved_special_token_83|>', '<|reserved_special_token_84|>', '<|reserved_special_token_85|>', '<|reserved_special_token_86|>', '<|reserved_special_token_87|>', '<|reserved_special_token_88|>', '<|reserved_special_token_89|>', '<|reserved_special_token_90|>', '<|reserved_special_token_91|>', '<|reserved_special_token_92|>', '<|reserved_special_token_93|>', '<|reserved_special_token_94|>', '<|reserved_special_token_95|>', '<|reserved_special_token_96|>', '<|reserved_special_token_97|>', '<|reserved_special_token_98|>', '<|reserved_special_token_99|>', '<|reserved_special_token_100|>', '<|reserved_special_token_101|>', '<|reserved_special_token_102|>', '<|reserved_special_token_103|>', '<|reserved_special_token_104|>', '<|reserved_special_token_105|>', '<|reserved_special_token_106|>', '<|reserved_special_token_107|>', '<|reserved_special_token_108|>', '<|reserved_special_token_109|>', '<|reserved_special_token_110|>', '<|reserved_special_token_111|>', '<|reserved_special_token_112|>', '<|reserved_special_token_113|>', '<|reserved_special_token_114|>', '<|reserved_special_token_115|>', '<|reserved_special_token_116|>', '<|reserved_special_token_117|>', '<|reserved_special_token_118|>', '<|reserved_special_token_119|>', '<|reserved_special_token_120|>', '<|reserved_special_token_121|>', '<|reserved_special_token_122|>', '<|reserved_special_token_123|>', '<|reserved_special_token_124|>', '<|reserved_special_token_125|>', '<|reserved_special_token_126|>', '<|reserved_special_token_127|>', '<|reserved_special_token_128|>', '<|reserved_special_token_129|>', '<|reserved_special_token_130|>', '<|reserved_special_token_131|>', '<|reserved_special_token_132|>', '<|reserved_special_token_133|>', '<|reserved_special_token_134|>', '<|reserved_special_token_135|>', '<|reserved_special_token_136|>', '<|reserved_special_token_137|>', '<|reserved_special_token_138|>', '<|reserved_special_token_139|>', '<|reserved_special_token_140|>', '<|reserved_special_token_141|>', '<|reserved_special_token_142|>', '<|reserved_special_token_143|>', '<|reserved_special_token_144|>', '<|reserved_special_token_145|>', '<|reserved_special_token_146|>', '<|reserved_special_token_147|>', '<|reserved_special_token_148|>', '<|reserved_special_token_149|>', '<|reserved_special_token_150|>', '<|reserved_special_token_151|>', '<|reserved_special_token_152|>', '<|reserved_special_token_153|>', '<|reserved_special_token_154|>', '<|reserved_special_token_155|>', '<|reserved_special_token_156|>', '<|reserved_special_token_157|>', '<|reserved_special_token_158|>', '<|reserved_special_token_159|>', '<|reserved_special_token_160|>', '<|reserved_special_token_161|>', '<|reserved_special_token_162|>', '<|reserved_special_token_163|>', '<|reserved_special_token_164|>', '<|reserved_special_token_165|>', '<|reserved_special_token_166|>', '<|reserved_special_token_167|>', '<|reserved_special_token_168|>', '<|reserved_special_token_169|>', '<|reserved_special_token_170|>', '<|reserved_special_token_171|>', '<|reserved_special_token_172|>', '<|reserved_special_token_173|>', '<|reserved_special_token_174|>', '<|reserved_special_token_175|>', '<|reserved_special_token_176|>', '<|reserved_special_token_177|>', '<|reserved_special_token_178|>', '<|reserved_special_token_179|>', '<|reserved_special_token_180|>', '<|reserved_special_token_181|>', '<|reserved_special_token_182|>', '<|reserved_special_token_183|>', '<|reserved_special_token_184|>', '<|reserved_special_token_185|>', '<|reserved_special_token_186|>', '<|reserved_special_token_187|>', '<|reserved_special_token_188|>', '<|reserved_special_token_189|>', '<|reserved_special_token_190|>', '<|reserved_special_token_191|>', '<|reserved_special_token_192|>', '<|reserved_special_token_193|>', '<|reserved_special_token_194|>', '<|reserved_special_token_195|>', '<|reserved_special_token_196|>', '<|reserved_special_token_197|>', '<|reserved_special_token_198|>', '<|reserved_special_token_199|>', '<|reserved_special_token_200|>', '<|reserved_special_token_201|>', '<|reserved_special_token_202|>', '<|reserved_special_token_203|>', '<|reserved_special_token_204|>', '<|reserved_special_token_205|>', '<|reserved_special_token_206|>', '<|reserved_special_token_207|>', '<|reserved_special_token_208|>', '<|reserved_special_token_209|>', '<|reserved_special_token_210|>', '<|reserved_special_token_211|>', '<|reserved_special_token_212|>', '<|reserved_special_token_213|>', '<|reserved_special_token_214|>', '<|reserved_special_token_215|>', '<|reserved_special_token_216|>', '<|reserved_special_token_217|>', '<|reserved_special_token_218|>', '<|reserved_special_token_219|>', '<|reserved_special_token_220|>', '<|reserved_special_token_221|>', '<|reserved_special_token_222|>', '<|reserved_special_token_223|>', '<|reserved_special_token_224|>', '<|reserved_special_token_225|>', '<|reserved_special_token_226|>', '<|reserved_special_token_227|>', '<|reserved_special_token_228|>', '<|reserved_special_token_229|>', '<|reserved_special_token_230|>', '<|reserved_special_token_231|>', '<|reserved_special_token_232|>', '<|reserved_special_token_233|>', '<|reserved_special_token_234|>', '<|reserved_special_token_235|>', '<|reserved_special_token_236|>', '<|reserved_special_token_237|>', '<|reserved_special_token_238|>', '<|reserved_special_token_239|>', '<|reserved_special_token_240|>', '<|reserved_special_token_241|>', '<|reserved_special_token_242|>', '<|reserved_special_token_243|>', '<|reserved_special_token_244|>', '<|reserved_special_token_245|>', '<|reserved_special_token_246|>', '<|reserved_special_token_247|>']
            elif "gemma" in mname.lower():
                input_template = f"<bos><start_of_turn>user\n{instructions[args.lang]['ctx']} "+"{context}\n{current}<end_of_turn>"
                input_context_text = ins[f'prompt_ctx_{lang}'].split(f"{instructions[args.lang]['ctx']} ")[1].split('\n')[0]
                input_current_text = ins[f'prompt_ctx_{lang}'].split(f"{instructions[args.lang]['ctx']} ")[1].split('\n')[1].split('<end_of_turn>')[0]
                output_template = "<start_of_turn>model\n{current}"
                output_current_text = ins[f'response_ctx_{lang}']
                special_tokens_to_keep = ['<start_of_turn>','<end_of_turn>','<pad>', '<eos>', '<bos>', '<unk>', '\n']
                decoder_input_output_separator = '\n'
            elif "qwen" in mname.lower():
                input_template = f"<|im_start|>system\n{instructions[args.lang]['ctx']}<|im_end|>\n<|im_start|>user\n"+"{context}\n{current}\n<|im_end|>"
                input_context_text = ins[f'prompt_ctx_{lang}'].split('<|im_start|>user\n')[1].split('\n')[0]
                input_current_text = ins[f'prompt_ctx_{lang}'].split('<|im_start|>user\n')[1].split('\n')[1]
                output_template = "<|im_start|>assistant\n{current}"
                output_current_text = ins[f'response_ctx_{lang}']
                special_tokens_to_keep = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>', '<tool_call>', '</tool_call>', '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|fim_pad|>', '<|repo_name|>', '<|file_sep|>', "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]
                decoder_input_output_separator = '\n'
            else:
                raise ValueError("Not supported LLM.")

            contextless_input_current_text = input_template.replace("{context}", "")
            save_path = save_dir + f"{ins_id}_{args.lang}_{lang}.json"
            attribute_args = AttributeContextArgs(
                    model_name_or_path=args.mname,
                    input_context_text=input_context_text,
                    input_current_text=input_current_text,
                    output_template=output_template,
                    input_template=input_template,
                    contextless_input_current_text=contextless_input_current_text,
                    show_intermediate_outputs=False,
                    attributed_fn="contrast_prob_diff",
                    context_sensitivity_std_threshold=100000,
                    output_current_text=output_current_text,
                    attribution_method="saliency",
                    attribution_kwargs={"logprob": True},
                    save_path=save_path,
                    tokenizer_kwargs={"use_fast": True},
                    model_kwargs={
                        "device_map": 'auto',
                        "torch_dtype": torch.bfloat16,
                        "cache_dir": cache_dir
                        },
                    generation_kwargs={
                        "temperature": 1.0,
                        "top_k": 1,
                        "max_new_tokens": 100,
                        },
                    show_viz=False,
                    decoder_input_output_separator=decoder_input_output_separator,
                    special_tokens_to_keep=special_tokens_to_keep
                    )
            res = attribute_context_with_model(attribute_args, model_mirage)

if __name__ == "__main__":
    main()
