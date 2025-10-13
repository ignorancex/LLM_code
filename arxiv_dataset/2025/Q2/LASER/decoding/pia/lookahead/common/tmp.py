# cnt = 0
            # for i in range(-1, max_branch_length):
            #     if i == -1:
            #         logit_index = 0
            #     else:
            #         logit_index = mask_indices[0][i]+1

            #     next_token_logits = logits[:, logit_index]
            #     next_tokens_scores = logits_processor(update_input_ids, next_token_logits)
            #     if decoding_kwargs.get('do_sample', False):
            #         probs = nn.functional.softmax(next_tokens_scores, dim=-1)
            #         next_tokens = torch.multinomial(probs, num_samples=1)
            #     else:
            #         next_tokens = torch.argmax(next_tokens_scores, dim=-1, keepdim=True).long()
            #         _, topk_indices = torch.topk(next_tokens_scores, k=1, dim=-1)
            #         # print(topk_indices, topk_indices.shape) # (1,3)

            #     cnt += 1
            #     if i == -1: # 这个是一定有的
            #         update_input_ids = torch.cat([update_input_ids, next_tokens], dim=1)
            #         next_token_id = next_tokens.tolist()[0][0]
            #         next_token_list.append(next_token_id)

            #     logit_indices.append(logit_index)

            #     if i == max_branch_length-1:
            #         break

            #     update_mask_indices = []
            #     update_draft_branches = []
            #     old_update_draft_branches = []
            #     for j, branch in enumerate(draft_branches):
            #         if i == -1:
            #             if len(branch) > i+1 and branch[i+1] == next_token_id: # 这里
            #                 update_mask_indices.append(mask_indices[j])
            #                 update_draft_branches.append(branch)
            #         else:
            #             if len(branch) > i+1 and branch[i+1] in topk_indices: # 这里
            #                 update_mask_indices.append(mask_indices[j])
            #                 update_draft_branches.append(branch)

            #     if len(update_mask_indices) == 0:
            #         break

            #     # if i > -1:
            #     #     cnt += 1

            #     old_update_draft_branches = deepcopy(update_draft_branches)
            #     mask_indices = update_mask_indices
            #     draft_branches = update_draft_branches

            # # print('next_token_list', next_token_list)
            # print('logit_indices', logit_indices)

            # if len(old_update_draft_branches) and cnt > 0:
            #     more_token = old_update_draft_branches[0][1:cnt+1]
            #     next_token_list.extend(more_token)

            #     tmp = torch.LongTensor(more_token).to(input_ids.device).unsqueeze(0)
            #     update_input_ids = torch.cat([update_input_ids, tmp], dim=1)
            #     # print(update_input_ids)
            #     # print(update_input_ids.shape)