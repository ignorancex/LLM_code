import json
from data import CIRCODataset
import os

# load the json file, data/submission/CIRCO/val_sub_mask_pretrain_blip_w_ImageNet_MR50.json
with open('data/submission/CIRCO/val_sub_mask_pretrain_blip_w_ImageNet_MR50.json', 'r') as f:
    results = json.load(f)

data = CIRCODataset(split='val', mode='relative', img_transform=None, text_transform=None, test_split="query")

select_res = {}
for reference_img, reference_img_id, modifier, target_img_id, query_id in data:

    result_img_ids = results[query_id]
    # 挑选出 top 3 的图片，目标图像会包含原图
    top_k = 1 + 3
    if target_img_id in result_img_ids[:top_k]:
        select_res[query_id] = result_img_ids[:top_k]

        # save image
        reference_img.save(f'./select_img/{query_id}_reference.jpg')
        # copy image
        for i, result_img_id in enumerate(result_img_ids[:top_k]):
            result_img_path = data.img_paths[data.img_ids_indexes_map[str(result_img_id)]]
            os.system(f'cp  {result_img_path} ./select_img/{query_id}_{i}.jpg')
        target_img_path = data.img_paths[data.img_ids_indexes_map[str(target_img_id)]]
        # copy image
        os.system(f'cp  {target_img_path} ./select_img/{query_id}_target.jpg')
        # save modifier to txt
        with open(f'./select_img/{query_id}_modifier.txt', 'w+') as f:
            f.write(modifier)
# save the json file
json.dump(select_res, open('./val_sub_mask_pretrain_blip_w_ImageNet_MR50_select.json', 'w+'), sort_keys=True)

print(select_res)
