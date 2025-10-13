import os


result = os.system("python embedding_generator.py \
                    --seed 123 \
                    --data_folder ./../data/outdoor/images \
                    --data_csv ./../data/outdoor/train_outdoor_combined.csv \
                    --save_folder ./saved_embeddings/dinov2_224x224_outdoor_train \
                    --res_x 224 \
                    --res_y 224 \
                    --batch_size 2 \
                    --num_workers 8 \
                    --model dinov2")
result = os.system("python package_for_chungus.py \
                    --embedding_folder ./saved_embeddings/dinov2_224x224_outdoor_train \
                    --output_folder ./chungus_embeddings")


result = os.system("python embedding_generator.py \
                    --seed 123 \
                    --data_folder ./../data/outdoor/images \
                    --data_csv ./../data/outdoor/val_outdoor_combined.csv \
                    --save_folder ./saved_embeddings/dinov2_224x224_outdoor_valid \
                    --res_x 224 \
                    --res_y 224 \
                    --batch_size 2 \
                    --num_workers 8 \
                    --model dinov2")
result = os.system("python package_for_chungus.py \
                    --embedding_folder ./saved_embeddings/dinov2_224x224_outdoor_valid \
                    --output_folder ./chungus_embeddings")


result = os.system("python embedding_generator.py \
                    --seed 123 \
                    --data_folder ./../data/outdoor/images \
                    --data_csv ./../data/outdoor/all_outdoor_combined.csv \
                    --save_folder ./saved_embeddings/dinov2_224x224_outdoor_all \
                    --res_x 224 \
                    --res_y 224 \
                    --batch_size 2 \
                    --num_workers 8 \
                    --model dinov2")
result = os.system("python package_for_chungus.py \
                    --embedding_folder ./saved_embeddings/dinov2_224x224_outdoor_all \
                    --output_folder ./chungus_embeddings")
