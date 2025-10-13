import os


result = os.system("python embedding_generator.py \
                    --seed 123 \
                    --data_folder ./../data/sim/images \
                    --data_csv ./../data/sim/train_sim_combined.csv \
                    --save_folder ./saved_embeddings/dinov2_224x224_sim_train \
                    --res_x 224 \
                    --res_y 224 \
                    --batch_size 2 \
                    --num_workers 8 \
                    --model dinov2")
result = os.system("python package_for_chungus.py \
                    --embedding_folder ./saved_embeddings/dinov2_224x224_sim_train \
                    --output_folder ./chungus_embeddings")


result = os.system("python embedding_generator.py \
                    --seed 123 \
                    --data_folder ./../data/sim/images \
                    --data_csv ./../data/sim/val_sim_combined.csv \
                    --save_folder ./saved_embeddings/dinov2_224x224_sim_valid \
                    --res_x 224 \
                    --res_y 224 \
                    --batch_size 2 \
                    --num_workers 8 \
                    --model dinov2")
result = os.system("python package_for_chungus.py \
                    --embedding_folder ./saved_embeddings/dinov2_224x224_sim_valid \
                    --output_folder ./chungus_embeddings")


result = os.system("python embedding_generator.py \
                    --seed 123 \
                    --data_folder ./../data/sim/images \
                    --data_csv ./../data/sim/all_sim_combined.csv \
                    --save_folder ./saved_embeddings/dinov2_224x224_sim_all \
                    --res_x 224 \
                    --res_y 224 \
                    --batch_size 2 \
                    --num_workers 8 \
                    --model dinov2")
result = os.system("python package_for_chungus.py \
                    --embedding_folder ./saved_embeddings/dinov2_224x224_sim_all \
                    --output_folder ./chungus_embeddings")
