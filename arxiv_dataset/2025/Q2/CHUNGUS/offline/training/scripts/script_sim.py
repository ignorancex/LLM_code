import os


experiment_name = "experiments/sim_dinov2_224x224"
result = os.system("python train.py \
                    --seed 123 \
                    --experiment {} \
                    --train_embeddings ./../embedding_generation/saved_embeddings/dinov2_224x224_sim_train/embeddings.npy \
                    --valid_embeddings ./../embedding_generation/saved_embeddings/dinov2_224x224_sim_valid/embeddings.npy \
                    --lr 0.005 \
                    --lr_decay_gamma 0.25 \
                    --lr_decay_step 50 \
                    --prediction_head reconstructionbasic \
                    --emb_dim 384 \
                    --lrizz_L 0.5 \
                    --loss_fn reconstructlrizz \
                    --epochs 50 \
                    --patience 50 \
                    --train_num_embeddings -1 \
                    --weight_decay 0".format(experiment_name))
result = os.system("python test.py \
                    --experiment {} \
                    --train_embeddings ./../embedding_generation/saved_embeddings/dinov2_224x224_sim_train/embeddings.npy \
                    --valid_embeddings ./../embedding_generation/saved_embeddings/dinov2_224x224_sim_valid/embeddings.npy \
                    --seed 123".format(experiment_name))
