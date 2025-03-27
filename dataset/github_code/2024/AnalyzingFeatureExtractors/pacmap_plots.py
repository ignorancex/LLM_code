from PIL import Image
import pacmap
import numpy as np
import matplotlib.pyplot as plt

real_feat_path = "<your path>.npy"
real_X = np.load(real_feat_path, allow_pickle=True)
real_X = real_X.reshape(real_X.shape[0], -1)
for gen_name in ["celebaHQ","stylegan2"]:
    gen_feat_path = f"<your path>\\{gen_name}.npy"
    save_path = gen_name+"_pacmap.png"
    gen_X = np.load(gen_feat_path, allow_pickle=True)
    gen_X = gen_X.reshape(gen_X.shape[0], -1)
    print(gen_name,flush=True)
    X = np.concatenate((real_X,gen_X))                
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0,random_state=32) 
    X_transformed = embedding.fit_transform(X, init="pca")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(X_transformed[real_X.shape[0]:, 0], X_transformed[real_X.shape[0]:, 1], s=0.05,label=gen_name,c = ["r"]*gen_X.shape[0])
    ax.scatter(X_transformed[:real_X.shape[0], 0], X_transformed[:real_X.shape[0], 1],s=0.05,label="ffhq",c = ["b"]*real_X.shape[0])
    ax.legend()
    fig.savefig(save_path)
    plt.close()
