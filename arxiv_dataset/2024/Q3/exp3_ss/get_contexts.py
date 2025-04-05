from sentence_transformers import SentenceTransformer, models
from data import *
from numpy.random import Generator, PCG64
import numpy as np
from sklearn import decomposition
from matplotlib import rc
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel


model_dir = '../Data/semanticscholar/model/'
arms_file = '../Data/semanticscholar/processed_data.tsv'


def get_word_embedding(X):
    word_embedding_model = models.Transformer(model_dir, max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model.encode(X, convert_to_tensor=True)


def get_eps_stat():
    rg = Generator(PCG64(12345))
    cnt = 25
    epsilon = [0.25, 0.40, 0.50, 0.60, 0.75]
    for dt in dataset:
        dest_file = get_dest_file(dt)
        if dest_file.is_file():
            df = load_processed_data(dt,dest_file)
            Qm = df.query_text.to_numpy()
            X = get_word_embedding(Qm)
            ids = rg.choice(X.shape[0],cnt,replace=False)
            cos_sim = cosine_similarity(X[ids,:],X)
            for e in epsilon:
                print("epsilon: %f averae number of arms: %f" %(e, (cos_sim >= e).sum()/cnt))
            

def plot_context():
    import matplotlib.pyplot as plt
    rg = Generator(PCG64(12345))
    cnt = 5
    colormap = np.array(["orange","cyan","blue","purple","black"])

    for dt in dataset:
        dest_file = get_dest_file(dt)
        if dest_file.is_file():
            df = load_processed_data(dt,dest_file)
            df_sess = df[df.groupby("session_id")["session_id"].transform('size') > 10]
            session_ids = df_sess.session_id.unique()
            sample_session_ids =rg.choice(session_ids,cnt,replace=False)
            X = df_sess[df_sess['session_id'].isin(sample_session_ids)].query_text.to_numpy()
            y = df_sess[df_sess['session_id'].isin(sample_session_ids)].session_id.to_numpy() #ordered list of session ids
            colors = colormap[np.where(sample_session_ids==y[:,None])[1]]
            embeddings = get_word_embedding(X)
            #tsne = manifold.TSNE(n_components=2,init='pca',random_state=12345,perplexity=100)
            #Y = tsne.fit_transform(embeddings)
            pca = decomposition.PCA(n_components=2,random_state=12345)
            pca.fit(embeddings)
            Y = pca.transform(embeddings)
            with plt.style.context(("seaborn-darkgrid",)):
                fig, ax = plt.subplots(frameon=False)
                rc('mathtext',default='regular')
                rc('text', usetex=True)
                ax.scatter(Y[:,0],Y[:,1],color=colors)
                ax.tick_params(axis="both", colors="white")
                ax.set_xlabel(r'$x_1$')
                ax.set_ylabel(r'$x_2$')
                #ax.set_xlim(-0.65,1.18)
                #ax.set_ylim(-0.60,0.96)
                #plt.show()
                #fig.savefig('plot.pdf',format='pdf', dpi=fig.dpi)#, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor(), bbox_inches='tight')
                fig.savefig('plot.pdf',format='pdf',bbox_inches='tight')
                plt.close()
            
            return df
        print("%s file doesn't exist" %dest_file)
        exit(0)


def main():
    plot_context()
    #get_eps_stat()

if __name__ == '__main__':
    main()
