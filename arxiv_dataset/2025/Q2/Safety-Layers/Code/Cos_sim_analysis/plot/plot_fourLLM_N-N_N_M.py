import matplotlib.pyplot as plt
import pickle
import matplotlib
import fire
import numpy as np

def main(
    save_dir: str='pics/four.png',
    data_paths: list=[
        '../cos_sims/all_cos_llama3.pkl',
        '../cos_sims/all_cos_llama2.pkl',
        '../cos_sims/all_cos_phi-3.pkl',
        '../cos_sims/all_cos_gemma.pkl',

    ]
):
    font = {'family': 'sans-serif', 'weight': 'normal'}

    matplotlib.rc('font', **font)


    fig, axs = plt.subplots(1, 4, figsize=(24, 3.2),sharey=True)



    axs[0].set_title('Llama-3-8B-Instruct',fontsize='x-large')
    axs[0].set_ylabel('Cos_sim Value',fontsize='large')
    axs[1].set_title('Llama-2-7B-Chat',fontsize='x-large')
    axs[2].set_title('Phi-3-mini-4k-instruct',fontsize='x-large')
    axs[3].set_title('gemma-2b-it',fontsize='x-large')
    axs[0].set_yticks(np.arange(0.2, 1.1, 0.2))



    axs[0].set_xticks(np.arange(0, 33, 10))
    axs[1].set_xticks(np.arange(0, 33, 10))
    axs[2].set_xticks(np.arange(0, 33, 10))
    axs[3].set_xticks(np.arange(0, 18, 5))

    for i in range(4):
        with open(data_paths[i], 'rb') as f:
            datas = pickle.load(f)
            
        good=np.array(datas[0])
        mean_good= np.mean(good, axis=0)
        std_good = np.std(good, axis=0)
        good_up=mean_good+std_good
        good_down=mean_good-std_good

        good_bad=np.array(datas[2])
        mean_gb= np.mean(good_bad, axis=0)
        std_gb = np.std(good_bad, axis=0)
        gb_up=mean_gb+std_gb
        gb_down=mean_gb-std_gb
        axs[i].plot(mean_good,label='N-N Pairs',color='tab:orange',linewidth=2)
        axs[i].plot(mean_gb, label='N-M Pairs',color='tab:green',linewidth=2)
        axs[i].tick_params(axis='both', labelsize='large')
        axs[i].legend(fontsize='large')
        # axs[1].legend()
        if i==3:
            axs[i].fill_between(np.arange(18),good_up,good_down,where=(good_up > good_down), color='bisque', alpha=0.45)
            axs[i].fill_between(np.arange(18),gb_up,gb_down,where=(gb_up > gb_down), color='lightgreen', alpha=0.25)
        else:
            axs[i].fill_between(np.arange(32),good_up,good_down,where=(good_up > good_down),color='bisque', alpha=0.45)
            axs[i].fill_between(np.arange(32),gb_up,gb_down,where=(gb_up > gb_down), color='lightgreen', alpha=0.25)

    axs[0].grid(True,linewidth=0.8,linestyle='--')
    axs[1].grid(True,linewidth=0.8,linestyle='--')
    axs[2].grid(True,linewidth=0.8,linestyle='--')
    axs[3].grid(True,linewidth=0.8,linestyle='--')

    plt.tight_layout()
    plt.savefig('/mnt/lishen/baichuan/motivation/all_pics/exist1.png',dpi=500)

fire.Fire(main)