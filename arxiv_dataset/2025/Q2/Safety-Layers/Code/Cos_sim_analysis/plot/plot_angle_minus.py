import matplotlib.pyplot as plt
import pickle
import matplotlib
import fire
import numpy as np

def main(
    save_dir: str='pics/four_minus.png',
    data_paths: list=[
        '../cos_sims/all_cos_llama3.pkl',
        '../cos_sims/all_cos_llama2.pkl',
        '../cos_sims/all_cos_phi-3.pkl',
        '../cos_sims/all_cos_gemma.pkl',

    ]
):


    axs[0].set_ylabel('Angle Degree Value',fontsize='large')


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
        
        minus=np.degrees(np.arccos(np.array(mean_gb)))-np.degrees(np.arccos(np.array(mean_good)))
        
        axs[i].plot(minus,label='Mean Angular Difference',linewidth=2)
        # axs[i].plot(mean_gb, label='N-M Pairs',color='tab:green',linewidth=2)
        axs[i].tick_params(axis='both', labelsize='large')
        axs[i].legend(fontsize='large')
        # axs[1].legend()
    axs[0].grid(True,linewidth=0.8,linestyle='--')
    axs[1].grid(True,linewidth=0.8,linestyle='--')
    axs[2].grid(True,linewidth=0.8,linestyle='--')
    axs[3].grid(True,linewidth=0.8,linestyle='--')
    # fig.text(0.028, 0.5, 'Cos_sim Value', ha='center', va='center', rotation='vertical',fontsize='large')

    fig.tight_layout(rect=[0.,0.1,1,1])

    fig.text(0.51, 0.035, 'Layer ID', ha='center', va='center',fontsize='large')


    plt.tight_layout()
    plt.savefig(save_dir,dpi=500)


fire.Fire(main)