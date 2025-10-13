import matplotlib.pyplot as plt
import pickle
import numpy as np
import fire

def main(
    data_path: str='../cos_sims/all_cos.pkl',
    save_dir: str='pics/',
):

    fig, axs = plt.subplots(1, 3, figsize=(9, 2.08), sharey=True, sharex=True)
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)
        
    good=np.array(datas[0])
    mean_good= np.mean(good, axis=0)
    std_good = np.std(good, axis=0)
    good_up=mean_good+std_good
    good_down=mean_good-std_good


    bad=np.array(datas[1])
    mean_bad= np.mean(bad, axis=0)
    std_bad = np.std(bad, axis=0)
    bad_up=mean_bad+std_bad
    bad_down=mean_bad-std_bad


    good_bad=np.array(datas[2])
    mean_gb= np.mean(good_bad, axis=0)
    std_gb = np.std(good_bad, axis=0)
    gb_up=mean_gb+std_gb
    gb_down=mean_gb-std_gb


    len_layers=len(mean_bad)




    # 子图1  _cos_sim
    axs[0].plot(mean_good,linewidth=1.2)
    axs[0].fill_between(np.arange(len_layers),good_up,good_down,where=(good_up > good_down), color='lightblue', alpha=0.4)

    axs[0].tick_params(axis='both', labelsize='x-small')
    axs[0].set_title('Normal-Normal Query Pairs',fontsize='x-small')
    axs[0].set_xticks(np.arange(0, len_layers, 5))
    axs[0].set_yticks(np.arange(0.4, 1.01, 0.2))
    axs[0].grid(True,linewidth=0.5,linestyle='--')
    # axs[0].set_xlabel('x')
    # axs[0].set_ylabel('y')

    # 子图2
    axs[1].plot(mean_bad,linewidth=1.2)
    axs[1].tick_params(axis='both', labelsize='x-small')
    axs[1].fill_between(np.arange(len_layers),bad_up,bad_down,where=(bad_up > bad_down), color='lightblue', alpha=0.4)
    # axs[1].legend(fontsize='x-small')
    # axs[1].set_ylim([y_min, y_max])  # 设置y轴范围
    axs[1].grid(True,linewidth=0.5,linestyle='--')
    axs[1].set_title('Malicious-Malicious Query Pairs',fontsize='x-small')
    # axs[1].set_xlabel('x')

    # 子图3
    axs[2].plot(mean_gb,linewidth=1.2)
    axs[2].fill_between(np.arange(len_layers),gb_up,gb_down,where=(gb_up > gb_down), color='lightblue', alpha=0.4)
    # axs[2].set_ylim([y_min, y_max])  # 设置y轴范围
    fig.text(0.525, 0.15, 'Layer', ha='center', va='center',fontsize='x-small')
    fig.text(0.03, 0.53, 'Cos_sim value', ha='center', va='center', rotation='vertical',fontsize='x-small')
    axs[2].tick_params(axis='both', labelsize='x-small')
    axs[2].set_title('Normal && Malicious Query Pairs',fontsize='x-small')
    axs[2].grid(True,linewidth=0.5,linestyle='--')
    # axs[2].set_xlabel('x')
    labels=['Layer-wise Average Cosine Similarity']
    fig.tight_layout(rect=[0.02,0.08,1,1])
    fig.legend(labels, loc='lower center', bbox_to_anchor=(0.519, -0.015), ncol=1, frameon=True,fontsize='x-small') 
    save_dir=save_dir+'1.png'
    plt.savefig(save_dir,dpi=500)
    
fire.Fire(main)