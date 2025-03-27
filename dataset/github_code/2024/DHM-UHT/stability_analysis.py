from representation_analysis import*
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def get_mean_curve(dataset='sinusoid', curve='parameter_dist', scheme='', ax=None):
    split_list = ["train"]
    for train_or_test in split_list:
        cca_result_collector_seed_list = []
        for seed in range(1, 6):
            scheme_name = scheme + str(seed) + '_seed'
            cca_result_collector = np.load(
                './output/representation_analyse/' + dataset + '/' + curve + '_' + scheme_name + '_' + train_or_test + '.npy',
                allow_pickle=True)
            cca_result_collector_seed_list.append(cca_result_collector)

        result_average_seed = np.mean(cca_result_collector_seed_list, axis=0)

        for i in range(len(result_average_seed)):
            reduce_curve = result_average_seed[i][range(0, len(result_average_seed[i]), 5)]
            ax.plot(reduce_curve, label="L" + str(i), alpha=0.75, markersize=6)

        if curve == 'parameter_dist':
            if 'BP' in scheme:
                ax.set_ylim(0, 0.008)
            else:
                ax.set_ylim(0, 0.008)

        ax.set_title(scheme.split('_', -1)[0], fontsize=25)
        ax.tick_params(axis='both', labelsize=20)

def draw_compare_ned():
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
    for i, scheme in enumerate(['BP_vary_amp_0.1_10_amp_', 'ANIL_4_layer_3_head_layer_', 'MAML_vary_amp_0.1_10_amp_']):
        get_mean_curve(curve='parameter_dist', scheme=scheme, ax=axs[i])

    axs[1].set_xlabel('Epoch', fontsize=25)
    axs[0].set_ylabel('NED', fontsize=25)
    axs[2].legend(loc='lower center', bbox_to_anchor=(1.1, 0.25), ncol=1, fontsize=16)
    plt.tight_layout()
    plt.savefig('./output/paper_fig/' + 'compare_ned.pdf', bbox_inches='tight')
    plt.show()

def draw_compare_cca():
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
    for i, scheme in enumerate(['MTL_vary_amp_0.1_10_amp_', 'ANIL_4_layer_3_head_layer_', 'MAML_vary_amp_0.1_10_amp_']):
        get_mean_curve(curve='cca', scheme=scheme, ax=axs[i])

    axs[1].set_xlabel('Epoch', fontsize=25)  # 调整坐标轴标签字体大小
    axs[0].set_ylabel('CCA', fontsize=25)
    axs[2].legend(loc='lower center', bbox_to_anchor=(1.1, 0.25), ncol=1, fontsize=16)
    plt.tight_layout()
    plt.savefig('./output/paper_fig/' + 'compare_cca.pdf', bbox_inches='tight')
    plt.show()

def draw_grad_var():
    var_dict = {}
    var_dict['BP'] = np.load('./output/representation_analyse/sinusoid/MTL_direct_var.npy', allow_pickle=True)
    var_dict['MAML'] = np.load('./output/representation_analyse/sinusoid/MAML_direct_var.npy', allow_pickle=True)
    var_dict['ANIL'] = np.load('./output/representation_analyse/sinusoid/ANIL_direct_var.npy', allow_pickle=True)
    fig, axs = plt.subplots(figsize=(6, 4.5))  # 创建一个包含三个子图的大图

    for i, method in enumerate(['MTL', 'ANIL', 'MAML']):
        plt_data = np.mean(var_dict[method], axis=-1)
        plt_data = np.mean(plt_data, axis=1)
        axs.plot(plt_data, label=method, alpha=0.75, markersize=6)  # 调整标记大小

    # axs.set_title(method, fontsize=25)  # 调整标题字体大小
    axs.tick_params(axis='both', labelsize=20)  # 调整刻度标签字体大小

    axs.set_xlabel('Epoch', fontsize=25)  # 调整坐标轴标签字体大小
    axs.set_ylabel(r'$\eta$', fontsize=25)
    # axs.set_ylim(0, 60)
    axs.legend(fontsize=16)
    plt.tight_layout()  # 自动调整子图布局，以确保不重叠
    plt.savefig('./output/paper_fig/' + 'grad_cos_var.pdf', bbox_inches='tight')
    plt.show()

def draw_grad_norm():
    var_dict = {}
    var_dict['MTL'] = np.load('./output/representation_analyse/sinusoid/MTL_grad_norm.npy', allow_pickle=True)
    var_dict['MAML'] = np.load('./output/representation_analyse/sinusoid/MAML_grad_norm.npy', allow_pickle=True)
    var_dict['ANIL'] = np.load('./output/representation_analyse/sinusoid/ANIL_grad_norm.npy', allow_pickle=True)
    fig, axs = plt.subplots(figsize=(6, 4.5))  # 创建一个包含三个子图的大图

    for i, method in enumerate(['MTL', 'ANIL', 'MAML']):
        plt_data = np.mean(var_dict[method], axis=-1)
        plt_data = np.mean(plt_data, axis=0)
        axs.plot(plt_data, label=method, alpha=0.75, markersize=6)  # 调整标记大小

    # axs.set_title(method, fontsize=25)  # 调整标题字体大小
    axs.tick_params(axis='both', labelsize=20)  # 调整刻度标签字体大小

    axs.set_xlabel('Epoch', fontsize=25)  # 调整坐标轴标签字体大小
    axs.set_ylabel(r'$||\overline{g}||_2$', fontsize=25)
    axs.set_ylim(0, 60)
    axs.legend(fontsize=16)
    plt.tight_layout()  # 自动调整子图布局，以确保不重叠
    plt.savefig('./output/paper_fig/' + 'grad_norm.pdf', bbox_inches='tight')
    plt.show()