from glob import glob
import gzip
from os.path import abspath, exists, isdir, getsize
from sys import argv
from random import randrange
from numpy import array, mean, std, transpose, nanmean, isnan, nan_to_num
from sklearn.metrics import roc_auc_score
from os import makedirs
from common import load_properties, fmax_score
from pandas import concat, read_csv, DataFrame
from itertools import product
from scipy import stats
#from scipy.integrate import simps
from numpy import trapz, array


# # # # #
	
def resultsFE(RULE):
    file_name = '%s/RESULTS/FE/RESULTS_FE_%s_fmax.csv' % (project_path, RULE)
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    fe_labels = list(mydf.columns.values)[1:]
    fe_mean  = list(mean(mydf, axis=0))
    #fe_std = list(std(mydf,axis=0))
    fe_std = list(stats.sem(mydf.iloc[1:mydf.shape[0],1:mydf.shape[1]], axis = 0))
    fe_mean_steps = [fe_mean[step] for step in steps]
    fe_std_steps = [fe_std[step] for step in steps]

    y  = array(fe_mean_steps)
    #auc = trapz(y, dx=1)/float(max(fe_mean)*180)
    auc = trapz(y, dx=10)/float(max_num_clsf - 10)

    #from scipy.integrate import simps   
    #a2 = simps(y, dx=1)

    # debugging tranpz: this should return 0.5, because we look for the area under the diagonal, and it does
    #a3 = trapz(array([0, float(max(fe_mean)/2), max(fe_mean)]), dx = 90)/float(max(fe_mean)*180)
    #print a3

    return fe_labels, fe_mean, fe_std, fe_mean_steps, fe_std_steps, auc


def resultsFE_rnd(RULE):
    file_name = '%s/RESULTS/FE/RESULTS_FE_%s_rnd_fmax.csv' % (project_path, RULE)
    print(file_name)
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    fe_labels = list(mydf.columns.values)[1:]
    fe_mean  = list(mean(mydf, axis=0))
    #fe_std = list(std(mydf,axis=0))
    fe_std = list(stats.sem(mydf.iloc[1:mydf.shape[0],1:mydf.shape[1]], axis = 0))
    fe_mean_steps = [fe_mean[step] for step in steps]
    fe_std_steps = [fe_std[step] for step in steps]

    y  = array(fe_mean_steps)
    #auc = trapz(y, dx=1)/float(max(fe_mean)*180)
    auc = trapz(y, dx=10)/float(max_num_clsf - 10)

    #from scipy.integrate import simps   
    #a2 = simps(y, dx=1)

    # debugging tranpz: this should return 0.5, because we look for the area under the diagonal, and it does
    #a3 = trapz(array([0, float(max(fe_mean)/2), max(fe_mean)]), dx = 90)/float(max(fe_mean)*180)
    #print a3

    return fe_labels, fe_mean, fe_std, fe_mean_steps, fe_std_steps, auc


def resultsBEST():
    file_name = '%s/RESULTS/BP/RESULTS_BP_fmax.csv' % (project_path)
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    best_labels = list(mydf.columns.values)[1:]
    best_mean  = list(mean(mydf, axis=0))
    #bp_std = list(std(mydf,axis=0))
    best_std = list(stats.sem(mydf.iloc[1:mydf.shape[0],1:mydf.shape[1]], axis = 0))
    best_mean_steps = [best_mean[step] for step in steps]
    best_std_steps = [best_std[step] for step in steps]
    
    y  = array(best_mean_steps)
    auc = trapz(y, dx=10)/float(max_num_clsf - 10)
    #auc = trapz(y, dx=1)/float(max(best_mean)*180)

    return best_labels, best_mean, best_std, best_mean_steps, best_std_steps, auc


def resultsBEST_rnd():
    file_name = '%s/RESULTS/BP/RESULTS_BP_rnd_fmax.csv' % (project_path)
    print(file_name)
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    best_labels = list(mydf.columns.values)[1:]
    best_mean  = list(mean(mydf, axis=0))
    #bp_std = list(std(mydf,axis=0))
    best_std = list(stats.sem(mydf.iloc[1:mydf.shape[0],1:mydf.shape[1]], axis = 0))
    best_mean_steps = [best_mean[step] for step in steps]
    best_std_steps = [best_std[step] for step in steps]

    y  = array(best_mean_steps)
    auc = trapz(y, dx=10)/float(max_num_clsf - 10)
    #auc = trapz(y, dx=1)/float(max(best_mean)*180)
    return best_labels, best_mean, best_std, best_mean_steps, best_std_steps, auc



def resultsSTACK():
    file_name = '%s/RESULTS/STACK/RESULTS_STACK_fmax.csv' % (project_path)
    print(file_name)
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    stack_labels = list(mydf.columns.values)[1:]
    stack_mean  = list(mean(mydf, axis=0))
    #bp_std = list(std(mydf,axis=0))
    stack_std = list(stats.sem(mydf.iloc[1:mydf.shape[0],1:mydf.shape[1]], axis = 0))

    dim_file = '%s/RESULTS/STACK/RESULTS_STACK_dim.csv' % (project_path)
    dimdf = read_csv(dim_file)
    dim = list(mean(dimdf, axis=0))

    dim_steps = [("%.1f" % dim[step]) for step in steps]
    stack_mean_steps = [stack_mean[step] for step in steps]
    stack_std_steps = [stack_std[step] for step in steps]

    y  = array(stack_mean_steps)
    auc = trapz(y, dx=10)/float(max_num_clsf - 10)
    #auc = trapz(y, dx=1)/float(max(stack_mean)*180)
    return dim_steps, stack_mean, stack_std, stack_mean_steps, stack_std_steps, auc



def resultsCES(RULE): #pathRL/sl/RESULTS/CES/RESULTS_CES_WA_start-1_[fmax, dim].csv
    file_name = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_fmax.csv' % (project_path, RULE, '1')
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    ces_labels = list(mydf.columns.values)[1:]
    ces_mean  = list(mean(mydf, axis=0))
    ces_std = list(stats.sem(mydf.iloc[1:mydf.shape[0],1:mydf.shape[1]], axis = 0))

    dim_file = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_dim.csv' % (project_path, RULE, '1')
    dimdf = read_csv(dim_file)
    dim = list(mean(dimdf, axis=0))
    #dim_steps = [dim[step] for step in steps]
    dim_steps = [("%.2f" % dim[step]) for step in steps]
    ces_mean_steps = [ces_mean[step] for step in steps]
    ces_std_steps = [ces_std[step] for step in steps]

    y  = array(ces_mean_steps)
    auc = trapz(y, dx=10)/float(max_num_clsf - 10)
    #auc = trapz(y, dx=1)/float(max(ces_mean)*180)
    return dim_steps, ces_mean, ces_std, ces_mean_steps, ces_std_steps, auc





def resultsCES_v(ver, RULE): #pathRL/sl/RESULTS/CES/RESULTS_CES_WA_start-1_[fmax, dim].csv
    file_name = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_fmax_%s.csv' % (project_path, RULE, '1', ver)
    mydf = read_csv(file_name)
    mydf.fillna(0, inplace=True)
    ces_labels = list(mydf.columns.values)[1:]
    ces_mean  = list(mean(mydf, axis=0))
    ces_std = list(stats.sem(mydf.iloc[1:mydf.shape[0],1:mydf.shape[1]], axis = 0))

    dim_file = '%s/RESULTS/CES/RESULTS_CES_%s_start-%s_dim_%s.csv' % (project_path, RULE, '1', ver)
    dimdf = read_csv(dim_file)
    dim = list(mean(dimdf, axis=0))
    #dim_steps = [dim[step] for step in steps]
    dim_steps = [("%.2f" % dim[step]) for step in steps]
    ces_mean_steps = [ces_mean[step] for step in steps]
    ces_std_steps = [ces_std[step] for step in steps]

    y  = array(ces_mean_steps)
    auc = trapz(y, dx=10)/float(max_num_clsf - 10)
    #auc = trapz(y, dx=1)/float(max(ces_mean)*180)
    return dim_steps, ces_mean, ces_std, ces_mean_steps, ces_std_steps, auc


def resultsRL(epsilon, age, conv, exit, strategy, RULE, algo, start):
    results_file = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.fmax.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start)
    print(results_file)
    mydf = read_csv(results_file)
    #mydf.fillna(0, inplace=True)

    rl_labels = list(mydf.columns.values)[1:] 
    mydf = read_csv(results_file, index_col=0)
    #print mydf.iloc[:, 9:21]
    rl_mean  = list(nanmean(mydf, axis=0))
    #print rl_mean
    rl_std = list(stats.sem(mydf.iloc[0:mydf.shape[0],0:mydf.shape[1]], axis = 0))
    dim_file = '%s/RESULTS/RL/RESULTS_RL_epsilon%s_pre%s_conv%s_exit%s_%s_%s_%s_start-%s.dim.csv' % (project_path, epsilon, age, conv, exit, strategy, RULE, algo, start)
    dimdf = read_csv(dim_file)
    dim = list(mean(dimdf, axis=0))
    dim_steps = [("%.2f" % dim[step]) for step in steps]
    rl_mean_steps = [rl_mean[step] for step in steps]
    rl_std_steps = [rl_std[step] for step in steps]

    y  = array(rl_mean_steps)
    auc = trapz(y, dx=10)/float(max_num_clsf - 10)
    #auc = trapz(y, dx=10)/float(max(rl_mean)*180)    
    
    return dim_steps, rl_mean, rl_std, rl_mean_steps, rl_std_steps, auc

# # # # #

def write_plot():
    str = ("\n\n")
    for epsilon in epsilons:

        best_labels, best_mean, best_std, best_mean_steps, best_std_steps, best_auc = resultsBEST_rnd()
        str += ("    avg_best = %r\n"
		"    std_best = %r\n" % (best_mean_steps, best_std_steps))
        str += ("    best = plt.errorbar(%r, avg_best, yerr=std_best, color='%s', linewidth=0.5)\n" % (x_ticks, colorbpfe[0]))
        str += ("    best = plt.plot(%r, avg_best, color='%s', label='BEST BP (auEPC=%.4f)', linewidth=0.5)\n\n\n" % (x_ticks, colorbpfe[0], best_auc))


        #fe_labels, fe_mean, fe_std, fe_mean_steps, fe_std_steps, fe_auc = resultsFE_rnd('WA')
        #str += ("    avg_fe = %r\n"
	#        "    std_fe = %r\n" % (fe_mean_steps, fe_std_steps))
        #str += ("    fe = plt.errorbar(%r, avg_fe, yerr=std_fe, color='%s', linewidth=0.5)\n" % (x_ticks, colorbpfe[1]))
        #str += ("    fe = plt.plot(%r, avg_fe, color='%s', label='FE-WA (auEPC=%.4f)', linewidth=0.5)\n\n\n" % (x_ticks, colorbpfe[1], fe_auc))

        
        
        fe2_labels, fe2_mean, fe_std, fe2_mean_steps, fe2_std_steps, fe2_auc = resultsFE_rnd('L2')
        str += ("    avg_fe2 = %r\n"
                "    std_fe2 = %r\n" % (fe2_mean_steps, fe2_std_steps))
        str += ("    fe2 = plt.errorbar(%r, avg_fe2, yerr=std_fe2, color='%s', linewidth=0.05)\n" % (x_ticks, colorbpfe[1]))
        str += ("    fe2 = plt.plot(%r, avg_fe2, color='%s', label='FE-L2 (auEPC=%.4f)', linewidth=0.05)\n\n\n" % (x_ticks, colorbpfe[1], fe2_auc))



        stack_labels, stack_mean, stack_std, stack_mean_steps, stack_std_steps, stack_auc = resultsSTACK()
        str += ("    dim_stack = %r\n"
                "    avg_stack = %r\n"
                "    std_stack = %r\n" 
		"    x=0\n"
		"    for i,j in zip(%r, %s):\n"
                "        plt.annotate(dim_stack[x], xy=(i,j+0.001), fontsize = 5, color='black')\n"
                "        x+=1\n" % (stack_labels, stack_mean_steps, stack_std_steps, x_ticks, stack_mean_steps))
        str += ("    stack = plt.errorbar(%r, avg_stack, yerr=std_stack, color='black', linewidth=0.5)\n" % (x_ticks))
        str += ("    stack = plt.plot(%r, avg_stack, color='black', label='FE-L1 (STACKING) (auEPC=%.4f)', linewidth=1)\n\n\n" % (x_ticks, stack_auc))


        '''ces_labels, ces_mean, ces_std, ces_mean_steps, ces_std_steps, ces_auc = resultsCES_v('v4', 'WA')
        str += ("    dim_ces = %r\n"
                "    avg_ces = %r\n"
                "    std_ces = %r\n"
                "    x=0\n"
                "    for i,j in zip(%r, %s):\n"
                "        plt.annotate(dim_ces[x], xy=(i,j), fontsize = 7, color='orange')\n"
                "        x+=1\n" % (ces_labels, ces_mean_steps, ces_std_steps, x_ticks, ces_mean_steps))
        str += ("    ces = plt.errorbar(%r, avg_ces, yerr=std_ces, color='orange')\n" % (x_ticks))
        str += ("    ces = plt.plot(%r, avg_ces, color='orange', label='CES (AUC=%.4f)', linewidth=1)\n\n\n" % (x_ticks, ces_auc))'''



        index = 0
        for strategy in strategies:
	    # D I V E R S I T Y    R N D & V2 #
            if (strategy == 'diversityrnd') or (strategy == 'euclideanrnd') or (strategy == 'correlationrnd') or (strategy == 'yulernd') or (strategy == 'kapparnd'):	        
                conv = '10'
                age = 0
                rl_labels, mean_rl, std_rl, rl_mean_steps, rl_std_steps, rl_auc = resultsRL(epsilon, age, conv, exit, strategy, RULE, algo, start)
                str += ("    dim_rl = %r\n"
	  		"    avg_rl = %r\n"
			"    std_rl = %r\n" % (rl_labels, rl_mean_steps, rl_std_steps))
                if (annotation):
                    str += ("    x=0\n"
		            "    for i,j in zip(%r, avg_rl):\n"
			    "        plt.annotate(dim_rl[x], xy=(i,j), fontsize = %i, color = \'%s\')\n"
#			    "        x+=1\n" % (x_ticks, (16-2*index), color[index]))
                            "        x+=1\n" % (x_ticks, (10-index), color[index]))
                str += ("    rl = plt.errorbar(%r, avg_rl, yerr=std_rl, color=\'%s\', linewidth=0.5)\n" % (x_ticks, color[index]))
                str += ("    rl = plt.plot(%r, avg_rl, color=\'%s\', label=\'RL_%s_conv%s (auEPC=%.4f)\', linewidth=0.5)\n\n\n" % (x_ticks, color[index], strategy, conv, rl_auc))

		#strategy = strategy + "v2"
		#rl_labels, mean_rl, std_rl, rl_mean_steps, rl_std_steps, rl_auc = resultsRL(epsilon, age, conv, exit, strategy, RULE, algo, start)
		#str += ("    dim_rl = %r\n"
		#	"    avg_rl = %r\n"
		#	"    std_rl = %r\n" % (rl_labels, rl_mean_steps, rl_std_steps))
		#if (annotation and strategy == 'diversityrndv2'):
		#    str += ("    x=0\n"
	 	#	    "    for i,j in zip(%r, avg_rl):\n"
		#	    "        plt.annotate(dim_rl[x], xy=(i,j), fontsize = 5, color = \'%s\')\n"
		#	    "        x+=1\n" % (x_ticks, color[index]))
		#str += ("    rl = plt.errorbar(%r, avg_rl, yerr=std_rl, color=\'%s\', linewidth=0.5, linestyle = '--')\n" % (x_ticks, color[index]))
                #str += ("    rl = plt.plot(%r, avg_rl, color=\'%s\', label=\'RL_%s_conv%s (auEPC=%.4f)\', linewidth=0.5, linestyle = '--')\n\n\n" % (x_ticks, color[index], strategy, conv, rl_auc))
                index += 1
            elif ('pessim' in strategy) or (strategy == 'backtrackrnd'):
                conv = '0'
                rl_labels, mean_rl, std_rl, rl_mean_steps, rl_std_steps, rl_auc = resultsRL(epsilon, '500000', conv, exit, strategy, RULE, algo, start)
                str += ("    dim_rl = %r\n"
                        "    avg_rl = %r\n"
                        "    std_rl = %r\n" % (rl_labels, rl_mean_steps, rl_std_steps))

                if (annotation):
                    str += ("    x=0\n"
                            "    for i,j in zip(%r, avg_rl):\n"
                            "        plt.annotate(dim_rl[x], xy=(i,j), fontsize = 4, color = \'%s\')\n"
                            "        x+=1\n" % (x_ticks, color[index]))
                str += ("    rl = plt.errorbar(%r, avg_rl, yerr=std_rl, color=\'%s\', linewidth=0.5)\n" % (x_ticks, color[index]))
                str += ("    rl = plt.plot(%r, avg_rl, color=\'%s\', label=\'RL_%s_age0.5M (auEPC=%.4f)\', linewidth=0.5)\n\n\n" % (x_ticks, color[index], strategy, rl_auc))
                index += 1
            else:
                conv = '10'
                rl_labels, mean_rl, std_rl, rl_mean_steps, rl_std_steps, rl_auc = resultsRL(epsilon, '0', conv, exit, strategy, RULE, algo, start)
                str += ("    dim_rl = %r\n"
                        "    avg_rl = %r\n"
                        "    std_rl = %r\n" % (rl_labels, rl_mean_steps, rl_std_steps))
                if (annotation):
                    str += ("    x=0\n"
                            "    for i,j in zip(%r, avg_rl):\n"
                            "        plt.annotate(dim_rl[x], xy=(i,j), fontsize = 4, color = \'%s\')\n"
                            "        x+=1\n" % (x_ticks, color[index]))
                str += ("    rl = plt.errorbar(%r, avg_rl, yerr=std_rl, color=\'%s\', linewidth=0.5)\n" % (x_ticks, color[index]))
                str += ("    rl = plt.plot(%r, avg_rl, color=\'%s\', label=\'RL_%s_conv10 (auEPC=%.4f)\', linewidth=0.5)\n\n\n" % (x_ticks, color[index], strategy, rl_auc))	
                index += 1
        str += ("    plt.title('%s -- epsilon=%s, RL-%s')\n" % (proj, epsilon, RULE))
        str += ("    plt.margins(0.01)\n")
        str += ("    plt.ylim(%s)\n") % plot_ylim
        str += ("    plt.xlabel('Number of initial base predictors', fontsize=15)\n")
        str += ("    xticks = [20, 40, 60, 80, 100, 120, 140, 160, 180]\n")
        str += ("    yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]\n")
        str += ("    plt.xticks(xticks, fontsize=15)\n")
        str += ("    plt.xticks(xticks, fontsize=15)\n")
        str += ("    plt.ylabel('F-Max', fontsize=15)\n")
        str += ("    plt.legend(loc=%s, prop={\'size\':8})\n" % (legend_loc))
        str += ("    pdf.savefig()\n    plt.close()\n\n") #after each plot 
    return str





# # # # #

def get_data_rl(project_path):
    str = ""
    str+=(write_plot())
    str+=("\n")
    return str

def generate_script(project_path):
    header = ("from numpy import nan\n"
              "from matplotlib.backends.backend_pdf import PdfPages\n"
              "import matplotlib.pyplot as plt\n\n"
              "with PdfPages('%s_auEPerfC.pdf') as pdf:\n\n    plt.style.use('ggplot')\n" % (proj))
    script_name = "%s/PLOTS/%s_auEPerfC.py" % (project_path, proj)
    with open(script_name, "w+") as plot_file:
        plot_file.write(header)
        plot_file.write(get_data_rl(project_path))
    plot_file.close()
    print(script_name)






proj = argv[1].replace("/", "")
project_path = "path/RL/%s" % proj
p = load_properties(project_path)


fold_count = int(p['foldCount'])
seeds = int(p['seeds'])

inputFilename = p['inputFilename']
exp = inputFilename.split(".")[0]

epsilons     = ['0.01', '0.10', '0.25', '0.50']
epsilons     = ['0.01']
ages         = [500000, 0]
conv_iters   = ['10']
exits        = [0]
#greedyrnd is purple
strategies   = ['pessimisticrnd', 'pessimcos', 'pessimeuclid', 'pessimcorr', 'pessimyule', 'pessimkappa'] #"diversityrnd", "euclideanrnd", "correlationrnd", "yulernd", "kapparnd", "gsernd"]
#strategies = []
color        = ['darkgrey', 'cyan', 'blue', 'lightseagreen', 'orange', 'magenta', 'yellow']
colorbpfe    = ['green', 'red']


#pastel
#color = ['white', "purple", "#FFFEA3",  "#FF9F9A", "#D0BBFF", "#B0E0E6", 'azure', 'darkseagreen', "#3498db", 'bisque', 'lightslategrey']


annotation   = 1
RULE         = 'L2'
algos        = ['Q']
#start_states = ['0', 'best']
start_states =['0']
linestyle    = {0:'-', 1:':'}

max_num_clsf = 180
sizes        = range(1, max_num_clsf+1)
x_ticks      = range(10, max_num_clsf+1, 10) 
steps        = [step-1 for step in x_ticks]
exit = 0
algo = 'Q'
start = '0'

if "pf1" in project_path:
    plot_ylim = "0.21, 0.25"
    legend_loc = "\'lower center\'"
elif "pf2" in project_path:
    plot_ylim = "0.2, 0.3"
    legend_loc = "\'lower right\'"
elif "pf3" in project_path:
    plot_ylim = "0.24, 0.33"
    legend_loc = "\'lower right\'"
elif "thaliana" in project_path:
    plot_ylim = "0.35, 0.54"
    legend_loc = "\'lower center\'"
elif "pacificus" in project_path:
    plot_ylim = "0.5, 0.7"
    legend_loc = "\'lower center\'"
elif "remanei" in project_path:
    plot_ylim = "0.56, 0.71"
    legend_loc = "\'lower center\'"
elif "elegans" in project_path:
    plot_ylim = "0.5, 0.67"
    legend_loc = "\'lower center\'"
elif "drosophila" in project_path:
    plot_ylim = "0.5, 0.6"
    legend_loc = "\'lower left\'"
elif "sl" in project_path:
    plot_ylim = "0.28, 0.36"
    legend_loc = "\'lower center\'"
elif "diabetic" in project_path:
    plot_ylim = "0.38, 0.42"
    legend_loc = "\'lower center\'"
elif "nba" in project_path:
    plot_ylim = "0.76, 0.8"
    legend_loc = "\'lower center\'"
elif "santander" in project_path:
    plot_ylim = "0.45, 0.6"
    legend_loc = "\'lower center\'"
elif "scania" in project_path:
    plot_ylim = "0.55, 0.75"
    legend_loc = "\'lower center\'"
elif "exoplanet" in project_path:
    plot_ylim = "0.0, 0.2"
    legend_loc = "\'lower center\'"
elif "creditcard" in project_path:
    plot_ylim = "0.53, 0.55"
    legend_loc = "\'lower center\'"
elif "pfp1" in project_path:
    plot_ylim = "0.81, 0.85"
    legend_loc = "\'lower center\'"
elif "pfp2" in project_path:
    plot_ylim = "0.77, 0.80"
    legend_loc = "\'lower center\'"
elif "pfp3" in project_path:
    plot_ylim = "0.80, 0.82"
    legend_loc = "\'lower center\'"
elif "apply" in project_path:
    plot_ylim = "0.185, 0.21"
    legend_loc = "\'lower center\'"
elif "vehicle" in project_path:
    plot_ylim = "0.45, 0.6"
    legend_loc = "\'lower center\'"
else:
    plot_ylim = "0.0, 0.5"
    legend_loc = "\'lower center\'"
generate_script(project_path)

print("Done!")



