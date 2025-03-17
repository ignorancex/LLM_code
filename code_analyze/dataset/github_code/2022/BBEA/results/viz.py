from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import time
import copy

from .analysis import *  
from .viz_util import *

from pylab import rcParams
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import rankdata
import scipy.stats as sps


def debug(args):
    #print(args)
    pass


def draw_regret_curve(results, arms, repeats,
                          title=None, x_unit="Second",
                          alpha_fill=0.1, std_div=4, xlim=None, ylim=None,
                          width=14, height=8, x_steps=1, guidelines=[],
                          save_name=None, target_folder='./', y_scale=1,
                          sub_y_metric="test error", best_err=0.2153,
                          legend=None, l_order=None, style_format=None, max_err=None):

    if type(arms) is not list:
        arms = [arms]

    unlabeled_arms = set(arms)
    rcParams['mathtext.default'] = 'regular'
    rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    subplot.grid(alpha=0.5)
    t_max = None
    
    its = FontProperties()
    its.set_style('italic')
    its.set_size('x-large')

    if max_err != None and best_err != None:
        max_regret = max_err - best_err
    else:
        max_regret = 1.0
    
    for arm in arms:
        if not arm in results.keys():
            raise ValueError(
                "results log do not have record for {}".format(arm))
        if style_format == None:
            style_format = get_predefined_style

        marker, color, linestyle = style_format(arm)

        #errors_by_interval = { }
        regrets_by_interval = { } 

        subplot.set_yscale('log')
        subplot.set_xscale('log')

        t_max = None
        s_t = time.time()
        for i in range(repeats):
            selected = get_result(results, arm, i)
            if max_err == None:
                max_err = selected['error'][0]
                max_regret = max_err - best_err
            y_best_errors = np.array(get_best_errors(selected)) 
            

            t_times = get_total_times(selected, x_unit)
            
            if t_max == None:
                t_max = int(max(t_times)) + 1
            elif t_max > int(max(t_times)) + 1:
                t_max = int(max(t_times)) + 1
            
            t = 0
            n_step = 0
            cur_best_err = max_err 
            for j in range(len(t_times)):

                cur_time = t_times[j]
                
                while t < cur_time:
                    if not t in regrets_by_interval.keys():
                        regrets_by_interval[t] = []
                        
                    #errors_by_interval[t].append(cur_best_err)
                    regret = (cur_best_err - best_err) * y_scale
                    
                    if regret < 0:
                        print("Invalid reget at {}: cur {}, best {}".format(t, cur_best_err, best_err))
                        raise ValueError("Invalid best error value!")
                    
                    regrets_by_interval[t].append(regret)
                    t += 2**n_step
                    if n_step <= 10:
                        n_step += 1

                if y_best_errors[j] < cur_best_err:
                    cur_best_err = y_best_errors[j]
                    
        #print("{}: calc regret takes {}".format(arm, time.time() - s_t))
        
        s_t = time.time()
        x = np.array([])
        y = np.array([]) 
        q1 = np.array([])
        q3 = np.array([])

        for i in range(0, t_max):
            # TODO: move  i in log scale
            if not i in regrets_by_interval:
                continue
            
            #errors = errors_by_interval[i]
            regrets = regrets_by_interval[i]

            y_ = np.median(regrets)
            #print("mean regret: {}".format(y_))
            y = np.append(y, y_)

            q1_ = np.quantile(regrets, 0.25)
            q1 = np.append(q1, q1_)
            q3_ = np.quantile(regrets, 0.75)
            q3 = np.append(q3, q3_)

            x = np.append(x, i)
            
        #print("{}: data sampling time {}".format(arm, time.time() - s_t))        
        if arm in unlabeled_arms:
            add_quantile_fill_line(x, y, q1, q3, color, ax=subplot, marker=marker, y_scale=y_scale,
                                                label=arm, linestyle=linestyle, alpha_fill=alpha_fill)
            unlabeled_arms.remove(arm)
        else:
            add_quantile_fill_line(x, y, q1, q3, color, marker=marker, y_scale=y_scale,
                                                        ax=subplot, linestyle=linestyle, alpha_fill=alpha_fill)

    if title != None:
        subplot.set_title('{}'.format(title))
    bbox_to_anchor = None
    loc = None
    borderaxespad = None
    
    if legend is not None:
        if 'bbox_to_anchor' in legend:
            bbox_to_anchor = legend['bbox_to_anchor']
        if 'loc' in legend:
            loc = legend['loc']
        if 'borderaxespad' in legend:
            borderaxespad = legend['borderaxespad']
    
        if l_order is not None:
            handles, labels = subplot.get_legend_handles_labels()        
            plt.legend([handles[idx] for idx in l_order], [labels[idx] for idx in l_order],
            prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
                loc=loc, borderaxespad=borderaxespad)        
        else:
            plt.legend(prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
                loc=loc, borderaxespad=borderaxespad)
    
    x_pos = 50
    if xlim is not None:
        plt.xlim(xlim)
        x_pos = xlim[-1]

    if ylim is not None:
        plt.ylim(ylim)        


    for s in guidelines:
        
        if 'regret' in s:
            pos = s['regret']            
            label = "{}".format(pos)
            if 'label' in s:
                label = "{}".format(s['label']) 
            plt.text(x_pos, pos, label, ha='right', size='x-large', color='black', fontproperties=its)
            plt.axhline(y=pos, color='red', linestyle=':')     

    plt.ylabel("Intermediate regret ({})".format(sub_y_metric), fontsize=15)
    plt.xlabel(x_unit, fontsize=15)
    if save_name is not None:
        plt.tight_layout()
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)
    else:
        plt.show()
        return plt



def draw_mean_regret_curve(results, arms, repeats,
                          title=None, x_unit="Hour",
                          alpha_fill=0.1, std_div=4, xlim=None, ylim=None,
                          width=14, height=8, x_steps='Hour', guidelines=[],
                          save_name=None, target_folder='./', y_scale=1,
                          sub_y_metric="error rate", best_err=0.001,
                          legend=None, l_order=None, style_format=None, max_err=None):

    if type(arms) is not list:
        arms = [arms]

    unlabeled_arms = set(arms)
    rcParams['figure.figsize'] = width, height
    rcParams['mathtext.default'] = 'regular'
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    subplot.grid(alpha=0.5)
    t_max = None
    x_pos = 50
    its = FontProperties()
    its.set_style('italic')
    its.set_size('x-large')

    if max_err != None and best_err != None:
        max_regret = max_err - best_err
    else:
        max_regret = 1.0
    
    for arm in arms:
        if not arm in results.keys():
            raise ValueError(
                "results log do not have record for {}".format(arm))
        if style_format == None:
            style_format = get_predefined_style

        marker, color, linestyle = style_format(arm)

        errors_by_interval = { }
        regrets_by_interval = { } 

        subplot.set_yscale('log')
        subplot.set_xscale('log')

        t_max = None
        
        for i in range(repeats):
            selected = get_result(results, arm, i)
            if max_err == None:
                max_err = selected['error'][0]
                max_regret = max_err - best_err
            y_best_errors = np.array(get_best_errors(selected)) 
            
            
            if x_unit == 'Second':
                if x_steps == 'Minute': 
                    t_times = get_total_times(selected, 'Minute')
                elif x_steps == 'Hour':
                    t_times = get_total_times(selected, 'Hour')
                else:
                    t_times = get_total_times(selected, '10min')
            else:
                t_times = get_total_times(selected, x_unit)
            
            if t_max == None:
                t_max = int(max(t_times)) + 1
            elif t_max > int(max(t_times)) + 1:
                t_max = int(max(t_times)) + 1
               
            
            t = 0
            cur_best_err = max_err 
            for j in range(len(t_times)):

                cur_time = t_times[j]

                while t < cur_time:
                    if not t in regrets_by_interval.keys():
                        errors_by_interval[t] = []
                        regrets_by_interval[t] = []
                    errors_by_interval[t].append(cur_best_err)
                    regret = (cur_best_err - best_err) * y_scale
                    if regret < 0:
                        print("Invalid reget at {}: cur {}, best {}".format(t, cur_best_err, best_err))
                    regrets_by_interval[t].append(regret)
                    t += 1

                if y_best_errors[j] < cur_best_err:
                    cur_best_err = y_best_errors[j]
                    
        #print("{}:{}".format(arm, regrets_by_interval.keys()))
                       
        x = np.array([])
        y = np.array([]) 
        q1 = np.array([])
        q3 = np.array([])

        for i in range(0, t_max):
            errors = errors_by_interval[i]
            regrets = regrets_by_interval[i]

            y_ = np.mean(regrets)
            #print("mean regret: {}".format(y_))
            y = np.append(y, y_)

            q1_ = np.quantile(regrets, 0.25)
            q1 = np.append(q1, q1_)
            q3_ = np.quantile(regrets, 0.75)
            q3 = np.append(q3, q3_)
            
            if x_unit == 'Second':
                if x_steps == 'Minute':
                    x = np.append(x, i*60)
                elif x_steps == 'Hour':
                    x = np.append(x, i*3600)
                else:
                    x = np.append(x, i*6)
            else:
                x = np.append(x, i)
            
        #print("{}: size of x: {}".format(arm, len(x)))        
        if arm in unlabeled_arms:
            add_no_fill_line(x, y, q1, q3, color, ax=subplot, marker=marker, y_scale=y_scale,
                                                label=arm, linestyle=linestyle, alpha_fill=alpha_fill)
            unlabeled_arms.remove(arm)
        else:
            add_no_fill_line(x, y, q1, q3, color, marker=marker, y_scale=y_scale,
                                                        ax=subplot, linestyle=linestyle, alpha_fill=alpha_fill)
        x_pos = np.min(x) + 50

    if title != None:
        subplot.set_title('{}'.format(title))
    bbox_to_anchor = None
    loc = None
    borderaxespad = None
    
    if legend is not None:
        if 'bbox_to_anchor' in legend:
            bbox_to_anchor = legend['bbox_to_anchor']
        if 'loc' in legend:
            loc = legend['loc']
        if 'borderaxespad' in legend:
            borderaxespad = legend['borderaxespad']
    
        if l_order is not None:
            handles, labels = subplot.get_legend_handles_labels()        
            plt.legend([handles[idx] for idx in l_order], [labels[idx] for idx in l_order],
            prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
                loc=loc, borderaxespad=borderaxespad)        
        else:
            plt.legend(prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
                loc=loc, borderaxespad=borderaxespad)
    
    
    if xlim is not None:
        plt.xlim(xlim)
        x_pos = xlim[0] + 50
        
    if ylim is not None:
        plt.ylim(ylim)

    for s in guidelines:
        
        if 'regret' in s:
            pos = s['regret']            
            label = "\n {}".format(pos)
            if 'label' in s:
                label = "\n {}".format(s['label']) 
            plt.text(x_pos, pos, label, size='x-large', color='black', fontproperties=its)
            plt.axhline(y=pos, color='red', linestyle=':')        
        
    plt.ylabel("Intermediate regret ({})".format(sub_y_metric), fontsize=15)
    plt.xlabel(x_unit, fontsize=15)
    if save_name is not None:
        plt.tight_layout()
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)
    else:
        plt.show()
        return plt


def draw_success_rate_fig(results, target_goal, x_max, 
                        num_runs=None, x_unit='Hour', step_size=1,
                        title=None, indi=None, div=None, ada=None,
                        save_name=None, target_folder='./',
                        indi_max=None, div_max=None, ada_max=None,
                        indi_scale=1, div_scale=1, ada_scale=1, name_map=None,
                        avgs=None, parallel=None, l_order=None,
                        width=10, height=6, legend=None, style_format=None,
                        show_marginal_best=False):
    opt_iterations = {}
    opt_successes = {}
    opts = list(sorted(results.keys()))

    x = range(0, x_max, step_size)

    if name_map is None:
        def map_names(name):
            return name
        name_map = map_names
    
    if num_runs == None:
        num_runs = len(results[opts[0]].keys())

    if style_format is None:
        style_format = get_predefined_style

    for opt in list(sorted(results.keys())):
        x_values = None
        if x_unit is 'Iteration':
            x_values = np.array(get_num_iters_over_threshold(
                results[opt], num_runs, target_goal))
        else:
            x_values = np.array(get_exec_times_over_threshold(
                results[opt], num_runs, target_goal, unit=x_unit))
        debug("x_values: {}".format(x_values[0]))
        opt_iterations[opt] = x_values
        successes = []
        for i in x:
            failure = (x_values[x_values > i].shape[0] / float(num_runs))
            success = 1.0 - failure
            successes.append(success)

        opt_successes[opt] = successes

    rcParams['mathtext.default'] = 'regular'
    rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(alpha=0.5)

    if parallel is not None:
        for g in parallel:
            if 'opt' in g:
                opt = g['opt']
                name = opt
                marker, color, linestyle = style_format(get_label(opt))
                sr = opt_successes[opt]
                x_ = x
                _max = x_max

                _scale = 1
                if 'scale' in g:
                    _scale = g['scale']
                    # linestyle=':'
                    name += ' [scaling]'

                if _scale * _max > x_max:
                    _max = int(x_max / _scale) + 1

                x_ = [x * _scale for x in range(0, _max, step_size)]
                sr = np.asarray(opt_successes[opt])

                max_index = min(len(x_), len(sr))
                if x_unit == 'Hour' and 'max_hour' in g:
                    max_index = g['max_hour'] + 1

                if sr.ndim == 1:
                    if x_max < max_index:
                        max_index = x_max
                    if len(x_) < max_index:
                        max_index = len(x_)
                    sr = sr.tolist()
                    debug("x: {}, y: {}, x_max: {}, max_index: {}".format(
                        len(x_), len(sr), x_max, max_index))

                    ax.plot(x_[:max_index], sr[:max_index],
                            marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))
                else:
                    for j in range(sr.ndim):
                        ax.plot(x_[:max_index], sr[:max_index, j],
                                marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))


    if indi is not None:
        for opt in indi:
            marker, color, linestyle = style_format(get_label(opt))
            sr = opt_successes[opt]
            x_ = x
            if indi_max is not None:
                x_ = range(0, indi_max, step_size)
                if indi_scale > 1:
                    if indi_scale * indi_max > x_max:
                        indi_max = int(x_max / indi_scale) + 1
                    x_ = [x * indi_scale for x in range(0, indi_max, step_size)]
                sr = opt_successes[opt][:indi_max]
            ax.plot(x_, sr,
                    marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))

    if ada is not None:
        for opt in ada:
            marker, color, linestyle = style_format(get_label(opt))
            sr = opt_successes[opt]
            x_ = x
            if ada_max is not None:
                x_ = range(0, div_max, step_size)
                sr = opt_successes[opt][:div_max]

            ax.plot(x_, sr,
                    marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))
                
    if avgs is not None and show_marginal_best:
        best_failures = []
        opt = 'xN-Div-I'
        for avg in sorted(avgs):
            list_size = len(opt_successes[avg])
            if len(best_failures) == 0:                
                best_failures = [ 1.0 for i in range(list_size)]
            #print("{}:{}".format(avg, opt_successes[avg]))
            for i in range(list_size):
                s = opt_successes[avg][i]
                f = 1.0 - s
                best_failures[i] *= f
        best_successes = []
        for bf in best_failures:
            best_successes.append(1.0 - bf)

        marker, color, linestyle = style_format(get_label(opt))
        ax.plot(x, best_successes, 
                marker='*', color=color, linestyle=linestyle, label=name_map(get_label(opt)))                   

    if div is not None:
        for opt in div:
            marker, color, linestyle = style_format(get_label(opt))
            sr = opt_successes[opt]
            x_ = x
            if div_max is not None:
                x_ = range(0, div_max, step_size)
                if div_scale > 1:
                    if div_scale * div_max > x_max:
                        div_max = int(x_max / div_scale) + 1
                    x_ = [x * div_scale for x in range(0, div_max, step_size)]
                    #linestyle = ':'

                sr = opt_successes[opt][:div_max]
            ax.plot(x_, sr,
                    marker=marker, color=color, linestyle=linestyle, label=name_map(get_label(opt)))

    if x_unit == "10min":
        subset_x = []
        for i in range(len(x)):
            if i % 6 == 0:
                subset_x.append(x[i])
        ax.set_xticks(subset_x)
        xlabels = [ x * 10 for x in subset_x ]
        ax.set_xticklabels(xlabels)
        x_unit = 'Minute'
    else:
        ax.set_xticks(x)

    minor_ticks = np.arange(0, 1.1, 0.1)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.3)

    if title is not None:
        ax.set_title(title)

    plt.xlabel("{}".format(x_unit), size=20)
    plt.ylabel("Success rate", fontsize=20)
    bbox_to_anchor = None
    loc = None
    borderaxespad = None
    if legend is not None:
        if 'bbox_to_anchor' in legend:
            bbox_to_anchor = legend['bbox_to_anchor']
        if 'loc' in legend:
            loc = legend['loc']
        if 'borderaxespad' in legend:
            borderaxespad = legend['borderaxespad']
        if l_order is not None:
            handles, labels = ax.get_legend_handles_labels()        
            plt.legend([handles[idx] for idx in l_order], [labels[idx] for idx in l_order],
            prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
                loc=loc, borderaxespad=borderaxespad)        
        else:
            plt.legend(prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
                loc=loc, borderaxespad=borderaxespad)

    if save_name is not None:
        plt.tight_layout()
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)
    else:
        return plt


def draw_trials_curve(results, arm, run_index,
                      x_unit='Hour', guidelines=[], g_best_acc=None,
                      xlim=None, ylim=None, title=None, save_name=None, max_err=None,
                      loc=3, width=10, height=6, metric='Test error'):
    selected = get_result(results, arm, run_index)
    x_time = get_total_times(selected, x_unit)
    y_errors = selected['error']
    if max_err == None:
        max_err = y_errors[0]
    
    if g_best_acc != None:
        g_best_err = 1.0 - g_best_acc
        y_errors = []
        for y in selected['error']:
            if y != None:
                y_errors.append(y - g_best_err)
            else:
                y_errors.append(max_err)  

    line_best_errors = np.array(get_best_errors(selected))
    if g_best_acc != None:
        line_best_errors = line_best_errors - g_best_err
        max_err = g_best_acc
    preamble = np.array([max_err])
    line_best_errors = np.concatenate((preamble, line_best_errors))
    rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    plot_func = subplot.semilogy
    if g_best_acc != None:
        plot_func = subplot.plot 

    available_arms = set([arm])
    #marker, color, linestyle = get_style(arm, results.keys())
    if 'select_trace' in selected:
        available_arms = set(selected['select_trace'])
    unlabeled_arms = set([arm])
    if len(available_arms) > 0:
        unlabeled_arms = copy.copy(available_arms)
    available_arms = list(sorted(available_arms))
    color = 'black'
    marker = '.'
    opacity = 1.0
    default_marker_size = 6.0
    marker_size = default_marker_size * opacity
    
    for i in range(len(x_time)):
        if len(available_arms) > 0:
            if 'select_trace' in selected:            
                arm = selected['select_trace'][i]

            marker, color, _ = get_style(arm, available_arms)
            if 'train_epoch' in selected:
                opacity = float(selected['train_epoch'][i] / max(selected['train_epoch']))
                marker_size = float(default_marker_size * opacity * 1.2)
                marker = 'o' # XXX: set same marker

            if 'model_index' in selected:
                if selected['model_index'][i] < 0:
                    color = 'red'

            if "epoch" in selected:
                if selected['epoch'][i] <= 3:
                    marker = '.'
                elif selected['epoch'][i] < 10:
                    marker = '*'
                else:
                    marker = 'o'
        # remove None result

        x = x_time[i]
        y = y_errors[i]
        if y != None:
            if arm in unlabeled_arms:
                plot_func(x, y, 
                    color=color, linestyle='', alpha=opacity,
                    marker=marker, markersize=marker_size,
                    label=get_label(arm))
                unlabeled_arms.remove(arm)
            else:
                plot_func(x, y,
                        color=color, linestyle='', alpha=opacity,
                        marker=marker, markersize=marker_size)

    # line plot for best error
    plot_func([0] + x_time, line_best_errors, color='blue',
                     linestyle='--', label='best error')

    if title is not None:
        subplot.set_title(title)
    if ylim is None:
        plt.ylim(ymax=max_err)
    else:
        plt.ylim(ylim)
    x_range = [0, 0]
    if xlim is not None:
        plt.xlim(xlim)
        x_range = list(xlim)
        x_ticks = [x for x in range(x_range[0], x_range[-1] + 1)]
        plt.xticks(x_ticks)

    for s in guidelines:
        label = ""
        pos = s['error']
        if g_best_acc != None and "regret" in s:
            pos = s['regret']

        if "rank" in s:
            label = "Top-{}".format(s['rank'])
        elif 'difficulty' in s:
            label = "Top {:.2f}%".format(s['difficulty']*100)

        plt.text(x_range[-1] + 0.1, pos, label, size=12)
        plt.axhline(y=pos, color='gray', linestyle=':')

    if g_best_acc != None:
        plt.ylabel("Intermidiate regret", fontsize=15)
    else:
        plt.ylabel(metric, fontsize=15)
    plt.xlabel(x_unit, size=15)
    plt.legend(loc=loc, prop={'size': 15})

    if save_name is not None:
        # plt.tight_layout()
        target_folder = '../../../figs/'
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)
    else:
        plt.show()


def draw_best_error_curve(results, arms, repeats,
                          guidelines=[], summary=False, title=None, x_unit="Hour",
                          xlim=None, ylim=(.001, 1), alpha_fill=0.1, std_div=4,
                          width=14, height=8, x_steps=1, plot_func='semilogy',
                          save_name=None, target_folder='.', y_scale=1,
                          x_ticks=None, y_ticks=None, sub_y_metric="test error",
                          legend=None, l_order=None, style_format=None, max_err=None):

    if type(arms) is not list:
        arms = [arms]

    unlabeled_arms = set(arms)
    rcParams['figure.figsize'] = width, height
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    subplot.grid(alpha=0.5)
    t_max = None
    
    for arm in arms:
        if not arm in results.keys():
            raise ValueError(
                "results log do not have record for {}".format(arm))
        if style_format == None:
            style_format = get_predefined_style

        marker, color, linestyle = style_format(arm)
        if summary is False:
            best_errors = []
            for i in range(repeats):
                selected = get_result(results, arm, i)
                if max_err == None:
                    max_err = selected['error'][0]
                x_time = get_total_times(selected, x_unit)
                y_best_errors = np.array(get_best_errors(selected)) * y_scale
                best_errors.append({'x': x_time, 'y': y_best_errors.tolist() })

            pfunc = subplot.plot
            if plot_func == 'semilogy':
                pfunc = subplot.semilogy
            elif plot_func == 'loglog':
                pfunc = subplot.loglog
                subplot.set_xscale('log',  nonposx='clip')

            for best_error in best_errors:
                if arm in unlabeled_arms:
                    pfunc([0] + best_error['x'], 
                        ([max_err] + best_error['y']), 
                        color=color, linestyle=linestyle, label=arm, marker=marker)
                    unlabeled_arms.remove(arm)
                else:
                    pfunc([0] + best_error['x'], ([max_err] + best_error['y']), color=color, linestyle=linestyle, marker=marker)
        else:
            errors_by_interval = { }            
            #subplot.set_yscale('log')
            if plot_func == 'semilogy':
                subplot.set_yscale('log')
            elif plot_func == 'loglog':
                subplot.set_yscale('log')
                subplot.set_xscale('log')

            for i in range(repeats):
                selected = get_result(results, arm, i)
                if max_err == None:
                    max_err = selected['error'][0]
                y_best_errors = np.array(get_best_errors(selected)) * y_scale
                t_times = get_total_times(selected, x_unit)
                t_max = int(max(t_times)) + 1
                
                if i == 0:
                    debug("t_times: {}".format(t_times[0]))
                    debug("y_best_errors: {}".format(y_best_errors[0]))

                t = 0
                cur_best_err = max_err * y_scale
                for j in range(len(t_times)):
                    
                    cur_time = t_times[j]
                    
                    while t < cur_time:
                        if not t in errors_by_interval.keys():
                            errors_by_interval[t] = []                          
                        errors_by_interval[t].append(cur_best_err)
                        t += 1

                    if y_best_errors[j] < cur_best_err:
                        cur_best_err = y_best_errors[j] 

                if i == 0:
                    debug("errors_by_interval: {}".format(errors_by_interval[0]))
                        
                        
            x = np.array([])
            y = np.array([]) 
            yerr = np.array([])
            
            for i in range(0, t_max):
                errors = errors_by_interval[i]
                
                y_ = np.mean(errors)
                y = np.append(y, y_)
                
                sd = np.std(errors)/std_div
               
                yerr = np.append(yerr, sd)
                x = np.append(x, i)
                
            
            if arm in unlabeled_arms:
                add_error_fill_line(x, y, yerr, color, ax=subplot, marker=marker, y_scale=y_scale,
                                    label=arm, linestyle=linestyle, alpha_fill=alpha_fill)
                unlabeled_arms.remove(arm)
            else:
                add_error_fill_line(x, y, yerr, color, marker=marker, y_scale=y_scale,
                    ax=subplot, linestyle=linestyle, alpha_fill=alpha_fill)
    if title != None:
        subplot.set_title('{}'.format(title))
    bbox_to_anchor = None
    loc = None
    borderaxespad = None
    
    if legend is not None:

        if 'bbox_to_anchor' in legend:
            bbox_to_anchor = legend['bbox_to_anchor']
        if 'loc' in legend:
            loc = legend['loc']
        if 'borderaxespad' in legend:
            borderaxespad = legend['borderaxespad']
    
    if l_order is not None:
        handles, labels = subplot.get_legend_handles_labels()        
        plt.legend([handles[idx] for idx in l_order], [labels[idx] for idx in l_order],
        prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
               loc=loc, borderaxespad=borderaxespad)        
    else:
        plt.legend(prop={'size': 15}, bbox_to_anchor=bbox_to_anchor,
               loc=loc, borderaxespad=borderaxespad)
    x_range = [0, 0]
    
    if x_ticks == None and xlim is not None:
        plt.xlim(xlim)
        x_range = list(xlim)
        x_ticks = [x for x in range(x_range[0], x_range[-1] + 1, x_steps)]
    
    if x_ticks != None:
        plt.xticks(x_ticks)

    if y_ticks != None:
        plt.yticks(y_ticks)

    if ylim is not None:
        plt.ylim(ylim)

    for s in guidelines:
        label = ""
        if "label" in s:
            label = s['label']
        elif "rank" in s:
            label = "Top-{}".format(s['rank'])
        elif 'difficulty' in s:
            label = "Top {:.2f}%".format(s['difficulty']*100)
        
        
        if 'perplexity' in s:
            plt.text(x_range[0] + .1, s['perplexity'], label)
            plt.axhline(y=s['perplexity'], color='gray', linestyle=':')
        else:
            plt.text(x_range[0] + .1, s['error'], label)
            plt.axhline(y=s['error'], color='gray', linestyle=':')

    plt.ylabel("Min Function Value ({})".format(sub_y_metric), fontsize=15)
    plt.xlabel(x_unit, fontsize=15)
    if save_name is not None:
        plt.tight_layout()
        plt.savefig(target_folder + save_name + '.png', format='png', dpi=300)
    else:
        plt.show()
        return plt


def add_error_fill_line(x, y, yerr, color=None, linestyle='-',
                        alpha_fill=0.3, ax=None, label=None, marker=None, y_scale=1.0):
    #ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr

    if np.max(y) < 1.0:
        ymin = np.maximum(y_scale * 0.001, ymin)
        ymax = np.minimum(y_scale * 1.0, ymax)
    
    #print("ymin: {}".format(ymin[0]))
    #print("ymax: {}".format(ymax[-1]))
    
    ax.semilogy(x, y, color=color, linestyle=linestyle, label=label, marker=marker)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

    
def add_quantile_fill_line(x, y, q1, q3, color=None, linestyle='-',
                        alpha_fill=0.3, ax=None, label=None, marker=None, y_scale=1.0):
    #ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()

    ax.plot(x, y, color=color, linestyle=linestyle, label=label, marker=marker)
    ax.fill_between(x, q1, q3, color=color, alpha=alpha_fill)

    
def add_no_fill_line(x, y, q1, q3, color=None, linestyle='-',
                        alpha_fill=0.3, ax=None, label=None, marker=None, y_scale=1.0):
    #ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()

    ax.plot(x, y, color=color, linestyle=linestyle, label=label, marker=marker)
