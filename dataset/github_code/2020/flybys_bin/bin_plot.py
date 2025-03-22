import bin_tools
import numpy as np

try:
    from matplotlib import pyplot
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def plot_function_single(cmd_args,data):
    args,data_av,data_nbody = data

    fig=pyplot.figure(figsize=(12,12))
    Np=9
    Nr=3
    Nc=3
    plots = []
    for i in range(Np):
        plot=fig.add_subplot(Nr,Nc,i+1)
        plots.append(plot)
    colors = ['k','r','b']
    
    a_print = data_av["a_print"]
    e_print = data_av["e_print"]
    i_print = data_av["i_print"]
    times = data_av["times"]/args.tend
    e1_sol = data_av["e_print"][0]
    e2_sol = data_av["e_print"][1]
    i1_sol = data_av["i_print"][0]
    i2_sol = data_av["i_print"][1]
    
    color='k'

    linestyles=['dashed','solid']
    linewidths = [2.0,1.0]
    colors=['k','g']
    labels=["$\mathrm{Inner\,averaged}$","$\mathrm{Four-body}$"]
    for i in range(2):
        if i==0:
            data = data_av
        elif i==1:
            data = data_nbody
        linestyle=linestyles[i]
        linewidth=linewidths[i]
        color = colors[i]
        label=labels[i]

        plots[0].plot(times,data["a_print"][0],color=color,linestyle=linestyle,linewidth=linewidth)
        plots[1].plot(times,data["a_print"][1],color=color,linestyle=linestyle,linewidth=linewidth)
        plots[2].plot(times,data["a_print"][2],color=color,linestyle=linestyle,linewidth=linewidth,label=labels[i])

        plots[3].plot(times,data["e_print"][0],color=color,linestyle=linestyle,linewidth=linewidth)
        plots[4].plot(times,data["e_print"][1],color=color,linestyle=linestyle,linewidth=linewidth)
        plots[5].plot(times,data["e_print"][2],color=color,linestyle=linestyle,linewidth=linewidth)

        plots[6].plot(times,(180.0/np.pi)*data["i_print"][0],color=color,linestyle=linestyle,linewidth=linewidth)
        plots[7].plot(times,(180.0/np.pi)*data["i_print"][1],color=color,linestyle=linestyle,linewidth=linewidth)
        plots[8].plot(times,(180.0/np.pi)*data["i_print"][2],color=color,linestyle=linestyle,linewidth=linewidth)

    linestyle='dotted'
    a1 = args.a1
    a2 = args.a2
    m1 = args.m1
    m2 = args.m2
    m3 = args.m3
    m4 = args.m4
    
    plots[2].axhline(y=args.a3,color='r',linestyle=linestyle,label="$\mathrm{Analytic}$")
    
    Q = args.Q
    e3 = args.e3
    e1x,e1y,e1z,j1x,j1y,j1z = bin_tools.orbital_elements_to_orbital_vectors(args.e1,args.i1,args.AP1,args.LAN1)
    e2x,e2y,e2z,j2x,j2y,j2z = bin_tools.orbital_elements_to_orbital_vectors(args.e2,args.i2,args.AP2,args.LAN2)

    delta_e3 = 0.0
    delta_e3 += bin_tools.delta_e3_function(a1,Q,e3,m1,m2,e1x,e1y,e1z,j1x,j1y,j1z)
    delta_e3 += bin_tools.delta_e3_function(a2,Q,e3,m3,m4,e2x,e2y,e2z,j2x,j2y,j2z)
    #print("delta_e3",delta_e3)
    
    plots[5].axhline(y=e3+delta_e3,color='r',linestyle=linestyle)
    
    delta_i3 = bin_tools.delta_i3_function(a1,a2,Q,e3,m1,m2,m3,m4,e1x,e1y,e1z,j1x,j1y,j1z,e2x,e2y,e2z,j2x,j2y,j2z)
    #print("delta_i3*180.0/np.pi",delta_i3*180.0/np.pi)
    plots[8].axhline(y=delta_i3*180.0/np.pi,color='r',linestyle=linestyle)

    ylabels=["$a_1/\mathrm{au}$","$a_2/\mathrm{au}$","$a_3/\mathrm{au}$","$e_1$","$e_2$","$e_3$","$i_1/\mathrm{deg}$","$i_2/\mathrm{deg}$","$i_3/\mathrm{deg}$"]
    labelsize=14
    fontsize=18
    for i,plot in enumerate(plots):
        plot.set_title(ylabels[i],fontsize=fontsize)
        plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
        if i in [6,7,8]:
            plot.set_xlabel("$t/t_\mathrm{end}$",fontsize=fontsize)


    loc = "upper right"
    handles,labels = plots[2].get_legend_handles_labels()
    plots[2].legend(handles,labels,loc=loc,fontsize=0.6*fontsize)

    if cmd_args.annotation != "":
        plots[0].annotate("$\mathrm{" + args.annotation + "}$",xy=(0.1,0.9),xycoords='axes fraction',fontsize=fontsize)


    a=0.3
    fig.subplots_adjust(hspace=a,wspace=a)
    fig.savefig(args.fig_filename)#,dpi=200)

    pyplot.show()

def plot_function_series(cmd_args,data_series):
    args,data_series_av,data_series_nbody = data_series
    
    values = data_series_av["values"]
    series_description = data_series_av["series_description"]
    plot_values = values
    
    if series_description=="i2":    
        plot_values = values*180.0/np.pi
        
    xscale="log"
    if series_description=="i2":
        xscale="linear"
        
    fig=pyplot.figure(figsize=(10,10))
    plot1=fig.add_subplot(2,1,1,yscale="log",xscale=xscale)
    plot2=fig.add_subplot(2,1,2,yscale="log",xscale=xscale)
    
    linestyles=['solid','dashed','dotted']

    colors = ['k','k','k']
    linewidths = [1.0,1.5,2.0]
    index_plot=0
    fontsize=cmd_args.fontsize
    labelsize=cmd_args.labelsize

    
    plot1.plot(plot_values,data_series_av["Delta_e1s"],color='k',linestyle='solid',linewidth=3,label="$\mathrm{Inner\,averaged}$")
    if cmd_args.do_nbody == True:
        plot1.scatter(plot_values,data_series_nbody["Delta_e1s"],color='g',label="$\mathrm{Four-body}$")
    elif cmd_args.compare_backreaction==True:
        plot1.plot(plot_values,data_series_nbody["Delta_e1s"],color='k',linestyle='dashed',label="$\mathrm{Inner\,averaged\,(no\,backreaction)}$")
    plot2.plot(plot_values,np.fabs(data_series_av["Delta_e2s"]),color='k',linestyle='solid',linewidth=3)
    if args.do_nbody == True:
        plot2.scatter(plot_values,np.fabs(data_series_nbody["Delta_e2s"]),color='g')
    elif args.compare_backreaction==True:
        plot2.plot(plot_values,data_series_nbody["Delta_e2s"],color='k',linestyle='dashed',label="$\mathrm{Inner\,averaged\,(no\,backreaction)}$")
        
        
    a1 = args.a1
    Q = args.Q
    e3 = args.e3
    m1 = args.m1
    m2 = args.m2
    M1 = m1+m2
    M2 = args.M2
    
    Delta_e1s = []
    Delta_e2s = []
    for index,value in enumerate(values):
        a1 = args.a1
        a2 = args.a2
        m3 = args.m3
        m4 = args.m4
        i2 = args.i2
        if series_description=="a2":
            a2 = value
        if series_description=="q2":
            q2 = value
            m3 = M2/(1.0+q2)
            m4 = m3*q2
        if series_description=="i2":
            i2 = value
        if series_description=="a1":
            a1 = value
            
        e1x,e1y,e1z,j1x,j1y,j1z = bin_tools.orbital_elements_to_orbital_vectors(args.e1,args.i1,args.AP1,args.LAN1)
        e2x,e2y,e2z,j2x,j2y,j2z = bin_tools.orbital_elements_to_orbital_vectors(args.e2,i2,args.AP2,args.LAN2)

        eps_SA1 = bin_tools.compute_eps_SA(M1,M2,a1/Q,e3)
        eps_SA2 = bin_tools.compute_eps_SA(M2,M1,a2/Q,e3)
        eps_oct1 = bin_tools.compute_eps_oct(m1,m2,M1,a1/Q,e3)
        eps_oct2 = bin_tools.compute_eps_oct(m3,m4,M2,a2/Q,e3)

        Delta_e1 = 0.0
        
        if args.quad==True:
            Delta_e1 += bin_tools.integrated_quad_function(eps_SA1,a1,Q,e3,m1,m2,e1x,e1y,e1z,j1x,j1y,j1z)
            if cmd_args.verbose==True:
                print("eps_SA1",eps_SA1,eps_SA1**2,eps_SA1**3,"eps_oct",((m1-m2)/M1)*(a1/Q))
                
        if args.oct==True:
            Delta_e1 += bin_tools.integrated_oct_function(eps_SA1,a1,Q,e3,m1,m2,e1x,e1y,e1z,j1x,j1y,j1z)
 
        if args.hex==True:
            Delta_e1 += bin_tools.integrated_hex_function(eps_SA1,a1,Q,e3,m1,m2,e1x,e1y,e1z,j1x,j1y,j1z)
        
        if cmd_args.verbose==True:
            print("quad",bin_tools.integrated_quad_function(eps_SA1,a1,Q,e3,m1,m2,e1x,e1y,e1z,j1x,j1y,j1z),"hex",bin_tools.integrated_hex_function(eps_SA1,a1,Q,e3,m1,m2,e1x,e1y,e1z,j1x,j1y,j1z),"hex cross",bin_tools.integrated_hex_function_cross(eps_SA1,a2,Q,e3,m3,m4,e1x,e1y,e1z,j1x,j1y,j1z,e2x,e2y,e2z,j2x,j2y,j2z))

        if index==0:
            plot1.axhline(y=Delta_e1,color='r',linestyle='-.',linewidth=3,label="$\mathrm{Analytic\,repl.}$",zorder=0)

        if args.hex_cross==True:
            Delta_e1 += bin_tools.integrated_hex_function_cross(eps_SA1,a2,Q,e3,m3,m4,e1x,e1y,e1z,j1x,j1y,j1z,e2x,e2y,e2z,j2x,j2y,j2z)

        Delta_e1s.append(Delta_e1)


        Delta_e2 = 0.0
        if args.quad==True:
            Delta_e2 += bin_tools.integrated_quad_function(eps_SA2,a2,Q,e3,m3,m4,e2x,e2y,e2z,j2x,j2y,j2z)
            if cmd_args.verbose==True:
                print("e2 eps",eps_SA2,eps_SA2**2,eps_SA2**3,"e3",e3)
        if args.oct==True:
            Delta_e2 += bin_tools.integrated_oct_function(eps_SA2,a2,Q,e3,m3,m4,e2x,e2y,e2z,j2x,j2y,j2z)
 
        if args.hex==True:
            Delta_e2 += bin_tools.integrated_hex_function(eps_SA2,a2,Q,e3,m3,m4,e2x,e2y,e2z,j2x,j2y,j2z)

        if args.hex_cross==True:
            Delta_e2 += bin_tools.integrated_hex_function_cross(eps_SA2,a1,Q,e3,m1,m2,e2x,e2y,e2z,j2x,j2y,j2z,e1x,e1y,e1z,j1x,j1y,j1z)

        Delta_e2s.append(Delta_e2)
        
    Delta_e1s = np.array(Delta_e1s)
    Delta_e2s = np.array(Delta_e2s)
    plot1.plot(plot_values,Delta_e1s,color='r',linestyle='dotted',linewidth=3,label="$\mathrm{Analytic}$")
    plot2.plot(plot_values,np.fabs(Delta_e2s),color='r',linestyle='dotted',linewidth=3)


    if series_description=="a2":
        alphas = [1,2,3,4]
        for alpha in alphas:
            a2_MMC = args.a1*pow(alpha**2 * args.M2/args.M1,1.0/3.0)
            plot1.axvline(x=a2_MMC,color='k',linestyle='dashed')
            plot2.axvline(x=a2_MMC,color='k',linestyle='dashed')
            plot1.annotate("$1:%s$"%alpha,xy=(1.02*a2_MMC,data_series_nbody["Delta_e1s"][-1]),fontsize=0.5*fontsize)

    if series_description=="a2":
        xlabel = "$a_2/\mathrm{au}$"
    elif series_description=="q2":
        xlabel = "$q_2$"
    elif series_description=="i2":
        xlabel = "$i_2/\mathrm{deg}$"
    elif series_description=="a1":
        xlabel = "$a_1$"

    plot1.set_ylabel("$\Delta e_1$",fontsize=fontsize)
    plot2.set_ylabel("$\Delta e_2$",fontsize=fontsize)
    
    plot2.set_xlabel(xlabel,fontsize=fontsize)
                
    plot1.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot2.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

    plot1.set_xticklabels([])

    plot1.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot2.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)


    loc = "upper left"
    if series_description=="i2":
        loc="lower left"

    handles,labels = plot1.get_legend_handles_labels()
    plot1.legend(handles,labels,loc=loc,fontsize=0.6*fontsize)

    fig.subplots_adjust(hspace=0.0,wspace=0.0)
    fig.savefig(cmd_args.series_fig_filename)#,dpi=200)

    pyplot.show()


def plot_function_overview(cmd_args):
        
   
    import os
    
    linestyles=['solid','dotted','-.','dashed']

        
    index_plot=0
    fontsize=cmd_args.fontsize
    labelsize=cmd_args.labelsize

    
    m1 = 10.0
    m2 = 8.0
    m3 = 6.0
    m4 = 4.0
    
    a1 = 1.0
    a2_values = np.array([0.1,1.0,5.0])
    Q_values = pow(10.0,np.linspace(1.0,3.0,100))

    e3 = 1.5
    e3_values = [1,10]
    m1_values = [10.0,8.0]
    m2_values = [10.0,2.0]
    m3_values = [5.0,6.0]
    m4_values = [5.0,4.0]
    
    linewidths = [1.0,2.0,3.0]
    colors = ['k','r','g','b']
    
    

    for index_m in range(len(m1_values)):
        m1 = m1_values[index_m]
        m2 = m2_values[index_m]
        m3 = m3_values[index_m]
        m4 = m4_values[index_m]
    
        for index_e3,e3 in enumerate(e3_values):
            fig=pyplot.figure(figsize=(9,7))
            xscale="log"
            plot=fig.add_subplot(1,1,1,yscale="log",xscale=xscale)
    
            for index_a2,a2 in enumerate(a2_values):

                Deltas = [[] for x in range(4)]
                for index_Q,Q in enumerate(Q_values):
                    M1 = m1+m2
                    M2 = m3+m4
                    eps_SA1 = bin_tools.compute_eps_SA(M1,M2,a1/Q,e3)
                    eps_oct = bin_tools.compute_eps_oct(m1,m2,M1,a1/Q,e3)
                    eps_hex = bin_tools.compute_eps_hex(m1,m2,M1,a1/Q,e3)
                    eps_hex_cross = bin_tools.compute_eps_hex_cross(m3,m4,M2,a2/Q,e3)
                   
                    Deltas[0].append(eps_SA1)
                    Deltas[1].append(eps_SA1*eps_oct)
                    Deltas[2].append(eps_SA1*eps_hex)
                    Deltas[3].append(eps_SA1*eps_hex_cross)

                if index_a2==0:
                    linewidth=2
                    plot.plot(Q_values/a1,Deltas[0],color=colors[0],linestyle=linestyles[0],linewidth=linewidth,label="$\mathrm{Quad.}$")
                    plot.plot(Q_values/a1,Deltas[1],color=colors[1],linestyle=linestyles[1],linewidth=linewidth,label="$\mathrm{Oct.}$")
                    plot.plot(Q_values/a1,Deltas[2],color=colors[2],linestyle=linestyles[2],linewidth=linewidth,label="$\mathrm{Hex.}$")
                    
                linewidth=linewidths[index_a2]
                plot.plot(Q_values/a1,Deltas[3],color=colors[3],linestyle=linestyles[3],linewidth=linewidth,label="$\mathrm{Hex.\,cross;}\,a_2=%s\,\mathrm{au}$"%a2)

            plot.set_xlabel("$Q/a_1$",fontsize=fontsize)
            plot.set_ylabel("$\sim\Delta e_1; \,\sim \Delta i_1$",fontsize=fontsize)

            loc = "lower left"
            handles,labels = plot.get_legend_handles_labels()
            plot.legend(handles,labels,loc=loc,fontsize=0.7*fontsize)

                
            plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            plot.annotate("$a_1 = %s\,\mathrm{au}$"%a1,xy=(0.75,0.85),xycoords='axes fraction',fontsize=fontsize)
            plot.set_title("$m_1 = %s; \, m_2 = %s; \, m_3 = %s;\, m_4 = %s; \, E = %s$"%(m1,m2,m3,m4,e3),fontsize=fontsize)
            
            fig.savefig(cmd_args.fig_dir + "overview_index_m_" + str(index_m) + "_index_e3_" + str(index_e3) + ".pdf")#,dpi=200)
    
    pyplot.show()
