import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.forward import Heat
from src.probability import Prior

def main():
    np.random.seed(19)

    n = 800
    sig = 5e-2
    N = 500
    L = 1
    delta_t = 1e-5
    alpha = 1e-8
    gamma = -1.2
    acceleration = 3

    pr = Prior(N=N, L=L, transform='dst', gamma=gamma, delta=0.)
    u = pr.sample(n_sample=100)
    ind = np.where(np.all(u > 0, axis=1))[0]
    u = u[ind[0],:]


    us = [u]
    times = [0]
    for k in range(n):
        dt = delta_t*k**acceleration
        fwd = Heat(N=N, L=L, transform='dst', alpha=alpha, time=dt)
        u = fwd(u)
        us.append(u)
        times.append(times[-1] + dt)
    us = np.vstack(us)
    print('Done forward simulation')

    fig, ax = plt.subplots()
    line, = ax.plot(fwd.x, us[0, :], color='k')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)
    #ax.set_ylim(np.min(u), np.max(u))
    ax.set_xlabel('Position')
    ax.set_ylabel('Temperature')
    plt.savefig('latex/figs/forward_heat_equation.png')

    time_template = 'Time elapsed {}' 
    time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top')

    def update(frame):
        tt = times[frame]#tt = frame**acceleration * time    
        tt = int(tt)
        time_text.set_text(time_template.format(tt))
        line.set_ydata(us[frame, :])
        return line, time_text

    ani = animation.FuncAnimation(fig, update, frames=n, blit=True)
    ani.save('latex/figs/forward_heat_equation.mp4', writer='ffmpeg', fps=30)
    print('Done forward animation')
    
    ######################################
    ##### Reverse heat equation!!!  ######
    ######################################

    plt.close('all')
    fig, ax = plt.subplots()
    line, = ax.plot(fwd.x, us[-1, :], color='k')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Temperature')
    plt.savefig('latex/figs/reverse_heat_equation.png')

    u_reversed = us[::-1]
    time_reversed = np.array(times)[::-1]
    time_template = 'Time elapsed {}' 
    time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top')

    def update(frame):
        tt = time_reversed[frame]#tt = frame**acceleration * time   
        tt = int(tt)
        time_text.set_text(time_template.format(tt))
        line.set_ydata(u_reversed[frame, :])
        return line, time_text

    ani = animation.FuncAnimation(fig, update, frames=n, blit=True)
    ani.save('latex/figs/reverse_heat_equation.mp4', writer='ffmpeg', fps=30)
    print('Done backwards animation')
    plt.show()


main()
