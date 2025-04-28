from runutils import run_
import matplotlib.pyplot as plt
import numpy as np


def post_process(ladle, Campaign_number):
    ladle.dump_state()
#     ladle.save_db()
    ladle.plot()
    ladle.plot3()
    font = {'weight': 'bold',
            'size': 22}
    plt.rc('font', **font)
    fig, ax = plt.subplots()
    ax.plot(ladle.XU, ladle.yerode*180.0, label='Erosion', linewidth=4)
    ax.grid()
    ax.set_xlabel('Position [Brick ID]', fontsize=20)
    ax.set_ylabel('Erosion [mm]', fontsize=20)
    fig.legend(fontsize=20)
    fig.set_size_inches(18.5,10.5)
    plt.show()

run_(7654321, post_process=post_process)
