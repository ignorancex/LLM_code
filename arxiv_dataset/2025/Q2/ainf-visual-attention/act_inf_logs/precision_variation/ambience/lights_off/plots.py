import matplotlib.pyplot as plt
import numpy as np

err_presence = np.genfromtxt("error_presence.csv",delimiter=',')
fe_presence = np.genfromtxt("free_energy_presence.csv",delimiter=',')
err_uniform = np.genfromtxt("error_uniform.csv",delimiter=',')
fe_uniform = np.genfromtxt("free_energy_uniform.csv",delimiter=',')
position_log = np.genfromtxt("position_log.csv", delimiter=",")
uniform_log = np.genfromtxt("uniform_log.csv", delimiter=",")

plt.plot(err_presence,label="err_presence")
plt.plot(err_uniform,label="err_uniform")
plt.legend()
plt.show()

plt.plot(fe_presence,label="fe_presence")
plt.plot(fe_uniform,label="fe_uniform")
plt.legend()
plt.show()

targets_u = uniform_log[:,2]
projections_u = uniform_log[:,6]
targets_p = position_log[:,2]
projections_p = position_log[:,6]

red_intentions_u = np.abs(targets_u - 16)
red_projections_u = np.abs(projections_u - 16)
red_intentions_p = np.abs(targets_p - 16 )
red_projections_p = np.abs(projections_p - 16)

# red_intentions_u = np.linalg.norm(red_intentions_u,axis=1)
# red_projections_u = np.linalg.norm(red_projections_u,axis=1)
# red_intentions_p = np.linalg.norm(red_intentions_p,axis=1)
# red_projections_p = np.linalg.norm(red_projections_p,axis=1)

max_tick = np.max((red_intentions_u, red_projections_u, red_intentions_p, red_projections_p))

plt.plot(red_projections_u,color="red",label="Red ball projection",linewidth=2)
plt.plot(red_intentions_u,color="pink",label="Red ball perception",linewidth=2)
plt.plot(red_projections_p,color="blue",label="Red ball projection",linewidth=2)
plt.plot(red_intentions_p,color="skyblue",label="Red ball perception",linewidth=2)

plt.subplots_adjust(left=0.08,
                    bottom=0.11, 
                    right=0.98, 
                    top=0.98, 
                    wspace=0.4, 
                    hspace=0.4)


plt.yticks(np.arange(0, max_tick+1, max_tick//10))
plt.xlabel("Steps")
plt.ylabel("Pixel distance")
plt.legend()

plt.show()

plt.plot(np.abs(red_projections_p-red_intentions_p), label="position")
plt.plot(np.abs(red_projections_u-red_intentions_u), label="uniform")
plt.legend()
plt.show()