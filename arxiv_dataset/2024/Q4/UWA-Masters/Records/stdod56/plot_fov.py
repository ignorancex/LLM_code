import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

bro = mpatches.Wedge((0, 0), 1.0, 0, 180, ec="none", alpha = 0.4)
dude = mpatches.Arc((0.0, 0.0), 0.4, 0.4, angle= 0.0, theta1 = 0.0, theta2 = 180.0, linewidth = 3)
fig, ax = plt.subplots()
ax.scatter(0.0, 0.0, color = 'k', s = 100)
ax.add_artist(bro)
ax.add_artist(dude)
ax.set(
	aspect= 1,
	xlim = (-1.1, 1.1),
	ylim = (-1.1, 1.1)
)

ax.arrow(0., 0., 0., .7, head_width = 60*0.001, color = 'k')
ax.annotate(r"$\hat{v}_i$", (0.0, 0.), xytext = (0.1,0.7), size = 40)
ax.annotate(r"$\phi$", (0.0, 0.0), xytext = (0.15, 0.15), size = 40) 
ax.set_axis_off()
plt.tight_layout(pad = 0.0, w_pad = 0.0)
#plt.show()
fig.savefig("stdod56_fov.pdf", bbox_inches = 'tight')
