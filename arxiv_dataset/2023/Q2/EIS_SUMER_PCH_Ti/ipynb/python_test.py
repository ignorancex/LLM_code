import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,6))
ax.plot([0,1,2],[0,1,2])
ax.get_xticklines()[0].set_color("red")
plt.show()