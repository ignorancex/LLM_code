import numpy as np
import matplotlib.pyplot as plt

def main():
    # Set up figure
    fig = plt.figure(figsize=plt.figaspect(1/3))

    # Make data
    r = 1
    c = 0
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v)) + c
    y = r * np.outer(np.sin(u), np.sin(v)) + c
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + c

    # Corner points: parametric in later parts
    neg_point = -1/np.sqrt(3)
    pos_point = 1/np.sqrt(3)

    #####
    # First subplot: One-hot encoding
    #####
    ax = fig.add_subplot(1,3,1,projection='3d')
    # Plot sphere
    ax.plot_surface(x, y, z, alpha=0.2)
    # Plot equator
    ax.plot(r*np.cos(u)+c, r*np.sin(u)+c, zs=c*np.ones_like(u), linestyle='-', color="tab:blue", alpha=0.6)
    # Plot one-hot vectors
    ax.plot([0,0], [0,1], zs= [1,0], c='tab:orange', marker='o') #(0,0,1) to (0,1,0)
    ax.plot([0,1], [1,0], zs= [0,0], c='tab:orange', marker='o') #(0,1,0) to (1,0,0)
    ax.plot([1,0], [0,0], zs= [0,1], c='tab:orange', marker='o') #(1,0,0) to (0,0,1)
    # Corresponding binary vectors
    ax.text(0.07, 0.07, 1, s=r"$ \mathbf{e}_3}$", horizontalalignment="left", verticalalignment="top")
    ax.text(0.07, 1.07, 0, s=r"$\mathbf{e}_2}$", horizontalalignment="left", verticalalignment="center")
    ax.text(1, 0, -.075, s=r"$\mathbf{e}_1}$", horizontalalignment="center", verticalalignment="top")
    # Plot cube
    ax.plot([neg_point,neg_point],[neg_point,neg_point],zs=[neg_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,neg_point],[neg_point,pos_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,pos_point],[neg_point,neg_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[pos_point,pos_point],zs=[pos_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[pos_point,neg_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,neg_point],[pos_point,pos_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[neg_point,neg_point],zs=[neg_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[neg_point,pos_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,neg_point],[pos_point,pos_point],zs=[pos_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,pos_point],[pos_point,pos_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,pos_point],[neg_point,neg_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,neg_point],[neg_point,pos_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    # Clean up plot
    ax.set_xlabel("x")
    ax.set_xticks([neg_point,pos_point])
    ax.set_xticklabels([r"$-\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{3}}$"])
    ax.set_yticks([neg_point,pos_point])
    ax.set_yticklabels([r"$-\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{3}}$"])
    ax.set_zticks([neg_point,pos_point])
    ax.set_zticklabels([r"$-\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{3}}$"])
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_title("(a)", y=-.1)
    ax.set_xlim([-1.01,1.01])
    ax.set_ylim([-1.01,1.01])
    ax.set_zlim([-1.01,1.01])
    ax.view_init(15,-40,0)
    ax.set_aspect('equal')


    #####
    # Second subplot: 2 points
    #####
    ax = fig.add_subplot(1,3,2, projection='3d')
    # Plot the sphere
    ax.plot_surface(x, y, z, alpha=0.2)
    # Plot equator
    ax.plot(r*np.cos(u)+c, r*np.sin(u)+c, zs=c*np.ones_like(u), linestyle='-', color="tab:blue", alpha=0.6)
    # Plot lines between 2 points: (0,0,0), and (1,1,1)
    ax.plot([neg_point, pos_point], [neg_point, pos_point], zs= [neg_point, pos_point], c='tab:orange', marker='o') #(0,0,0) to (1,1,1)
    # Corresponding binary vectors
    ax.text(neg_point-.07, neg_point-.07, neg_point, s="(0,0,0)", horizontalalignment="right", verticalalignment="center")
    ax.text(pos_point+.07, pos_point+.07, pos_point, s="(1,1,1)", horizontalalignment="left", verticalalignment="center")
    # Plot cube
    ax.plot([neg_point,neg_point],[neg_point,neg_point],zs=[neg_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,neg_point],[neg_point,pos_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,pos_point],[neg_point,neg_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[pos_point,pos_point],zs=[pos_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[pos_point,neg_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,neg_point],[pos_point,pos_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[neg_point,neg_point],zs=[neg_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[neg_point,pos_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,neg_point],[pos_point,pos_point],zs=[pos_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,pos_point],[pos_point,pos_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,pos_point],[neg_point,neg_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,neg_point],[neg_point,pos_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    # Clean up plot
    ax.set_xlabel("x")
    ax.set_xticks([neg_point,pos_point])
    ax.set_xticklabels([r"$-\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{3}}$"])
    ax.set_yticks([neg_point,pos_point])
    ax.set_yticklabels([r"$-\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{3}}$"])
    ax.set_zticks([neg_point,pos_point])
    ax.set_zticklabels([r"$-\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{3}}$"])
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_title("(b)", y=-.1)
    ax.set_xlim([-1.01,1.01])
    ax.set_ylim([-1.01,1.01])
    ax.set_zlim([-1.01,1.01])
    ax.view_init(15,-40,0)
    ax.set_aspect('equal')



    #####
    # Third subplot: 4 points
    #####
    ax = fig.add_subplot(1,3,3, projection='3d')
    # Plot the sphere
    ax.plot_surface(x, y, z, alpha=0.2)
    # Plot equator
    ax.plot(r*np.cos(u)+c, r*np.sin(u)+c, zs=c*np.ones_like(u), linestyle='-', color="tab:blue", alpha=0.6)
    # Plot lines between 4 points: (0,0,0), (0,1,1), (1,0,1), (1,1,0)
    ax.plot([neg_point, neg_point], [neg_point, pos_point], zs= [neg_point, pos_point], c='tab:orange', marker='o') #(0,0,0) to (0,1,1)
    ax.plot([neg_point, pos_point], [neg_point, neg_point], zs= [neg_point, pos_point], c='tab:orange', marker='o') #(0,0,0) to (1,0,1)
    ax.plot([neg_point, pos_point], [neg_point, pos_point], zs= [neg_point, neg_point], c='tab:orange', marker='o') #(0,0,0) to (1,1,0)
    ax.plot([neg_point, pos_point], [pos_point, neg_point], zs= [pos_point, pos_point], c='tab:orange', marker='o') #(0,1,1) to (1,0,1)
    ax.plot([neg_point, pos_point], [pos_point, pos_point], zs= [pos_point, neg_point], c='tab:orange', marker='o') #(0,1,1) to (1,1,0)
    ax.plot([pos_point, pos_point], [neg_point, pos_point], zs= [pos_point, neg_point], c='tab:orange', marker='o') #(1,0,1) to (1,1,0)
    # Corresponding binary vectors
    ax.text(neg_point-.07, neg_point-.07, neg_point, s="(0,0,0)", horizontalalignment="right", verticalalignment="center")
    ax.text(neg_point+.07, pos_point+.07, pos_point, s="(0,1,1)", horizontalalignment="left", verticalalignment="bottom")
    ax.text(pos_point-0.15, neg_point-0.15, pos_point, s="(1,0,1)", horizontalalignment="right", verticalalignment="center")
    ax.text(pos_point+.07, pos_point+.07, neg_point, s="(1,1,0)", horizontalalignment="left", verticalalignment="center")
    # Plot cube
    ax.plot([neg_point,neg_point],[neg_point,neg_point],zs=[neg_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,neg_point],[neg_point,pos_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,pos_point],[neg_point,neg_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[pos_point,pos_point],zs=[pos_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[pos_point,neg_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,neg_point],[pos_point,pos_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[neg_point,neg_point],zs=[neg_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([pos_point,pos_point],[neg_point,pos_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,neg_point],[pos_point,pos_point],zs=[pos_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,pos_point],[pos_point,pos_point],zs=[neg_point,neg_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,pos_point],[neg_point,neg_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    ax.plot([neg_point,neg_point],[neg_point,pos_point],zs=[pos_point,pos_point], linestyle=':', c='k', alpha=0.7)
    # Clean up plot
    ax.set_xlabel("x")
    ax.set_xticks([neg_point,pos_point])
    ax.set_xticklabels([r"$-\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{3}}$"])
    ax.set_yticks([neg_point,pos_point])
    ax.set_yticklabels([r"$-\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{3}}$"])
    ax.set_zticks([neg_point,pos_point])
    ax.set_zticklabels([r"$-\frac{1}{\sqrt{3}}$", r"$\frac{1}{\sqrt{3}}$"])
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_title("(c)", y=-.1)
    ax.set_xlim([-1.01,1.01])
    ax.set_ylim([-1.01,1.01])
    ax.set_zlim([-1.01,1.01])
    ax.view_init(15,-40,0)
    ax.set_aspect('equal')
    
    #####
    # Plot
    #####
    plt.rcParams.update(
        {"xtick.direction" : "in",
        "xtick.major.size" : 3,
        "xtick.major.width" : 0.5,
        "xtick.minor.size" : 1.5,
        "xtick.minor.width" : 0.5,
        "xtick.minor.visible" : True,
        "xtick.top" : True,
        "xtick.labelsize" : 10,
        "ytick.direction" : "in",
        "ytick.major.size" : 3,
        "ytick.major.width" : 0.5,
        "ytick.minor.size" : 1.5,
        "ytick.minor.width" : 0.5,
        "ytick.minor.visible" : True,
        "ytick.right" : True,
        "ytick.labelsize" : 10,
        "font.family": "serif",
        "font.serif" : "Times",
        "font.size" : 18,
        "legend.fontsize" : 12,
        "text.usetex" : True,
        "text.latex.preamble" : ['\\usepackage{amsmath}'],
        "figure.figsize" : [3.5, 2.625]
        }
    )

    
    plt.show()

if __name__ == "__main__":#
    main()