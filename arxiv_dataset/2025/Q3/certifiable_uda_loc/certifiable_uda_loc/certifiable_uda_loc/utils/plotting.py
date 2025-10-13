def get_plot_colormap(n: int, name: str = "Dark2"):
    from matplotlib import cm

    interval = [lv1 for lv1 in range(n)]
    cmap = cm.get_cmap(name)
    colors = [cmap(x) for x in interval]
    return colors
