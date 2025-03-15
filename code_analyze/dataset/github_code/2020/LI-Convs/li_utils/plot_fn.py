import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def visualize_features(
		fea_list,
		fig_title,
		save_path=None,
		fea_title_list=None,
		normalize_data=False,
		val_min=None,
		val_max=None,
		figsize=(20,5,),
		cmap='Greys'):

	v_min = val_min or np.min(fea_list)
	v_max = val_max or np.max(fea_list)

	if normalize_data:

		norm = matplotlib.colors.Normalize(
			vmin=np.min(v_min),
			vmax=np.max(v_max),
			clip=True)
		mapper = matplotlib.cm.ScalarMappable(
			norm=norm, cmap=cmap)

		fea_list = [mapper.to_rgba(fea) for fea in fea_list]

		v_min,v_max = (0.0,1.0,)

	plt.gca().set_axis_off()
	fig, axes = plt.subplots(figsize=figsize,nrows=1, ncols=len(fea_list))
	plt.axis('off')

	fig.suptitle(fig_title)
	for idx, ax in enumerate(axes.flat):

		im = ax.imshow(
			fea_list[idx], vmin=v_min, vmax=v_max, cmap=cmap)
		ax.axis('off')

		if fea_title_list is not None:
			ax.title.set_text(fea_title_list[idx])

	# set colorbar
	fig.subplots_adjust(right=0.82)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.75])
	fig.colorbar(im, cbar_ax)

	if save_path is None:
		plt.show()
	else:
		fig.savefig(save_path, dpi=fig.dpi)

	plt.close(fig)



def visualize_heatmap(
		probs,
		save_path=None):

	cdict3 = {'red': ((0.0, 0.0, 0.0),
	                  (0.25, 0.0, 0.0),
	                  (0.5, 0.8, 1.0),
	                  (0.75, 1.0, 1.0),
	                  (1.0, 0.4, 1.0)),
	          'green': ((0.0, 0.0, 0.0),
	                    (0.25, 0.0, 0.0),
	                    (0.5, 0.9, 0.9),
	                    (0.75, 0.0, 0.0),
	                    (1.0, 0.0, 0.0)),

	          'blue': ((0.0, 0.0, 0.4),
	                   (0.25, 1.0, 1.0),
	                   (0.5, 1.0, 0.8),
	                   (0.75, 0.0, 0.0),
	                   (1.0, 0.0, 0.0))
	          }

	blue_red = LinearSegmentedColormap('BlueRed1', cdict3)

	max_pos = np.max(probs)
	probs = probs / abs(max_pos)

	# a colormap and a normalization instance
	cmap = blue_red
	norm = plt.Normalize(vmin=-1.0, vmax=1.0)
	probs = cmap(norm(probs))

	plt.imsave(save_path, probs, cmap=cmap)
