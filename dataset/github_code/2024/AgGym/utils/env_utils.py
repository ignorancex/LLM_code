import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from matplotlib.lines import Line2D
import numpy as np
import matplotlib as mpl
import copy

def set_growth_stage_dict(self):
    gs_keys = self.growth_stages.keys()
    gs_vals = self.growth_stages.values()
    gs_sum = 0
    self.gs_reverse = {}
    for k, v in zip(gs_keys, gs_vals):
        gs_sum += v
        self.gs_reverse[gs_sum] = k
    self.gs_end = list(self.gs_reverse.keys())[-1]

def retrieve_growth_stage(self):
    for i in self.gs_reverse.keys():
        if i > self.timestep:
            self.gs_title = self.gs_reverse[i]
            break

def plot_grid(self, title):
    _, ax_dict = plt.subplot_mosaic("A", figsize=(20,20), constrained_layout=True, clear=True)
    ax_dict = self.render(ax_dict)
    # self.threat.draw(ax_dict["A"])

    if self.mode == "eval":
        (Path(self.result_path) / 'eval' / f'agent_{self.best_agent}').mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(self.result_path) / 'eval' / f'agent_{self.best_agent}' / f"{title}.png", dpi=300)
    elif self.mode == "train" and self.episode % 1000 == 0:
        (Path(self.result_path) / f"ep_{self.episode}").mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(self.result_path) / f"ep_{self.episode}" / f"{title}.png", dpi=300)
    plt.close()
    

def plot_field(self, title):
    pest_palette = sns.light_palette("seagreen")
    plot_state_cmap = sns.color_palette("rocket", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=0, vmax=3)
    action_dict = {0: "N/A", 1: "Low", 2: "Medium", 3: "High"}

    total_crops = self.state_space
    healthy_counts = np.full(len(self.infect_counts), total_crops)
    infect_counts = copy.deepcopy(self.infect_counts)
    dead_counts = np.array(self.dead_counts) + np.array(infect_counts)

    color_list = []
    color_dict = {0: "black", 1: pest_palette[1], 2: pest_palette[3], 3: pest_palette[5]}
    height_list = []
    for pest in self.action_list:
        color_list.append(color_dict[pest])
        if pest > 0:
            height_list.append(-1)
        else:
            height_list.append(0)

    ax_dict = plt.figure(num=1, figsize=(12,12), constrained_layout=True, clear=True).subplot_mosaic(
    """
    AB
    CB
    DD
    EE
    """)

    # Plot reward over time
    sns.lineplot(x=np.arange(len(self.reward_list)), y=self.reward_list, ax=ax_dict["A"])
    ax_dict["A"].spines[:].set_visible(False)
    ax_dict["A"].tick_params(axis="both", which="both", bottom=False, left=False)
    ax_dict["A"].set_ylabel("Reward")
    ax_dict["A"].set_xlabel("Planting Days")
    ax_dict["A"].set_title("Reward Earned Across One Trajectory")

    # # Plot heatmap
    # sns.heatmap(self.plot_state, vmin=0, vmax=3, ax=ax_dict["B"], linewidths=.5)
    # ax_dict["B"].tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    # ax_dict["B"].set_title(f"Current Plot State, Growth Stage: {self.gs_title}, Day {self.timestep}")
    # colorbar = ax_dict["B"].collections[0].colorbar
    # colorbar.set_ticks([0., 1., 2., 3.])
    # colorbar.set_ticklabels(["Unallocated", "Dead", "Infected", "Healthy"])
    # colorbar.ax.tick_params(size=0)
    
    ax_dict = self.render(ax_dict, "B")
    # sns.heatmap(self.plot_state, vmin=0, vmax=3, ax=ax_dict["B"], linewidths=.5)
    ax_dict["B"].tick_params(axis="both", which="both", bottom=False, left=False, labelleft=False, labelbottom=False)
    ax_dict["B"].set_title(f"Current Plot State, Growth Stage: {self.gs_title}, Day {self.timestep}")
    # colorbar = ax_dict["B"].collections[0].colorbar
    # colorbar.set_ticks([0., 1., 2., 3.])
    # colorbar.set_ticklabels(["Unallocated", "Dead", "Infected", "Healthy"])
    # colorbar.ax.tick_params(size=0)

    # Plot Text of current agent action
    ax_dict["C"].text(0.5, 0.5, f"Current Action {action_dict[self.action]}", size=24, horizontalalignment="center")
    ax_dict["C"].spines[:].set_visible(False)
    ax_dict["C"].tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # Plot Percentage of crop
    if len(healthy_counts) == 0:
        healthy_counts = [0]
        infect_counts = [0]
        dead_counts = [0]
    sns.barplot(x=np.arange(len(healthy_counts)), y=healthy_counts, color=plot_state_cmap(norm(3)), ax=ax_dict["D"])
    sns.barplot(x=np.arange(len(dead_counts)), y=dead_counts, color=plot_state_cmap(norm(1)), ax=ax_dict["D"])
    sns.barplot(x=np.arange(len(infect_counts)), y=infect_counts, color=plot_state_cmap(norm(2)), ax=ax_dict["D"])
    ax_dict["D"].spines[:].set_visible(False)
    ax_dict["D"].tick_params(axis="both", which="both", bottom=False, left=False)
    ax_dict["D"].set_xticks(range(0, self.timestep, 10))
    ax_dict["D"].set_ylabel("Number of Crops")
    ax_dict["D"].set_title("Total Number of Crops")
    custom_lines = [Line2D([0], [0], color=plot_state_cmap(norm(3)), lw=4),
                    Line2D([0], [0], color=plot_state_cmap(norm(2)), lw=4),
                    Line2D([0], [0], color=plot_state_cmap(norm(1)), lw=4)]
    ax_dict["D"].legend(custom_lines, ["Healthy", "Infected", "Dead"], loc="upper right", bbox_to_anchor=(1,1))

    # Plot actions over time
    if len(height_list) == 0:
        height_list = [0]
        color_list = pest_palette
    sns.barplot(x=np.arange(len(height_list)),y=height_list, hue=np.arange(len(height_list)), palette=color_list, ax=ax_dict["E"], legend=False)
    ax_dict["E"].spines[:].set_visible(False)
    ax_dict["E"].tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False)
    ax_dict["E"].set_ylabel("Application")
    ax_dict["E"].set_yticks([-1,0], labels=("None", "Applied"))
    ax_dict["E"].set_xticks(range(0, self.timestep, 10))
    custom_lines = [Line2D([0], [0], color=pest_palette[1], lw=4),
                    Line2D([0], [0], color=pest_palette[3], lw=4),
                    Line2D([0], [0], color=pest_palette[5], lw=4)]
    ax_dict["E"].legend(custom_lines, ["Low", "Medium", "High"], loc="upper right", bbox_to_anchor=(1,1))

    if self.mode == "eval":
        (Path(self.result_path) / 'eval' / f'agent_{self.best_agent}').mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(self.result_path) / 'eval' / f'agent_{self.best_agent}' / f"{title}.png", dpi=300)
    elif self.mode == "train" and self.episode % 1000 == 0:
        (Path(self.result_path) / f"ep_{self.episode}").mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(self.result_path) / f"ep_{self.episode}" / f"{title}.png", dpi=300)
    plt.close()

def make_gif(self):
    if self.mode == "eval":
        (Path(self.result_path) / 'eval' / f'agent_{self.best_agent}').mkdir(parents=True, exist_ok=True)
        generator = [i for i in (Path(self.result_path) / 'eval' / f'agent_{self.best_agent}').glob(f"*.png")]
        generator.sort()
        frames = [Image.open(image) for image in generator]
        frame_one = frames[0]
        frame_one.save(Path(self.result_path) / 'eval' / f'agent_{self.best_agent}' / f"summary.gif", format="GIF", append_images=frames,
                   save_all=True, duration=200, loop=0)
    elif self.mode == "train" and self.episode % 1000 == 0 and self.plot_progression == 'True':
        generator = [i for i in (Path(self.result_path) / f"ep_{self.episode}").glob(f"*.pdf")]
        generator.sort()
        frames = [Image.open(image) for image in generator]
        frame_one = frames[0]
        frame_one.save(Path(self.result_path) / f"ep_{self.episode}" / f"summary.gif", format="GIF", append_images=frames,
                   save_all=True, duration=200, loop=0)
