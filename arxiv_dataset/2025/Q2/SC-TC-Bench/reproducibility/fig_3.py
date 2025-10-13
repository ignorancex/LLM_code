import argparse
from utils import compute_percentage, process_df
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def choose_marker_color_label(model, lang):
    if model == "baichuan2":
        marker = "^"
    elif model == "chatglm2":
        marker = "<"
    elif model == "llama3-70b":
        marker = "o"
    elif model == "llama3-8b":
        marker = "o"
    elif model == "qwen":
        marker = ">"
    elif model == 'taiwan-llm':
        marker = "D"
    elif model == 'breeze':
        marker = "D"
    elif model == "gpt4o":
        marker = "s"
    elif model == "gpt4":
        marker = "p"
    elif model == "gpt3.5":
        marker = "D"
    elif model == "ds":
        marker = "*"
    
    if lang == "english":
        color = "#0072B2"
    elif lang == "simplified":
        color = "#E69F00"
    elif lang == "traditional":
        color = "#009E73"

    return color, marker, f"{model}, {lang}"

def batch_load(models, langs, prompt_id):
    d_res = dict()
    for model in models:
        for lang in langs:
            df = process_df(file_type='response', sub_file_type='final', task='name', lang=lang, prompt_id=prompt_id, llm=model, action='load')
            mc, t, na = compute_percentage(df=df)
            try:
                color, marker, label = choose_marker_color_label(model=model, lang=lang)
            except:
                print(model, lang)
            d_res[f"{model} {lang}"] = [mc, t, na, color, marker, label]
    
    return d_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_id', type=int, default=0)


    parser.add_argument('--trial_num', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--marker_size', type=int, default=130)
    parser.add_argument('--arrow', default=False, action='store_true')
    args = parser.parse_args()

    models = ["baichuan2", "chatglm2", "qwen", "breeze", "taiwan-llm", "llama3-8b", "llama3-70b", "gpt4o", "gpt4", "gpt3.5", "ds"]
    langs = [ "simplified" , "traditional", "english"]
    epsilon = 1e-5
    
    group2_legend = [
        Line2D([0], [0], color='black', marker='*', lw=0, mec='black', mew=1, label='DeepSeek-R1'),
        Line2D([0], [0], color='white', marker='s', lw=0, mec='black', mew=1, label='gpt4o'),
        Line2D([0], [0], color='white', marker='p', lw=0, mec='black', mew=1, label='gpt4'),
        Line2D([0], [0], color='white', marker='D', lw=0, mec='black', mew=1, label='gpt3.5'),
        Line2D([0], [0], color='white', marker='o', lw=0, mec='black', mew=1, label='llama3-70b'),
        Line2D([0], [0], color='black', marker='o', lw=0, mec='black', mew=1, label='llama3-8b', fillstyle='top'),
        ]
        
    group3_legend = [
    Line2D([0], [0], color='white', marker='^', lw=0, mec='black', mew=1, label='baichuan2'),
    Line2D([0], [0], color='white', marker='<', lw=0, mec='black', mew=1, label='chatglm2'),
    Line2D([0], [0], color='white', marker='>', lw=0, mec='black', mew=1, label='qwen'),
    ]
    
    group4_legend = [
    Line2D([0], [0], color='black', marker='D', lw=0, mec='black', mew=1, label='breeze', fillstyle='bottom'),
    Line2D([0], [0], color='black', marker='D', lw=0, mec='black', mew=1, label='taiwan-llm', fillstyle='right'),
    ]

    d_res = batch_load(models, langs, prompt_id=args.prompt_id)
    
    if args.arrow:
        assert args.prompt_id in [2, 3]
        d_res_arrow = batch_load(models=models, langs=langs, prompt_id=1)
        
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    for idx, lang in enumerate(langs):
        for model in models:
            mc, t, na, color, marker, label = d_res[f"{model} {lang}"]

            if mc + t == 0:
                div = epsilon
            else:
                div = mc + t
            axes[idx].axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='y=0.5')
            axes[idx].text(-0.11, 0.5, "0.5", va='center', ha='right',  color="black")
        
            
            if 'llama3-8b' in label:
                axes[idx].plot(1-na, mc / div, color='black', marker=marker, markersize=10, fillstyle='top', markeredgecolor="black", linewidth=1, label=label, alpha=args.alpha)
            elif 'taiwan-llm' in label:
                axes[idx].plot(1-na, mc / div, color='black', marker=marker, markersize=10, fillstyle='right', markeredgecolor="black", linewidth=1, label=label, alpha=args.alpha)
            elif 'breeze' in label:
                axes[idx].plot(1-na, mc / div, color='black', marker=marker, markersize=10, fillstyle='bottom', markeredgecolor="black", linewidth=1, label=label, alpha=args.alpha)
            elif 'ds' in label:
                axes[idx].plot(1-na, mc / div, color='black', marker=marker, markersize=10, markeredgecolor="black", linewidth=1, label=label, alpha=args.alpha)
            else:
                axes[idx].scatter(1-na, mc / div, color='white', marker=marker, s=args.marker_size, edgecolor="black", linewidth=1, label=label, alpha=args.alpha)

            if args.arrow:
                mc_arrow, t_arrow, na_arrow, _, _, _ = d_res_arrow[f"{model} {lang}"]
                
                if mc_arrow + t_arrow == 0:
                    div_arrow = epsilon
                else:
                    div_arrow = mc_arrow + t_arrow
                
                if mc_arrow / div_arrow < mc / div:
                    axes[idx].annotate("", xy=(1-na, mc / div), xytext=(1-na_arrow, mc_arrow / div_arrow), arrowprops=dict(arrowstyle='->', color='red', lw=2),size =20)
                else:
                    axes[idx].annotate("", xy=(1-na, mc / div), xytext=(1-na_arrow, mc_arrow / div_arrow), arrowprops=dict(arrowstyle='->', color='blue', lw=2, linestyle='--'),size =20)
            

        axes[idx].set_ylabel("Mainland Chinese Name Rate")
        axes[idx].set_xlabel("Rate of Valid Responses")
        axes[idx].set_ylim(-0.1, 1.1)
        axes[idx].set_xlim(-0.1, 1.1)
        
        if lang == "english":
            title = "English"
        elif lang == "simplified":
            title = "Simplified Chinese"
        elif lang == 'traditional':
            title = "Traditional Chinese"
        axes[idx].set_title(f'Prompted in {title}', weight="bold")

        axes[idx].grid(which='both', linestyle='--', linewidth=0.5, color='gray')

        annotation_text_size = 9
        axes[idx].text(-0.484, 1, "Biased towards\n" + r"Mainland Chinese Names $\rightarrow$", va='center', ha='left', rotation='vertical', color="black", fontsize=annotation_text_size)
        axes[idx].text(-0.37, 0, "Biased towards\n" + r"$\leftarrow$ Taiwanese Names", va='center', ha='right', rotation='vertical', color="black", fontsize=annotation_text_size)
        axes[idx].text(1.1, -0.4, "High Instruction\n" + r"Adherence $\rightarrow$", va='center', ha='right',  color="black", fontsize=annotation_text_size)
        axes[idx].text(0.26, -0.4, "Low Instruction\n" + r"$\leftarrow$ Adherence", va='center', ha='right',  color="black", fontsize=annotation_text_size)
            
    
    legend2 = axes[0].legend(handles=group2_legend, loc='upper left', title='English Oriented LLMs', bbox_to_anchor=(0.05, -0.35), ncol=3)
    axes[0].add_artist(legend2)

    legend3 = axes[1].legend(handles=group3_legend, loc='upper left', title='Simplified Chinese Oriented LLMs', bbox_to_anchor=(0.2, -0.35), ncol=2)
    axes[1].add_artist(legend3)

    axes[2].legend(handles=group4_legend, loc='upper left', title='Traditional Chinese Oriented LLMs', bbox_to_anchor=(-0.1, -0.35), ncol=2)
    

    plt.tight_layout()
    plt.savefig(f"figure/regional_name/scatter_{args.prompt_id}_{args.arrow}.pdf")
