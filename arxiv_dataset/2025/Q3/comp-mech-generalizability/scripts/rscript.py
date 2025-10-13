import subprocess
import warnings
warnings.filterwarnings("ignore")

subprocess.run(
        [
            "Rscript",
            # "../src_figure/plot_logit_lens.R",
            # "../src_figure/plot_logit_attribution.R",
            "../src_figure/plot_head_pattern.R",
            # "../src_figure/PaperPlot.R",
            # f"../results/copyVSfact/logit_lens/gpt2_full",
        ]
    )