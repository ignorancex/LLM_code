import pandas as pd
from utils import process_df

mark_scheme = {'qwen':"mark=triangle, mark options={rotate=270}",
                    'baichuan2':"mark=triangle",
                    'chatglm2':"mark=triangle, mark options={rotate=90}",
                    'breeze':"mark=halfsquare*",
                    'taiwan-llm':"mark=halfsquare right*",
                    'gpt4o':"mark=square",
                    'gpt4':"mark=pentagon",
                    'gpt3.5':"mark=square, mark options={rotate=45}",
                    'llama3-70b':"mark=o",
                    'llama3-8b':"mark=halfcircle*"}

def get_mc_prop(df):
    return len(df[(df['region'] == 1) ]) / len(df) * 100

if __name__ == '__main__':

    langs = ['simplified', 'traditional', 'english']
    models = ['qwen', 'baichuan2', 'chatglm2', 'breeze', 'taiwan-llm', 'gpt4o', 'gpt4', 'gpt3.5', 'llama3-70b', 'llama3-8b']

    for lang in langs:
        for model in models:
            s = f"\\addplot[only marks, color=black, {mark_scheme[model]}] coordinates {{\n"
            t = ''
            for permut in range(20, 400, 20):
                df = process_df(file_type='response', sub_file_type='permutation_final', task='name', lang=lang, prompt_id=permut, llm=model, action='load')
                prop_mc = get_mc_prop(df=df)
                s += f"({permut}, {prop_mc:.2f}) "
                t += f"({permut}, {prop_mc:.2f}) "
            s += "};\n"
            s += "\\addplot[color=black, mark=none] coordinates {\n"
            s += t
            s += "};"
            print(s)
        print()
        print()
        print()
    

    """
    Note that the output is not directly a figure. Instead, the output is the source code for generating the figure in latex. 
    To generate the figure, please replace the placeholders in the following latex code.

    Latex source code:

    \begin{figure*}[ht]
    \centering
    \begin{subfigure}[t]{\textwidth}
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        width=\linewidth,
        height=5cm,
        xlabel={Number of Permutations},
        ylabel={\% Mainland Chinese Name},
        grid=major,
        legend entries={\scriptsize{\qwen},,\scriptsize{\baichuan},,\scriptsize{\chatglm},,\scriptsize{\breeze},,\scriptsize{\taiwanllm},,\scriptsize{\gptivo},,\scriptsize{\gptiv},,\scriptsize{\gptiii},,\scriptsize{\llamas},,\scriptsize{\llamae}},
        legend style={at={(0.5,1.15)},anchor=north, legend columns=-1}
    ]
    {{placeholder}}
    \draw[dashed, thick, color=black] (axis cs:180, \pgfkeysvalueof{/pgfplots/ymin}) -- (axis cs:180, \pgfkeysvalueof{/pgfplots/ymax});
    \end{axis}
    \end{tikzpicture}
    \caption{Prompted in Simplified Chinese.}
        \label{fig:permute_simplified_chinese}
        \end{subfigure}
        \hfill
    \begin{subfigure}[t]{\textwidth}
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        width=\linewidth,
        height=5cm,
        xlabel={Number of Permutations},
        ylabel={\% Mainland Chinese Name},
        grid=major,
        legend entries={\scriptsize{\qwen},,\scriptsize{\baichuan},,\scriptsize{\chatglm},,\scriptsize{\breeze},,\scriptsize{\taiwanllm},,\scriptsize{\gptivo},,\scriptsize{\gptiv},,\scriptsize{\gptiii},,\scriptsize{\llamas},,\scriptsize{\llamae}},
        legend style={at={(0.5,1.15)},anchor=north, legend columns=-1}
    ]
    {{placeholder}}
    \draw[dashed, thick, color=black] (axis cs:180, \pgfkeysvalueof{/pgfplots/ymin}) -- (axis cs:180, \pgfkeysvalueof{/pgfplots/ymax});
    \end{axis}
    \end{tikzpicture}
    \caption{Prompted in Traditional Chinese.}
        \label{fig:permute_traditional_chinese}
        \end{subfigure}
    \hfill
    \begin{subfigure}[t]{\textwidth}
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        width=\linewidth,
        height=5cm,
        xlabel={Number of Permutations},
        ylabel={\% Mainland Chinese Name},
        grid=major,
        legend entries={\scriptsize{\qwen},,\scriptsize{\baichuan},,\scriptsize{\chatglm},,\scriptsize{\breeze},,\scriptsize{\taiwanllm},,\scriptsize{\gptivo},,\scriptsize{\gptiv},,\scriptsize{\gptiii},,\scriptsize{\llamas},,\scriptsize{\llamae}},
        legend style={at={(0.5,1.15)},anchor=north, legend columns=-1}
    ]
    {{placeholder}}
    \draw[dashed, thick, color=black] (axis cs:180, \pgfkeysvalueof{/pgfplots/ymin}) -- (axis cs:180, \pgfkeysvalueof{/pgfplots/ymax});
    \end{axis}
    \end{tikzpicture}
    \caption{Prompted in English.}
        \label{fig:permute_english}
        \end{subfigure}
    \caption{The number of permutations used in the regional name task experiments (180, at the dotted vertical line) yield results for our primary metric of interest (\% Mainland Chinese Names that are selected) that are comparable to the asymptotic rates from running the experiment for more permutations.}
    \label{fig:permute}
    \end{figure*}

    """
    
