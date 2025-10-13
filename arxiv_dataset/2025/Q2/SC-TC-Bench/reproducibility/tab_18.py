import pandas as pd

if __name__ == '__main__':
    sources = ['baidu-baike', 'map-cc', 'mcc4', 'tw-wiki', 'cctw', 'ootc', 'twc4', 'twchat', 'c4']

    target_terms = ['鐵路警察', '路警', '月臺', '捷運', '月臺門', '轉運站', '大衆運輸', '車掌','交流道',
                    '家庭車','白牌車','腳踏車', '機車','行車距離', '追撞', '罰鍰', '觀光局', '共乘',
                    '協力車', '夭然瓦斯管道', '瓦斯表', '保存期限', '自動販賣機' ,'褲裙', 
                    '太白粉','柳橙','鮪魚', '衛生套/保險套','三温暖']
    
    df_gt = pd.read_csv('source_data/regional_term_and_definition.csv')
    
    target_terms_in_simp = []
    for term in target_terms:
        tem = df_gt[df_gt['Taiwan']==term].reset_index()['Mainland China'][0]
        target_terms_in_simp.append(tem)

    nonmisaligned_terms = df_gt[~df_gt['Taiwan'].isin(target_terms)]['Taiwan'].tolist()

    nonmisaligned_terms_in_simp = []
    for term in nonmisaligned_terms:
        tem = df_gt[df_gt['Taiwan']==term].reset_index()['Mainland China'][0]
        nonmisaligned_terms_in_simp.append(tem)

    for source in sources:
        s = ''
        s += f'{{\\tt {source}}}' + ' & '

        df_corpus_s = pd.read_csv(f"source_data/corpus_count/regional_term/mc_{source}.csv")
        df_corpus_t = pd.read_csv(f"source_data/corpus_count/regional_term/tw_{source}.csv")

        ns = df_corpus_s[df_corpus_s['word'].isin(target_terms_in_simp)]['count'].mean()
        nt = df_corpus_t[df_corpus_t['word'].isin(target_terms)]['count'].mean()

        n_mean = ((df_corpus_s[df_corpus_s['word'].isin(target_terms_in_simp)]['count']+0.001) / (df_corpus_t[df_corpus_t['word'].isin(target_terms)]['count'] + 0.001)).mean()

        ms = df_corpus_s[df_corpus_s['word'].isin(nonmisaligned_terms_in_simp)]['count'].mean()
        mt = df_corpus_t[df_corpus_t['word'].isin(nonmisaligned_terms)]['count'].mean()

        m_mean = ((df_corpus_s[df_corpus_s['word'].isin(nonmisaligned_terms_in_simp)]['count']+0.001) / (df_corpus_t[df_corpus_t['word'].isin(nonmisaligned_terms)]['count'] + 0.001)).mean()

        s += f'{ns:.2f} & {nt:.2f} & {ns/nt:.2f} & {ms:.2f} & {mt:.2f} & {ms/mt:.2f} \\\\'
        print(s)


        """
        Note that the output is not directly a table. Instead, the output is the source code for generating the table in latex. 
        To generate the table, please replace the placeholders in the following latex code with the printed s.

        Latex source code:

        \begin{table*}[t]
        \centering
        \adjustbox{max width=\textwidth}{
        \begin{tabular}{lcccccc}
        \toprule[1.1pt]
        Corpus & \begin{tabular}[c]{@{}l@{}} Misaligned Items Written \\ in Simplified Chinese\end{tabular} & \begin{tabular}[c]{@{}l@{}} Misaligned Items Written \\ in Traditional Chinese\end{tabular} &  \begin{tabular}[c]{@{}l@{}} Misaligned Ratio \\ (Simplified/Traditional)\end{tabular} & \begin{tabular}[c]{@{}l@{}} Non-misaligned Items Written \\ in Simplified Chinese\end{tabular}  & \begin{tabular}[c]{@{}l@{}} Non-misaligned Items Written \\ in Traditional Chinese\end{tabular}& \begin{tabular}[c]{@{}l@{}} Non-misaligned Ratio \\ (Simplified/Traditional)\end{tabular} \\ \midrule
        {{s_placeholder}}
        \bottomrule[1.1pt]
        \end{tabular}
        }
        \end{table*}
         """