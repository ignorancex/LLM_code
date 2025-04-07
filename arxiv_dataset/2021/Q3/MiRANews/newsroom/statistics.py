#coding=utf8

INPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_clean.jsonl'
OUTPUT_FILE = '/scratch/xxu/multi-multi/multi_multi_stat.jsonl'

def novel_ngrams(document, gold_summ, ngram):
    doc_grams = set()
    for i in range(len(document)-ngram+1):
        doc_grams.add(' '.join(document[i:i+ngram]))
    count = 0
    novel = 0
    for i in range(len(gold_summ)-ngram+1):
        gram = ' '.join(gold_summ[i:i+ngram])
        if gram not in doc_grams:
            novel += 1
        count += 1
    return novel, count

if __name__ == '__main__':
    import json

    source_dict = {}
    cluster_size = {}
    cluster_size_sum = 0
    cluster_num = 0

    summary_length = 0
    summary_sent_num = 0
    summary_num = 0
    document_length = 0
    document_sent_num = 0
    document_num = 0
    extract_num = 0

    document_vocabs = set()
    summary_vocabs = set()

    novel_unigrams_num = 0
    unigrams_num = 0
    novel_bigrams_num = 0
    bigrams_num = 0
    novel_trigrams_num = 0
    trigrams_num = 0
    novel_fgrams_num = 0
    fgrams_num = 0

    foreign_novel_unigrams_num = 0
    foreign_unigrams_num = 0
    foreign_novel_bigrams_num = 0
    foreign_bigrams_num = 0
    foreign_novel_trigrams_num = 0
    foreign_trigrams_num = 0
    foreign_novel_fgrams_num = 0
    foreign_fgrams_num = 0

    summ_novel_unigrams_num = 0
    summ_unigrams_num = 0
    summ_novel_bigrams_num = 0
    summ_bigrams_num = 0
    summ_novel_trigrams_num = 0
    summ_trigrams_num = 0
    summ_novel_fgrams_num = 0
    summ_fgrams_num = 0

    fpout = open(OUTPUT_FILE, 'w')

    for line in open(INPUT_FILE):
        flist = line.strip().split('\t')
        cluster_id = flist[0]
        summ_url = flist[1]
        pairs = flist[2]
        pair_obj = json.loads(pairs)
        # Cluster num
        cluster_num += 1

        # Cluster size
        if len(pair_obj) not in cluster_size:
            cluster_size[len(pair_obj)] = 1
        else:
            cluster_size[len(pair_obj)] += 1
        cluster_size_sum += len(pair_obj)

        documents = []; summaries = []
        for pid in pair_obj:
            pair = pair_obj[pid]
            document = pair['[DOCUMENT]']
            summary = pair['[SUMMARY]']
            title = pair['[TITLE]']
            url = pair['[URL]']
            if url.find('newser.com') > -1:
                summary = title + summary
            source = pair['[SORUCE]'].replace(':80', '')
            timestamp = pair['[DATE]']
            year = timestamp[:4]
            month = timestamp[4:6]
            date = timestamp[6:]

            # News source
            if source not in source_dict:
                source_dict[source] = 1
            else:
                source_dict[source] += 1

            # Document length
            document_sent_num += len(document)
            document = ' '.join(document).lower()
            document_toks = document.split()
            document_length += len(document_toks)
            document_num += 1
            document_vocabs |= set(document_toks)

            # Summary length
            summary_sent_num += len(summary)
            summary = ' '.join(summary).lower()
            summary_toks = summary.split()
            summary_length += len(summary_toks)
            summary_num += 1
            summary_vocabs |= set(summary_toks)

            documents.append(document_toks)
            summaries.append(summary_toks)

            # Novelty
            novel, count = novel_ngrams(document_toks, summary_toks, 1)
            novel_unigrams_num += novel
            unigrams_num += count
            novel, count = novel_ngrams(document_toks, summary_toks, 2)
            novel_bigrams_num += novel
            bigrams_num += count
            novel, count = novel_ngrams(document_toks, summary_toks, 3)
            novel_trigrams_num += novel
            trigrams_num += count
            novel, count = novel_ngrams(document_toks, summary_toks, 4)
            novel_fgrams_num += novel
            fgrams_num += count

            # Extract summary
            if document.lower().find(' '.join(summary_toks[:min(len(summary_toks), 10)]).lower()) > -1:
                extract_num += 1

        # Foreign novelty
        for i, summary_toks in enumerate(summaries):
            document_toks = [item for sublist in documents[:i]+documents[i+1:] for item in sublist]
            #document_toks = [item for sublist in documents for item in sublist]
            novel, count = novel_ngrams(document_toks, summary_toks, 1)
            foreign_novel_unigrams_num += novel
            foreign_unigrams_num += count
            novel, count = novel_ngrams(document_toks, summary_toks, 2)
            foreign_novel_bigrams_num += novel
            foreign_bigrams_num += count
            novel, count = novel_ngrams(document_toks, summary_toks, 3)
            foreign_novel_trigrams_num += novel
            foreign_trigrams_num += count
            novel, count = novel_ngrams(document_toks, summary_toks, 4)
            foreign_novel_fgrams_num += novel
            foreign_fgrams_num += count

        # Foreign novelty
        tmp_novel_unigrams_num = 0
        tmp_unigrams_num = 0
        tmp_novel_bigrams_num = 0
        tmp_bigrams_num = 0
        tmp_novel_trigrams_num = 0
        tmp_trigrams_num = 0
        tmp_novel_fgrams_num = 0
        tmp_fgrams_num = 0

        for i, summary_toks in enumerate(summaries):
            document_toks = [item for sublist in summaries[:i]+summaries[i+1:] for item in sublist]
            document_toks = [item for sublist in summaries[:i]+summaries[i+1:] for item in sublist]
            novel, count = novel_ngrams(document_toks, summary_toks, 1)
            summ_novel_unigrams_num += novel
            summ_unigrams_num += count
            tmp_novel_unigrams_num += novel
            tmp_unigrams_num += count

            novel, count = novel_ngrams(document_toks, summary_toks, 2)
            summ_novel_bigrams_num += novel
            summ_bigrams_num += count
            tmp_novel_bigrams_num += novel
            tmp_bigrams_num += count

            novel, count = novel_ngrams(document_toks, summary_toks, 3)
            summ_novel_trigrams_num += novel
            summ_trigrams_num += count
            tmp_novel_trigrams_num += novel
            tmp_trigrams_num += count

            novel, count = novel_ngrams(document_toks, summary_toks, 4)
            summ_novel_fgrams_num += novel
            summ_fgrams_num += count
            tmp_novel_fgrams_num += novel
            tmp_fgrams_num += count

        tmp_scores = []
        tmp_scores.append(str(tmp_novel_unigrams_num/tmp_unigrams_num))
        tmp_scores.append(str(tmp_novel_bigrams_num/tmp_bigrams_num))
        tmp_scores.append(str(tmp_novel_trigrams_num/tmp_trigrams_num))
        tmp_scores.append(str(tmp_novel_fgrams_num/tmp_fgrams_num))
        fpout.write(line.strip() + '\t' + ' '.join(tmp_scores)+'\n')

    print ('Number for cluster:', cluster_num)
    print ('Dataset Size (number of doc-summary pairs)', cluster_size_sum)
    print ('Avg cluster size', cluster_size_sum/cluster_num)
    print ('Cluster size:', cluster_size)

    print ('Extract percentage:', extract_num / summary_num)
    print ('Avg Summary length (word):', summary_length / summary_num)
    print ('Avg Document length (word):', document_length / document_num)
    print ('Avg Summary length (sentence):', summary_sent_num / summary_num)
    print ('Avg Document length (sentence):', document_sent_num / document_num)
    print ('Document vocab size:', len(document_vocabs))
    print ('Summary vocab size:', len(summary_vocabs))

    print ('Novel unigrams (1:1):', novel_unigrams_num/unigrams_num)
    print ('Novel bigrams (1:1):', novel_bigrams_num/bigrams_num)
    print ('Novel trigrams (1:1):', novel_trigrams_num/trigrams_num)
    print ('Novel 4-grams (1:1):', novel_fgrams_num/fgrams_num)

    print ('Novel unigrams (n-1:1):', foreign_novel_unigrams_num/foreign_unigrams_num)
    print ('Novel bigrams (n-1:1):', foreign_novel_bigrams_num/foreign_bigrams_num)
    print ('Novel trigrams (n-1:1):', foreign_novel_trigrams_num/foreign_trigrams_num)
    print ('Novel 4-grams (n-1:1):', foreign_novel_fgrams_num/foreign_fgrams_num)

    print ('Novel unigrams (summary):', summ_novel_unigrams_num/summ_unigrams_num)
    print ('Novel bigrams (summary):', summ_novel_bigrams_num/summ_bigrams_num)
    print ('Novel trigrams (summary):', summ_novel_trigrams_num/summ_trigrams_num)
    print ('Novel 4-grams (summary):', summ_novel_fgrams_num/summ_fgrams_num)

    print ('Number of source:', len(source_dict))
    print ('\n')

    for item in sorted(source_dict.items(), key = lambda d:d[1], reverse=True):
        if item[1] < 1000:
            break
        print (item[0] + '\t' + str(item[1]))
    fpout.close()
