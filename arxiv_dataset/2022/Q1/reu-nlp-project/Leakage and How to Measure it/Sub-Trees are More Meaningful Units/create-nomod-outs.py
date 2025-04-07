import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document

def strip_doc(doc):
   doc = doc.to_dict()
   new_doc = []
   for sent in doc:
      new_sent = []
      for word in sent:
         new_word = {"id":word["id"], "text":word["text"]}
         new_sent.append(new_word)
      new_doc.append(new_sent)
   return Document(new_doc)

german_doc = strip_doc(CoNLL.conll2doc("../datasets/Universal-Dependencies-2.8.1/ud-treebanks-v2.8/UD_German-HDT/de_hdt-ud-test.conllu"))
english_doc = strip_doc(CoNLL.conll2doc("../datasets/Universal-Dependencies-2.8.1/ud-treebanks-v2.8/UD_English-EWT/en_ewt-ud-test.conllu"))



en_nosubjmod = stanza.Pipeline(lang="en", processors="depparse,lemma,tokenize,pos", depparse_model_path="saved_models/depparse/no-nsubj-amod/en_ewt_parser.pt", depparse_pretrain_path="/home/nkrasner/stanza_resources/en/pretrain/combined.pt", tokenize_pretokenized=True)

en_noobjmod = stanza.Pipeline(lang="en", processors="depparse,lemma,tokenize,pos", depparse_model_path="saved_models/depparse/no-obj-amod/en_ewt_parser.pt", depparse_pretrain_path="/home/nkrasner/stanza_resources/en/pretrain/combined.pt", tokenize_pretokenized=True)

de_nosubjmod = stanza.Pipeline(lang="de", processors="depparse,lemma,tokenize,pos", depparse_model_path="saved_models/depparse/no-nsubj-amod/de_hdt_parser.pt", depparse_pretrain_path="/home/nkrasner/stanza_resources/de/pretrain/gsd.pt", tokenize_pretokenized=True)

de_noobjmod = stanza.Pipeline(lang="de", processors="depparse,lemma,tokenize,pos", depparse_model_path="saved_models/depparse/no-obj-amod/de_hdt_parser.pt", depparse_pretrain_path="/home/nkrasner/stanza_resources/de/pretrain/gsd.pt", tokenize_pretokenized=True)

en_nosubjmod_outs = en_nosubjmod(english_doc)
en_noobjmod_outs = en_noobjmod(english_doc)
de_nosubjmod_outs = de_nosubjmod(german_doc)
de_noobjmod_outs = de_noobjmod(german_doc)

CoNLL.write_doc2conll(en_nosubjmod_outs, "single-mod-outs/en_nosubjmod_outs.conllu")
CoNLL.write_doc2conll(en_noobjmod_outs, "single-mod-outs/en_noobjmod_outs.conllu")
CoNLL.write_doc2conll(de_nosubjmod_outs, "single-mod-outs/de_nosubjmod_outs.conllu")
CoNLL.write_doc2conll(de_noobjmod_outs, "single-mod-outs/de_noobjmod_outs.conllu")
