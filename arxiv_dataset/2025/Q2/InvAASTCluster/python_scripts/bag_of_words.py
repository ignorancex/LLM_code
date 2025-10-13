#!/usr/bin/python
#Title			: bag_of_words.py
#Usage			: python bag_of_words.py -h
#Author			: pmorvalho
#Date			: May 03, 2021
#Description    	: Computes bag of words using sutdent submissions. There are three options for Bag of Words using: Programs' ASTs, Programs' code itselt, Programs' invariants (syntactic)
#Notes			:
#Python Version	: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================

import argparse
import functools
import glob
import numpy as np
import pandas as pd
import operator
import os
import pickle
from pycparser import c_ast, parse_file
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sys import argv

class BoW(object):
    """BoW (BagOfWords) class
       Three types of BoW:
       - Syntax of programs' code       (syntactic = True)
       - Syntax of programs' AST        (ast = True)
       - Syntax of programs' invariants (invariants = True)
       - Syntax of programs' AST and invariants  (ast = True AND invariants = True)"""

    def __init__(self, syntactic=False, ast=False, invariants=False, invariants_scope_name=False, invariants_scope_depth=False, dir=None, ngrams=3):
        super(BoW, self).__init__()
        self.syntactic = syntactic
        self.invariants = invariants
        self.ast = ast
        self.pickle_file_name = ""
        if self.syntactic:
            self.pickle_file_name += "syntax-"
        if self.ast:
            self.pickle_file_name += "ast-"
        self.ngrams = ngrams
        self.invariants_scope_name = invariants_scope_name
        self.invariants_scope_depth = invariants_scope_depth
        if self.invariants:
            self.pickle_file_name += "invs-"
            self.inv_file = 'invariants'
            if self.invariants_scope_name:
                self.pickle_file_name += "sn-"
                self.inv_file += "-scope_name"
            if self.invariants_scope_depth:
                self.pickle_file_name += "dp-"
                self.inv_file += "-scope_depth"
            self.inv_file += '.pickle.gz'

            if dir is None:
                self.dir = "daikon-invariants/"

        if dir is None:
            self.dir="injected_programs/" if self.syntactic else "ASTs/"
        else:
            self.dir = dir

        self.pickle_file_name += "ng{n}".format(n=str(self.ngrams))
        # names, vectors and vectorizer itself
        self.invs = []
        self.files = None
        self.programs = []
        self.asts = []
        self.vectors = []
        self.vectorizer = None
        self.vocab_size = None

        self.asts_dict = dict()
        self.invariants_dict = dict()
        self.programs_dict = dict()

        self.programs_submissions_names = []
        self.asts_submissions_names = []
        self.invariants_submissions_names = []

        self.pickle_file_name += '.pickle'

        if self.invariants and self.ast:
            self.get_invariants_corpus()

    def get_submissions_names(self):
        if self.syntactic:
            return self.programs_submissions_names
        elif self.ast:
            return self.asts_submissions_names
        elif self.invariants:
            return self.invariants_submissions_names

    def get_program_syntax(self, p):
        f = open(p, "r+").read()
        # remove comments //
        f = re.sub("//.*\\n", "", f)
        # remove comments /* */
        f = re.sub("/\*(.|\n)*?\*/", "", f)
        # f = re.sub("\(|\)", " ", f)
        f = re.sub(",|;", " ", f)
        f = re.sub("\n|\t", " ", f)
        return f

    def get_AST(self, file):
        ast = open(file, "r+").read()
        ast = re.sub("ID: .*?\n", "ID:\n", ast)
        ast = re.sub("Decl: .*?,", "Decl:,", ast)
        ast = re.sub("TypeDecl: .*?,", "TypeDecl:,", ast)
        return ast

    def program_dealer(self, f, sub_name=None):
        if self.syntactic:
            return self.get_program_syntax(f)
        elif self.ast and self.invariants:
            try:
                return self.get_AST(f) + self.invariants_dict[sub_name]
            except Exception as e:
                print("** WARNING ** No invariants: ", sub_name)
                return self.get_AST(f)
        elif self.ast:
            return self.get_AST(f)

    def get_submissions_name(self, l, syntax=False):
        t = "-correct"
        if "incorrect" in l:
            t = "-incorrect"
        if syntax:
            d = l.split("/")[-1].split(".")[0]
            e = l.split("/")[-3]
            lab = l.split("/")[-4]
            ds = l.split("/")[-5]
            return "{ds}-{lab}-{e}-{d}{ty}".format(ds=ds, lab=lab, e=e, d=d, ty='' if ds == "C-Pack-IPAs" else t)
        elif l[-3:] != '.gz':
            d = l.split("/")[-1].split(".")[0]
            e = l.split("/")[-2]
            lab = l.split("/")[-3]
            ds = l.split("/")[-4]
            return "{ds}-{lab}-{e}-{d}{ty}".format(ds=ds, lab=lab, e=e, d=d, ty='' if ds == "C-Pack-IPAs" else t)
        else:
            e = l.split("/")[-2]
            lab = l.split("/")[-3]
            ds = l.split("/")[-4]
            return "{ds}-{lab}-{e}-".format(ds=ds, lab=lab, e=e)
            
    def get_program_corpus(self):
        self.files = glob.glob(self.dir+'/**/*.c', recursive=True) if self.syntactic else glob.glob(self.dir+'/**/*.ast', recursive=True)
        dir = re.sub("incorrect_submissions", "**", self.dir)
        dir = re.sub("correct_programs_clara_supports", "**", dir)
        files = glob.glob(dir+'/**/*.c', recursive=True) if self.syntactic else glob.glob(dir+'/**/*.ast', recursive=True)
        for l in files:
            sub_name = self.get_submissions_name(l, syntax=self.syntactic)
            self.programs_submissions_names.append(sub_name)
            self.programs.append(self.program_dealer(l))
            if l in self.files:
                self.programs_dict[sub_name] = self.programs[-1]

    def get_ast_corpus(self):
        self.files = glob.glob(self.dir+'/**/*.ast', recursive=True)
        dir = re.sub("incorrect_submissions", "**", self.dir)
        dir = re.sub("correct_programs_clara_supports", "**", dir)
        files = glob.glob(dir+'/**/*.ast', recursive=True)
        for l in files:
            sub_name = self.get_submissions_name(l)
            self.asts_submissions_names.append(sub_name)
            self.asts.append(self.program_dealer(l, sub_name=sub_name))
            if l in self.files:
                self.asts_dict[sub_name] = self.asts[-1]

    def get_invariants_corpus(self):
        dir = re.sub("ASTs/", "daikon-invariants/", self.dir)
        self.files = glob.glob(dir+'/**/'+self.inv_file, recursive=True)
        dir = re.sub("incorrect_submissions", "**", dir)
        dir = re.sub("correct_programs_clara_supports", "**", dir)
        files = glob.glob(dir+'/**/'+self.inv_file, recursive=True)
        for l in files:
            exercises = pd.read_pickle(l)
            subs = exercises[0]
            invariants = exercises[1]
            for s in range(len(subs)):
                sub_name = self.get_submissions_name(l) + subs[s]
                if "itsp" in sub_name:
                    sub_name = sub_name + "-incorrect" if "incorrect" in l else sub_name + "-correct"
                self.invariants_submissions_names.append(sub_name)
                self.invs.append(" ".join(map(str, functools.reduce(operator.iconcat, invariants[s], []))))
                # if "one of" in self.invs[-1]:
                    # print()
                    # self.invs[-1] =re.sub("_.*? one of {.*?}", "", self.invs[-1])
                if l in self.files or self.ast: # this because when we want both AST and Invariants we need to save all invariants from incorrect_submissions
                # and correct submissions, otherwise the dictionary has no invariants of correct or incorrect and the BOW will have a smaller vocabulary
                    self.invariants_dict[sub_name] = self.invs[-1]

    def get_syntax_BOW(self): #syntactic-based BOW
        vectorizer = TfidfVectorizer(token_pattern=r'\b[\w\[\]]+\b[\[\]\)\(]*|[=><!]{1,2}', stop_words={' ', ';', '{', '}'}, ngram_range=(1,self.ngrams))
                                                    # '\b(\w+|[^\w\s]+)\b'
        self.get_program_corpus()
        X = vectorizer.fit_transform(self.programs)
        return vectorizer,X

    def get_AST_BOW(self): #syntactic-based BOW
        vectorizer = TfidfVectorizer(token_pattern=r'\b[\w\[\]]+\b[\[\]\)\(]*|[=><!]{1,2}', stop_words={' ', ';', '{', '}'}, ngram_range=(1,self.ngrams))
                                                    # '\b(\w+|[^\w\s]+)\b'
        self.get_ast_corpus()
        X = vectorizer.fit_transform(self.asts)
        return vectorizer,X

    def get_invariant_BOW(self): #invariant-based BOW
        vectorizer = TfidfVectorizer(token_pattern=r'\b[\w\[\]]+\b[\[\]\)\(]*|[=><!]{1,2}', stop_words={' '}, ngram_range=(1,self.ngrams)) # , '{', '}'
                                                    #\b beginning of the word, \w Matches any word character (alphanumeric & underscore [A-Za-z0-9_])
                                                    # [\[\]\)\(]* for [] and () after a variable
                                                    # [=><!]{1,2} for comparison operators
        self.get_invariants_corpus()
        vectors = vectorizer.fit_transform(self.invs)
        return vectorizer, vectors

    def get_BOW(self):
        if self.ast:
            self.vectorizer, self.vectors = self.get_AST_BOW()
        elif self.syntactic:
            self.vectorizer, self.vectors = self.get_syntax_BOW()
        else:
            self.vectorizer, self.vectors = self.get_invariant_BOW()
        # vectorizer = TfidfVectorizer(stop_words={' ', ';', '{', '}'}, ngram_range=(1,3))
                                    # regex for tokens      #which words are ignored    # ngram (min_len,max_len)
        print("Vocabulary Size: ", len(self.vectorizer.vocabulary_))
        print("Stop words: ", self.vectorizer.get_stop_words())
        print("Dealt with ", self.vectors.shape[0], " programs")
        return self.vectorizer, self.vectors

    def get_vocabulary_size(self):
        if self.vocab_size is None:
            self.vocab_size = len([i for i in self.vectorizer.vocabulary_.keys() if " " not in i])
        return self.vocab_size

    def save_bow(self, output_dir, vectorizer):
        pickle_dict = dict()
        if self.asts_submissions_names:
            pickle_dict['submissions'] = [e[0] for e in self.asts_dict.items()]
            pickle_dict['vectors'] = [vectorizer.transform([e[1]]) for e in self.asts_dict.items()]
        elif self.invariants_submissions_names:
            pickle_dict['submissions'] = [e[0] for e in self.invariants_dict.items()]
            pickle_dict['vectors'] = [vectorizer.transform([e[1]]) for e in self.invariants_dict.items()]
        elif self.programs_submissions_names:
            pickle_dict['submissions'] = [e[0] for e in self.programs_dict.items()]
            pickle_dict['vectors'] = [vectorizer.transform([e[1]]) for e in self.programs_dict.items()]
        with open(output_dir+"/"+self.pickle_file_name, "wb") as file:
            pickle.dump(pickle_dict, file)

        os.system("gzip -f "+output_dir+"/"+self.pickle_file_name)

def parser():
    parser = argparse.ArgumentParser(prog='bag_of_words.py', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dir', nargs='?', default="daikon-invariants/",
                        help="directory with submissions for lab submissions")
    parser.add_argument('-s', '--syntactic', action="store_true", default=False,
                        help="creates a bag of words based on program's syntax")
    parser.add_argument('-a', '--ast', action="store_true", default=False,
                        help="creates a bag of words based on program's AST syntax")
    parser.add_argument('-i', '--invariants', action="store_true", default=False,
                        help="creates a bag of words based on program's invariants")
    parser.add_argument('-sn','--invariants_scope_name', action="store_true", default=False, help="creates a bag of words based on program's invariants with scopes' names")
    parser.add_argument('-dp', '--invariants_scope_depth', action="store_true", default=False, help="creates a bag of words based on program's invariants with scopes' depth")
    parser.add_argument('-ng', '--ngrams', default=3, type=int,
                        help="Computes ngrams of max length --ng. Ngrams with length from 1 to --ng")
    parser.add_argument('-o', '--output_dir', nargs='?', help='the name of the output file.')
    args = parser.parse_args(argv[1:])
    return args

if __name__ == '__main__':
    args = parser()
    bow = BoW(args.syntactic, args.ast, args.invariants, args.invariants_scope_name, args.invariants_scope_depth, args.dir, args.ngrams)
    vectorizer, X = bow.get_BOW()
    if args.output_dir:
        os.system("mkdir -p "+args.output_dir)
        bow.save_bow(args.output_dir, vectorizer)
    # names=vectorizer.get_feature_names()
    # program_vector=vectorizer.transform([programs[0]]).toarray()
