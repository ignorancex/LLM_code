#!/usr/bin/python
#Title			: get_asts.py
#Usage			: python get_asts.py -h
#Author			: pmorvalho
#Date			: May 06, 2021
#Description    	:
#Notes			:
#Python Version	: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================

import argparse
from contextlib import redirect_stdout
import glob
import io
import os
from pycparser import c_ast, parse_file
import re
import sys

def saves_AST(file, output_file):
    ast = parse_file(file, use_cpp=True,
            cpp_path='gcc',
            cpp_args=['-E', '-Iutils/fake_libc_include'])
    v = c_ast.NodeVisitor()
    v.visit(ast)
    outputdir = re.sub("/[^/]*.ast", "", output_file)
    if not os.path.exists(outputdir):
        os.system("mkdir -p "+outputdir)

    with open(output_file, "w+") as o:
        new_stdout = io.StringIO()
        with redirect_stdout(new_stdout):
            ast.show(buf=new_stdout)
            out = new_stdout.getvalue()
            # to remove code inserted by pycparser
            out = out.split("Typedef: xcb_visualid_t, [], [\'typedef\']\n    TypeDecl: xcb_visualid_t, []\n      IdentifierType: [\'uint32_t\']\n")[1]
            o.write(out)

def get_ASTs_corpus():
    for l in glob.glob(args.dir+'/**/*.c', recursive=True):
        saves_AST(l, re.sub(args.dir, args.output_dir, l).replace(".c", ".ast"))

def parser():
    parser = argparse.ArgumentParser(prog='compute_bag_of_words.py', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dir',
                        help="directory with submissions for lab programs")
    parser.add_argument('-o', '--output_dir', help='the name of the output file.')
    args = parser.parse_args(sys.argv[1:])
    if args.output_dir is None:
        args.output_dir = "ASTs/"+args.dir
    return args

if __name__ == '__main__':
    args = parser()
    get_ASTs_corpus()
