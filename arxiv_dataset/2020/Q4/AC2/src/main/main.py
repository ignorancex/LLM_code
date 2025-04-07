# from ..file_map_engine.engine import parse
import os
import ast
from os import path
from pprint import pprint
import json
import ast
from flask import Flask, render_template, redirect, request, url_for
import sys
from file_map_engine.engine import get_calls
from file_map_engine.call_dir import dir_walk
from file_map_engine.ast_engine import make_ast
from file_map_engine.ast_helper import wrapper_get
from file_map_engine.metrics import metric_a2a
from file_map_engine.metrics import metric_c2c
import pickle

cur_dir = os.getcwd()
# from test import make_ast
debug = 1

app = Flask(__name__)

@app.route('/')
def new_page():
    return render_template('home.html')

@app.route('/download_data', methods=['POST'])
def download():
    github = request.form['github']
    print(github)

    cmd = "git clone "+github
    cmd2 = 'git tag --sort=committerdate > ../tags'


    test_dir = cur_dir+'/target_repo_dir'
    os.chdir(test_dir)
    os.system(cmd)

    cur = os.getcwd()
    try:
        repo_dir = cur + '/' + github.split('/')[-1].split('.')[0]
        os.chdir(repo_dir)
    except:
        repo_dir = cur + '\\' + github.split('/')[-1].split('.')[0]
        os.chdir(repo_dir)

    os.system(cmd2)
    os.chdir(test_dir)

    tags = open('tags').readlines()

    os.chdir(repo_dir)

    ulti_dict = {}
    ulti_tree = {}

    num = 0

    for i in tags:
        num += 1
        print("\n\n\n New Tag: ", i, "\n\n")


        cmd3 = 'git checkout ' + i
        os.system(cmd3)


        os.chdir(test_dir)
        try:
            cmdd = 'lcom '+ './' + github.split('/')[-1].split('.')[0] +' > ../static/cohesion_'+github.split('/')[-1].split('.')[0]+'_'+i
            print(test_dir)
            print(cmdd)
            os.system(cmdd)
        except:
            cmdd = 'lcom '+ '.\\' + github.split('/')[-1].split('.')[0] +' > ..\\static\\cohesion_'+github.split('/')[-1].split('.')[0]+'_'+i + '.txt'
            print(test_dir)
            print(cmdd)
            os.system(cmdd)

        new_dir = github.split('/')[-1].split('.')[0]
        tree = dir_walk(new_dir)


        returned_tree = get_dictionary(tree)
        ulti_tree[str(num)] = tree
        ulti_dict[str(num)] = returned_tree

        if debug == 1:
            print(tree)
            print("------------------------------")
            print(returned_tree)
            print("------------------------------")
        os.chdir(repo_dir)



    os.chdir(cur_dir)
    try:
        pickle.dump( ulti_dict, open( "./directory_map.dat", "wb" ))
        pickle.dump( ulti_tree, open( "./ast_map.dat", "wb" ))
    except:
        pickle.dump( ulti_dict, open( "directory_map.dat", "wb" ))
        pickle.dump( ulti_tree, open( "ast_map.dat", "wb" ))
    # ulti_tree = pickle.load( open( "ast_map.dat", "rb" ) )
    # ulti_dict = pickle.load( open( "directory_map.dat", "rb" ))
    # new_ulti_dict = json.dumps(ulti_dict)

    a2a = metric_a2a(ulti_tree)
    c2c = metric_c2c(ulti_dict)
    try:
        pickle.dump( a2a, open( "./metric_a2a.dat", "wb" ))
        pickle.dump( c2c, open( "./metric_c2c.dat", "wb" ))
    except:
        pickle.dump( a2a, open( "metric_a2a.dat", "wb" ))
        pickle.dump( c2c, open( "metric_c2c.dat", "wb" ))

    return redirect('/run')

@app.route('/run')
def hello_world():

    script_dir = os.getcwd()

    new_dir = os.path.join(os.getcwd(), 'target_repo_dir')


    tree = dir_walk(new_dir)
    try:
        ulti_dict = pickle.load( open( "./directory_map.dat", "rb" ))
    except:
        ulti_dict = pickle.load( open( "directory_map.dat", "rb" ))
    new_ulti_dict = json.dumps(ulti_dict)

    try:
        ulti_tree = pickle.load( open( "./ast_map.dat", "rb" ) )
    except:
        ulti_tree = pickle.load( open( "ast_map.dat", "rb" ) )
    tags = open('target_repo_dir/tags.txt').readlines()

    try:
        metric_a2a = pickle.load( open( "./metric_a2a.dat", "rb" ) )
        metric_c2c = pickle.load( open( "./metric_c2c.dat", "rb" ) )
    except:
        metric_a2a = pickle.load( open( "metric_a2a.dat", "rb" ) )
        metric_c2c = pickle.load( open( "metric_c2c.dat", "rb" ) )

    return render_template('index.html', tags = tags, ulti_tree = {'body': ulti_tree} ,ulti_dict = new_ulti_dict, tree = tree, metric_a2a = {'body':metric_a2a}, metric_c2c = {'body':metric_c2c})

def get_dictionary(tree):

    new_calls_tree = {}

    for k in tree.keys():
        print(k, "\n-------------")
        for i in tree[k]:
            if i['type'] == 'file':
                if i['value'].split('.')[-1] == 'py':
                    # print(i['value'])
                    code = open(i['value'], 'r', encoding='utf-8').read()
                    new_tree = make_ast(code)
                    final_obj = wrapper_get(new_tree, i['value'])
                    # print(final_obj)
                    print("successful")
                    if i['value'] not in new_calls_tree.keys():
                        new_calls_tree[i['value']] = dict(final_obj)

    # pickle.dump( new_calls_tree, open( "test2.dat", "wb" ))

    return new_calls_tree

if __name__ == '__main__':

    app.run(debug = False)