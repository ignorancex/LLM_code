from flask import render_template
from flask import make_response
from app import app
from elasticsearch import Elasticsearch, helpers
import json
from datetime import datetime
from time import sleep


#Default index works awesome.
@app.route('/')
def index():
    client = Elasticsearch("localhost:9200")
    query_all = {
    'size' : 5,
    'query': {
    'match_all' : {}
    }
    }
    print ("\nSleeping for a few seconds to wait for indexing request to finish.")
    sleep(1)
    # pass the query_all dict to search() method
    try:
        resp = client.search(
        index = "market_asean",
        body = query_all
        )

        print(type(resp))
        temp = json.dumps(resp)
        user = json.loads(temp)
        print(user['hits']['hits'][0]["_source"])
        return render_template('index.html', title='Home', user=user)
    except:
        return make_response(render_template("404.html"), 404)

#This works great!
@app.route('/<index>')
@app.route('/index/<index>')
def index2(index):
    client = Elasticsearch("localhost:9200")
    query_all = {
    'size' : 5,
    'query': {
    'match_all' : {}
    }
    }
    print ("\nSleeping for a few seconds to wait for indexing request to finish.")
    sleep(1)
    # pass the query_all dict to search() method
    try:
        resp = client.search(
        index = index,
        body = query_all
        )

        print(type(resp))
        temp = json.dumps(resp)
        user = json.loads(temp)
        print(user['hits']['hits'][0]["_source"])
        return render_template('index.html', title='Home', user=user)
    except:
        return make_response(render_template("ES404.html"), "test")

#This is for the search area and works pretty well once we figure out the index stuff...
#maybe just make the url pass the entire market_elitemarket or market_xxxxxx to save time
@app.route('/<index>/search/<searchTerms>')
def aseanSearch(index, searchTerms):
    client = Elasticsearch("localhost:9200")
    query_all = {
    'size' : 5,
        'query': {
            'match' : {
                "title": searchTerms
            }
        }
    }
    print ("\nSleeping for a few seconds to wait for indexing request to finish.")
    sleep(1)
    # pass the query_all dict to search() method
    try:
        resp = client.search(
        index = index,
        body = query_all
        )

        print(type(resp))
        temp = json.dumps(resp)
        user = json.loads(temp)
        print(user['hits']['hits'][0]["_source"])
        return render_template('index.html', title='Home', user=user)
    except:
        return make_response(render_template("ES404.html"), "test")


# 
# FIX THESE LATER
# 
@app.errorhandler(404)
def not_found():
    """Page not found."""
    return make_response(render_template("404.html"), 404)

@app.errorhandler(400)
def bad_request():
    """Bad request."""
    return make_response(render_template("400.html"), 400)


@app.errorhandler(500)
def server_error():
    """Internal server error."""
    return make_response(render_template("500.html"), 500)