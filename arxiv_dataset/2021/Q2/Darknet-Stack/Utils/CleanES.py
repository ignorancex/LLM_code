from elasticsearch import Elasticsearch, helpers
import json
from datetime import datetime
from time import sleep


#CREATE INDICE - WORKING
#Create Client
es = Elasticsearch()
# ignore 400 cause by IndexAlreadyExistsException when creating an index
#es.indices.create(index='test-index', ignore=400)
# ignore 404 and 400

es.indices.delete(index='elitemarket', ignore=[400, 404])
es.indices.delete(index='market_icarus', ignore=[400, 404])
es.indices.delete(index='market_asean', ignore=[400, 404])