import csv
from rdflib import Graph
from rdflib.namespace import RDF, FOAF
from rdflib import URIRef, BNode, Literal

loc_value=[]
foaf_value=[]
type_value=[]
name=''
location=URIRef("https://schema.org/location")

#reading model store csv result
input_file = csv.DictReader(open("model_csv_output.csv"))


for row in input_file:
    # convert it from an OrderedDict to a regular dict
    row = dict(row)
    # print(row['location'])
    loc_value=[str(row['location']).split(',')]
    foaf_value = [str(row['foaf']).split(',')]
    type_value= [str(row['type']).split(',')]
    name = str(row['name'])

uri = URIRef("https://www.wikidata.org/wiki/" + name.replace(' ','_'))


g = Graph()
#add name of target entity
g.add( (uri, FOAF.name, Literal(name)) )

#Adding top-3 entailing type with highest tf-idf values
g.add( (uri, RDF.type, Literal(type_value[0][0])) )
g.add( (uri, RDF.type, Literal(type_value[0][1])) )
g.add( (uri, RDF.type, Literal(type_value[0][2])) )

#Adding top-5 location with highest cosine similarity value
g.add( (uri, location, Literal(loc_value[0][0])) )
g.add( (uri, location, Literal(loc_value[0][1])) )
g.add( (uri, location, Literal(loc_value[0][2])) )
g.add( (uri, location, Literal(loc_value[0][3])) )
g.add( (uri, location, Literal(loc_value[0][4])) )


#Adding top-4 associated entities with highest cosine similarity value
g.add( (uri, FOAF.knows, Literal(foaf_value[0][0])) )
g.add( (uri, FOAF.knows, Literal(foaf_value[0][1])) )
g.add( (uri, FOAF.knows, Literal(foaf_value[0][2])) )
g.add( (uri, FOAF.knows, Literal(foaf_value[0][3])) )


g.serialize(destination='rdf_graph.ttl', format='turtle')
# print (g.serialize(format='turtle'))