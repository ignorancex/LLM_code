import pickle 
from eventregistry import *   

def get_uri_list(api_key, category_uri, source_uri, date_start, date_end):

    #construct the query using the specified values
    e_reg = EventRegistry(apiKey = api_key)
    e_qry = QueryArticlesIter(sourceUri = source_uri, categoryUri=category_uri, dateStart = date_start, dateEnd = date_end, dataType = "news")

    #execute the query
    #maxItem=-1 for all events
    resutls = e_qry.execQuery(e_reg, sortBy = "date", maxItems = -1)

    uri_lst = []

    for result in resutls:
        uri_lst.append(result['uri'])

    return(uri_lst)

def generate_uri_list(api_key, category_uri_lst, source_uri_lst, date_start, date_end):

    uri_dict = {}

    for category_uri in category_uri_lst:
        for source_uri in source_uri_lst:
            uri_lst = get_uri_list(api_key, category_uri, source_uri, date_start, date_end)

            for uri in uri_lst:

                if uri not in uri_dict.keys():
                    uri_dict[uri] = []
                uri_dict[uri].append(source_uri.split('.')[0])

    with open("data/intermediate/uri_dict.txt", "ab") as myFile:
        pickle.dump(uri_dict, myFile)