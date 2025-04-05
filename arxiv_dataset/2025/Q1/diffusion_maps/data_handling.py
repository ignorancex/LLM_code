from __future__ import division
import os
import pandas as pd
import numpy as np
import shapefile

def load_shapefile_with_index(filename):
        #Loads a shapfile reader and returns it along with an index of OAs
        shp=shapefile.Reader(filename)
        
        index=index_of_shapes(shp)

        return(shp,index)
        
        

def index_of_shapes(reader):
        #Finds index of OAs in shapefile. Returns - Dicionary of {OA:index}
        recIndex={}
        for i,rec in enumerate(reader.iterRecords()):
                recIndex[rec[1]]=i

        return(recIndex)
