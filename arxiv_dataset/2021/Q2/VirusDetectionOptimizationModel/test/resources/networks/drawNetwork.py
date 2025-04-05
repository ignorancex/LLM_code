import networkx
import pandas
import pylab

print("Enter the network file name (without extension):")
networkfile = input()

G = networkx.Graph()
df = pandas.read_csv(networkfile+".txt", header=None)
G.add_edges_from(df.to_records(index=False))
networkx.draw(G, with_labels=True)

print("Enter title:")
title = input()

pylab.savefig(networkfile+".png", dpi=600, metadata={"Title":title, "Author":"Sudesh Agrawal (sudesh@utexas.edu)", "Copyright":"All rights reserved."})