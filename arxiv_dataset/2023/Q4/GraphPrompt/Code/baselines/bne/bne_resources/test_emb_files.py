import os

sc = open(os.path.join("embeddings/BioEmb","Emb_SGsc.txt"),"r")
w = open(os.path.join("embeddings/BioEmb","Emb_SGw.txt"),"r")
umls = open(os.path.join("embeddings","UMLS_output_BNE_SGw.txt"),"r")

sc = set([str(item.replace("\n","").split(" ")[0]) for item in sc])
w = set([str(item.replace("\n","").split(" ")[0]) for item in w])
umls = set([str(item.replace("\n","").split(" ")[0]) for item in umls])

print("Length of SGsc is ", len(sc))
print("Length of SGw is ", len(w))
print("Length of UMLS is ", len(umls))

print("Length of union is ", len(sc | w))
print("Length of intersection is ", len(sc & w))

print("Length of union umls and sc is", len(sc | umls))
print("Length of union umls and w is", len(w | umls))

print("Length of interesection umls and sc is", len(sc & umls))
print("Length of intersection umls and w is", len(w & umls))