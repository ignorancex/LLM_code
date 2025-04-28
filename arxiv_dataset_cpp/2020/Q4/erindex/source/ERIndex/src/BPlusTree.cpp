/*
 * BPlusTree.cpp
 *
 *  Created on: 25/set/2014
 *      Author: fernando
 */

#include "BPlusTree.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include "Bookmark.h"
#include "LZIndex.h"
#include "EncryptedFileBitWriter.h"

namespace std {

class LZIndex;

BPlusTreeNode::BPlusTreeNode(int t1, bool leaf1){
	t = t1;
	leaf = leaf1;
	//Per i vettori delle chiavi, dei valori e dei puntatori viene allocata memoria per un elemento in più,
	//in modo da permettere l'overflow
	keys = new int[2*t];
	if (leaf)
		values=new vector<Value>[2*t];
	C = new BPlusTreeNode*[2*t+1];
	for (int i=0;i<2*t+1;i++)
		C[i]=NULL;
	n = 0;
	previousSibling=NULL;
	nextSibling=NULL;
	parent=NULL;
}


// traverse all nodes in a subTree rooted with this node
void BPlusTreeNode::traverse(){
	int i;
	if (leaf)
		for (i = 0; i < n; i++)
					cout << " " << keys[i];
	else
		for (i = 0; i < n+1; i++)
					C[i]->traverse();  //il sottoalbero C[i] contiene chiavi <= rispetto a keys[i]


}

// traverse all nodes in a subTree rooted with this node
void BPlusTreeNode::nodeStats(int *leafNodes, int *indexNodes){
	int i;
	if (leaf)
		(*leafNodes)++;
	else{
		(*indexNodes)++;
		for (i = 0; i < n+1; i++)
			C[i]->nodeStats(leafNodes,indexNodes);
	}//il sottoalbero C[i] contiene chiavi <= rispetto a keys[i]
}


int int_log2(int u)    // compute # bits to represent u
{
  int i = 1;
  int r = 1;

  while((i<=32) && (r<u)){
    r=2*r+1;
    i = i+1;
  }

  return i;
}

BPlusTreeNode* BPlusTreeNode::searchInsertionLeaf(int k){
	if (leaf == true)  //spostato dopo il while
			return this;
	int i = 0;
	while (i < n && k > keys[i])   //cerca i tale che la chiave keys[i] >= k
		i++;
	if (C[i]==NULL)
		C[i]=tree->getNode(CNN[i]);
	return C[i]->searchInsertionLeaf(k);
}


void BPlusTreeNode::split(){
	//BPlusTreeNode *z = new BPlusTreeNode(t, leaf);
	BPlusTreeNode *z = tree->newNode(leaf);
	z->parent=parent;  //il nuovo nodo ha lo stesso parent del nodo in overflow
	z->tree=tree;
	z->n = t;

	//chiave da copiare o spostare nel nodo superiore

	int kk=keys[t];
	if (leaf){
		for (int j = 0; j < t; j++)			//mette nel nuovo nodo tutte le chiavi che occupano le posizioni dal t in poi, insieme ai corrispondenti valori
				z->keys[j] = keys[j+t];
		for (int j = 0; j < t; j++)
			z->values[j] = values[j+t];
	}
	else{
		for (int j = 0; j < t-1; j++)
					z->keys[j] = keys[j+t+1];
		for (int j = 0; j < t; j++){
			z->C[j] = C[j+t+1];
			z->C[j]->parent=z;
			//C[j+t+1]->parent=z;
		}
		z->n=t-1;
	}

	n = t;							//reimposta a t il numero di chiavi presenti nel nodo da splittare



	if (leaf){
		//inserisce il nuovo nodo nella lista doppiamente linkata delle foglie
		if (nextSibling!=NULL)
			nextSibling->previousSibling=z;
		z->previousSibling=this;
		z->nextSibling=nextSibling;
		nextSibling=z;

		if (tree->lastLeaf==this)
			tree->lastLeaf=z;

	}
	/*else{
		//cancella la chiave kk dal nuovo nodo z
		for (int j=0; j< z->n-1; j++){
			z->keys[j] = z->keys[j+1];
			z->C[j] = C[j+1];
		}
		z->n--;
	}*/
	//effettua il copy up:la chiave più piccola del nuovo nodo viene copiata nel parent
	//insieme al puntatore al nuovo nodo
	if (parent==NULL){
		//parent = new BPlusTreeNode(t, false);
		parent = tree->newNode(false);
		z->parent=parent;
		parent->C[0]=this;
		parent->keys[0]=kk;
		parent->C[1]=z;
		parent->n=1;
		tree->root=parent;
		parent->tree=tree;
		//index->dumpTree();
	}
	else{
		parent->insert(kk,z);
	}
}



void BPlusTreeNode::insert(int k,BPlusTreeNode *c){
	int i = n-1;
	//Sposta di una posizione a destra tutte le chiavi maggiori di quella da inserire,
	//per far posto ad essa
	while (i >= 0 && keys[i] > k){
		keys[i+1] = keys[i];
		C[i+2]=C[i+1];
		i--;
	}
	keys[i+1] = k;
	C[i+2] = c;
	n = n+1;
	//index->dumpTree();
	if (n==2*t)
		split();
}

//Restituisce la posizione della prima chiave >=k tra tutte quelle presenti nel nodo
//Ritorna il valore true se tale chiave coincide con k (quindi se k è presente nel nodo)
inline bool BPlusTreeNode::binarySearchInNode(int k, int *position){
	int p=0; int u=n-1;
	*position=-1;
	while (*position ==-1 && p<=u){
			int m=(p+u)/2;
			if (keys[m] > k )
					u=m-1;
			else if (keys[m] < k)
					p=m+1;
			else{
			   *position=m;
			   return true;
			}
	}
	*position=p;
	return false;
}

void BPlusTreeNode::insert(int k,Value value){
	int posK;

	//search the key k in this node and, if not found, add it to the node keys
	bool found=binarySearchInNode(k,&posK);
	if (!found){
		//Sposta di una posizione a destra tutte le chiavi, a partire dalla posizione position (è la prima maggiore di k)
		int i=n-1;
		while (i >= posK){
			keys[i+1] = keys[i];
			values[i+1]=values[i];
			i--;
		}
		//Inserisce la nuova chiave nel nodo
		keys[posK] = k;
		values[posK].clear();
		//Incrementa di 1 il numero delle chiavi presenti nel nodo
		n = n+1;
	}
	//Inserisce il valore nella lista dei valori associati alla chiave
	values[posK].push_back(value);

	if (n==2*t)
		split();
}

bool BPlusTreeNode::search(int k,vector<Value>* value){
	int i = 0;
	while (i < n && k > keys[i])   //cerca i tale che la chiave keys[i] >= k
		i++;
	if (leaf){
		if (keys[i] == k){			  //se k coincide con keys[i] la ricerca ha avuto successo
			value=&values[i];
			return true;
		}
		else
			return false;
	}
	else{
		return C[i]->search(k,value);
	}
}


void BPlusTree::setNumberOfIndividuals(int numberOfIndividuals){
	this->numberOfIndividuals=numberOfIndividuals;
}

void BPlusTree::setMaximumNumberOfFactorsPerIndividual(int maximumNumberOfFactorsPerIndividual){
	this->maximumNumberOfFactorsPerIndividual=maximumNumberOfFactorsPerIndividual;
}

void BPlusTree::setMaximumIndividualId(int maximumIndividualId){
	this->maximumIndividualId=maximumIndividualId;
}

void BPlusTreeNode::saveKeys(FileBitWriter &bitWriter){
	//Save n (number of keys really contained in this node)
	int nBits=int_log2(2*t);
	bitWriter.write(nBits,n);

	//Save the first key absolute value
	int firstKey=*keys;
	nBits=int_log2(tree->referenceLength+1);
	bitWriter.write(nBits,*keys+1);

	//Save the other keys relative values
	int lastKey=keys[n-1];
	nBits=int_log2(lastKey-firstKey);
	bitWriter.writeInt(nBits);
	for (int i=1;i<n;i++)
		bitWriter.write(nBits,keys[i]-firstKey);

}


void BPlusTreeNode::loadKeys(FileBitReader *bitReader){
	//Save n (number of keys really contained in this node)
	int nBits=int_log2(2*t);
	n=bitReader->read(nBits);

	//Save the first key absolute value

	nBits=int_log2(tree->referenceLength+1);
	*keys=bitReader->read(nBits)-1;
	int firstKey=*keys;

	//Read the other keys' values
	nBits=bitReader->getInt();
	for (int i=1;i<n;i++)
		keys[i]=bitReader->read(nBits)+firstKey;

}


//vedi Algoritmo di compressione "Invariable coding" descritto nel paper"Benchmarking a B-tree compression method"
void BPlusTreeNode::save(EncryptedFileBitWriter &bitWriter){
	int nBits;
	if (!leaf)
		//SAVE ALL THE INDEX NODES IN THE SUB-TREE ROOTED IN THIS NODE
		for (int i = 0; i < n+1; i++)
				if (!C[i]->leaf)
				 		C[i]->save(bitWriter);

	nodeNumber=tree->savedNodes;

	tree->savedNodes++;
	//The general part of the node (i.e. the part that isn't individual-specific") will be encrypted using the system key and the nonce treeBaseNonce + nodeNumber+1
	EncryptionContext *nodeEncryptionContext=new EncryptionContext(tree->index->userPortfolio->getSystemKey()->getClearValue(),tree->baseNonce + nodeNumber+1);
	bitWriter.setEncryptionContext(nodeEncryptionContext);

	//SAVE KEYS
	saveKeys(bitWriter);

	if (leaf){  //SAVE VALUES
		int maxFactorId=std::numeric_limits<int>::min(); //=-2;
		int minFactorId=std::numeric_limits<int>::max();
		int maxDuplicatesPerKey=0;
		int numberOfValues=0;
		int duplicatedKeys=0;
		int numberOfDistinctIndividuals=0;
		bool duplicatedKeysBitmap[n];  //1 for keys with duplicates
		for (int i=0;i<n;i++){
			if (values[i].size()>1){
				duplicatedKeys++;
				duplicatedKeysBitmap[i]=1;
				if (values[i].size()>maxDuplicatesPerKey){
					maxDuplicatesPerKey=values[i].size();
				}
			} else{
				duplicatedKeysBitmap[i]=0;
			}

			for (int j=0;j<values[i].size();j++){
				if (values[i][j].factorId >maxFactorId)
					maxFactorId=values[i][j].factorId;
				if (values[i][j].factorId <minFactorId)
					minFactorId=values[i][j].factorId;
				int64_t iid=values[i][j].individualId;
				if (individualRefs.count(iid)==0){
					//add the individual id to the map of the individuals, computing its reference
					//as the first available referenceId (the referenceId are in the range
					//[0,numberOfDistinctIndividuals-1]
					individualRefs[iid]=numberOfDistinctIndividuals;
					//init the encryption context for the individual
					individualEncryptionContexts[iid]=new EncryptionContext(
							tree->index->userPortfolio->getIndividualKey(iid)->getClearValue(),tree->baseNonce + nodeNumber+1);
					numberOfDistinctIndividuals++;
				}
				numberOfValues++;
			}
		}

		//SAVE THE INDIVIDUALS APPEARING IN THIS LEAF
		//the values of each individual will be encrypted using the individual specific key and the nonce
		//tree->baseNonce + nodeNumber+1


		//number of bits to encode the number of the individual appearing in this leaf (at most
		//all the individual appearing in the LZ-index)
		nBits=int_log2(tree->numberOfIndividuals);
		//write the number of individuals appearing in this leaf
		bitWriter.write(nBits,numberOfDistinctIndividuals);
		//write individual ids (TODO: encryption)
		nBits=int_log2(tree->maximumIndividualId);

		int i=0;
		for (auto it=individualRefs.begin(); it!=individualRefs.end(); ++it){
			//reassign the individualIds to disk
		    bitWriter.write(nBits,it->first);
		    //reassign the references in order of individualId (the map is sorted by individualId)
		    it->second=i;
		    i++;
		}

		//NUMBER OF BITS NEEDED TO ENCODE THE INDIVIDUAL REFERENCES
		//int individualNBits=int_log2(int_log2(individualRefs.size()-1));  //bits needed to encode nBits
		int individualNBits=int_log2(individualRefs.size()-1);  //bits needed to encode nBits

		//MININUM VALUE OF THE factorId
		nBits=int_log2(tree->maximumNumberOfFactorsPerIndividual);
		bitWriter.write(nBits,minFactorId);

		//FACTOR ID RELATIVE VALUES LENGTH
		int nBitsNeededBits=int_log2(int_log2(tree->maximumNumberOfFactorsPerIndividual));  //bits needed to encode nBits
		int factorNBits=int_log2(maxFactorId-minFactorId);
		bitWriter.write(nBitsNeededBits,factorNBits);

		//DUPLICATES BIT
		//this bit indicates if in this leaf exists at least a key with duplicates
		bitWriter.write(1,duplicatedKeys==0?0:1);

		if (duplicatedKeys){
			//BITMAP indicating which keys have duplicates
			for (int i=0;i<n;i++)
				bitWriter.write(1,duplicatedKeysBitmap[i]);
			//TABLE containing, for each duplicated key, the number of its values
			nBits=int_log2(maxDuplicatesPerKey);
			bitWriter.writeInt(nBits);
			for (int i=0;i<n;i++){
				if (duplicatedKeysBitmap[i])
					bitWriter.write(nBits,values[i].size());
			}
			//ARRAY OF VALUES (those of the first key preceed that of the second key and so on)
			for (int i=0;i<n;i++)  //for each key
				for (int j=0;j<values[i].size();j++){  //for each value
					bitWriter.write(individualNBits,individualRefs[values[i][j].individualId]);  //individual
					bitWriter.setEncryptionContext(individualEncryptionContexts[values[i][j].individualId]);
					bitWriter.write(factorNBits,values[i][j].factorId-minFactorId);	//factor
					bitWriter.setEncryptionContext(nodeEncryptionContext);
				}

		} else{
			for (int i=0;i<n;i++){
				bitWriter.write(individualNBits,individualRefs[values[i][0].individualId]);
				bitWriter.setEncryptionContext(individualEncryptionContexts[values[i][0].individualId]);
				bitWriter.write(factorNBits,values[i][0].factorId-minFactorId); //there is exactly one value per key
				bitWriter.setEncryptionContext(nodeEncryptionContext);
			}
		}

		//free the memory allocated for the individual encryption contexts objects
		for (auto it=individualEncryptionContexts.begin(); it!=individualEncryptionContexts.end(); ++it){
			delete it->second;
		}


	} else{
		//SAVE POINTERS (node numbers)
		nBits=int_log2(tree->nodesNumber);
		for (int i = 0; i < n+1; i++)
			bitWriter.write(nBits,C[i]->nodeNumber);

	}
	bitWriter.flush();
	//ADD TO THE TREE DIRECTORY THE START POSITION OF THIS NODE IN THE B-TREE BYTESTREAM
	Bookmark* bm=bitWriter.getBookmark();
	uint64_t p=bm->getPosition();
	if (tree->savedNodes < tree->nodesNumber)
		tree->treeDirectory.nodesStartsPositions.push_back(p);
	else
		tree->treeDirectoryOffset=p;

	delete nodeEncryptionContext;
}


void BPlusTreeNode::load(EncryptedFileBitReader *bitReader){
	//The general part of the node (i.e. the part that isn't individual-specific") will be encrypted using the system key and the nonce treeBaseNonce+nodeNumber+1
	EncryptionContext *nodeEncryptionContext=new EncryptionContext(tree->index->userPortfolio->getSystemKey()->getClearValue(),tree->baseNonce + nodeNumber+1);
	bitReader->setEncryptionContext(nodeEncryptionContext);

	//Create an unknown encryption context (i.e. an encryption context with a NULL key)
	u8 *key=NULL;
	EncryptionContext *unknownEncryptionContext=new EncryptionContext(key,0);


	//LOAD KEYS
	loadKeys(bitReader);
	int nBits;

	if (leaf){
		//LOAD INDIVIDUALS APPEARING IN THIS LEAF
		//number of bits to encode the number of the individual appearing in this leaf (at most
		//all the individuals appearing in the LZ-index)
		nBits=int_log2(tree->numberOfIndividuals);
		//write the number of individuals appearing in this leaf
		int numberOfIndividuals=bitReader->read(nBits);
		//read the individual ids
		nBits=int_log2(tree->maximumIndividualId);
		for (int i=0;i<numberOfIndividuals;i++){
			int64_t iid=bitReader->read(nBits);
			individualRefs[i]=iid;
			IndividualKey *ik=tree->index->userPortfolio->getIndividualKey(iid);
			//init the encryption context for the individual

			if (ik!=NULL)
			  individualEncryptionContexts[iid]=new EncryptionContext(
						ik->getClearValue(),tree->baseNonce + nodeNumber+1);
			else
			  individualEncryptionContexts[iid]=NULL;
		}

		//LOAD VALUES

		//NUMBER OF BITS NEEDED TO DECODE THE INDIVIDUAL REFERENCES
		//int individualNBits=int_log2(int_log2(individualRefs.size()-1));  //bits needed to encode nBits
		int individualNBits=int_log2(individualRefs.size()-1);  //bits needed to encode nBits

		//MININUM VALUE OF THE factorId
		nBits=int_log2(tree->maximumNumberOfFactorsPerIndividual);
		int minFactorId=bitReader->read(nBits);

		//FACTOR ID RELATIVE VALUES LENGTH
		nBits=int_log2(int_log2(tree->maximumNumberOfFactorsPerIndividual));
		int factorNBits=bitReader->read(nBits);

		//DUPLICATES BIT
		//this bit indicates if in this leaf exists at least a key with duplicates
		bool duplicatedKeys=bitReader->read(1);

		uint64_t iid;
		if (duplicatedKeys){
			bool duplicatedKeysBitmap[n];
			int numberOfValuesPerKey[n];
			//BITMAP indicating which keys have duplicates
			for (int i=0;i<n;i++)
				duplicatedKeysBitmap[i]=bitReader->read(1);
			//TABLE containing, for each duplicated key, the number of its values
			nBits=bitReader->getInt();
			for (int i=0;i<n;i++){
				if (duplicatedKeysBitmap[i]){
					//int numberOfValues=bitReader->read(nBits);
					//values[i].resize(numberOfValues);
					numberOfValuesPerKey[i]=bitReader->read(nBits);
				} else
					//values[i].resize(1);
					numberOfValuesPerKey[i]=1;
			}
			//ARRAY OF VALUES (those of the first key preceed that of the second key and so on)
			for (int i=0;i<n;i++)
				for (int j=0;j<numberOfValuesPerKey[i];j++){
					int iref=bitReader->read(individualNBits);
					iid=individualRefs[iref];
					if (individualEncryptionContexts[iid]!=NULL){
						Value v;
						v.individualId=iid;
						bitReader->setEncryptionContext(individualEncryptionContexts[iid]);
						v.factorId=bitReader->read(factorNBits)+minFactorId;
						values[i].push_back(v);
						bitReader->setEncryptionContext(nodeEncryptionContext);
					}
					else{
						bitReader->setEncryptionContext(unknownEncryptionContext); //simulates the first encryption context switch
						bitReader->read(factorNBits); //skip the factorId
						bitReader->setEncryptionContext(nodeEncryptionContext); //simulates the second encryption context switch
					}
				}
		} else{
			//ARRAY OF VALUES (there is exactly one value per key)
			for (int i=0;i<n;i++){
				//values[i].resize(1);
				iid=individualRefs[bitReader->read(individualNBits)];
				if (individualEncryptionContexts[iid]!=NULL){
					Value v;
					v.individualId=iid;
					bitReader->setEncryptionContext(individualEncryptionContexts[iid]);
					v.factorId=bitReader->read(factorNBits)+minFactorId;
					bitReader->setEncryptionContext(nodeEncryptionContext);
					values[i].push_back(v);
				}
				else{
					bitReader->setEncryptionContext(unknownEncryptionContext); //simulates the first encryption context switch
					bitReader->read(factorNBits); //skip the factorId
					bitReader->setEncryptionContext(nodeEncryptionContext); //simulates the second encryption context switch
				}
			}
		}
		//destroy the individual encryption contexts
		for (int i=0;i<numberOfIndividuals;i++){
			iid=individualRefs[i];
			if (individualEncryptionContexts[iid]!=NULL)
			   delete individualEncryptionContexts[iid];
		}

	}else{
		CNN = new int[2*t+1];
		//LOAD POINTERS (Node Numbers)
		nBits=int_log2(tree->treeDirectory.nodesNumber);
		for (int i = 0; i < n+1; i++)
			   CNN[i]=bitReader->read(nBits);

	}

	//destroy the node encryption context
	delete nodeEncryptionContext;
	delete unknownEncryptionContext;

}





BPlusTree::BPlusTree(LZIndex *index, int _t,uint64_t referenceLength){
	this->index=index;

	root = NULL;
	t = _t;

	treeDirectoryOffset=0;

	nodesNumber=0;
	this->referenceLength=referenceLength;

	maximumIndividualId=std::numeric_limits<int>::min();
}

BPlusTree::~BPlusTree() {
	// TODO Auto-generated destructor stub
}

BPlusTreeNode *BPlusTree::newNode(bool leaf){
	nodesNumber++;
	return new BPlusTreeNode(t,leaf);
}


void BPlusTree::traverse(){
	BPlusTreeNode *currentLeaf;
	currentLeaf=firstLeaf;
	while (currentLeaf !=NULL){
		for (int i = 0; i < currentLeaf->n; i++)
				cout << " " << currentLeaf->keys[i];
		currentLeaf=currentLeaf->nextSibling;
	}
}

void BPlusTree::treeTraverse(){
	if (root!=NULL)
		root->traverse();
}

void BPlusTree::treeStats(int* leafNodes,int *indexNodes){
	*leafNodes=0;
	*indexNodes=0;
	if (root!=NULL)
		root->nodeStats(leafNodes,indexNodes);
}




BPlusTreeNode* BPlusTree::searchInsertionLeaf(int k){
	return (root == NULL)? NULL : root->searchInsertionLeaf(k);
}


void BPlusTreeNode::dumpSubTree(int level){
	cout << std::string(3*level, ' ');
	for (int i=0;i<n;i++){
		if (leaf){
			cout << keys[i] <<  " (";
			for (int j=0;j< values[i].size();j++){
			  cout << (values[i])[j];
			  if (j<values[i].size()-1)
				  cout << ",";
			}
            cout << ") ";
		}
		else
			cout << keys[i] <<  " ";

	}
	cout << (leaf?"[leaf]":"") << " (parent=" << (parent==NULL?-1:parent->keys[0]) << "...)" << endl;
	if (!leaf)
		for (int i=0;i<n+1;i++){
			if (C[i]==NULL)
					C[i]=tree->getNode(CNN[i]);
			C[i]->dumpSubTree(level+1);
		}

}


bool BPlusTree::search(int k,vector<Value>* value){
	return (root == NULL)? false : root->search(k,value);
}


void BPlusTree::insert(int k,Value value){
	BPlusTreeNode *insertionLeaf;
	if (root == NULL){
		//Crea il nodo radice
		//root = new BPlusTreeNode(t, true);
		root = newNode(true);
		root->tree=this;
		firstLeaf=root;
		lastLeaf=root;
		insertionLeaf=root;
	}
	else
		insertionLeaf=root->searchInsertionLeaf(k);
	insertionLeaf->insert(k,value);
	//dumpTree();

	//adjust the maximum individualID: it will be used to encode the individualIds when saving the leafs
	if (value.individualId>maximumIndividualId)
		maximumIndividualId=value.individualId;

}

void BPlusTree::dumpTree(){
	if (root != NULL){
		root->dumpSubTree(0);
	}
}



void BPlusTree::setBaseNonce(uint32_t baseNonce){
	this->baseNonce=baseNonce;
}

uint64_t BPlusTree::save(EncryptedFileBitWriter &bitWriter, uint64_t *treeDirectoryOffset){
	savedNodes=0;
	Bookmark* bm=bitWriter.getBookmark();
	uint64_t p=bm->getPosition();
	treeDirectory.treeOffset=p;
	if (root != NULL){
		//The following encryption context has been initialized from the caller
		//with the system key and a nonce depending on the specific tree
		treeDirectory.nodesNumber=nodesNumber;

		//SAVE THE LEAF NODES (as the nodes with nodeNumber in [0,leafNodesNumber-1])
		BPlusTreeNode *currentLeaf=firstLeaf;
		while (currentLeaf !=NULL){
			currentLeaf->save(bitWriter);
			currentLeaf=currentLeaf->nextSibling;
		}
		treeDirectory.lastLeafNodeNumber=savedNodes-1;

		//SAVE THE INDEX NODES: the root node will be saved lastly
		root->save(bitWriter);
		Bookmark* bm=bitWriter.getBookmark();
		uint64_t p=bm->getPosition();

		*treeDirectoryOffset=p;
		//SAVE THE TREE DIRECTORY (with the baseNonce)
		EncryptionContext *treeEncryptionContext=new EncryptionContext(index->userPortfolio->getSystemKey()->getClearValue(),baseNonce);
		bitWriter.setEncryptionContext(treeEncryptionContext);
		//treeOffset: offset of this tree in the LZ index bytestream
		bitWriter.writeInt(treeDirectory.treeOffset);
		//number of nodes
		bitWriter.writeInt(treeDirectory.nodesNumber);
		//last leaf node
		int nBits=int_log2(treeDirectory.nodesNumber-1);
		bitWriter.write(nBits,treeDirectory.lastLeafNodeNumber);
		//start positions of all the nodes following the first
		nBits=int_log2(*treeDirectoryOffset-treeDirectory.treeOffset);

		for (int i = 0; i < treeDirectory.nodesNumber-1; i++)
			bitWriter.write(nBits,treeDirectory.nodesStartsPositions[i]-treeDirectory.treeOffset);

		bitWriter.flush();


		return bitWriter.getBookmark()->getPosition()-treeDirectory.treeOffset;  //return the B-Tree size in bytes

	}
}


void BPlusTree::loadAllNodesInMemory(){
	//double mean=0;
	//double meanOccupancy=0;
	//int numberOfLeaf=0;
	for (int i=0;i<treeDirectory.nodesNumber-1;i++){  //root has already been loaded
		BPlusTreeNode *n=getNode(i);
		/*
		if (n->leaf){
			mean+=n->individualRefs.size();
			meanOccupancy+=n->n;
			double meanOfValuesPerIndividual=0;
			map<int,int> m;
			int numberOfValues=0;
			for (int ii=0;ii<n->n;ii++){ //for each key
				for (int j=0;j<n->values[ii].size();j++){
					if (m.count(n->values[ii][j].individualId)==0)
						m[n->values[ii][j].individualId]=0;
					m[n->values[ii][j].individualId]++;
					numberOfValues++;
				}

			}
			double med=0;
			for (auto it=m.begin(); it!=m.end(); ++it)
				med+=it->second;
			med=med/m.size();


			cout << "node " << i << " med " << ":"<< med << endl;
			numberOfLeaf++;
		}
		*/
    }
	//mean=(double)mean/numberOfLeaf;
	//meanOccupancy=(double)meanOccupancy/numberOfLeaf;

	//cout << "Mean: " << mean << endl;
	//cout << "Mean occupancy: " << mean << endl;
}

void BPlusTree::load(EncryptedFileBitReader *bitReader, uint64_t treeDirectoryOffset){
	this->bitReader=bitReader;
	this->treeDirectoryOffset=treeDirectoryOffset;
	//reads the treeDirectory
	bitReader->gotoBookmark(new Bookmark(this->treeDirectoryOffset,0));
	EncryptionContext *treeEncryptionContext=new EncryptionContext(index->userPortfolio->getSystemKey()->getClearValue(),baseNonce);
	bitReader->setEncryptionContext(treeEncryptionContext);

	//treeOffset: offset of this tree in the LZ index bytestream
	treeDirectory.treeOffset=bitReader->getInt();
	//number of nodes
	treeDirectory.nodesNumber=bitReader->getInt();
	//last leaf node
	int nBits=int_log2(treeDirectory.nodesNumber-1);
	treeDirectory.lastLeafNodeNumber=bitReader->read(nBits);

	//start positions of all the nodes following the first
	nBits=int_log2(this->treeDirectoryOffset-treeDirectory.treeOffset);
	treeDirectory.nodesStartsPositions.resize(treeDirectory.nodesNumber);
	nodes.resize(treeDirectory.nodesNumber);

	for (unsigned int i = 0; i < treeDirectory.nodesNumber-1; i++){
		treeDirectory.nodesStartsPositions[i]=bitReader->read(nBits)+treeDirectory.treeOffset;
		nodes[i]=NULL;
	}

	root=getNode(treeDirectory.nodesNumber-1);

}


typedef pair<int,vector<Value>> KeyValuePair;

vector<pair<int,vector<Value>>> BPlusTree::rangeQueryAllInMemory(int l,int u){
	vector<pair<int,vector<Value>>> result;
    if (root!=NULL){
    	//search for the start leaf, i.e. that:
    	// - containing l, if l exists;
    	// - would contain l, il l doesn't exist
    	BPlusTreeNode *currentLeaf=root->searchInsertionLeaf(l);
    	//find the first key value greater or equal than l
    	int firstKey=-2; //-1 can be a value
    	int i=0; //currentLeaf current key position
    	bool foundFirstValue=false;
    	bool outOfRange=false;
    	while (!outOfRange){
    		if (i>currentLeaf->n-1){
    			if (currentLeaf->nextSibling==NULL)
    				outOfRange=true;
    			else{
    				currentLeaf=currentLeaf->nextSibling;
    				i=0;
    			}
    		}
    		if (!outOfRange){ //there is a key to examine
    			int k=currentLeaf->keys[i];
    			if (k>u)
    				outOfRange=true;
    			else if (k>=l)
    				result.push_back(KeyValuePair(k,currentLeaf->values[i]));
    			i++;
    		}
    	}
    }
	return result;
}

inline BPlusTreeNode *BPlusTree::getNode(int nodeNumber){
	if (nodes[nodeNumber]==NULL){
		uint64_t nodeStartPosition;
		if (nodeNumber>0)
			nodeStartPosition=treeDirectory.nodesStartsPositions[nodeNumber-1];
		else
			nodeStartPosition=treeDirectory.treeOffset;
		bitReader->gotoBookmark(new Bookmark(nodeStartPosition,0));
		bool leaf=nodeNumber <= treeDirectory.lastLeafNodeNumber;
		BPlusTreeNode *node=new BPlusTreeNode(t,leaf);
		node->tree=this;
		node->nodeNumber=nodeNumber;
		node->load(bitReader);
		nodes[nodeNumber]=node;
	}
	return nodes[nodeNumber];
}

bool BPlusTree::isValueInRange(int l,int u,Value value){
	bool found=false;
	root=getNode(treeDirectory.nodesNumber-1);
	if (root!=NULL){
		//search for the start leaf, i.e. that:
		// - containing l, if l exists;
		// - would contain l, il l doesn't exist
		BPlusTreeNode *currentLeaf=root->searchInsertionLeaf(l);
		int currentLeafNodeNumber=currentLeaf->nodeNumber;
		//find the first key value greater or equal than l
		int firstKey=-2; //-1 can be a value
		int i=0; //currentLeaf current key position
		bool outOfRange=false;

		while (!outOfRange && !found){
			if (i>currentLeaf->n-1){
				if (currentLeafNodeNumber==treeDirectory.lastLeafNodeNumber)
					outOfRange=true;
				else{
					currentLeafNodeNumber++;
					currentLeaf=getNode(currentLeafNodeNumber);
					i=0;
				}
			}
			if (!outOfRange){ //there is a key to examine
				int k=currentLeaf->keys[i];
				if (k>u)
					outOfRange=true;
				else if (k>=l){
					int j=0;
					int nov=currentLeaf->values[i].size();
					while (j<nov && !found){
						if (currentLeaf->values[i][j]==value)
							found=true;
						else
							j++;
					}
				}
				i++;
			}
		}
	}
	return found;
}


//Returns, for each individual,a vector of pairs.
//The first element of each pair is a key k and the second is a factorId
map<int,vector<pair<int,int>>> BPlusTree::rangeQuery(int l,int u){
	map<int,vector<pair<int,int>>> result;
    if (root!=NULL){
    	//search for the start leaf, i.e. that:
    	// - containing l, if l exists;
    	// - would contain l, il l doesn't exist
    	BPlusTreeNode *currentLeaf=root->searchInsertionLeaf(l);
    	int currentLeafNodeNumber=currentLeaf->nodeNumber;
    	//find the first key value greater or equal than l
    	int firstKey=-2; //-1 can be a value
    	int i=0; //currentLeaf current key position
    	bool foundFirstValue=false;
    	bool outOfRange=false;
    	while (!outOfRange){
    		if (i>currentLeaf->n-1){
    			if (currentLeafNodeNumber==treeDirectory.lastLeafNodeNumber)
    				outOfRange=true;
    			else{
    				currentLeafNodeNumber++;
    				currentLeaf=getNode(currentLeafNodeNumber);
    				i=0;
    			}
    		}
    		if (!outOfRange){ //there is a key to examine
    			int k=currentLeaf->keys[i];
    			if (k>u)
    				outOfRange=true;
    			else if (k>=l){
    				for (unsigned int j=0;j<currentLeaf->values[i].size();j++){
    					//cout << currentLeaf->values[i][j].individualId << "," << currentLeaf->values[i][j].factorId <<"," << currentLeafNodeNumber << endl;
    					result[currentLeaf->values[i][j].individualId].push_back(pair<int,int>(k,currentLeaf->values[i][j].factorId));
    				}
    			}
    			i++;
    		}
    	}
    }
	return result;
}


/*
// Main
int main()

{

    BPlusTree t(3,1000);
    t.insert(10,Value(0,0));
    t.insert(10,Value(1,0));
    t.insert(20,Value(0,1));
    t.insert(5,Value(0,2));
    t.insert(6,Value(0,3));
    t.insert(12,Value(1,4));
    t.insert(30,Value(1,5));
    t.insert(30,Value(2,4));
    t.insert(30,Value(2,5));
    t.insert(3,Value(0,6));
    t.insert(1,Value(0,10));

    cout << "Traversal of the constucted tree is ";
    t.traverse();
    cout<<endl;
    int k = 6;
    cout<<k<<" is ";
    return 0;

}

*/







} /* namespace std */
