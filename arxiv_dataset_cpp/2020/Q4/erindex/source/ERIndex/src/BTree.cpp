/*
 * BTree.cpp
 *
 *  Created on: 25/set/2014
 *      Author: fernando
 */

#include "BTree.h"
#include <iostream>

namespace std {

BTreeNode::BTreeNode(int t1, bool leaf1){
		t = t1;
		leaf = leaf1;
		keys = new int[2*t-1];
		C = new BTreeNode *[2*t];
		n = 0;
}


// traverse all nodes in a subtree rooted with this node
void BTreeNode::traverse(){
	int i;
	for (i = 0; i < n; i++){
		if (leaf == false)
			C[i]->traverse();  //il sottoalbero C[i] contiene chiavi <= rispetto a keys[i]
		cout << " " << keys[i];
	}

	if (leaf == false)
		C[i]->traverse();  //rimane da attraversare il sottoalbero più a destra (il numero dei sottoalberi supera di 1 il numero delle chiavi)

}

void BTreeNode::insertNonFull(int k){
	int i = n-1;
	//Se il nodo è foglia la chiave va inserita in esso
	if (leaf == true){
		//Sposta di una posizione a destra tutte le chiavi maggiori di quella da inserire, per far posto ad essa
		while (i >= 0 && keys[i] > k){
			keys[i+1] = keys[i];
			i--;
		}
		keys[i+1] = k;
		n = n+1;
	} else{  //il nodo non è foglia
		while (i >= 0 && keys[i] > k)
			i--;
		if (C[i+1]->n == 2*t-1){
			splitChild(i+1, C[i+1]);
			if (keys[i+1] < k)
				i++;
		}
		C[i+1]->insertNonFull(k);
	}
}

void BTreeNode::splitChild(int i, BTreeNode *y){
	BTreeNode *z = new BTreeNode(y->t, y->leaf);
	z->n = t - 1;
	for (int j = 0; j < t-1; j++)			//mette nel nuovo nodo tutte le chiavi che occupano le posizioni dal t in poi
		z->keys[j] = y->keys[j+t];
	if (y->leaf == false){
		for (int j = 0; j < t; j++)
			z->C[j] = y->C[j+t];
	}
	y->n = t - 1;							//reimposta a t-1 il numero di chiavi presenti nel nodo da splittare
	for (int j = n; j >= i+1; j--)			//sposta di 1 a destra tutti i nodi figli per far posto al nodo appena creato, che deve essere
		C[j+1] = C[j];						//l' (i+1)^ figlio
	C[i+1] = z;
	for (int j = n-1; j >= i; j--)			//sposta di 1 a destra tutte le chiavi per far posto a quella in posizione t-1 nel nodo da splittare,
		keys[j+1] = keys[j];				//che è centrale tra tale nodo e quello appena creato

	keys[i] = y->keys[t-1];
	n = n + 1;
}

BTreeNode *BTreeNode::search(int k){
	int i = 0;
	while (i < n && k > keys[i])   //cerca i tale che la chiave keys[i] >= k
		i++;
	if (keys[i] == k)			  //se k coincide con keys[i] la ricerca ha avuto successo
		return this;
	if (leaf == true)		 //se invece k è maggiore di keys[i] e il nodo non è foglia, allora cerca nel nodo di sinistra, che contiene le chiavi più piccole di keys[i]
		return NULL;
	return C[i]->search(k);
}

BTree::BTree(int _t){
	root = NULL;
	t = _t;
}

BTree::~BTree() {
	// TODO Auto-generated destructor stub
}


void BTree::traverse(){
	if (root != NULL)
		root->traverse();
}

BTreeNode* BTree::search(int k){
	return (root == NULL)? NULL : root->search(k);
}

void BTree::insert(int k){
	if (root == NULL){
		//Crea un nuovo nodo, memorizzandovi la chiave nella prima posizione
		root = new BTreeNode(t, true);
		root->keys[0] = k;
		root->n = 1;
	}
	else{
		if (root->n == 2*t-1){
			BTreeNode *s = new BTreeNode(t, false);
			s->C[0] = root;
			s->splitChild(0, root);
			int i = 0;
			if (s->keys[0] < k)
				i++;
			s->C[i]->insertNonFull(k);
			root = s;
		}
		else
			root->insertNonFull(k);  //il nodo non è pieno: la chiave va inserita in esso
	}
}











} /* namespace std */
