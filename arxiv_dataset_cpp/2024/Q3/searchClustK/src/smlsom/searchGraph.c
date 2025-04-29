#include "link.h"
#include "searchGraph.h"

void bfs(int node, int depth, link neighs, int nnode) {
  int k,d;
  link l;
  link que1 = make_dummy();
  link que2 = make_dummy();
  boolean *chk_array = (boolean *)malloc(nnode * sizeof(boolean));
  for (k = 0; k < nnode; k++)
    chk_array[k] = false;

  d = 0;
  pushTo(que2,node);
  chk_array[node] = true;
  while(que2->next != NULL && d <= depth) {
    d++;
    que1->next = que2->next;
    que2->next = NULL;
    while(que1->next != NULL) {
      k = getTopFrom(que1);
      pushTo(neighs,k);
      l = adjsNodes[k];
      while((l = l->next) != NULL) {
        if (!chk_array[l->node]) {
          pushTo(que2,l->node);
          chk_array[l->node] = true;
        }
      }
    }
  }
  free(que1); free_que(que2);
  free(chk_array);
}

int bfs_dist(int node, int nnode) {
  int k,d;
  link l;
  link que1 = make_dummy();
  link que2 = make_dummy();
  boolean *chk_array = (boolean *)malloc(nnode * sizeof(boolean));
  for (k = 0; k < nnode; k++)
    chk_array[k] = false;

  d = 0;
  pushTo(que2,node);
  chk_array[node] = true;
  while(que2->next != NULL) {
    d++;
    que1->next = que2->next;
    que2->next = NULL;
    while(que1->next != NULL) {
      k = getTopFrom(que1);
      l = adjsNodes[k];
      while((l = l->next) != NULL) {
        if (!chk_array[l->node]) {
          pushTo(que2,l->node);
          chk_array[l->node] = true;
        }
      }
    }
  }
  free(que1); free_que(que2);
  free(chk_array);

  return d;
}

void graph2adjmat(int *adjmatrix, int M) {
  int m,i,j;
  link li;

  for (m = 0; m < M; m++) {
    if (getkFrom(availNodes,m)) i = 0; else i = -1;
    for (j = 0; j < M; j++) {
      adjmatrix[m*M + j] = i;
    }
    li = adjsNodes[m];
    while ((li = li->next) != NULL) {
      j = li->node;
      adjmatrix[m*M + j] = 1;
    }
  }
}
