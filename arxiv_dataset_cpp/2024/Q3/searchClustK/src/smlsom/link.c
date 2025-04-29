#include "link.h"

link *adjsNodes;
link availNodes;

void pushTo(link que, int node) {
  link new = (link)malloc(sizeof(list));
  new->node = node;
  new->next = que->next;
  que->next = new;
}

int getTopFrom(link que) {
  link l = que->next;
  if (l == NULL)
    return -1;

  int node = l->node;
  que->next = l->next;
  free(l);
  return node;
}

boolean getkFrom(link que, int k) {
  link l,bl;
  for (bl = que, l=que->next; l != NULL; bl=l,l = l->next) {
    if (l->node == k) {
      bl->next = l->next;
      free(l);
      return true;
    }
  }
  return false;
}

int getDim(link que) {
  link l = que;
  int dim = 0;
  while ((l = l->next) != NULL) dim++;
  return dim;
}

boolean isIn(link que, int k) {
  link l = que;
  while ((l = l->next) != NULL) {
    if (l->node == k)
      return true;
  }
  return false;
}

void free_que(link que) {
  while(getTopFrom(que) != -1) ;
  free(que);
}

link make_dummy() {
  link l = (link)malloc(sizeof(list));
  l->node = -1;
  l->next = NULL;
  return l;
}

void init_list(int nnode) {
  int i;

  adjsNodes = (link *)malloc(sizeof(link) * nnode);
  for (i = 0; i < nnode; i++)
    adjsNodes[i] = make_dummy();

  availNodes = make_dummy();
  for (i = nnode-1; i >= 0; i--)
    pushTo(availNodes,i);
}

void free_list(int nnode) {
  link bl,l;
  int i;
  for (i = 0; i < nnode; i++) {
    bl = adjsNodes[i];
    l = bl->next;
    while(l != NULL) {
      free(bl);
      bl = l; l = l->next;
    }
  }
  free(adjsNodes);

  while(getTopFrom(availNodes) != -1) ;
  free(availNodes);
}

// create direct edge k->node
void createEdge(int k, int node) {
  pushTo(adjsNodes[k],node);
}

boolean deleteEdge(int k, int node) {
  link l = adjsNodes[k];
  link bl;
  for (bl = l, l = l->next; l != NULL; bl = l, l = l->next) {
    if (l->node == node) {
      bl->next = l->next;
      free(l);
      return true;
    }
  }
  return false;
}

boolean deleteNode(int k) {
  if(!getkFrom(availNodes,k)) {
    fprintf(stderr,"node %d not available\n",k);
    return false;
  }

  link bl = adjsNodes[k];
  link l;
  while(bl->next != NULL) {
    l = bl->next;
    deleteEdge(l->node,k);
    bl->next = l->next;
    free(l);
  }
  return true;
}

void print_list(int nnode) {
  int i;
  link l;

  printf("**adjacents\n");
  for (i = 0; i < nnode; i++) {
    l = adjsNodes[i];
    printf("node %d : ", i);
    while((l = l->next) != NULL)
      printf("%d ",l->node);
    printf("\n");
  }

  printf("**available nodes : ");
  l = availNodes;
  while((l = l->next) != NULL)
    printf("%d ",l->node);
  printf("\n");
}

void print_que(link que) {
  link l = que;

  printf("print que : ");
  while((l = l->next) != NULL)
    printf("%d ",l->node);
  printf("\n");
}
