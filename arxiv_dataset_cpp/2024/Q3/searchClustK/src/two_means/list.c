#include "list.h"

void pushTo(list que, int ind) {
  list new= (list)malloc(sizeof(list));
  new->ind = ind;
  new->next = que->next;
  que->next = new;
}

int getTopFrom(list que) {
  list l = que->next;
  if (l == NULL)
    return -1;

  int ind = l->ind;
  que->next = l->next;
  free(l);
  return ind;
}

void rmAllEnt(list que) {
  int i;
  while((i = getTopFrom(que)) != -1);
}

boolean getkFrom(list que, int k) {
  list l,bl;
  for (bl = que, l=que->next; l != NULL; bl=l,l = l->next) {
    if (l->ind == k) {
      bl->next = l->next;
      free(l);
      return true;
    }
  }
  return false;
}

int getDim(list que) {
  list l = que;
  int dim = 0;
  while ((l = l->next) != NULL) dim++;
  return dim;
}

boolean isIn(list que, int k) {
  list l = que;
  while ((l = l->next) != NULL) {
    if (l->ind == k)
      return true;
  }
  return false;
}

void free_que(list que) {
  while(getTopFrom(que) != -1) ;
  free(que);
}

list make_dummy() {
  list l = (list)malloc(sizeof(list));
  l->ind = -1;
  l->next = NULL;
  return l;
}

void print_que(list que) {
  list l = que;

  printf("print que : ");
  while((l = l->next) != NULL)
    printf("%d ",l->ind);
  printf("\n");
}

int list2vec(int *vec, list l)
{
  int i = 0;
  while((l = l->next) != NULL) {
    vec[i] = l->ind;
    i++;
  }
  return i;
}
