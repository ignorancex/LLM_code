/*----------------------------------------------------------------------------*/
/*------------------------------  PRINT QUEUE  -------------------------------*/
/*----------------------------------------------------------------------------*/

#include "plane.h"

/*----------------------------------------------------------------------------*/

void print_queue_init(print_queue_t *queue)
{
  queue->num_vecs = 0;
}

void print_queue_clear(print_queue_t *queue)
{
  queue->num_vecs = 0; // For safety
}

/*----------------------------------------------------------------------------*/

void print_queue_purge(print_queue_t *queue, FILE *fp)
{
  int i;
  // char s[128];
  coef_vec_t vv;
  for (i=0; i<queue->num_vecs; i++)
  {
    vv = (queue->vecs)[i];
    fprintf(fp,"%llu\n",(unsigned long long)vv);
  }
  queue->num_vecs = 0;
}

/*----------------------------------------------------------------------------*/

void print_queue_write(print_queue_t *queue, coef_vec_t v, FILE *fp)
{
  int ind = queue->num_vecs;
  (queue->vecs)[ind] = v;
  (queue->num_vecs)++;
  if (queue->num_vecs == QUEUE_VECS) print_queue_purge(queue,fp);
}

