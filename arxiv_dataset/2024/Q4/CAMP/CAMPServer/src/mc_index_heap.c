/* heap_index.c */

/* This is an implementation of the mc_index.h interface using a GreedyDualSize
   eviction policy with a heap with a node for each cost/size value, the
   node pointing to an lru queue. */ 
#include <stdlib.h>
#include <math.h> /* round */
#include <mc_core.h>
#include <bit_ops.h>
#include <mc_index.h>
#include "implicit_heap.h"

static inline void update_minimum(index_struct* index);
static inline uint32_t index_priority(index_struct* index, struct item* it);
static inline void lru_append(struct lruq* queue, struct item* it);
static inline void lru_delete(struct lruq* queue, struct item* it);
static inline void item_set_prev(struct item* it, struct item** prev);
static inline void item_set_next(struct item* it, struct item* next);
static inline struct item** item_prev(struct item* it);
static inline struct item** item_next(struct item* it);
static inline uint32_t item_priority(struct item* it);
static inline uint64_t item_inflated_priority(struct item* it);

/* Public methods */

int index_init(index_struct* index) {
	/* range = max_priority / min_priority */
	uint32_t precision = 11;
	/* min cost/size as a fraction: numerator, denominator */
	uint32_t min_priority[] = { 40, 1 };
	/* max cost/size as a fraction: numerator, denominator */
	uint32_t max_priority[] = { 250000, 1 };

  /* range = max_priority / min_priority */
  index->range = (max_priority[0] * min_priority[1]) /
    (max_priority[1] * min_priority[0]) + 1;
  index->minimum = 0;
  index->precision = precision;
  index->min_priority[0] = min_priority[0];
  index->min_priority[1] = min_priority[1];
  index->queue = pq_create();
  index->lruqs = (struct lruq *) calloc(index->range, sizeof(struct lruq));
  if (index->lruqs == NULL)
    return MC_ERROR;
  uint32_t i;
  for (i = 0; i < index->range; i++)
    index->lruqs[i].tail = &index->lruqs[i].head;
  return MC_OK;
}

int index_deinit(index_struct* index) {
  pq_destroy(index->queue); 
  free(index->lruqs);
  return MC_OK;
}

int index_insert(index_struct* index, struct item* it) {
  index_metadata *meta = (index_metadata *) item_get_metadata(it);
  meta->priority = index_priority(index, it);
  meta->inflated_priority = meta->priority + index->minimum;
  struct lruq* lru = &index->lruqs[meta->priority];
  lru_append(lru, it);
  if (lru->node == NULL)
    lru->node = pq_insert(index->queue, lru, meta->inflated_priority);
  return MC_OK;
}

int index_delete(index_struct* index, struct item* it) {
  uint64_t old_priority, new_priority;
  struct lruq* lru = &index->lruqs[item_priority(it)];
  old_priority = item_inflated_priority(lru->head);
  lru_delete(lru, it);
  /* update heap if necessary */
  if (lru->head == NULL) {
    pq_delete(index->queue, lru->node);
    lru->node = NULL;
  } else {
    new_priority = item_inflated_priority(lru->head);
    if (new_priority != old_priority) {
      pq_increase_key(index->queue, lru->node, new_priority);
    }
  }
  update_minimum(index);
  return MC_OK;
}

struct item* index_get_next(index_struct* index, struct item* prev_item) {
  UNUSED(index);
  struct item** next = item_next(prev_item);
  /* try to return next item with same key */
  if (next != NULL)
    return *next;
  /* try to return strict successor of current key if any */
  uint32_t oldcost = item_priority(prev_item);
  uint32_t cost = oldcost;
  while (++cost < index->range && index->lruqs[cost].head == NULL);
  if (cost < index->range)
    return index->lruqs[cost].head;
  /* try to return minimum */
  cost = -1;
  while (++cost < oldcost && index->lruqs[cost].head == NULL);
  if (cost < oldcost)
    return index->lruqs[cost].head;
  return NULL; /* empty */
}

struct item* index_get_lowest(index_struct* index) {
  pq_node_type* node = pq_find_min(index->queue);
  if (node == NULL)
    return NULL;
  return pq_get_item(index->queue, node)->head;
}

int index_touch(struct item* it) {
  UNUSED(it);
  /* No-op. This technique does not require any action for index_touch. */
  return MC_OK;
}

/* Static methods */

static inline void update_minimum(index_struct* index) {
  if (pq_empty(index->queue))
    index->minimum = 0;
  else
    index->minimum = pq_get_key(index->queue, pq_find_min(index->queue));
}

static inline uint32_t index_priority(index_struct* index, struct item* it) {
  index_metadata* meta = (index_metadata*) item_get_metadata(it);
  /* calculate priority: (cost/size) / min_priority */
  uint32_t size = item_size(it);
  uint32_t priority = (meta->cost * index->min_priority[1]) / 
    (size * index->min_priority[0]);
  if (priority >= index->range)
    priority = index->range - 1;
  uint32_t p = index->precision;
  if (p != (uint32_t) -1) {
    /* round down, keeping only a number of bits specified by precision. */
    uint32_t m = bo_bits_in_32(priority);
    uint32_t n = (m > p) ? m - p : 0;
    priority = (priority >> n) << n;
  }
  return priority;
}

static inline void lru_append(struct lruq* queue, struct item* it) {
  index_metadata *meta = (index_metadata *) item_get_metadata(it);
  meta->next = NULL;
  meta->prev = queue->tail;
  *queue->tail = it;
  queue->tail = &meta->next;
}

static inline void lru_delete(struct lruq* queue, struct item* it) {
  index_metadata *meta = (index_metadata *) item_get_metadata(it);
  struct item** prev = meta->prev;
  struct item* next = meta->next;
  if (next != NULL)
    item_set_prev(next, prev);
  else
    queue->tail = prev;
  *prev = next;
}

static inline void item_set_prev(struct item* it, struct item** prev) {
  ((index_metadata*) item_get_metadata(it))->prev = prev;
}

static inline void item_set_next(struct item* it, struct item* next) {
  ((index_metadata*) item_get_metadata(it))->next = next;
}

static inline struct item** item_prev(struct item* it) {
	return ((index_metadata*) item_get_metadata(it))->prev;
}

static inline struct item** item_next(struct item* it) {
	return &((index_metadata*) item_get_metadata(it))->next;
}

static inline uint32_t item_priority(struct item* it) {
  return ((index_metadata*) item_get_metadata(it))->priority;
}

static inline uint64_t item_inflated_priority(struct item* it) {
  return ((index_metadata*) item_get_metadata(it))->inflated_priority;
}

int index_set_cost(struct item* it, cost_t cost) {
	index_metadata* meta = (index_metadata*) item_get_metadata(it);
	meta->cost = cost;
	return MC_OK;
}
