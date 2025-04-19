/* rtpq_index.c */

/* This is an implementation of the mc_index.h interface using a GreedyDualSize
   eviction policy with an approximate radix tree as the priority queue. */ 
/* Possible areas for speed improvement:
   1. replace bit_ops operations nlz and ntz by hardware operation on
      machines that support it.
   2. a lot of statements in rtpq_index.c are checks against invalid inputs.
      These can be replaced by assert statements after being plugged in
      to twemcache and after correct inputs are insured. Example: calling
      deletes on items not in the index */
#include <stdlib.h>
#include <math.h> /* round */
#include <mc_core.h>
#include <mc_items.h>
#include <bit_ops.h>
//#include <rtpq_index.h>
#include <stdio.h>

static void index_update_minimum(index_struct* index);
static uint32_t index_priority(index_struct* index, struct item* it);

/* range: upper bound on possible values of cost/size
 * precision: number of bits used the queue will use to keep track of the
 * cost/size values. For example:
 *   precision = 0: all items have same cost/size values, imprecise queue
 *   precision = #bits in range: cost/size faithfully represented, exact queue
 */
int index_init(index_struct* index) {
	/* range = max_priority / min_priority */
	uint32_t precision = 8;
	/* min cost/size as a fraction: numerator, denominator */
	uint32_t min_priority[] = { 40, 1 };
	/* max cost/size as a fraction: numerator, denominator */
	uint32_t max_priority[] = { 250000, 1 };

	index->range = (max_priority[0] * min_priority[1]) /
			(max_priority[1] * min_priority[0]);
	index->minimum = 0;
	index->precision = precision;
	index->min_priority[0] = min_priority[0];
	index->min_priority[1] = min_priority[1];
	int status = rtpq_init(&index->queue, bo_bits_in_32(index->range));
	//printf("range: %u, min: %u, precision: %u\n", index->range, index->minimum, index->precision);
	return (status == RTPQ_OK) ? MC_OK : MC_ERROR;
}

int index_deinit(index_struct* index) {
  int status = rtpq_deinit(&index->queue); 
  return (status == RTPQ_OK) ? MC_OK : MC_ERROR;
}

int index_insert(index_struct* index, struct item* it) {
  /* insert into queue and save insertion point */
  uint32_t priority = index_priority(index, it);
  ((index_metadata*) item_get_metadata(it))->location
    = rtpq_insert(&index->queue, priority, it);
  //printf("insert priority: %u\n", priority);
  return MC_OK;
}

int index_delete(index_struct* index, struct item* it) {
  index_metadata* meta = (index_metadata*)item_get_metadata(it);
  if (rtpq_empty(&index->queue) == RTPQ_ERROR)
    return MC_ERROR;
  /* delete from queue */
  if (rtpq_delete(&index->queue, meta->location) == RTPQ_ERROR)
    return MC_ERROR;
  //printf("delete %s, priority %u\n", item_key(it), meta->location.key);
  /* update minimum value */
  index_update_minimum(index);
  return MC_OK;
}

struct item* index_get_next(index_struct* index, struct item* prev_item) {
  UNUSED(index);
  rtpq_keyval kv;
  index_metadata* meta = (index_metadata*)item_get_metadata(prev_item);
  /* try to return next item with same key */
  if (meta->next != NULL)
    return meta->next;
  /* try to return strict successor of current key if any */
  if (rtpq_successor(&index->queue, meta->location.key, &kv, 1) == RTPQ_OK) {
    if (meta->location.key < index->minimum && index->minimum <= kv.key)
      return NULL;
    return kv.value;
  }
  /* try to return minimum */
  if (rtpq_minimum(&index->queue, &kv) == RTPQ_OK) {
    if (index->minimum <= kv.key)
      return NULL;
    return kv.value;
  }
  return NULL; /* empty */
}

struct item* index_get_lowest(index_struct* index) {
  rtpq_keyval kv;
  if (rtpq_successor(&index->queue, index->minimum, &kv, 0) == RTPQ_OK)
    return kv.value;
  if (rtpq_minimum(&index->queue, &kv) == RTPQ_OK)
    return kv.value;
  return NULL;
}

int index_touch(struct item* it) {
  UNUSED(it);
  /* No-op. This technique does not require any action for index_touch. */
  return MC_OK;
}

void index_print(index_struct* index, FILE *out, uint32_t flags) {
  rtpq_print(&index->queue, out, flags);
}

static void index_update_minimum(index_struct* index) {
  //uint32_t oldmin = index->minimum;
  rtpq_keyval kv;
  if (rtpq_successor(&index->queue, index->minimum, &kv, 0) == RTPQ_OK)
    index->minimum = kv.key;
  else if (rtpq_minimum(&index->queue, &kv) == RTPQ_OK)
    index->minimum = kv.key;
  else 
    index->minimum = 0;
  //if (index->minimum != oldmin)
    //printf("min: %u\n", index->minimum);
}

static uint32_t index_priority(index_struct* index, struct item* it) {
  index_metadata* meta = (index_metadata*) item_get_metadata(it);
//  uint32_t size = meta->size;
  uint32_t size = item_size(it);
  /* calculate priority: (cost/size) / min_priority */
  uint32_t priority = (meta->cost * index->min_priority[1]) /
    (size * index->min_priority[0]);
  if (priority > index->range)
    priority = index->range;
  /* SIC: precision is assumed to be between -1 and 10, -1 for no rounding */
  uint32_t p = index->precision;
  if (p != -1) {
    /* round down, keeping only a number of bits specified by precision. */
    uint32_t m = bo_bits_in_32(priority);
    uint32_t n = (m > p) ? m - p : 0;
    priority = (priority >> n) << n;
  }
  return (priority + index->minimum) % (index->range + 1);
}

int index_set_cost(struct item* it, cost_t cost) {
	index_metadata* meta = (index_metadata*) item_get_metadata(it);
	meta->cost = cost;
	return MC_OK;
}
