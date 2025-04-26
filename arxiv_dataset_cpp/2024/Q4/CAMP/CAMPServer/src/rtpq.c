/* rtpq.c */

#include <stdlib.h>     /* malloc */
#include <math.h>       /* pow, sqrt */
#include <mc_index.h> /* rtpq_index_metadata */
#include <bit_ops.h>
#include <rtpq.h>



/* Private rtpq functions */
static inline uint32_t rtpq_child(uint32_t parent, uint32_t rank);
static inline uint32_t rtpq_parent(uint32_t child);
static inline uint32_t rtpq_rank(uint32_t index);
static        void     rtpq_update_tree(rtpq* pq, uint32_t key);
static        void     rtpq_print_contents(rtpq* pq, FILE *stream);
static        void     rtpq_print_distrib(rtpq* pq, FILE *stream, bool verbose);


/* rtpq_node interface as used by rtpq */
static inline rtpq_node* rtpq_node_prev(rtpq_node* node);
static inline rtpq_node* rtpq_node_next(rtpq_node* node);
static inline void       rtpq_node_set_prev(rtpq_node* node, rtpq_node* prev);
static inline void       rtpq_node_set_next(rtpq_node* node, rtpq_node* next);
static inline rtpq_value rtpq_node_item(rtpq_node* node);
static inline rtpq_node* rtpq_node_make_node(rtpq_value value);
static inline char*      rtpq_node_string(rtpq_node* node);



/* Allocator and initializer with argument specifying the number of keys. */
int rtpq_init(rtpq* pq, uint32_t num_bits_range) { 
  int length;
  uint32_t level;
  int i;


  /* require that num_bits_range be non-zero to have at least 1 tree level */
  if (num_bits_range == 0)
    return RTPQ_ERROR;

  /* SUBTLE: ceil( num_bits_range / BO_LOG_WORD_SIZE ) */
  pq->max_level = (num_bits_range - 1) / BO_LOG_WORD_SIZE + 1;

  /* initialize tree_levels */
  pq->tree_levels = (uint32_t**) malloc(sizeof(uint32_t*) * pq->max_level);
  if (pq->tree_levels == NULL)
    return RTPQ_ERROR;
 for (level = 0, length = 1; level < pq->max_level; level++) {
    pq->tree_levels[level] = (uint32_t*) malloc(sizeof(uint32_t) * length);
    if (pq->tree_levels[level] == NULL)
      goto error;
    for (i = 0; i < length; i++)
      pq->tree_levels[level][i] = 0;
    length <<= BO_LOG_WORD_SIZE; /* length = 32^level */
  }

  /* initialize leaf_level */
  pq->leaf_level = (rtpq_node**) malloc(sizeof(rtpq_node*) * 2*length);
  if (pq->leaf_level == NULL) {
    length >>= BO_LOG_WORD_SIZE;
    level--;
    goto error;
  }
  for (i = 0; i < 2*length; i++)
    pq->leaf_level[i] = NULL;
  pq->leaf_level_tail = &pq->leaf_level[length];
  return RTPQ_OK;

error:
  for (; level > pq->max_level; level--)
    free(pq->tree_levels[level]);
  free(pq->tree_levels);
  return RTPQ_ERROR;
}



/* Deallocator. */
int rtpq_deinit(rtpq* pq) {
  uint32_t level;
  free(pq->leaf_level);
  for (level = 0; level < pq->max_level; level++)
    free(pq->tree_levels[level]);
  free(pq->tree_levels);
  return RTPQ_OK;
}



/* Insert. */
rtpq_location rtpq_insert(rtpq* pq, rtpq_key key, rtpq_value value) {
  rtpq_node* new_node = rtpq_node_make_node(value);
  if (pq->leaf_level[key] == NULL)
    pq->leaf_level[key] = new_node;
  else
    rtpq_node_set_next(pq->leaf_level_tail[key], new_node);
  rtpq_node_set_prev(new_node, pq->leaf_level_tail[key]);
  pq->leaf_level_tail[key] = new_node;
  rtpq_update_tree(pq, key);
  rtpq_location loc = { key, new_node };
  return loc;
}



/* Delete the item at given location. */
int rtpq_delete(rtpq* pq, rtpq_location location) {
  ASSERT(pq->leaf_level[location.key] != NULL);
  rtpq_node* next = rtpq_node_next(location.node);
  rtpq_node* prev = rtpq_node_prev(location.node);
  if (prev == NULL) {
	  ASSERT(pq->leaf_level[location.key] == location.node);
    pq->leaf_level[location.key] = next;
  } else {
    rtpq_node_set_next(prev, next);
  }
  if (next == NULL) {
	  ASSERT(pq->leaf_level_tail[location.key] == location.node);
    pq->leaf_level_tail[location.key] = prev;
  } else {
    rtpq_node_set_prev(next, prev);
  }
  rtpq_update_tree(pq, location.key);
  return RTPQ_OK;
}



/* Minimum. */
int rtpq_minimum(rtpq* pq, rtpq_keyval* kv) {
  if (rtpq_empty(pq))
    return RTPQ_ERROR;

  uint32_t level;
  uint32_t word;
  uint32_t index = 0;

  for (level = 0; level < pq->max_level; level++) {
    word = pq->tree_levels[level][index];
    index = rtpq_child(index, bo_lsb(word));
  }
  kv->key = index;
  kv->value = rtpq_node_item(pq->leaf_level[index]);
  return RTPQ_OK;
}



/* Maximum. */
int rtpq_maximum(rtpq* pq, rtpq_keyval* kv) {
  if (rtpq_empty(pq)) 
    return RTPQ_ERROR;

  uint32_t level;
  uint32_t word;
  uint32_t index = 0;

  for (level = 0; level < pq->max_level; level++) {
    word = pq->tree_levels[level][index];
    index = rtpq_child(index, bo_msb(word));
  }
  kv->key = index;
  kv->value = rtpq_node_item(pq->leaf_level[index]);
  return RTPQ_OK;
}



/* Successor. */
int rtpq_successor(rtpq* pq, rtpq_key key, rtpq_keyval* kv, bool strict) {
  /* if there is a value with this key, return that value */
  if (pq->leaf_level[key] != NULL && !strict) {
    kv->key = key;
    kv->value = rtpq_node_item(pq->leaf_level[key]);
    return RTPQ_OK;
  }

  /* ascend tree until ancestor has a non-empty successor subtree */
  uint32_t level;
  uint32_t rank;
  uint32_t index = key;
  uint32_t word;
  uint32_t sibling_rank = BO_NULL_BIT_POS;
  for (level = pq->max_level - 1; level < pq->max_level; level--) {
    rank = rtpq_rank(index);
    index = rtpq_parent(index);
    word = pq->tree_levels[level][index];
    sibling_rank = bo_nbl(word, rank);
    if (sibling_rank != BO_NULL_BIT_POS)
      break;
  }

   /* check for unsuccessful search */
  if (sibling_rank == BO_NULL_BIT_POS)
    return RTPQ_ERROR;

  /* find minimum in this subtree */
  index = rtpq_child(index, sibling_rank); 
  /* -- set level back to child's level */
  for (level += 1; level < pq->max_level; level++) {
    word = pq->tree_levels[level][index];
    rank = bo_lsb(word);
    index = rtpq_child(index, rank);
  }
  kv->key = index;
  kv->value = rtpq_node_item(pq->leaf_level[index]);
  return RTPQ_OK;
}



/* Predecessor. */
int rtpq_predecessor(rtpq* pq, rtpq_key key, rtpq_keyval* kv, bool strict) {
  /* if there is a value with this key, return that value */
  if (pq->leaf_level[key] != NULL && !strict) {
    kv->key = key;
    kv->value = rtpq_node_item(pq->leaf_level[key]);
    return RTPQ_OK;
  }

  /* ascend tree until ancestor has a non-empty predecessor subtree */
  uint32_t level;
  uint32_t rank;
  uint32_t index = key;
  uint32_t word;
  uint32_t sibling_rank = BO_NULL_BIT_POS;
  for (level = pq->max_level - 1; level < pq->max_level; level--) {
    rank = rtpq_rank(index);
    index = rtpq_parent(index);
    word = pq->tree_levels[level][index];
    sibling_rank = bo_nbr(word, rank);
    if (sibling_rank != BO_NULL_BIT_POS)
      break;
  }

  /* check for unsuccessful search */
  if (sibling_rank == BO_NULL_BIT_POS)
    return RTPQ_ERROR;

  /* find maximum in this subtree */
  index = rtpq_child(index, sibling_rank); 
  /* -- set level back to child's level */
  for (level += 1; level < pq->max_level; level++) {
    word = pq->tree_levels[level][index];
    rank = bo_msb(word);
    index = rtpq_child(index, rank);
  }
  kv->key = index;
  kv->value = rtpq_node_item(pq->leaf_level[index]);
  return RTPQ_OK;
}



/* Return whether the queue is empty or not. */
inline bool rtpq_empty(rtpq* pq) {
  return pq->tree_levels[0][0] == 0; 
}



/* Print to stream a representation of the contents of the container. */
void rtpq_print(rtpq* pq, FILE *stream, uint32_t flags) {
  if ((flags & 1) != 0)
    rtpq_print_contents(pq, stream);
  if ((flags & 2) != 0)
    rtpq_print_distrib(pq, stream, 1);
  if ((flags & 4) != 0)
    rtpq_print_distrib(pq, stream, 0);
}



/* Implementation of private rtpq functions */



/* Return the index of a node given its parent's index and its rank
   among its siblings. */
static inline uint32_t rtpq_child(uint32_t parent, uint32_t rank) {
  return (parent << BO_LOG_WORD_SIZE) + rank;
}



/* Return the index of a node given the index of one of its children. */
static inline uint32_t rtpq_parent(uint32_t child) {
  return child >> BO_LOG_WORD_SIZE;
}



/* Return the number representing the position of a node among its siblings. */
static inline uint32_t rtpq_rank(uint32_t index) {
  return index & (BO_WORD_SIZE - 1);
}



/* Update the internal nodes of the tree on the path from the leaf node at
   the given key to the root node. */
static void rtpq_update_tree(rtpq* pq, uint32_t key) {
  /* update the parent of the leaf node */
  uint32_t level = pq->max_level - 1;
  uint32_t child_index = key;
  uint32_t child_rank = rtpq_rank(child_index);
  uint32_t index = rtpq_parent(child_index);
  uint32_t* word = &pq->tree_levels[level][index];
  /* test if child bit in parent word is consistent with whether child is
     empty */
  bool contains_key = pq->leaf_level[key] != NULL;
  if (contains_key == bo_bit(*word, child_rank))
    return;
  *word = bo_toggle(*word, child_rank);
  level--;

  /* update the ancestors of the leaf node (besides parent) */
  uint32_t* child_word;
  while (level < pq->max_level) {   /* while(level >= 0) if level were an int */
    child_word = word;
    child_index = index;
    child_rank = rtpq_rank(child_index);
    index = rtpq_parent(child_index);
    word = &pq->tree_levels[level][index];
    /* test if child bit in parent word is consistent with whether child 
       word is 0 */
    if ((*child_word != 0) == bo_bit(*word, child_rank))
      return;
    *word = bo_toggle(*word, child_rank);
    level--;
  }
}



/* Print to stream a representation of the contents of the container. */
static void rtpq_print_contents(rtpq* pq, FILE *stream) {
  int i;
  uint32_t level;
  int length;
  rtpq_node* node;
//  char b[BO_WORD_SIZE + 1];

  /* print the nonleaf nodes */
  for (level = 0, length = 1; level < pq->max_level; level++) {
//    fprintf(stream, "level %d: ", level);
//    for (i = 0; i < length; i++) {
//      bo_binary_repr(pq->tree_levels[level][i], b);
//      fprintf(stream, "%s, ", b);
//    }
    length <<= BO_LOG_WORD_SIZE;
//    fprintf(stream, "\n\n");
  }

  /* print the contents of non-empty leaf nodes */
  for (i = 0; i < length; i++) {

    if ((node = pq->leaf_level[i]) == NULL)
      continue;
    fprintf(stream, "%u: ", i);
    while (node != NULL) {
      fprintf(stream, "%s, ", rtpq_node_string(node));
      node = rtpq_node_next(node);
    }
    fprintf(stream, "\n");
  }
  fprintf(stream, "\n");
}



/* Print to stream the distribution of items over the keys. */
static void rtpq_print_distrib(rtpq* pq, FILE *stream, bool verbose) {
  /* num_buckets = WORD_SIZE ^ max_level */
  int num_buckets = 1 << (BO_LOG_WORD_SIZE * pq->max_level); 
  int bucket_count[num_buckets];
  int i, j, max_count, sum_bucket_count, num_nonempty;
  rtpq_node* n;

  /* count number of items in each bucket */
  sum_bucket_count = 0;
  num_nonempty = 0;
  for (i = 0, max_count = 0; i < num_buckets; i++) {
    j = 0;
    n = pq->leaf_level[i];
    while (n != NULL) {
      j++;
      n = rtpq_node_next(n);
    }
    bucket_count[i] = j;
    sum_bucket_count += j;
    if (j > 0) num_nonempty++; 
    max_count = max_count < j ? j : max_count;
  }

  /* create histogram */
  int count_bucket[max_count];
  for (i = 0; i < max_count; i++)
    count_bucket[max_count] = 0;
  for (i = 0; i < num_buckets; i++)
    count_bucket[bucket_count[i]]++;

  /* print number of buckets (if > 0) with a given number of items */
  if (verbose) {
    fprintf(stream, "number of items, number of buckets with that many items\n");
    for (i = 0; i < max_count; i++)
      if (count_bucket[i] > 0) {
        fprintf(stream, "%10d, %10d\n", i, count_bucket[i]);
      }
    fprintf(stream, "number of buckets used: %d\n", num_nonempty);
  } else
    /* max number of items any bucket has */
    fprintf(stream, ",%d", max_count);
    /* mean and standard deviation of number of items among non-empty buckets */
    double average = ((double) sum_bucket_count) / num_nonempty;
    double variance = 0;
    for (i = 0; i < num_buckets; i++)
      if (bucket_count[i] > 0)
        variance += pow(bucket_count[i] - average, 2);
    double stddev = sqrt(variance / num_nonempty);
    fprintf(stream, ",%.0f,%.0f,%d", average, stddev, num_nonempty);
}




/* Implementation of rtpq_node interface with rtpq_node = struct item */

static inline rtpq_node* rtpq_node_prev(rtpq_node* node) {
	return ((rtpq_index_metadata*)item_get_metadata(node))->prev;
}

static inline rtpq_node* rtpq_node_next(rtpq_node* node) {
	return ((rtpq_index_metadata*)item_get_metadata(node))->next;
}

static inline void rtpq_node_set_prev(rtpq_node* node, rtpq_node* prev) {
	((rtpq_index_metadata*)item_get_metadata(node))->prev = prev;
}

static inline void rtpq_node_set_next(rtpq_node* node, rtpq_node* next) {
	((rtpq_index_metadata*)item_get_metadata(node))->next = next;
}

static inline rtpq_value rtpq_node_item(rtpq_node* node) {
  return node;
}

static inline rtpq_node* rtpq_node_make_node(rtpq_value value) {
	rtpq_index_metadata* meta = (rtpq_index_metadata*)item_get_metadata(value);
  meta->prev = NULL;
  meta->next = NULL;
  return value;
}

static inline char* rtpq_node_string(rtpq_node* node) {
  return item_key(node);
}
