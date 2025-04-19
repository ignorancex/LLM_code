/*
 * mc_memory.c
 *
 *  Created on: Nov 7, 2013
 *      Author: jyap
 */
#include <stdlib.h>
#include <mc_core.h>

//static uint64_t mem_allocatedbytes;
//static uint64_t mem_maxbytes;
//
//pthread_mutex_t mem_lock;          		/* lock protecting memory manager */

void mem_init(struct mem_instance* inst, size_t max_cachesize) {
	ASSERT(inst != NULL);

	/* Keep track of the current and max cache size */
	inst->allocated_bytes = 0;
	inst->max_bytes = max_cachesize;
}

void mem_deinit(struct mem_instance* inst) {
	ASSERT(inst != NULL);
	/* Empty for now. Using malloc/free */
}

struct item* mem_alloc(struct mem_instance* inst, size_t size) {
	ASSERT(inst != NULL);
	struct item* it = NULL;

	pthread_mutex_lock(&inst->lock);

	/* Check if the allocated space exceeds the max bytes allowed */
	if (inst->allocated_bytes + size > inst->max_bytes) {
		goto done;
	}

	/* Perform the actual allocation */
	it = (struct item*) malloc(size);

	/* Could not allocate memory */
	if (it == NULL) {
		goto done;
	}

	inst->allocated_bytes += size;

	/* Initialize item header */
	item_hdr_init(it, 0, 1);

done:
	pthread_mutex_unlock(&inst->lock);
	return it;
}

void mem_free(struct mem_instance* inst, struct item* it) {
	ASSERT(it != NULL);

	//Get size from: item_size(it);
	pthread_mutex_lock(&inst->lock);
	inst->allocated_bytes -= item_size(it);
	pthread_mutex_unlock(&inst->lock);

	/* De-allocate item and re-claim free space */
	free(it);
}
