/*
 * mc_index.c
 *
 *  Created on: Nov 7, 2013
 *      Author: jyap
 */


#include <stdlib.h>
#include <mc_core.h>

int index_init(index_struct* index) {
	TAILQ_INIT(&index->lruq);
	return MC_OK;
}

int index_deinit(index_struct* index) {
	// TODO(jyap): cleanup.
	return MC_OK;
}

int index_insert(index_struct* index, struct item* it) {
//	TAILQ_INSERT_TAIL(&index->lruq, it, i_tqe);

	index_metadata* meta = (index_metadata*)item_get_metadata(it);
//	i_tqe = &(meta->i_tqe);

	meta->i_tqe.tqe_next = ((void *)0);
	meta->i_tqe.tqe_prev = (&index->lruq)->tqh_last;
	*(&index->lruq)->tqh_last = (it);
	(&index->lruq)->tqh_last = &(meta->i_tqe.tqe_next);

	return MC_OK;
}

int index_delete(index_struct* index, struct item* it) {
//	TAILQ_REMOVE(&index->lruq, it, i_tqe);

//	TAILQ_ENTRY(item) *i_tqe = &((index_metadata*)item_get_metadata(it))->i_tqe;
	index_metadata* meta = (index_metadata*)item_get_metadata(it);
	index_metadata* meta_next = NULL;

	void **oldnext = (void *)&(meta->i_tqe.tqe_next);
	void **oldprev = (void *)&(meta->i_tqe.tqe_prev);
	if (((meta->i_tqe.tqe_next)) != ((void *)0)) {
		meta_next = (index_metadata*)item_get_metadata(meta->i_tqe.tqe_next);
		meta_next->i_tqe.tqe_prev = meta->i_tqe.tqe_prev;
	} else {
		(&index->lruq)->tqh_last = meta->i_tqe.tqe_prev;
	}
	*meta->i_tqe.tqe_prev = (meta->i_tqe.tqe_next);
	(*oldnext) = (void *) ((void *)0);
	(*oldprev) = (void *) ((void *)0);

	return MC_OK;
}

struct item* index_get_next(index_struct* index, struct item* prev_item) {
//	return TAILQ_NEXT(prev_item, i_tqe);
	index_metadata* meta = (index_metadata*)item_get_metadata(prev_item);
	return meta->i_tqe.tqe_next;
}

struct item* index_get_lowest(index_struct* index) {
//	return TAILQ_FIRST(&index->lruq);
	return ((&index->lruq)->tqh_first);
}

int index_touch(struct item* it) {
	return MC_OK;
}

int index_set_cost(struct item* it, cost_t cost) {
	index_metadata* meta = (index_metadata*) item_get_metadata(it);
	meta->cost = cost;
	return MC_OK;
}
