/*
 * twemcache - Twitter memcached.
 * Copyright (c) 2012, Twitter, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of the Twitter nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>
#include <stdio.h>

#include <mc_core.h>
#include <mc_sqltrig.h>

extern struct settings settings;

/*
 * We only reposition items in the lru q if they haven't been
 * repositioned in this many seconds. That saves us from churning
 * on frequently-accessed items
 */
#define ITEM_UPDATE_INTERVAL    60

#define ITEM_LRUQ_MAX_TRIES     50

/* 2MB is the maximum response size for 'cachedump' command */
#define ITEM_CACHEDUMP_MEMLIMIT (2 * MB)

pthread_mutex_t cache_lock;                     /* lock protecting lru q and hash */
struct item_tqh item_lruq[SLABCLASS_MAX_IDS];   /* lru q of items */
struct item_tqh reserved_item_lruq[SLABCLASS_MAX_IDS];   /* lru q of reserved items */
static uint64_t cas_id;                         /* unique cas id */

struct mem_instance memory_manager;
struct mem_instance reserved_memory_manager;

index_struct p_index;
index_struct reserved_p_index;

#define QUARANTINE_LIST_MAX		100


/*
 * Returns the next cas id for a new item. Minimum cas value
 * is 1 and the maximum cas value is UINT64_MAX
 */
static uint64_t
item_next_cas(void)
{
    if (settings.use_cas) {
        return ++cas_id;
    }

    return 0ULL;
}

static bool
item_expired(struct item *it)
{
    ASSERT(it->magic == ITEM_MAGIC);

    return (it->exptime > 0 && it->exptime < time_now()) ? true : false;
}

void
item_init(void)
{
	int ret = -1;

    log_debug(LOG_DEBUG, "item hdr size %d", ITEM_HDR_SIZE);

    pthread_mutex_init(&cache_lock, NULL);

//    uint8_t i;
//    for (i = SLABCLASS_MIN_ID; i <= SLABCLASS_MAX_ID; i++) {
//        TAILQ_INIT(&item_lruq[i]);
//        TAILQ_INIT(&reserved_item_lruq[i]);
//    }

    ret = index_init(&p_index);
    if (ret != MC_OK) {
    	log_debug(LOG_ERR, "error initializing index. Code: %d", ret);
    }

    ret = index_init(&reserved_p_index);
    if (ret != MC_OK) {
    	log_debug(LOG_ERR, "error initializing reserved index. Code: %d", ret);
    }

    mem_init(&memory_manager, settings.maxbytes);
    mem_init(&reserved_memory_manager, settings.reserved_maxbytes);

    cas_id = 0ULL;
}

void
item_deinit(void)
{
	// TODO(jyap): Free up all allocated items. Clean the index structure?

	index_deinit(&p_index);
	mem_deinit(&memory_manager);
	mem_deinit(&reserved_memory_manager);
}

inline char *
item_key(struct item *it)
{
    char *key;

    ASSERT(it->magic == ITEM_MAGIC);

    key = it->end;
    if (item_has_cas(it)) {
        key += sizeof(uint64_t);
    }

    key += sizeof(index_metadata);

    return key;
}

/*
 * Get start location of item payload
 */
char *
item_data(struct item *it)
{
    char *data;

    ASSERT(it->magic == ITEM_MAGIC);

//    if (item_is_raligned(it)) {
//        data = (char *)it + slab_item_size(it->id) - it->nbyte;
//    } else {
        data = it->end + it->nkey + 1; /* 1 for terminal '\0' in key */
        if (item_has_cas(it)) {
            data += sizeof(uint64_t);
        }

        data += sizeof(index_metadata);
//    }

    return data;
}

/*
 * Get the slab that contains this item.
 */
struct slab *
item_2_slab(struct item *it)
{
    struct slab *slab;

    ASSERT(it->magic == ITEM_MAGIC);
    ASSERT(it->offset < settings.slab_size);

    slab = (struct slab *)((uint8_t *)it - it->offset);

    ASSERT(slab->magic == SLAB_MAGIC);

    return slab;
}

static void
item_acquire_refcount(struct item *it)
{
    ASSERT(pthread_mutex_trylock(&cache_lock) != 0);
    ASSERT(it->magic == ITEM_MAGIC);

    it->refcount++;
//    slab_acquire_refcount(item_2_slab(it));
}

static void
item_release_refcount(struct item *it)
{
    ASSERT(pthread_mutex_trylock(&cache_lock) != 0);
    ASSERT(it->magic == ITEM_MAGIC);
    ASSERT(it->refcount > 0);

    it->refcount--;
//    slab_release_refcount(item_2_slab(it));
}

void
item_hdr_init(struct item *it, uint32_t offset, uint8_t id)
{
//    ASSERT(offset >= SLAB_HDR_SIZE && offset < settings.slab_size);

    it->magic = ITEM_MAGIC;
    it->offset = offset;
    it->id = id;
    it->refcount = 0;
    it->flags = 0;
}

/*
 * Add an item to the tail of the priority index.
 *
 * The index is sorted in ascending priority order - lowest to most highest.
 * For LRU, enqueuing at item to the tail of the lru q requires us to update
 * its last access time atime.
 *
 * The allocated flag indicates whether the item being re-linked is a newly
 * allocated or not. This is useful for updating the slab lruq, which can
 * choose to update only when a new item has been allocated (write-only) or
 * the opposite (read-only), or on both occasions (access-based).
 */
static void
item_link_q(struct item *it, bool allocated)
{
    uint8_t id = it->id;

    ASSERT(id >= SLABCLASS_MIN_ID && id <= SLABCLASS_MAX_ID);
    ASSERT(it->magic == ITEM_MAGIC);
    ASSERT(!item_is_slabbed(it));

    log_debug(LOG_VVERB, "link q it '%.*s' at offset %"PRIu32" with flags "
              "%02x id %"PRId8"", it->nkey, item_key(it), it->offset,
              it->flags, it->id);

    it->atime = time_now();
    if (item_is_lease_holder(it)) {
//    	TAILQ_INSERT_TAIL(&reserved_item_lruq[id], it, i_tqe);
//    	slab_reserved_lruq_touch(item_2_slab(it), allocated);
    	if (index_insert(&reserved_p_index, it) != MC_OK) {
    		log_debug(LOG_ERR, "error inserting it '%.*s' to reserved index",
    				it->nkey, item_key(it));
    	}
    } else {
//    	TAILQ_INSERT_TAIL(&item_lruq[id], it, i_tqe);
//    	slab_lruq_touch(item_2_slab(it), allocated);
    	if (index_insert(&p_index, it) != MC_OK) {
    		log_debug(LOG_ERR, "error inserting it '%.*s' to index",
    				it->nkey, item_key(it));
    	}
    }

    stats_slab_incr(id, item_curr);
    stats_slab_incr_by(id, data_curr, item_size(it));
    stats_slab_incr_by(id, data_value_curr, it->nbyte);
}

/*
 * Remove the item from the priority index.
 */
static void
item_unlink_q(struct item *it)
{
    uint8_t id = it->id;

    ASSERT(id >= SLABCLASS_MIN_ID && id <= SLABCLASS_MAX_ID);
    ASSERT(it->magic == ITEM_MAGIC);

    log_debug(LOG_VVERB, "unlink q it '%.*s' at offset %"PRIu32" with flags "
              "%02x id %"PRId8"", it->nkey, item_key(it), it->offset,
              it->flags, it->id);

    if(item_is_lease_holder(it)) {
//    	TAILQ_REMOVE(&reserved_item_lruq[id], it, i_tqe);
    	if (index_delete(&reserved_p_index, it) != MC_OK) {
    		log_debug(LOG_ERR, "error deleting it '%.*s' from reserved index",
    				it->nkey, item_key(it));
    	}
    } else {
//    	TAILQ_REMOVE(&item_lruq[id], it, i_tqe);
    	if (index_delete(&p_index, it) != MC_OK) {
    		log_debug(LOG_ERR, "error deleting it '%.*s' from index",
    				it->nkey, item_key(it));
    	}
    }

    stats_slab_decr(id, item_curr);
    stats_slab_decr_by(id, data_curr, item_size(it));
    stats_slab_decr_by(id, data_value_curr, it->nbyte);
}


/*
 * Make an item with zero refcount available for reuse by unlinking
 * it from the lru q and hash.
 *
 * Don't free the item yet because that would make it unavailable
 * for reuse.
 */
void
item_reuse(struct item *it)
{
    ASSERT(pthread_mutex_trylock(&cache_lock) != 0);
    ASSERT(it->magic == ITEM_MAGIC);
    ASSERT(!item_is_slabbed(it));
    ASSERT(item_is_linked(it));

    if (item_is_lease_holder(it)) {
    	ASSERT(it->refcount == 1);
    	item_unset_pinned(it);
    } else
    	ASSERT(it->refcount == 0);

    it->flags &= ~ITEM_LINKED;

    assoc_delete(item_key(it), it->nkey);
    item_unlink_q(it);

    stats_slab_incr(it->id, item_remove);
    stats_slab_settime(it->id, item_reclaim_ts, time_now());

    stats_thread_incr(items_unlink);

    log_debug(LOG_VERB, "reuse %s it '%.*s' at offset %"PRIu32" with id "
              "%"PRIu8" refcount ""%"PRIu32"", item_expired(it) ? "expired" : "evicted",
              it->nkey, item_key(it), it->offset, it->id, it->refcount);
}

/*
 * Find an unused (unreferenced) item from lru q.
 *
 * First try to find an expired item from the lru Q of item's slab
 * class; if all items are unexpired, return the one that is the
 * least recently used.
 *
 * We bound the search for an expired item in lru q, by only
 * traversing the oldest ITEM_LRUQ_MAX_TRIES items.
 */
static struct item *
item_get_from_index(uint8_t id, index_struct *index_ptr)
{
    struct item *it;  /* expired item */
    struct item *uit; /* unexpired item */
    uint32_t tries;

    if (!settings.use_lruq) {
        return NULL;
    }

//    for (tries = ITEM_LRUQ_MAX_TRIES, it = TAILQ_FIRST(&lruq[id]),
//         uit = NULL;
//         it != NULL && tries > 0;
//         tries--, it = TAILQ_NEXT(it, i_tqe)) {
    uit = NULL;
    it = index_get_lowest(index_ptr);
    for (tries = 0;
    		it != NULL && tries < ITEM_LRUQ_MAX_TRIES;
    		tries++) {

//    	log_debug(LOG_VERB, "|| get it '%.*s' from LRU slab %"PRIu8, it->nkey, item_key(it), it->id);

		/* Skip an un-pinned item if it's refcount is larger than 0.
		 * Skip a pinned item if it's refcount is larger than 1 (pinned
		 *  items always have 1 refcount to prevent the slab from being
		 *  evicted).
		 */
		if (it->refcount != 0) {
			if (!item_is_lease_holder(it)) {
			log_debug(LOG_VERB, "skip it '%.*s' at offset %"PRIu32" with "
					  "flags %02x id %"PRId8" refcount %"PRIu16"", it->nkey,
					  item_key(it), it->offset, it->flags, it->id,
					  it->refcount);
			}
			if (item_is_lease_holder(it) && it->refcount >= 2) {
				log_debug(LOG_VERB, "skip it '%.*s' at offset %"PRIu32" with "
						  "flags %02x id %"PRId8" refcount %"PRIu16"", it->nkey,
						  item_key(it), it->offset, it->flags, it->id,
						  it->refcount);
			}
			continue;
		}

		if (item_expired(it)) {
			/* first expired item wins */
			return it;
		} else if (uit == NULL) {
			/* otherwise, get the lru unexpired item */
			uit = it;
			break;
		}

		it = index_get_next(index_ptr, it);
    }

    return uit;
}

inline size_t
item_ntotal(uint8_t nkey, uint32_t nbyte, bool use_cas)
{
    size_t ntotal;

    ntotal = use_cas ? sizeof(uint64_t) : 0;
    ntotal += ITEM_HDR_SIZE + nkey + 1 + nbyte + CRLF_LEN +
    		sizeof(index_metadata);

    return ntotal;
}

char* item_get_metadata(struct item* it) {
	ASSERT(it->magic == ITEM_MAGIC);

	char* metadata = it->end;
	if (item_has_cas(it)) {
		metadata += sizeof(uint64_t);
	}
	return metadata;
}

uint8_t item_slabid2(uint8_t nkey, uint32_t nbyte)
{
    size_t ntotal;
    uint8_t id;

    ntotal = item_ntotal(nkey, nbyte, settings.use_cas);

    id = slab_id(ntotal);
    if (id == SLABCLASS_INVALID_ID) {
        log_debug(LOG_NOTICE, "slab class id out of range with %"PRIu8" bytes "
                  "key, %"PRIu32" bytes value and %zu item chunk size", nkey,
                  nbyte, ntotal);
    }

    return id;
}

/**
 * Given the size of the key and value, this function returns the expected size of
 * the item that will hold this key-value pair.
 */
size_t item_expected_size(uint8_t nkey, uint32_t nbyte)
{
	return item_ntotal(nkey, nbyte, settings.use_cas);
}

static void
item_free(struct item *it, bool lock_slab)
{
    ASSERT(it->magic == ITEM_MAGIC);

    if(item_is_lease_holder(it)) {
//    	slab_put_reserved_item(it, lock_slab);
    	stats_thread_decr(num_lease_token);

    	if(item_has_qlease(it)) {
    		stats_thread_decr(num_q_lease_token);
    	} else {
    		stats_thread_decr(num_i_lease_token);
    	}

    	mem_free(&reserved_memory_manager, it);
    } else {
//    	slab_put_item(it);
    	mem_free(&memory_manager, it);
    }
    stats_thread_incr(items_free);
}

/*
 * Link an item into the hash table and lru q
 */
static void
_item_link(struct item *it)
{
    ASSERT(it->magic == ITEM_MAGIC);
    ASSERT(!item_is_linked(it));
    ASSERT(!item_is_slabbed(it));

    log_debug(LOG_DEBUG, "link it '%.*s' at offset %"PRIu32" with flags "
              "%02x id %"PRId8"", it->nkey, item_key(it), it->offset,
              it->flags, it->id);

    it->flags |= ITEM_LINKED;
    item_set_cas(it, item_next_cas());

    assoc_insert(it);
    item_link_q(it, true);
}

/*
 * Unlinks an item from the lru q and hash table. Free an unlinked
 * item if it's refcount is zero.
 */
static void
_item_unlink(struct item *it)
{
    ASSERT(it->magic == ITEM_MAGIC);
    ASSERT(item_is_linked(it));


    if (item_is_lease_holder(it))
    	item_unset_pinned(it);

    log_debug(LOG_DEBUG, "unlink it '%.*s' at offset %"PRIu32" with flags "
              "%02x id %"PRId8" refcount %"PRIu16"", it->nkey, item_key(it), it->offset,
              it->flags, it->id, it->refcount);

    if (item_is_linked(it)) {
        it->flags &= ~ITEM_LINKED;

        assoc_delete(item_key(it), it->nkey);

        item_unlink_q(it);

        if (it->refcount == 0) {
            item_free(it, true);
        }

        stats_thread_incr(items_unlink);
    }
}


/*
 * Decrement the refcount on an item. Free an unliked item if its refcount
 * drops to zero.
 */
static void
_item_remove2(struct item *it, bool lock_slab)
{
    ASSERT(it->magic == ITEM_MAGIC);
    ASSERT(!item_is_slabbed(it));

    log_debug(LOG_DEBUG, "remove it '%.*s' at offset %"PRIu32" with flags "
              "%02x id %"PRId8" refcount %"PRIu16"", it->nkey, item_key(it),
              it->offset, it->flags, it->id, it->refcount);

    if (it->refcount != 0) {
        item_release_refcount(it);
    }

    if (it->refcount == 0 && !item_is_linked(it)) {
        item_free(it, lock_slab);
    }
}


static void
_item_remove(struct item *it)
{
	_item_remove2(it, true);
}

/*
 * Evict enough items to fit the incoming item.
 */
static struct item*
_item_allocate_or_evict(size_t size_required, bool reserved_item) {
	struct item *oit = NULL;
	struct item *it = NULL;

	uint8_t id = 1;
	int num_evict = 0;

	do {
		/* Try to allocate first */
		if (reserved_item) {
			it = mem_alloc(&reserved_memory_manager, size_required);
		} else {
			it = mem_alloc(&memory_manager, size_required);
		}
		if (it != NULL) {
			break;
		}

		/* Failed to allocate. Try freeing up some older items */
		num_evict++;
		if(reserved_item) {
			oit = item_get_from_index(id, &reserved_p_index); /* expired / unexpired lru item */
		} else {
			oit = item_get_from_index(id, &p_index); /* expired / unexpired lru item */
		}

		/* Remove the item from the LRU Q */
		if (oit != NULL) {
			_item_unlink(oit);
		}
	} while (oit != NULL && num_evict < ITEM_LRUQ_MAX_TRIES && it == NULL);

	if (num_evict > 0) {
		stats_thread_incr_by(items_evicted, num_evict);
	}
	return it;
}

/*
 * Allocate an item. We allocate an item either by -
 *  1. Reusing an expired item from the lru Q of an item's slab class. Or,
 *  2. Consuming the next free item from slab of the item's an slab class
 *
 * On success we return the pointer to the allocated item. The returned item
 * is refcounted so that it is not deleted under callers nose. It is the
 * callers responsibilty to release this refcount when the item is inserted
 * into the hash + lru q or freed.
 */
static struct item *
_item_alloc(size_t item_size, const char *key, uint8_t nkey, uint32_t dataflags,
		rel_time_t exptime, uint32_t nbyte, bool lock_slab, bool reserved_item)
{
    struct item *it;  /* item */
//    struct item *uit; /* unexpired lru item */

//    uint8_t id = item_slabid(0, (uint32_t) item_size);
    uint8_t id = 1;
    /* TODO(jyap): set item->id = id */

//    ASSERT(id >= SLABCLASS_MIN_ID && id <= SLABCLASS_MAX_ID);
//
//    /*
//     * We try to obtain an item in the following order:
//     *  1)  by acquiring an expired item;
//     *  2)  by getting a free slot from the last slab in current class;
//     *  3)  by evicting a slab, if slab eviction(s) are enabled;
//     *  4)  by evicting an item, if item lru eviction is enabled.
//     */
//    if(reserved_item) {
//    	it = item_get_from_lruq(id, reserved_item_lruq); /* expired / unexpired lru item */
//    } else {
//    	it = item_get_from_lruq(id, item_lruq); /* expired / unexpired lru item */
//    }
//
//    if (it == NULL) {
//    	if (reserved_item)
//    		log_debug(LOG_VERB, "cannot get item '%.*s' from LRU reserveditem=true", nkey, key);
//    	else
//    		log_debug(LOG_VERB, "cannot get item '%.*s' from LRU reserveditem=false", nkey, key);
//    }
//
//    if (it != NULL && item_expired(it)) {
//        /* 1) this is an expired item, always use it */
//        stats_slab_incr(id, item_expire);
//        stats_slab_settime(id, item_expire_ts, it->exptime);
//
////        if(item_is_lease_holder(it)) {
////        	/* Allow this expired gumball to be re-used */
////        	item_unset_pinned(it);
////        }
//
//        /* Count expired leases */
//        if(it->lease_token > LEASE_SENTINEL) {
//            stats_thread_incr(expired_leases);
//            if (item_has_xlease(it)) {
//                stats_thread_incr(expired_x_leases);
//            } else {
//                stats_thread_incr(expired_s_leases);
//            }
//        }
//
//        item_reuse(it);
//        goto done;
//    }
//
//    uit = (settings.evict_opt & EVICT_LRU)? it : NULL; /* keep if can be used */
//
//    if (reserved_item) {
////    	it = slab_get_reserved_item(id, lock_slab);
//    } else {
////    	it = slab_get_item(id);
//    }

    //it = mem_alloc(item_size);
    it = _item_allocate_or_evict(item_size, reserved_item);
    if (it != NULL) {
        /* 2) or 3) either we allow random eviction a free item is found */
        goto done;
    }

//    if (uit != NULL) {
//        /* 4) this is an lru item and we can reuse it */
//        it = uit;
//        stats_slab_incr(id, item_evict);
//        stats_slab_settime(id, item_evict_ts, time_now());
//
//        item_reuse(it);
//        goto done;
//    }
//
//    // finally, try evict slab if we cannot find any item in EVICT_LRU option
//    if ((reserved_item == false) && (settings.evict_opt & EVICT_LRU)) {
//    	it = slab_get_item_by_evict_slab(id);
//
//    	if (it != NULL)
//    		goto done;
//    }
//
//    if (reserved_item == true)
//    	log_warn("server error on allocating item in slab %"PRIu8" key '%.*s', reserveditem=true", id, nkey, key, reserved_item);
//
//    if (reserved_item == false)
//       	log_warn("server error on allocating item in slab %"PRIu8" key '%.*s', reserveditem=false", id, nkey, key, reserved_item);
//
//    stats_thread_incr(server_error);

    return NULL;

done:
    ASSERT(it->id == id);
    ASSERT(!item_is_linked(it));
//    ASSERT(!item_is_slabbed(it));
//    ASSERT(it->offset != 0);
	ASSERT(it->refcount == 0);

    item_acquire_refcount(it);

    it->flags = settings.use_cas ? ITEM_CAS : 0;
    it->dataflags = dataflags;
    it->nbyte = nbyte;
    it->exptime = exptime;
    it->nkey = nkey;
    it->num_of_trans_q_inv_leases = 0;

    /* Set inserted time of the new item to now */
    clocktime_now(item_created_time(it));

//#if defined MC_MEM_SCRUB && MC_MEM_SCRUB == 1
//    memset(it->end, 0xff, slab_item_size(it->id) - ITEM_HDR_SIZE);
//#endif
    memcpy(item_key(it), key, nkey);

    stats_slab_incr(id, item_acquire);

    log_debug(LOG_VERB, "alloc it '%.*s' at offset %"PRIu32" with id %"PRIu8
              " expiry %u refcount %"PRIu16"", it->nkey, item_key(it),
              it->offset, it->id, it->exptime, it->refcount);

    return it;
}

struct item *
item_alloc(size_t item_size, char *key, uint8_t nkey, uint32_t dataflags,
           rel_time_t exptime, uint32_t nbyte)
{
    struct item *it;

    pthread_mutex_lock(&cache_lock);
    it = _item_alloc(item_size, key, nkey, dataflags, exptime, nbyte, true, false);
    pthread_mutex_unlock(&cache_lock);

    return it;
}

/*
 * Decrement the refcount on an item. Free an unliked item if its refcount
 * drops to zero.
 */
void
item_remove(struct item *it)
{
    pthread_mutex_lock(&cache_lock);
    _item_remove(it);
    pthread_mutex_unlock(&cache_lock);
}

/*
 * Unlink an item and remove it (if its recount drops to zero).
 */
void
item_delete(struct item *it)
{
    pthread_mutex_lock(&cache_lock);
    _item_unlink(it);
    _item_remove(it);
    pthread_mutex_unlock(&cache_lock);
}

/*
 * Touch the item by moving it to the tail of lru q only if it wasn't
 * touched ITEM_UPDATE_INTERVAL secs back.
 */
static void
_item_touch(struct item *it)
{
    ASSERT(it->magic == ITEM_MAGIC);
    ASSERT(!item_is_slabbed(it));

    if (it->atime >= (time_now() - ITEM_UPDATE_INTERVAL)) {
        return;
    }
    index_touch(it);

    log_debug(LOG_VERB, "update it '%.*s' at offset %"PRIu32" with flags "
              "%02x id %"PRId8"", it->nkey, item_key(it), it->offset,
              it->flags, it->id);

    if (item_is_linked(it)) {
        item_unlink_q(it);
        item_link_q(it, false);
    }
}

void
item_touch(struct item *it)
{
    if (it->atime >= (time_now() - ITEM_UPDATE_INTERVAL)) {
        return;
    }

    pthread_mutex_lock(&cache_lock);
    _item_touch(it);
    pthread_mutex_unlock(&cache_lock);
}

/*
 * Replace one item with another in the hash table and lru q.
 */
static void
_item_replace(struct item *it, struct item *nit)
{
    ASSERT(it->magic == ITEM_MAGIC);
    ASSERT(!item_is_slabbed(it));

    ASSERT(nit->magic == ITEM_MAGIC);
    ASSERT(!item_is_slabbed(nit));

    log_debug(LOG_VERB, "replace it '%.*s' at offset %"PRIu32" id %"PRIu8" "
              "with one at offset %"PRIu32" id %"PRIu8"", it->nkey,
              item_key(it), it->offset, it->id, nit->offset, nit->id);

    _item_unlink(it);
    _item_link(nit);
}

// TODO: make this work with the reserved lruq as well.
static char *
_item_cache_dump(uint8_t id, uint32_t limit, uint32_t *bytes)
{
    const size_t memlimit = ITEM_CACHEDUMP_MEMLIMIT;
    char *buffer;
    uint32_t bufcurr;
    struct item *it;
    uint32_t len;
    uint32_t shown = 0;
    char key_temp[KEY_MAX_LEN + 1];
    char temp[KEY_MAX_LEN * 2];

    buffer = mc_alloc(memlimit);
    if (buffer == NULL) {
        return NULL;
    }

    for (bufcurr = 0, it = TAILQ_FIRST(&item_lruq[id]);
         it != NULL && (limit == 0 || shown < limit);
         it = TAILQ_NEXT(it, i_tqe)) {

        ASSERT(it->nkey <= KEY_MAX_LEN);
        /* copy the key since it may not be null-terminated in the struct */
        strncpy(key_temp, item_key(it), it->nkey);
        key_temp[it->nkey] = '\0'; /* terminate */
        len = snprintf(temp, sizeof(temp), "ITEM %s [%d b; %lu s]\r\n",
                       key_temp, it->nbyte,
                       (unsigned long)it->exptime + time_started());
        if (len >= sizeof(temp)) {
            log_debug(LOG_WARN, "item log was truncated during cache dump");
        }
        if (bufcurr + len + (sizeof("END\r\n") - 1) > memlimit) {
            break;
        }
        memcpy(buffer + bufcurr, temp, len);
        bufcurr += len;
        shown++;
    }

    memcpy(buffer + bufcurr, "END\r\n", sizeof("END\r\n") - 1);
    bufcurr += sizeof("END\r\n") - 1;

    *bytes = bufcurr;

    return buffer;
}

char *
item_cache_dump(uint8_t id, uint32_t limit, uint32_t *bytes)
{
    char *ret;

    pthread_mutex_lock(&cache_lock);
    ret = _item_cache_dump(id, limit, bytes);
    pthread_mutex_unlock(&cache_lock);

    return ret;
}

/* Allocate an item with value size 0 that will act as the lease holder */
static struct item*
_item_create_reserved_item(const char* key, uint8_t nkey, uint32_t vlen, const lease_token_t* token, bool lock_slab)
{
    uint32_t flags = 0;
    time_t exptime = settings.lease_token_expiry / 1000;
    struct item *it;
//    uint8_t id;

    if(exptime == 0) {
        /* Minimum 1 second expiry time */
        exptime = 1;
    }

    if(exptime > 0) {
        /* Add 1 second to account for rounding errors */
        exptime++;
    }

//    id = item_slabid(nkey, vlen);
    it = _item_alloc(item_expected_size(nkey, vlen), key, nkey,
    		flags, time_reltime(exptime), vlen, lock_slab, true);
    if (it != NULL) {
        item_set_lease_holder(it);
        item_set_pinned(it);
        if(token == NULL) {
            it->lease_token = lease_next_token();
        } else {
            it->lease_token = *token;
        }
    }

    return it;
}


/* Allocate an item with value size 0 that will act as the lease holder */
static struct item*
_item_create_lease_holder(const char* key, uint8_t nkey, const lease_token_t* token, bool lock_slab)
{
	uint32_t flags = 0, vlen = 0;
	time_t exptime = settings.lease_token_expiry / 1000;
	struct item *it;
//	uint8_t id;

	if(exptime == 0) {
		/* Minimum 1 second expiry time */
		exptime = 1;
	}

	if(exptime > 0) {
		/* Add 1 second to account for rounding errors */
		exptime++;
	}

//	id = item_slabid(nkey, vlen);
	it = _item_alloc(item_expected_size(nkey, vlen), key, nkey,
			flags, time_reltime(exptime), vlen, lock_slab, true);
	if (it != NULL) {
		item_set_lease_holder(it);
		item_set_pinned(it);
		if(token == NULL) {
			it->lease_token = lease_next_token();
		} else {
			it->lease_token = *token;
		}
	}

	return it;
}

/*
 * Return an item if it hasn't been marked as expired, lazily expiring
 * item as-and-when needed
 *
 * When a non-null item is returned, it's the callers responsibily to
 * release refcount on the item
 */
static struct item *
_item_get(const char *key, size_t nkey)
{
    struct item *it;

    it = assoc_find(key, nkey);
    if (it == NULL) {
        log_debug(LOG_VERB, "get it '%.*s' not found", nkey, key);
        return NULL;
    }

    log_debug(LOG_VERB, "get it Lease Token: %d; Time: %d; Exptime: %d",
        		it->lease_token,
        		time_now(), it->exptime);

    if (it->exptime != 0 && it->exptime <= time_now()) {
    	if(it->lease_token > LEASE_SENTINEL) {
    	    stats_thread_incr(expired_leases);
    	    if (item_has_qlease(it)) {
    	        stats_thread_incr(expired_q_leases);
    	    } else {
    	        stats_thread_incr(expired_i_leases);
    	    }
    	}

        _item_unlink(it);
        stats_slab_incr(it->id, item_expire);
        stats_slab_settime(it->id, item_reclaim_ts, time_now());
        stats_slab_settime(it->id, item_expire_ts, it->exptime);
        log_debug(LOG_VERB, "get it '%.*s' expired and nuked", nkey, key);
        return NULL;
    }

    if (settings.oldest_live != 0 && settings.oldest_live <= time_now() &&
        it->atime <= settings.oldest_live) {
        _item_unlink(it);
        stats_slab_incr(it->id, item_evict);
        stats_slab_settime(it->id, item_evict_ts, time_now() );
        log_debug(LOG_VERB, "it '%.*s' nuked", nkey, key);
        return NULL;
    }

    item_acquire_refcount(it);

    log_debug(LOG_VERB, "get it '%.*s' found at offset %"PRIu32" with flags "
              "%02x id %"PRIu8" refcount %"PRIu32"", it->nkey, item_key(it),
              it->offset, it->flags, it->id, it->refcount);

    return it;
}

struct item *
item_get(const char *key, size_t nkey)
{
    struct item *it;

    pthread_mutex_lock(&cache_lock);
    it = _item_get(key, nkey);
    pthread_mutex_unlock(&cache_lock);

    return it;
}

/*
 * Flushes expired items after a "flush_all" call. Expires items that
 * are more recent than the oldest_live setting
 *
 * TODO: make this work with reserved lruq as well
 */
static void
_item_flush_expired(void)
{
    uint8_t i;
    struct item *it, *next;

    if (settings.oldest_live == 0) {
        return;
    }

    for (i = SLABCLASS_MIN_ID; i <= SLABCLASS_MAX_ID; i++) {
        /*
         * Lru q is sorted in ascending time order -- oldest to most recent.
         * Since time is computed at granularity of one sec, we would like
         * to flush items which are accessed during this one sec. So we
         * proactively flush most recent ones in a given queue until we hit
         * an item older than oldest_live. Older items in this queue are then
         * lazily expired by oldest_live check in item_get.
         */
        TAILQ_FOREACH_REVERSE_SAFE(it, &item_lruq[i], item_tqh, i_tqe, next) {
            ASSERT(!item_is_slabbed(it));

            if (it->atime < settings.oldest_live) {
                break;
            }

            _item_unlink(it);

            stats_slab_incr(it->id, item_evict);
            stats_slab_settime(it->id, item_evict_ts, time_now());
        }
    }
}

void
item_flush_expired(void)
{
    pthread_mutex_lock(&cache_lock);
    _item_flush_expired();
    pthread_mutex_unlock(&cache_lock);
}

/*
 * Store an item in the cache according to the semantics of one of the
 * update commands - {set, add, replace, append, prepend, cas}
 */
static item_store_result_t
_item_store(struct item *it, req_type_t type, struct conn *c, bool lock_slab)
{
    item_store_result_t result;  /* item store result */
    bool store_it;               /* store item ? */
    char *key;                   /* item key */
    struct item *oit, *nit;      /* old (existing) item & new item */
//    uint8_t id;                  /* slab id */
    size_t new_item_size;
    uint32_t total_nbyte;

    result = NOT_STORED;

    // if new item has no value, not store
    if (it->nbyte == 0 && !item_is_lease_holder(it))
    	return result;

    store_it = true;

    key = item_key(it);
    nit = NULL;
    oit = _item_get(key, it->nkey);
    if (oit == NULL) {
        switch (type) {
        case REQ_IQSET:
        	// If someone calls set without first calling q Lease or Get, there
        	// won't be an existing key. Prevent the set from succeeding.
        	// TODO: revisit if this is the right decision.
        	store_it = false;
            break;

        case REQ_SET:
        	stats_slab_incr(it->id, set_success);
        	break;

        case REQ_ADD:
            stats_slab_incr(it->id, add_success);
            break;

        case REQ_REPLACE:
            stats_thread_incr(replace_miss);
            store_it = false;
            break;

        case REQ_APPEND:
            stats_thread_incr(append_miss);
            store_it = false;
            break;

        case REQ_PREPEND:
            stats_thread_incr(prepend_miss);
            store_it = false;
            break;

        case REQ_CAS:
            stats_thread_incr(cas_miss);
            result = NOT_FOUND;
            store_it = false;
            break;

        case REQ_IQGET:
        case REQ_DELETE:
        case REQ_QAREG:
        case REQ_QAREAD:
        	store_it = true;
        	break;
        default:
            NOT_REACHED();
        }
    } else {
        switch (type) {
        case REQ_IQSET:
        	if (it->lease_token != oit->lease_token) {
            	/* Incoming lease token is not equal to existing lease token.
            	 * Don't insert (either a delete occurred or the token had expired) */
            	store_it = false;
            	stats_thread_incr(set_miss_token);
            } else {
            	stats_slab_incr(it->id, set_success);

            	/* Compute the cost to generate the data for this item */
            	cost_t cost = clocktime_diff_milli(item_created_time(it),
            			item_created_time(oit));
            	index_set_cost(it, cost);

//            	log_debug(LOG_VVERB,
//            			"Item cost for key '%.*s'"
//            			" with key size %"PRIu8" and value size %"PRIu32
//            			" cost is %"PRIu64,
//            			oit->nkey, item_key(oit), oit->nkey, it->nbyte, cost);
            }
            break;

        case REQ_SET:
        	// keep old token so that operations with leases are not affected by this set interleaved
			it->lease_token = oit->lease_token;

        	stats_slab_incr(it->id, set_success);
        	break;

        case REQ_ADD:
        	if (oit->nbyte == 0) {
        		// get the lease token from the old value
        		it->lease_token = oit->lease_token;

        		store_it = true;
        		stats_slab_incr(oit->id, add_success);
        		break;
        	}

            stats_thread_incr(add_exist);
            /*
             * Add only adds a non existing item. However we promote the
             * existing item to head of the lru q
             */
            _item_touch(oit);
            store_it = false;
            break;

        case REQ_REPLACE:
        	if (oit->nbyte == 0) {
        		store_it = false;
        	} else {
        		it->lease_token = oit->lease_token;

				stats_slab_incr(oit->id, replace_hit);
				stats_slab_incr(it->id, replace_success);
        	}
            break;

        case REQ_APPEND:
            stats_slab_incr(oit->id, append_hit);

            total_nbyte = oit->nbyte + it->nbyte;
//            id = item_slabid(oit->nkey, total_nbyte);
//            if (id == SLABCLASS_INVALID_ID) {
//                /* FIXME: logging client error but not sending CLIENT ERROR
//                 * to the client because we are inside the item module, which
//                 * technically shouldn't directly handle commands. There is not
//                 * a proper return status to indicate such an error.
//                 * This can only be fixed by moving the command-aware logic
//                 * into a separate module.
//                 */
//                log_debug(LOG_NOTICE, "client error on c %d for req of type %d"
//                          " with key size %"PRIu8" and value size %"PRIu32,
//                          c->sd, c->req_type, oit->nkey, total_nbyte);
//
//                stats_thread_incr(cmd_error);
//                store_it = false;
//                break;
//            }
            new_item_size = item_expected_size(oit->nkey, total_nbyte);

            /* if oit is large enough to hold the extra data and left-aligned,
             * which is the default behavior, we copy the delta to the end of
             * the existing data. Otherwise, allocate a new item and store the
             * payload left-aligned.
             */

            // nbyte = 0 means that item does not exist
            if (oit->nbyte == 0) {
            	store_it = false;
            	break;
            }

			nit = _item_alloc(new_item_size, key, oit->nkey, oit->dataflags,
                                  oit->exptime, total_nbyte, true, false);

			if (nit == NULL) {
				store_it = false;
				break;
			}

			memcpy(item_data(nit), item_data(oit), oit->nbyte);
			memcpy(item_data(nit) + oit->nbyte, item_data(it), it->nbyte);
			nit->lease_token = oit->lease_token;
			it = nit;

            store_it = true;
            stats_slab_incr(it->id, append_success);
            break;

        case REQ_PREPEND:
            stats_slab_incr(oit->id, prepend_hit);

            /*
             * Alloc new item - nit to hold both it and oit
             */
            total_nbyte = oit->nbyte + it->nbyte;
//            id = item_slabid(oit->nkey, total_nbyte);
//            if (id == SLABCLASS_INVALID_ID) {
//                log_debug(LOG_NOTICE, "client error on c %d for req of type %d"
//                          " with key size %"PRIu8" and value size %"PRIu32,
//                          c->sd, c->req_type, oit->nkey, total_nbyte);
//
//                stats_thread_incr(cmd_error);
//                store_it = false;
//                break;
//            }
            new_item_size = item_expected_size(oit->nkey, total_nbyte);

            /* if oit is large enough to hold the extra data and is already
             * right-aligned, we copy the delta to the front of the existing
             * data. Otherwise, allocate a new item and store the payload
             * right-aligned, assuming more prepends will happen in the future.
             */

            // nbyte = 0 means that item does not exist
            if (oit->nbyte == 0) {
            	store_it = false;
            	break;
            }

			nit = _item_alloc(new_item_size, key, oit->nkey, oit->dataflags,
                                  oit->exptime, total_nbyte, true, false);
            if (nit == NULL) {
                store_it = false;
                break;
            }

			nit->flags |= ITEM_RALIGN;
			memcpy(item_data(nit), item_data(it), it->nbyte);
			memcpy(item_data(nit) + it->nbyte, item_data(oit), oit->nbyte);
			nit->lease_token = oit->lease_token;
			it = nit;

            store_it = true;
            stats_slab_incr(it->id, prepend_success);
            break;

        case REQ_CAS:
            if (item_cas(it) != item_cas(oit)) {
                log_debug(LOG_DEBUG, "cas mismatch %"PRIu64" != %"PRIu64 "on "
                          "it '%.*s'", item_cas(oit), item_cas(it), it->nkey,
                          item_key(it));
                stats_slab_incr(oit->id, cas_badval);
                result = EXISTS;
                break;
            }

            stats_slab_incr(oit->id, cas_hit);
            stats_slab_incr(it->id, cas_success);
            break;

        case REQ_DELETE:
        	// Lease holder replacing an existing entry
        	store_it = true;
        	break;

        case REQ_QAREG:
        	store_it = true;
        	break;

        case REQ_QAREAD:
        	store_it = true;
        	break;

        default:
            NOT_REACHED();
        }
    }

    if (result == NOT_STORED && store_it) {
        if (oit != NULL) {
            _item_replace(oit, it);
        } else {
            _item_link(it);
        }
        result = STORED;

        log_debug(LOG_VERB, "store it '%.*s'at offset %"PRIu32" with flags %02x"
                  " id %"PRId8"", it->nkey, item_key(it), it->offset, it->flags,
                  it->id);

        /* After set succeeds, assign the DEFAULT_TOKEN to allow this item to be replaced
         * by future sets that didn't obtain a lease_token from a get miss (as happens in
         * the refresh client case). */
        if (type == REQ_IQSET || type == REQ_SAR || type == REQ_DAR) {
			if (it->lease_token != LEASE_DELETED && !item_is_lease_holder(it)) {
				it->lease_token = DEFAULT_TOKEN;
			}
        }
    } else {
    	log_debug(LOG_VERB, "did not store it '%.*s'at offset %"PRIu32" with flags %02x"
                  " id %"PRId8"", it->nkey, item_key(it), it->offset, it->flags,
                  it->id);
    }

    if (type != REQ_SET) {
		log_debug(LOG_VERB, "store it Lease Token: %d; Time: %d; Exptime: %d",
						it->lease_token,
						time_now(), it->exptime);
    }

    /* release our reference, if any */
    if (oit != NULL) {
        _item_remove2(oit, lock_slab);
    }

    if (nit != NULL) {
        _item_remove(nit);
    }

    return result;
}

item_store_result_t
item_store(struct item *it, req_type_t type, struct conn *c)
{
    item_store_result_t ret = NOT_STORED;

    pthread_mutex_lock(&cache_lock);
    ret = _item_store(it, type, c, true);
    pthread_mutex_unlock(&cache_lock);

    return ret;
}

/*
 * Add a delta value (positive or negative) to an item.
 */
static item_delta_result_t
_item_add_delta(struct conn *c, char *key, size_t nkey, bool incr,
                int64_t delta, char *buf)
{
    int res;
    char *ptr;
    uint64_t value;
    struct item *it;

    it = _item_get(key, nkey);
    if (it == NULL) {
        return DELTA_NOT_FOUND;
    }

    ptr = item_data(it);

    if (!mc_strtoull_len(ptr, &value, it->nbyte)) {
        _item_remove(it);
        return DELTA_NON_NUMERIC;
    }

    if (incr) {
        value += delta;
    } else if (delta > value) {
        value = 0;
    } else {
        value -= delta;
    }

    if (incr) {
        stats_slab_incr(it->id, incr_hit);
    } else {
        stats_slab_incr(it->id, decr_hit);
    }

    res = snprintf(buf, INCR_MAX_STORAGE_LEN, "%"PRIu64, value);
    ASSERT(res < INCR_MAX_STORAGE_LEN);
    if (res > it->nbyte) { /* need to realloc */
        struct item *new_it;
//        uint8_t id;

//        id = item_slabid(it->nkey, res);
//        if (id == SLABCLASS_INVALID_ID) {
//            log_debug(LOG_NOTICE, "client error on c %d for req of type %d"
//            " with key size %"PRIu8" and value size %"PRIu32, c->sd,
//            c->req_type, it->nkey, res);
//        }

        new_it = _item_alloc(item_expected_size(it->nkey, res), item_key(it),
        		it->nkey, it->dataflags, it->exptime, res, true, false);
        if (new_it == NULL) {
            _item_remove(it);
            return DELTA_EOM;
        }
        if (incr) {
            stats_slab_incr(new_it->id, incr_success);
        } else {
            stats_slab_incr(new_it->id, decr_success);
        }

        memcpy(item_data(new_it), buf, res);
        _item_replace(it, new_it);
        _item_remove(new_it);
    } else {
        /*
         * Replace in-place - when changing the value without replacing
         * the item, we need to update the CAS on the existing item
         */
        if (incr) {
            stats_slab_incr(it->id, incr_success);
        } else {
            stats_slab_incr(it->id, decr_success);
        }

        item_set_cas(it, item_next_cas());
        memcpy(item_data(it), buf, res);
        it->nbyte = res;
    }

    _item_remove(it);

    return DELTA_OK;
}

item_delta_result_t
item_add_delta(struct conn *c, char *key, size_t nkey, int incr,
               int64_t delta, char *buf)
{
    item_delta_result_t ret;

    pthread_mutex_lock(&cache_lock);
    ret = _item_add_delta(c, key, nkey, incr, delta, buf);
    pthread_mutex_unlock(&cache_lock);

    return ret;
}

static void
item_handle_lease_failure(const char *key, uint8_t nkey, const lease_token_t *token)
{
	/* Lease token creation failed. Prevent other inserts on this key from succeeding by
	 *  advancing the gumball_adjust_timestamp. Any get miss that occurred before the current
	 *  time, will fail to insert it's value since it's miss_timestamp is older than the
	 *  gumball_adjust_timestamp. Side effect: this affects all keys, not just the key
	 *  for which the gumball creation failed.
	 */
	stats_thread_incr(lease_error);
}


struct item*
item_create_lease_holder(char* key, uint8_t nkey)
{
	struct item* it = NULL;
	pthread_mutex_lock(&cache_lock);
	it = _item_create_lease_holder(key, nkey, NULL, true);
	pthread_mutex_unlock(&cache_lock);

	return it;
}

/* Similar to item_get_and_release except here we just remove the lease
 * token rather than deleting the item.
 */
rstatus_t
item_get_and_unlease(char* key, uint8_t nkey, lease_token_t token_val, struct conn *c) {
	struct item *it = NULL; /* item for this key */
	rstatus_t status = MC_OK;

	log_debug(LOG_VERB, "get_and_unlease for '%.*s'", nkey, key);

	pthread_mutex_lock(&cache_lock);
	it = _item_get(key, nkey);
	if (it == NULL) {
		/* No need to do anything. Item did not exist */
		/* TODO: print warning message? should not get to this state */
	} else {
		if (it->lease_token == token_val) {
			// Refresh version: just remove the lease on the item. This allows someone
			// else to acquire an qLease.
			it->lease_token = LEASE_DELETED;

			stats_thread_decr(num_lease_token);
			if (item_has_qlease(it)) {
			    item_unset_qlease(it);
			    stats_thread_decr(num_q_lease_token);

				// copy item back to the main memory space
			    //item_slabid(it->nkey, it->nbyte)
				struct item* new_it = _item_alloc(
						item_expected_size(it->nkey, it->nbyte),
						item_key(it), it->nkey,
						it->dataflags, 0, it->nbyte, true, false);

				if (new_it != NULL) {
					memcpy(item_data(new_it), item_data(it), it->nbyte);
					new_it->lease_token = LEASE_DELETED;

					_item_replace(it, new_it);
					_item_remove(new_it);
				} else {
					// TODO(jyap): Is this right? This is different from the main branch.
					// Delete the old item
					_item_unlink(it);
				}
			} else {
			    // Assume it is a shared lease
			    stats_thread_decr(num_i_lease_token);

			    // if item has no value on it, delete item
			    if (it->nbyte == 0) {
			    	_item_unlink(it);
			    }
			}
		} else {
			/* Don't do anything. Lease token did not match. */
		}

		_item_remove(it);
	}
	pthread_mutex_unlock(&cache_lock);

	return status;
}

/**
 * Get I lease for a key. If it fails because someone is holding lease for this key,
 * markedVal is set to 0. Otherwise, markedVal = 1
 */
rstatus_t
item_quarantine_and_register(char* tid, uint32_t ntid, char* key,
		uint8_t nkey, u_int8_t *markedVal,struct conn *c) {
	struct item *it = NULL;	/* item for this key */
	struct item *trans_it = NULL;
	trig_cursor_t cursor;
	char* k;
	size_t klen;

	log_debug(LOG_VERB, "quarantine_and_register for '%.*s'", nkey, key);

	pthread_mutex_lock(&cache_lock);

	// get the item from the KVS
	it = _item_get(key, nkey);

	// check for item status first
	if (it != NULL && (it->flags & ITEM_QLEASE)) {
		// decrement ref count
		_item_remove(it);

		// release all item in this transaction
		trans_it = _item_get(tid, ntid);

		if (trans_it == NULL) {
			// set marked value
			*markedVal = 0;

			pthread_mutex_unlock(&cache_lock);

			return MC_OK;
		}

		char *data = item_data(trans_it);

		// loop through keys and delete
		while (trig_keylist_next(data, trans_it->nbyte, &cursor, &k, &klen) == TRIG_OK) {
			// TODO(hieu): use a different name for it here.
			it = _item_get(key, nkey);

			if (it == NULL) {
				// Item may have been evicted due to the cache running out of memory.
				// We should continue to try to delete the rest of the keys.
				log_debug(LOG_VERB, "quarantine_and_register item not found %s %s", key, tid);
				continue;
			}

			// because the trans will be rollbacked, no need to delete item
			// just do the unset i release
			item_unset_q_inv_lease(it);

			if (it->num_of_trans_q_inv_leases > 0) {
				it->num_of_trans_q_inv_leases--;

				if (it->num_of_trans_q_inv_leases > 0) {
					_item_remove(it);
					continue;
				}
			}

			_item_unlink(it);
			_item_remove(it);
		}

		// remove transaction
		_item_unlink(trans_it);
		_item_remove(trans_it);

		// tell DBMS to abort transaction
		*markedVal = 0;

		pthread_mutex_unlock(&cache_lock);
		return MC_OK;
	}

	// look up for the transaction item
	trans_it = _item_get(tid, ntid);

	// at this time, item for "key" is not granted a q lease (in refresh session), and
	// trans_it should be available
	if (it == NULL) {
		// allocate new item for "key" and value is set to null
		it = _item_create_lease_holder(key, nkey, NULL, true);

		if (it == NULL) {
			log_debug(LOG_VERB, "quarantine_and_register cannot allocate new memory for key %s", key);
			return MC_ERROR;
		}

		// store this item
		//TODO(hieu): check return value of item_store and handle failures
		_item_store(it, REQ_QAREG, c, true);
	} else {
	    // Assign a new lease token if the item existed previously.
	    it->lease_token = lease_next_token();
	}

	ASSERT(it != NULL);
//	ASSERT(trans_it != NULL);
	ASSERT((it->flags & ITEM_QLEASE) == false);

	// Now it and trans_it are ready, three possible cases raises:
	// 1. item already has a i lease on it, need to expand the expire time, change lease token
	// and increment counter
	// 2. item already has a s lease (shared lease) on it, replace the shared lease by a
	// i lease
	// 3. item has no lease on it, just need to grant a i lease (add lease token, increment
	// counter and set expire time)
	//
	// In all cases, item keys will be added to the trans_it list
	item_set_q_inv_lease(it);
	_item_touch(it);

	// add key to transaction item
	trig_listlen_t old_keylistlen = 0;

	if (trans_it != NULL) {
	    old_keylistlen = trans_it->nbyte;
	}

	// If the trans_it does not exist or if it does but the new key is not contained
	// in the existing keylist
	if (trans_it == NULL ||
	        trig_check_keylist(item_data(trans_it), old_keylistlen, key,
	                (trig_keylen_t) nkey) != TRIG_OK) {
	    // add counter
		it->num_of_trans_q_inv_leases++;

		u_int32_t num_of_bytes = old_keylistlen + trig_new_keylist_size(key, nkey);
		trig_listlen_t new_listlen = 0;

		// allocate memory for new item
		struct item* new_trans_it = _item_create_reserved_item(tid, ntid,
				num_of_bytes, NULL, true);

		if (trans_it != NULL) {
			// copy old items to new items and add new key
			//if (trig_check_keylist(item_data(trans_it), keylistlen, key, (trig_keylen_t) nkey) != TRIG_OK)
			trig_keylist_addkey(item_data(new_trans_it), &new_listlen, new_trans_it->nbyte,
				item_data(trans_it), trans_it->nbyte, key, nkey);
		} else {
			// There was no old keylist so create a new one with just the new key.
			trig_keylist_addkey(item_data(new_trans_it), &new_listlen, new_trans_it->nbyte,
				NULL, 0, key, nkey);
		}

		// replace old item
		_item_store(new_trans_it, REQ_QAREG, c, true);

		// decrement refcount
		_item_remove(new_trans_it);
	}

	_item_remove(it);

	if (trans_it != NULL) {
		_item_remove(trans_it);
	}

	pthread_mutex_unlock(&cache_lock);

	*markedVal = 1;
	return MC_OK;
}

rstatus_t
item_delete_and_release(char* tid, u_int32_t ntid, struct conn *c) {
	struct item* trans_it = NULL;
	struct item* it = NULL;
	trig_cursor_t cursor;

	log_debug(LOG_VERB, "delete_and_release for '%.*s'", ntid, tid);

	pthread_mutex_lock(&cache_lock);

	trans_it = _item_get(tid, ntid);
	if (trans_it == NULL) {
		log_debug(LOG_VERB, "delete_and_release trans item not found %s", tid);

		pthread_mutex_unlock(&cache_lock);
		return MC_INVALID;
	}

	char *data = item_data(trans_it);
	char *key;
	size_t nkey;

	// loop through keys and delete
	cursor = NULL;
	while (trig_keylist_next(data, trans_it->nbyte, &cursor, &key, &nkey) == TRIG_OK) {
		it = _item_get(key, nkey);

		if (it == NULL) {
			// Item may have been evicted due to the cache running out of memory.
			// We should continue to try to delete the rest of the keys.
			log_debug(LOG_VERB, "delete_and_release item not found %s", key);
			continue;
		}

		if (it->num_of_trans_q_inv_leases > 0) {
			it->num_of_trans_q_inv_leases--;

			if (it->num_of_trans_q_inv_leases == 0) {
				_item_unlink(it);
				_item_remove(it);
			} else {
				// delete value by allocating new item with same key but has no value
				struct item* new_it = _item_create_lease_holder(key, nkey, NULL, true);

				new_it->num_of_trans_q_inv_leases = it->num_of_trans_q_inv_leases;
				u_int8_t flags = it->flags;

				// TODO(hieu): Check return code for item_store and handle failures
				// replace old item
				_item_store(new_it, REQ_QAREG, c, true);

				new_it->flags = flags;
				new_it->exptime = it->exptime;

				// Since the flags value was overwritten above, ensure that this item
				// is marked as a reserve_item by setting it as a lease_holder.
				// This needs to be marked as a reserve_item since it was allocated from the
				// reserve slab.
				item_set_lease_holder(new_it);

				_item_remove(it);
				_item_remove(new_it);
			}

			continue;
		}

		_item_unlink(it);
		_item_remove(it);
	}

	_item_unlink(trans_it);
	_item_remove(trans_it);

	pthread_mutex_unlock(&cache_lock);

	return MC_OK;
}

/**
 * Try to quarantine and read the item.
 * Return MC_OK if it successfully quarantine and read the item
 */
rstatus_t
item_quarantine_and_read(char* key, uint8_t nkey, lease_token_t lease_token, struct conn *c) {
	struct item* it = NULL;
	rstatus_t status = MC_OK;

	log_debug(LOG_VERB, "quarantine_and_read for '%.*s'", nkey, key);

	pthread_mutex_lock(&cache_lock);
	it = _item_get(key, nkey);

	if (it == NULL) {
		// create a lease holder and grant a q lease
		it = _item_create_reserved_item(key, nkey, 0, NULL, true);

		if (it == NULL) {
			status = MC_ERROR;
		} else {
			stats_thread_incr_by(num_q_lease_token , 1);
			stats_thread_incr(total_lease_token);

			// return the lease and no value
			item_set_qlease(it);
			_item_store(it, REQ_QAREAD, c, true);

			status = MC_OK;
		}
	} else {
		// if item has lease on it, return INVALID
		if (item_is_lease_holder(it)) {
			// if item does not have write lease, it must have read lease
			// TODO: it MUST have straight way to check if it is a read lease
			if (!item_has_q_inv_lease(it) && !item_has_qlease(it)) {
				status = MC_OK;

				it->lease_token = lease_next_token();
				item_set_qlease(it);

				_item_remove(it);
				pthread_mutex_unlock(&cache_lock);
				return status;
			} else {		// item is currently has q-lease
				if (lease_token == NO_LEASE) {
					status = MC_INVALID;
					_item_remove(it);
					stats_thread_incr(qlease_aborts);
					stats_thread_incr(backoff);
					pthread_mutex_unlock(&cache_lock);
					return status;
				}

				if (lease_token != it->lease_token)	// lease token provided not match, back-off
					status = MC_INVALID;
				else
					status = MC_OK;
					
				if (status == MC_INVALID) {
					stats_thread_incr(qlease_aborts);
					stats_thread_incr(backoff);
				}
			}
		} else {		// item has no lease
			// if item has no lease and has value, move this item to reserved space and grant a q lease
			struct item* new_it = _item_create_reserved_item(key, nkey, it->nbyte, NULL, true);

			if (new_it == NULL) {
				status = MC_ERROR;
			} else {
				stats_thread_incr_by(num_q_lease_token, 1);
				stats_thread_incr(total_lease_token);

				new_it->dataflags = it->dataflags;

				memcpy(item_data(new_it), item_data(it), it->nbyte);

				item_set_qlease(new_it);

				_item_replace(it, new_it);

				_item_remove(new_it);

				status = MC_OK;
			}
		}
	}

	if (it != NULL) {
		_item_remove(it);
	}

	pthread_mutex_unlock(&cache_lock);

	return status;
}

item_store_result_t
item_swap_and_release(struct item* it, struct conn *c) {
	struct item *oit = NULL;
	item_store_result_t status = STORED;

	log_debug(LOG_VERB, "swap_and_release for '%.*s'", it->nkey, item_key(it));

	pthread_mutex_lock(&cache_lock);
	oit = _item_get(item_key(it), it->nkey);
	if (oit == NULL) {
		status = NOT_FOUND;
	} else {
		if (oit->lease_token != it->lease_token) {
			status = NOT_FOUND;
		} else {
			if (oit->nbyte == 0 && it->nbyte == 0) {
				// just delete the item
				_item_unlink(oit);
				status = NOT_STORED;
			} else {
				// create new item at normal space and replace old item
				_item_replace(oit, it);

				// no need to keep lease token anymore, so reset it to zero
				it->lease_token = 0;
				status = STORED;
			}
		}
	}

	if (oit != NULL) {
		_item_remove(oit);
	}

	pthread_mutex_unlock(&cache_lock);

	return status;
}

item_store_result_t
item_get_and_delete(char* key, uint8_t nkey, struct conn *c)
{
	struct item *it = NULL; /* item for this key */
	item_store_result_t found;

	log_debug(LOG_VERB, "get_and_delete for '%.*s'", nkey, key);

	pthread_mutex_lock(&cache_lock);
	it = _item_get(key, nkey);

	if (it != NULL) {
		found = EXISTS;

		_item_unlink(it);
		_item_remove(it);
	} else
		found = NOT_FOUND;

	pthread_mutex_unlock(&cache_lock);

	return found;
}


struct item*
item_get_and_lease(const char* key, uint8_t nkey, struct conn *c, lease_token_t* lease_token,
		bool overwrite_lease)
{
	struct item *it = NULL; /* item for this key */
	item_store_result_t ret;
	time_t exptime;

	pthread_mutex_lock(&cache_lock);
	it = _item_get(key, nkey);
	if (it == NULL) {
		stats_thread_incr(delete_miss);

		/* Item not found, create a lease token */
		it = _item_create_lease_holder(key, nkey, NULL, true);
		if (it == NULL) {
			ret = STORE_ERROR;
		} else {
			*lease_token = it->lease_token;
			ret = _item_store(it, c->req_type, c, true);
		}

		switch(ret) {
		// TODO: handle failure scenarios
		case STORED:
			stats_thread_incr_by(num_lease_token, 1);
			stats_thread_incr_by(num_i_lease_token, 1);
			stats_thread_incr(total_lease_token);
			break;
		default:
			item_handle_lease_failure(key, nkey, NULL);
			break;
		}
	} else {
		if(it->lease_token == LEASE_DELETED || overwrite_lease) {
			/* Assign with new lease token and update expiry time */
			it->lease_token = lease_next_token();
			*lease_token = it->lease_token;

			exptime = settings.lease_token_expiry / 1000;

			if(exptime == 0) {
				/* Minimum 1 second expiry time */
				exptime = 1;
			}

			if(exptime > 0) {
				/* Add 1 second to account for rounding errors */
				exptime++;
			}

			it->exptime = time_reltime(exptime);
			clocktime_now(item_created_time(it));
			_item_touch(it);

			if (!item_is_lease_holder(it)) {
                stats_thread_incr(num_lease_token);
                stats_thread_incr(num_i_lease_token);
                stats_thread_incr(total_lease_token);
			}
		} else {
			/* Someone else already has the lease, return hot_miss */
			*lease_token = LEASE_HOTMISS;
		}
	}

	// item should not be _item_remove'd here because it will be returned 
	// and accessed by the caller.
//	if (it != NULL) {
//		_item_remove(it);
//	}

	pthread_mutex_unlock(&cache_lock);
	return it;
}

void
item_unset_pinned(struct item *it)
{
	item_release_refcount(it);
}


void
item_set_pinned(struct item *it)
{
	item_acquire_refcount(it);
}
