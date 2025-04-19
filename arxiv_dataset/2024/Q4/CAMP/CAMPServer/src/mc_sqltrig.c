/*
 * mc_sqltrig.c
 *
 *  Created on: Apr 10, 2013
 *      Author: Jason Yap
 */

#include <stdlib.h>
#include <mc_sqltrig.h>


/***
 * Allocate a memory block to hold a registration entry.
 * Copies the data in trigger_text and internal_token to the end of the
 *  block and points to them with the text and internal_token members
 *  of the returned entry.
 */
trig_entry_t* 	trig_allocRegistrationEntry(
		char* trigger_text, uint32_t text_len,
		char* internal_token, uint32_t internal_token_len) {

	size_t total_size = TRIG_ENTRY_SIZE + text_len + internal_token_len;
	trig_entry_t* new_entry = malloc(total_size);

	char* cptr;
	if (new_entry == NULL) {
		return NULL;
	}

	new_entry->next = NULL;
	new_entry->template_id = 0;

	/* Copy Trigger text and Internal Token into allocated memory(after the struct) */
	cptr = (char*)new_entry;
	cptr += TRIG_ENTRY_SIZE;
	memcpy(cptr, trigger_text, text_len);
	new_entry->text = cptr;
	new_entry->text_len = text_len;
	cptr += text_len;

	memcpy(cptr, internal_token, internal_token_len);
	new_entry->internal_token = cptr;
	new_entry->internal_token_len = internal_token_len;
	cptr += internal_token_len;

	ASSERT(cptr - (char*)new_entry == total_size);

	return new_entry;
}

/***
 * Allocates a memory block to hold a registration collection.
 * Copies the key to end of the block and points to it with
 *  the member key in the returned collection.
 */
trig_coll_t*	trig_allocRegistrationColl(char* key, uint32_t key_len) {
	uint32_t total_size = TRIG_COLL_SIZE + key_len;
	trig_coll_t* new_coll = malloc(total_size);
	char* cptr;

	// unable to allocate memory
	if (new_coll == NULL) {
		return NULL;
	}

	new_coll->registration_list = NULL;
	new_coll->num_registrations = 0;

	cptr = (char*)new_coll;
	cptr += TRIG_COLL_SIZE;
	memcpy(cptr, key, key_len);
	new_coll->key = cptr;
	new_coll->key_len = key_len;
	cptr += key_len;

	ASSERT(cptr - (char*)new_coll == total_size);

	return new_coll;
}

trig_status_t trig_checkDuplicateRegistration(char* trigger, uint32_t trigger_len) {

	return TRIG_ERROR;
}

/***
 * Compare key1 with key2.
 * Returns 	< 0		if key2_len < key1_len or first non-matching character in key2 is smaller than in key1
 * 			> 0		if key2_len > key1_len or first non-matching character in key2 is larger than in key1
 * 			0		both keys match exactly
 */
static int32_t
trig_compare_key(
		char* key1, trig_keylen_t key1_len,
		char* key2, trig_keylen_t key2_len) {

	if (key2_len - key1_len != 0) {
		return key2_len - key1_len;
	}

	return memcmp(key1, key2, key1_len);
}

/***
 * Check if the keylist contains new_key.
 */
trig_status_t	trig_check_keylist(
		char* it_list, trig_listlen_t it_listlen,
		char* new_key, trig_keylen_t new_keylen ) {

	char	*cptr = it_list;
	char	*curr_key = NULL;
	trig_keylen_t	curr_keylen = 0;
	int32_t	compare_val = 0;

	ASSERT(it_list != NULL);

	if (it_listlen == 0) {
		return TRIG_NOTFOUND;
	}

	while (cptr - it_list < it_listlen) {
		curr_keylen = *(trig_keylen_t *)cptr;
		cptr += sizeof(curr_keylen);

		curr_key = cptr;
		cptr += curr_keylen;

		compare_val = trig_compare_key(curr_key, curr_keylen, new_key, new_keylen);
		if (compare_val == 0) {
			return TRIG_OK;
		} else if (compare_val < 0) {
			return TRIG_NOTFOUND;
		} else {
			/* continue to check the rest of the list */
		}
	}

	return TRIG_NOTFOUND;
}

/***
 * Add a new key into a keylist, stored at the destination (dest).
 * When an empty keylist is specified, a new keylist is generated at dest
 * 	with the contents being the new_key.
 */
trig_status_t	trig_keylist_addkey(
		char* dest, trig_listlen_t *dest_listlen, trig_listlen_t dest_maxsize,
		char* old_keylist, trig_listlen_t old_keylistlen,
		char* new_key, trig_keylen_t new_keylen) {

	char	*cptr = dest;
	trig_listlen_t	ins_pos = 0;
	trig_keylen_t	key_len = 0;
	int32_t	compare_val = 0;

	ASSERT(dest != NULL);
	ASSERT(dest_maxsize >= old_keylistlen + new_keylen + sizeof(new_keylen));

	if (old_keylist == NULL || old_keylistlen == 0) {
		/* No previous keylist. Treat this as if creating a new keylist */
		*(trig_keylen_t *)cptr = new_keylen;
		cptr += sizeof(new_keylen);

		memcpy(cptr, new_key, new_keylen);
		cptr += new_keylen;
	} else {
		/* Add new key to the existing keylist and store both into the destination */

		/* Find the position in the old keylist where this key should go */
		cptr = old_keylist;
		while (ins_pos < old_keylistlen) {
			key_len = *(trig_keylen_t *)cptr;
			cptr += sizeof(key_len);

			compare_val = trig_compare_key(cptr, key_len, new_key, new_keylen);
			if (compare_val < 0) {
				/* Done. This is where new_key should go */
				break;
			} else if (compare_val > 0) {
				/* new_key should be place after current pos. Move to next key */
				ins_pos += sizeof(key_len) + key_len;
			} else {
				/* Should never get here. The keylist should be checked for the
				 *  existence of the key before trying to merge since merge is unnecessary if
				 *  the key already exists. */
				ASSERT(false);
			}
		}

		/* Fill the dest with the old keylist and the new_key inserted into the proper position */
		cptr = dest;
		if (ins_pos > 0) {
			/* Copy the old keylist before the new_key. Only do this
			 *  if the destination is different from the old_keylist.
			 * Otherwise, the destination already contains the correct data. */
			if(cptr != old_keylist) {
				memcpy(cptr, old_keylist, ins_pos);
			}
			cptr += ins_pos;
		}

		/* Copy the rest of the of the keylist first.
		 * The caller may have specified the destination as the same array, so
		 *   data needs to be moved to its correct location before it is overwritten
		 *   by the incoming key. */
		if (ins_pos < old_keylistlen) {
			/* Copy the old keylist after the new_key */
			memcpy(cptr + new_keylen + sizeof(new_keylen),
					old_keylist + ins_pos, old_keylistlen - ins_pos);
			//cptr += old_keylistlen - ins_pos;
		}

		/* Copy the new_key */
		*(trig_keylen_t *)cptr = new_keylen;
		cptr += sizeof(new_keylen);

		memcpy(cptr, new_key, new_keylen);
		cptr += new_keylen;

		/* Skip over previously copied portion in dest */
		cptr += old_keylistlen - ins_pos;
	}

	*dest_listlen = old_keylistlen + new_keylen + sizeof(new_keylen);
	ASSERT(cptr - (char*)dest == old_keylistlen + new_keylen + sizeof(new_keylen));

	return TRIG_OK;
}

trig_listlen_t 		trig_new_keylist_size(
		char *key, trig_keylen_t key_len) {
	return key_len + sizeof(key_len);
}


trig_status_t	trig_keylist_merge_lists(
		char *dest_keylist, 	trig_listlen_t *dest_keylistlen, trig_listlen_t dest_maxlen,
		char *input_keylist, 	trig_listlen_t input_keylistlen) {

	trig_cursor_t cursor = NULL;
	char* key;
	size_t	nkey_sizet;
	trig_keylen_t nkey;
	while(trig_keylist_next(
			input_keylist, input_keylistlen,
			&cursor,
			&key, &nkey_sizet) == TRIG_OK) {

		/* trig_keylist_next expects a size_t but we want a trig_keylen_t for the next few calls. */
		nkey = (trig_keylen_t) nkey_sizet;

		/* Only add the new key if it doesn't already exist in the list */
		if(trig_check_keylist(dest_keylist, *dest_keylistlen, key, nkey) == TRIG_NOTFOUND) {
			trig_keylist_addkey(dest_keylist, dest_keylistlen, dest_maxlen,
					dest_keylist, *dest_keylistlen,
					key, nkey);

			if(*dest_keylistlen > dest_maxlen) {
				return TRIG_ERROR;
			}
		}
	}

	return TRIG_OK;
}

trig_status_t	trig_keylist_next(
    	char *keylist, 	trig_listlen_t keylistlen,
    	trig_cursor_t *delete_cursor,
    	char **key, 		size_t *nkey) {

	if (keylistlen == 0) {
		return TRIG_NOTFOUND;
	}

	/* Start at the head of the keylist */
	if(*delete_cursor == NULL) {
		*delete_cursor = keylist;
	} else {
		/* Move cursor forward based on the length of the key it is point to */
		*delete_cursor = *delete_cursor + (*(trig_keylen_t *) (*delete_cursor) + sizeof(trig_keylen_t));
	}

	/* Check if cursor has reached the end */
	if (*delete_cursor - keylist < keylistlen) {
		*nkey = *(trig_keylen_t *) (*delete_cursor);
		*key = *delete_cursor + sizeof(trig_keylen_t);
		return TRIG_OK;
	}

	return TRIG_NOTFOUND;
}
