/*
 * mc_index_test.c
 *
 *  Created on: Nov 20, 2013
 *      Author: jyap
 */


#include <stdlib.h>
#include <mc_core.h>

index_struct p_index;

static void
indextest_init(void) {
	index_init(&p_index);
}

static void
indextest_deinit(void) {
	struct item* it = index_get_lowest(&p_index);
	while (it != NULL) {
		ASSERT(index_delete(&p_index, it) == MC_OK);
		it = index_get_lowest(&p_index);
	}
}

static struct item*
find_key(char* key, uint8_t nkey) {
	struct item* it = index_get_lowest(&p_index);

	while (it != NULL) {
		if (it->nkey == nkey && memcmp(item_key(it), key, nkey) == 0) {
			// This item is a match for the key.
			break;
		}

		it = index_get_next(&p_index, it);
	}

	return it;
}

static void
indextest_insert(void) {
	indextest_init();

	char* key = "test1";
	uint8_t nkey = strlen(key);
	char* value = "value1";
	uint32_t nbyte = strlen(value);

	// Check that it doesn't exist before insertion
	struct item* ret_it = find_key(key, nkey);
	ASSERT(ret_it == NULL);

	struct item* it = item_alloc(item_expected_size(nkey, nbyte), key, nkey, 0, 0, nbyte);
	ASSERT(it != NULL);

	// Set the value
	memcpy(item_data(it), value, nbyte);

	// Insert the item
	ASSERT(index_insert(&p_index, it) == MC_OK);

	// Check that it exists after insertion
	ret_it = find_key(key, nkey);
	ASSERT(ret_it != NULL);

	// Check that the value matches
	ASSERT(it->nkey == ret_it->nkey);
	ASSERT(memcmp(item_data(it), item_data(ret_it), it->nbyte) == 0);

	// Insert a second key-value
	char* key2 = "test2a";
	uint8_t nkey2 = strlen(key2);
	char* value2 = "value2a";
	uint32_t nbyte2 = strlen(value2);

	// Check that it doesn't exist before insertion
	ret_it = find_key(key2, nkey2);
	ASSERT(ret_it == NULL);

	struct item* it2 = item_alloc(item_expected_size(nkey2, nbyte2), key2, nkey2, 0, 0, nbyte2);
	ASSERT(it2 != NULL);

	// Set the value
	memcpy(item_data(it2), value2, nbyte2);

	// Insert the item
	ASSERT(index_insert(&p_index, it2) == MC_OK);

	// Check that it exists after insertion
	ret_it = find_key(key2, nkey2);
	ASSERT(ret_it != NULL);

	// Check that the value matches
	ASSERT(it2->nkey == ret_it->nkey);
	ASSERT(memcmp(item_key(it2), item_key(ret_it), it2->nkey) == 0);

	indextest_deinit();
	printf("OK INDEX_INSERT\r\n");
}

static void
indextest_delete(void) {
	indextest_init();

	char* key = "test1";
	uint8_t nkey = strlen(key);
	char* value = "value1";
	uint32_t nbyte = strlen(value);

	// Check that it doesn't exist before insertion
	struct item* ret_it = find_key(key, nkey);
	ASSERT(ret_it == NULL);

	struct item* it = item_alloc(item_expected_size(nkey, nbyte), key, nkey, 0, 0, nbyte);
	ASSERT(it != NULL);

	// Set the value
	memcpy(item_data(it), value, nbyte);

	// Insert the item
	ASSERT(index_insert(&p_index, it) == MC_OK);

	// Insert a second key-value
	char* key2 = "test2a";
	uint8_t nkey2 = strlen(key2);
	char* value2 = "value2a";
	uint32_t nbyte2 = strlen(value2);

	// Check that it doesn't exist before insertion
	ret_it = find_key(key2, nkey2);
	ASSERT(ret_it == NULL);

	struct item* it2 = item_alloc(item_expected_size(nkey2, nbyte2), key2, nkey2, 0, 0, nbyte2);
	ASSERT(it2 != NULL);

	// Set the value
	memcpy(item_data(it2), value2, nbyte2);

	// Insert the item
	ASSERT(index_insert(&p_index, it2) == MC_OK);

	// Delete the first item
	ASSERT(index_delete(&p_index, it) == MC_OK);

	// Check that it is no longer there
	ret_it = find_key(key, nkey);
	ASSERT(ret_it == NULL);

	// Check that the other item still exists
	ret_it = find_key(key2, nkey2);
	ASSERT(ret_it != NULL);

	// Check that the value matches
	ASSERT(it2->nkey == ret_it->nkey);
	ASSERT(memcmp(item_key(it2), item_key(ret_it), it2->nkey) == 0);

	// Delete the first item
	ASSERT(index_delete(&p_index, it2) == MC_OK);

	// Check that it is no longer there
	ret_it = find_key(key2, nkey2);
	ASSERT(ret_it == NULL);

	// Check that insert after delete works
	// Insert the item
	ASSERT(index_insert(&p_index, it2) == MC_OK);

	// Check that it exists after insertion
	ret_it = find_key(key2, nkey2);
	ASSERT(ret_it != NULL);

	// Check that the value matches
	ASSERT(it2->nkey == ret_it->nkey);
	ASSERT(memcmp(item_key(it2), item_key(ret_it), it2->nkey) == 0);

	indextest_deinit();
	printf("OK INDEX_DELETE\r\n");
}

static void
indextest_get_lowest(void) {
	indextest_init();

	char* key = "test1";
	uint8_t nkey = strlen(key);
	char* value = "value1";
	uint32_t nbyte = strlen(value);

	// Check that it doesn't exist before insertion
	struct item* ret_it = find_key(key, nkey);
	ASSERT(ret_it == NULL);

	// There should be no items in the index right now
	ret_it = index_get_lowest(&p_index);
	ASSERT(ret_it == NULL);

	struct item* it = item_alloc(item_expected_size(nkey, nbyte), key, nkey, 0, 0, nbyte);
	ASSERT(it != NULL);

	// Set the value
	memcpy(item_data(it), value, nbyte);

	// Insert the item
	ASSERT(index_insert(&p_index, it) == MC_OK);

	// Retrieve the lowest item in the index (which should be the just
	// inserted item because that is the only one that exists)
	ret_it = index_get_lowest(&p_index);
	ASSERT(ret_it != NULL);

	// Check that the value matches
	ASSERT(it->nkey == ret_it->nkey);
	ASSERT(memcmp(item_data(it), item_data(ret_it), it->nbyte) == 0);

	// Insert a second key-value
	char* key2 = "test2a";
	uint8_t nkey2 = strlen(key2);
	char* value2 = "value2a";
	uint32_t nbyte2 = strlen(value2);

	// Check that it doesn't exist before insertion
	ret_it = find_key(key2, nkey2);
	ASSERT(ret_it == NULL);

	struct item* it2 = item_alloc(item_expected_size(nkey2, nbyte2), key2, nkey2, 0, 0, nbyte2);
	ASSERT(it2 != NULL);

	// Set the value
	memcpy(item_data(it2), value2, nbyte2);

	// Insert the item
	ASSERT(index_insert(&p_index, it2) == MC_OK);

	// Retrieve the lowest item in the index (could be either the first
	// or second item, depending on the replacement technique used)
	ret_it = index_get_lowest(&p_index);
	ASSERT(ret_it != NULL);

	// Check that the value matches either item
	ASSERT(it->nbyte == ret_it->nbyte || it2->nbyte == ret_it->nbyte );
	if (it->nbyte == ret_it->nbyte) {
		ASSERT(memcmp(item_data(it), item_data(ret_it), it->nbyte) == 0);
	} else {
		ASSERT(memcmp(item_data(it2), item_data(ret_it), it2->nbyte) == 0);
	}

	indextest_deinit();
	printf("OK INDEX_GET_LOWEST\r\n");
}

static void
indextest_get_next(void) {
	indextest_init();

	char* key = "test1";
	uint8_t nkey = strlen(key);
	char* value = "value1";
	uint32_t nbyte = strlen(value);

	// Check that it doesn't exist before insertion
	struct item* ret_it = find_key(key, nkey);
	ASSERT(ret_it == NULL);

	// There should be no items in the index right now
	ret_it = index_get_lowest(&p_index);
	ASSERT(ret_it == NULL);

	struct item* it = item_alloc(item_expected_size(nkey, nbyte), key, nkey, 0, 0, nbyte);
	ASSERT(it != NULL);

	// Set the value
	memcpy(item_data(it), value, nbyte);

	// Insert the item
	ASSERT(index_insert(&p_index, it) == MC_OK);

	// Retrieve the lowest item in the index (which should be the just
	// inserted item because that is the only one that exists)
	ret_it = index_get_lowest(&p_index);
	ASSERT(ret_it != NULL);

	// This should be the only item in the cache now.
	ret_it = index_get_next(&p_index, ret_it);
	ASSERT(ret_it == NULL);

	// Insert a second key-value
	char* key2 = "test2a";
	uint8_t nkey2 = strlen(key2);
	char* value2 = "value2a";
	uint32_t nbyte2 = strlen(value2);

	// Check that it doesn't exist before insertion
	ret_it = find_key(key2, nkey2);
	ASSERT(ret_it == NULL);

	struct item* it2 = item_alloc(item_expected_size(nkey2, nbyte2), key2, nkey2, 0, 0, nbyte2);
	ASSERT(it2 != NULL);

	// Set the value
	memcpy(item_data(it2), value2, nbyte2);

	// Insert the item
	ASSERT(index_insert(&p_index, it2) == MC_OK);

	// Retrieve the lowest item in the index (could be either the first
	// or second item, depending on the replacement technique used)
	ret_it = index_get_lowest(&p_index);
	ASSERT(ret_it != NULL);

	// Check that the value matches either item
	ASSERT(it->nbyte == ret_it->nbyte || it2->nbyte == ret_it->nbyte );
	if (it->nbyte == ret_it->nbyte) {
		ASSERT(memcmp(item_data(it), item_data(ret_it), it->nbyte) == 0);

		// Check that the next item is the other inserted item
		ret_it = index_get_next(&p_index, ret_it);
		ASSERT(ret_it != NULL);
		ASSERT(it2->nbyte == ret_it->nbyte );
		ASSERT(memcmp(item_data(it2), item_data(ret_it), it2->nbyte) == 0);
	} else {
		ASSERT(memcmp(item_data(it2), item_data(ret_it), it2->nbyte) == 0);

		// Check that the next item is the other inserted item
		ret_it = index_get_next(&p_index, ret_it);
		ASSERT(ret_it != NULL);
		ASSERT(it->nbyte == ret_it->nbyte );
		ASSERT(memcmp(item_data(it), item_data(ret_it), it->nbyte) == 0);
	}

	indextest_deinit();
	printf("OK INDEX_GET_NEXT\r\n");
}

void
indextest_run_all(void) {
	indextest_insert();
	indextest_delete();
	indextest_get_lowest();
	indextest_get_next();
}
