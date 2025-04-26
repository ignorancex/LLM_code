#include "ChunkedTable.hpp"
#include "Logger.hpp"
#include "util.hpp"
#include <algorithm>
#include <cstdlib>

#ifdef ENCLAVE
#include "ocalls_t.h"
#else
#include "ocalls.hpp"
#endif

/**
 * Add a chunk to a chunked table. Increases num_chunks by 1 and doubles chunk_capacity if num_chunks == chunk_capacity
 * at time of the call.
 *
 * Does not change current_chunk. This operation is independent of moving the current chunk index.
 * @param table
 */
void
add_chunk(chunked_table_t *table) {
#ifdef CHUNKED_TABLE_PREALLOC
    logger(WARN, "add_chunk called although preallocation compiler flag is set.");
#endif

    if (table->num_chunks == table->chunk_capacity) [[unlikely]] {
        // double capacity for chunk pointers
        // use realloc hoping that there is some free memory in place
        table->chunks = static_cast<table_chunk_t **>(
                realloc(table->chunks, table->chunk_capacity * sizeof(table_chunk_t *) * 2));
        malloc_check(table->chunks);
        table->chunk_capacity <<= 1;
    }
    const auto new_chunk = reinterpret_cast<table_chunk_t *>(malloc(sizeof(table_chunk_t)));
    new_chunk->num_tuples = 0;
    // Memory of tuples not written to is undefined. No memset!
    table->chunks[table->num_chunks] = new_chunk;
    table->num_chunks++;
}

/**
 * Initialize a chunked table with exactly one allocated chunk and capacity for 8 chunk pointers.
 * table->chunk_capacity = 8;
 * table->num_chunks = 1;
 * table->current_chunk = 0;
 * table->num_tuples = 0;
 * @param table
 */
void
init_chunked_table(chunked_table_t *table) {
    if (table->chunk_capacity > 0) [[unlikely]] {
        logger(ERROR, "Trying to initialize a chunked table that is already initialized!");
        return;
    }
    table->num_tuples = 0;
    table->num_chunks = 0;
    table->chunks = static_cast<table_chunk_t **>(aligned_alloc(64, 64)); // allocate exactly one cache line
    malloc_check(table->chunks)
    table->chunk_capacity = 64 / sizeof(table_chunk_t *); // should be 8
    add_chunk(table); // add the first chunk
    table->current_chunk = 0;
}

/**
 * Initializa a table with exacly num_chunks allocated chunks and capacity for num_chunks.
 * table->chunk_capacity = num_chunks;
 * table->num_chunks = num_chunks;
 * table->current_chunk = 0;
 * table->num_tuples = 0;
 * @param table
 * @param num_chunks
 */
void
init_chunked_table_prealloc(chunked_table_t *table, const uint64_t num_chunks) {
    if (table->chunk_capacity > 0) [[unlikely]] {
        logger(ERROR, "Trying to initialize a chunked table that is already initialized!");
        return;
    }

    table->num_tuples = 0;
    table->num_chunks = num_chunks;
    table->chunk_capacity = num_chunks;
    table->chunks = static_cast<table_chunk_t **>(aligned_alloc(64, sizeof(table_chunk_t *) * num_chunks));
    malloc_check(table->chunks)

    for (uint64_t i = 0; i < num_chunks; ++i) {
        const auto new_chunk = reinterpret_cast<table_chunk_t *>(malloc(sizeof(table_chunk_t)));
        malloc_check(new_chunk)
        new_chunk->num_tuples = 0;
        // Memory of tuples not written to is undefined. No memset!
        table->chunks[i] = new_chunk;
    }

    table->current_chunk = 0;
}

void
insert_output(chunked_table_t *table, type_key key, type_value Rpayload, type_value Spayload) {
    auto chunk = table->chunks[table->current_chunk];
    if (chunk->num_tuples >= TUPLES_PER_CHUNK) [[unlikely]] {
        // current chunk full
        if (table->current_chunk == table->num_chunks - 1) {
            // if the current chunk is the last allocated chunk, allocate a new chunk
            add_chunk(table);
        }
        ++table->current_chunk; // Increment the current chunk index
        chunk = table->chunks[table->current_chunk]; // move the chunk pointer.
    }

    chunk->tuples[chunk->num_tuples].key = key;
    chunk->tuples[chunk->num_tuples].Rpayload = Rpayload;
    chunk->tuples[chunk->num_tuples].Spayload = Spayload;
    chunk->num_tuples++;
}

void
finish_chunked_table(chunked_table_t *table) {
    for (uint64_t i = 0; i < table->num_chunks; ++i) {
        table->num_tuples += table->chunks[i]->num_tuples;
    }
}

/**
 * Frees all chunks and the chunk pointer array. Does not free the table itself.
 * @param table
 */
void
destroy_table(chunked_table_t *table) {
    for (uint64_t i = 0; i < table->num_chunks; ++i) {
        free(table->chunks[i]);
    }
    free(table->chunks);
    table->chunk_capacity = 0;
    table->num_chunks = 0;
    table->current_chunk = 0;
}

/**
 * Concatenates multiple chunked tables by concatenating their chunk pointers.
 *
 * In the resulting table, it is not guaranteed that chunks are full. Thus, the number of tuples in each chunk must be
 * checked before reading.
 * @param tables vector of chunked_table_t
 * @return Pointer to a chunked table that contains pointers to all chunks of the input tables.
 */
chunked_table_t *
concatenate(const std::vector<chunked_table_t> &tables) {
    auto result_table = static_cast<chunked_table_t *>(malloc(sizeof(chunked_table_t)));

    // Calculate required number of chunks and number of tuples
    result_table->num_chunks = 0;
    result_table->num_tuples = 0;
    for (const chunked_table_t &table : tables) {
        result_table->num_chunks += table.num_chunks;
        result_table->num_tuples += table.num_tuples;
    }

    // Allocate array for chunk pointers
    result_table->chunks = (table_chunk_t **) malloc(sizeof(table_chunk_t *) * result_table->num_chunks);
    malloc_check(result_table->chunks)
    result_table->chunk_capacity = result_table->num_chunks;

    // Copy chunk pointers
    uint64_t start = 0;
    for (const chunked_table_t &table: tables) {
        std::copy_n(table.chunks, table.num_chunks, result_table->chunks + start);
        start += table.num_chunks;
    }
    result_table->current_chunk = result_table->num_chunks - 1;
    return result_table;
}
