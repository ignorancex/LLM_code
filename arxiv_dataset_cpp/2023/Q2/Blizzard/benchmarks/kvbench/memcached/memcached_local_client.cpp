#include <libmemcached/memcached.h>
#include <stdio.h>
#include <string.h>

const unsigned long memcached_keys = 10000UL;
int key_size = 8;
int value_size = 8;
double frac_read = 0.5;


char key[8];
char value[8];

char *retrieved_value;
memcached_st *memc;
memcached_return rc;

void load(unsigned long keys){

	for(unsigned long i=0UL; i < keys; i++){

		snprintf(key,key_size,"k%lu",i);
		snprintf(value,value_size,"v%lu",i);

		rc = memcached_set(memc, key, strlen(key), value, strlen(value), (time_t)0, (uint32_t)0);
		if (rc != MEMCACHED_SUCCESS){
			fprintf(stderr, "Couldn't store key: %s\n", memcached_strerror(memc, rc));
			exit(1);
		}
	}

}

void bench(unsigned long keys){

	double coin;
	unsigned long random_key;
	unsigned long random_value;
	size_t value_length;
	uint32_t flags;

	//for(;;){
	for(int i=0; i<10;i++){
		coin = ((double)rand())/RAND_MAX;
		if(coin > frac_read){
			random_key = rand() % keys;
			random_value = rand() % keys;

			snprintf(key,key_size,"k%lu",random_key);
			snprintf(value,value_size,"v%lu",random_value);

			rc = memcached_set(memc, key, strlen(key), value, strlen(value), (time_t)0, (uint32_t)0);
			if (rc != MEMCACHED_SUCCESS){
				fprintf(stderr, "Couldn't store key: %s\n", memcached_strerror(memc, rc));
				exit(1);
			}

		}else{
			retrieved_value = memcached_get(memc, key, strlen(key), &value_length, &flags, &rc);
			if (rc == MEMCACHED_SUCCESS) {
				fprintf(stderr, "Key retrieved successfully\n");
				printf("The key '%s' returned value '%s'.\n", key, retrieved_value);
				free(retrieved_value);
			}
		}

	}
}


	int main(int argc, char **argv) {
		//memcached_servers_parse (char *server_strings);
		memcached_server_st *servers = NULL;

		memc = memcached_create(NULL);
		servers = memcached_server_list_append(servers, "localhost", 11211, &rc);
		rc = memcached_server_push(memc, servers);

		if (rc == MEMCACHED_SUCCESS){
			fprintf(stderr, "Added server successfully\n");
		}else{
			fprintf(stderr, "Couldn't add server: %s\n", memcached_strerror(memc, rc));
			exit(1);
		}
		load(memcached_keys);
		bench(memcached_keys);
		return 0;
	}
