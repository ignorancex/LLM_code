#include <pthread.h>
#include <mc_lease.h>

pthread_mutex_t lease_lock;                     /* lock protecting lru q and hash */
lease_token_t	lease_current_token;			/* Current token value */

void
lease_init(void) {
	pthread_mutex_init(&lease_lock, NULL);
	lease_current_token = LEASE_START;
}

lease_token_t
lease_next_token(void) {
	lease_token_t val;
	pthread_mutex_lock(&lease_lock);
	val = lease_current_token++;
	pthread_mutex_unlock(&lease_lock);

	return val;
}

