/*
 * mc_reg_thread.c
 *
 *  Created on: Mar 05, 2014
 *      Author: Jason Yap
 */

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <mc_reg_thread.h>

#define CMD_EXECUTE_QUERY 	101
#define CMD_EXECUTE_UPDATE 	401
#define BUFFER_SIZE			1024 * 20	// 20KB buffer to hold trigger text


static pthread_mutex_t enqueue_lock;    /* lock for adding items to the queue */
static pthread_mutex_t queue_lock;    /* lock for removing items from the queue */
static reg_queue_item* queue_head;
static reg_queue_item* queue_tail;
static int sock_conn;					/* Socket Connection to the JDBC forwarding server */

/* TODO: replace this implementation for maintaining the trigger mapping with
 *       a set or hashmap.
 */
static reg_queue_item* trigger_map[100];
static uint32_t trigger_map_num_items;

static rstatus_t
thread_create(thread_func_t func, void *arg)
{
    pthread_t thread;
    err_t err;

    err = pthread_create(&thread, NULL, func, arg);
    if (err != 0) {
        log_error("pthread create failed: %s", strerror(err));
        return MC_ERROR;
    }

    return MC_OK;
}

static rstatus_t
connect_to_server(char* jdbc_server_hostname) {
	struct sockaddr_in serv_addr;

	if((sock_conn = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		printf("\n Error : Could not create socket \n");
		return MC_ERROR;
	}

	memset(&serv_addr, '0', sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	
	/* TODO: fix hard-coded port number. Read as input argument from twemcache initialization */
	serv_addr.sin_port = htons(4747);

	if(inet_pton(AF_INET, jdbc_server_hostname, &serv_addr.sin_addr)<=0)
	{
		printf("\n inet_pton error occurred\n");
		return MC_ERROR;
	}

	if( connect(sock_conn, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
	{
	   printf("\n Error : Connect Failed \n");
	   return MC_ERROR;
	}

	return MC_OK;
}

static rstatus_t
send_message(int sockfd, char * buffer, uint32_t message_size) {
	int32_t bytes_sent = 0;
	int32_t n = 0;

	while (bytes_sent < message_size) {
		n = write(sockfd,buffer,message_size);
		if (n <= 0) {
			return MC_ERROR;
		}

		bytes_sent += n;
	}

	return MC_OK;
}

static int32_t
recv_message(int sockfd, char * buffer, uint32_t buffer_size) {
	return read(sockfd, buffer, buffer_size);
}

static void *
reg_thread_main(void *arg) {
	char send_buffer[BUFFER_SIZE];
	char recv_buffer[BUFFER_SIZE];
	int16_t command = CMD_EXECUTE_UPDATE;
	reg_queue_item *queue_item = NULL;
	uint32_t message_size;
	int32_t response_len;
	char* cptr;

	while (true) {
		message_size = 0;
		cptr = send_buffer;
		queue_item = NULL;

		// Pull an element from the queue and send it to the JDBC forwarding service.
		pthread_mutex_lock(&queue_lock);
		if (queue_head != NULL) {
			queue_item = queue_head;
			queue_head = queue_head->next;

			// If this was the only item in the queue, fix the queue_tail pointer as well.
			if (queue_head == NULL) {
				queue_tail = queue_head;
			}
		}
		pthread_mutex_unlock(&queue_lock);

		// No item obtained from the queue. Sleep and try again.
		if (queue_item == NULL) {
			sleep(1);
			continue;
		}

		// Construct the message.
		*(int16_t*) cptr = htons(command);
		cptr += sizeof(command);
		message_size += sizeof(command);

		*(uint32_t*) cptr = htonl(queue_item->trigger_len);
		cptr += sizeof(queue_item->trigger_len);
		message_size += sizeof(queue_item->trigger_len);

		memcpy(cptr, queue_item->trigger, queue_item->trigger_len);
		cptr += queue_item->trigger_len;
		message_size += queue_item->trigger_len;

		// Send it to the forwarding server.
		send_message(sock_conn, send_buffer, message_size);

		// Get the response.
		response_len = recv_message(sock_conn, recv_buffer, BUFFER_SIZE);
		ASSERT (response_len == sizeof(int32_t));

		if (*(int32_t*)recv_buffer != MC_OK) {
			// TODO: handle failure to register trigger.
		}

		// Check if this trigger needs to be added to the trigger map.
		if (!trigger_map_contains(queue_item->trigger, queue_item->trigger_len)) {
			trigger_map_add(queue_item);
		} else {
			// Clean up the queue item.
			free (queue_item);
		}
	}

	return NULL;
}

/***
 * Adds a trigger to a queue to get registered with the RDBMS by a
 * background thread.
 */
rstatus_t
reg_enqueue(char* str, uint32_t str_len) {
	reg_queue_item *queue_item = NULL;
	if (trigger_map_contains(str, str_len)) {
		return MC_OK;
	}

	// Create the queue item and copy the information in.
	queue_item = (reg_queue_item*)malloc(sizeof(reg_queue_item) + str_len);
	memcpy(queue_item->end, str, str_len);
	queue_item->trigger_len = str_len;
	queue_item->trigger = queue_item->end;
	queue_item->next = NULL;

	pthread_mutex_lock(&queue_lock);
	// Slot it into the queue. FIFO order.
	if (queue_head == NULL || queue_tail == NULL) {
		ASSERT (queue_head == queue_tail);

		queue_head = queue_item;
		queue_tail = queue_head;
	} else {
		queue_tail->next = queue_item;
		queue_tail = queue_item;
	}
	pthread_mutex_unlock(&queue_lock);
	return MC_OK;
}

void
reg_init(char* jdbc_server_hostname) {
	pthread_mutex_init(&enqueue_lock, NULL);
	pthread_mutex_init(&queue_lock, NULL);
	queue_head = NULL;
	queue_tail = NULL;

	rstatus_t status = connect_to_server(jdbc_server_hostname);
	ASSERT(status == MC_OK);

	thread_create(reg_thread_main, NULL);

	trigger_map_num_items = 0;

//	char* test = "update users set rescount = 44 where userid = 51";
//	reg_enqueue(test, strlen(test));
//
//	char* test2 = "update users set rescount = 54 where userid = 51";
//	reg_enqueue(test2, strlen(test2));
//
//	sleep(2);
//	reg_enqueue(test, strlen(test));
}

void
reg_deinit(void) {
	reg_queue_item *queue_item;
	pthread_mutex_lock(&queue_lock);
	while (queue_head != NULL) {
		queue_item = queue_head;
		queue_head = queue_head->next;
		if (queue_head == NULL) {
			queue_tail = queue_head;
		}

		free (queue_item);
	}
	pthread_mutex_unlock(&queue_lock);
	close(sock_conn);

}


/***
 * Returns true if the specified trigger has been registered.
 */
bool
trigger_map_contains(char* str, uint32_t str_len) {
	int i;

	// TODO: replace this array based implementation with a hashmap/set.
	pthread_mutex_lock(&queue_lock);
	for (i = 0; i < trigger_map_num_items; i++) {
		ASSERT(trigger_map[i] != NULL);
		if (str_len == trigger_map[i]->trigger_len) {
			if (strncmp(str, trigger_map[i]->trigger, str_len) == 0) {
				pthread_mutex_unlock(&queue_lock);
				return true;
			}
		}
	}
	pthread_mutex_unlock(&queue_lock);
	return false;
}


/***
 * Adds the queue item (specifying the trigger details) to the trigger map.
 *
 */
rstatus_t
trigger_map_add(reg_queue_item* item) {
	pthread_mutex_lock(&queue_lock);
	trigger_map[trigger_map_num_items] = item;
	trigger_map_num_items++;
	pthread_mutex_unlock(&queue_lock);
	return MC_OK;
}
