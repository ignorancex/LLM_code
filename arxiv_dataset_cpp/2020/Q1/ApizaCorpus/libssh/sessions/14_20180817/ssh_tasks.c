#define LIBSSH_STATIC 1
#include <libssh/libssh.h> 
#include <stdlib.h>
#include <stdio.h>
#include <libssh/sftp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

char * SERVER = "localhost";
char * PASSWORD = "password";

int show_remote_user(ssh_session session)
{
    return -1;
}

int sftp_operations(ssh_session session)
{
    return -1;
}

int main()
{
    printf("Hello world!\n");
    int ret = 0;
    //int state = 0;
    int hash = 0;
    int hlen = 0;
    char * p = NULL;
    char*  buf = NULL;
    int hexa = 0;
    int cmp = 0;
    int rc = 0;
    enum ssh_known_hosts_e state;
    ssh_key host_publickey = NULL;


    ssh_session new_session = ssh_new();

    if (new_session  == NULL) {
    	printf("Cannot create a new SSH session\n");
    	exit(-1);
    }
    
    else {
    	printf("Created a new SSH session\n");
    	ssh_options_set(new_session, SSH_OPTIONS_HOST, SERVER);

    	ret = ssh_connect(new_session);

    	if(ret == SSH_ERROR) {
    		printf("Cannot connect to %s\n", SERVER);
    	}
    	else if (ret = SSH_OK) {
    		printf("Connect to %s\n", SERVER);

    		rc = ssh_get_server_publickey(new_session, &host_publickey);
		    if (rc < 0) {
		        return -1;
		    }






    		state = ssh_is_server_known(new_session);
		   	switch (state) {
			       case SSH_SERVER_KNOWN_OK:
			           /* OK */
		    		printf("Host is known %s\n", SERVER);

			   //         break;
			       // case SSH_KNOWN_HOSTS_CHANGED:
			       //     fprintf(stderr, "Host key for server changed: it is now:\n");
			       //     ssh_print_hexa("Public key hash", hash, hlen);
			       //     fprintf(stderr, "For security reasons, connection will be stopped\n");
			       //     ssh_clean_pubkey_hash(&hash);
			       //     return -1;
			       // case SSH_KNOWN_HOSTS_OTHER:
			       //     fprintf(stderr, "The host key for this server was not found but an other"
			       //             "type of key exists.\n");
			       //     fprintf(stderr, "An attacker might change the default server key to"
			       //             "confuse your client into thinking the key does not exist\n");
			       //     ssh_clean_pubkey_hash(&hash);
			       //     return -1;
			       // case SSH_KNOWN_HOSTS_NOT_FOUND:
			       //     fprintf(stderr, "Could not find known host file.\n");
			       //     fprintf(stderr, "If you accept the host key here, the file will be"
			       //             "automatically created.\n");
			       //     /* FALL THROUGH to SSH_SERVER_NOT_KNOWN behavior */
			       // case SSH_KNOWN_HOSTS_UNKNOWN:
			       //     hexa = ssh_get_hexa(hash, hlen);
			       //     fprintf(stderr,"The server is unknown. Do you trust the host key?\n");
			       //     fprintf(stderr, "Public key hash: %s\n", hexa);
			       //     ssh_string_free_char(hexa);
			       //     ssh_clean_pubkey_hash(&hash);
			       //     p = fgets(buf, sizeof(buf), stdin);
			       //     if (p == NULL) {
			       //         return -1;
			       //     }
			       //     cmp = strncasecmp(buf, "yes", 3);
			       //     if (cmp != 0) {
			       //         return -1;
			       //     }
			       //     rc = ssh_session_update_known_hosts(new_session);
			       //     if (rc < 0) {
			       //         fprintf(stderr, "Error %s\n", strerror(errno));
			       //         return -1;
			       //     }
			       //     break;
			       // case SSH_KNOWN_HOSTS_ERROR:
			       //     fprintf(stderr, "Error %s", ssh_get_error(new_session));
			       //     ssh_clean_pubkey_hash(&hash);
			       //     return -1;
		   		}

    		ssh_disconnect(new_session);

    	}

    	free(new_session);
    }
}