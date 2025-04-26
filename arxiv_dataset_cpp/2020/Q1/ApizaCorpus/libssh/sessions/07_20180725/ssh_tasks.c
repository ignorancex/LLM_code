#define LIBSSH_STATIC 1
#include <libssh/libssh.h> 
#include <stdlib.h>
#include <stdio.h>
#include <libssh/sftp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

char * SERVER = "localhost";
char * PASSWORD = "password";

int show_remote_user(ssh_session session)
{
	ssh_channel channel = ssh_channel_new(session);
	if(channel == NULL){
		printf("Failed to create channel");
    	return -1;
	}
	printf("Created Channel!\n");

	int retCode = ssh_channel_open_session(channel);
	if(retCode != SSH_OK){
    	printf("Failed to open channel: %s\n", ssh_get_error(session));
    	return -1;
    }
	printf("Opened Channel!\n");
	

	retCode = ssh_channel_request_exec(channel, "who");
	if(retCode != SSH_OK){
    	printf("Failed to execute who command: %s\n", ssh_get_error(session));
    	return -1;
    }
	printf("Sent Command!\n");

	char* response;
	retCode = ssh_channel_read(channel, response, -1, 0);
	if(retCode == SSH_ERROR){
    	printf("Failed to get response: %s\n", ssh_get_error(session));
    	return -1;
    }
	printf("Got response: %s", response);

	retCode = ssh_channel_send_eof(channel);
	if(retCode == SSH_ERROR){
    	printf("Failed to send eof: %s\n", ssh_get_error(session));
    	return -1;
    }
	printf("Sent EOF\n");

    ssh_channel_close(channel);
    ssh_channel_free(channel);
    printf("Closed Channel!\n");

    return 1;
}

int sftp_operations(ssh_session session)
{
    sftp_session sftp = sftp_new(session);
	if(sftp == NULL){
		printf("Failed to create sftp server");
    	return -1;
	}
	printf("Created SFTP!\n");

	int retCode = sftp_init(sftp);
	if(retCode != SSH_OK){
    	printf("Failed to init sftp: %s\n", ssh_get_error(session));
    	return -1;
    }
	printf("Init SFTP!\n");

	retCode = sftp_mkdir(sftp, "Dir1", 1); // What value?
    return 1;
}

int main()
{
    ssh_session session = ssh_new();
    if ( session == NULL ) 
    {
    	printf("Failed to create session");
    	return -1;
    }

    ssh_options_set(session, SSH_OPTIONS_HOST, SERVER);

    int retCode = ssh_connect(session);
    if(retCode != SSH_OK){
    	printf("Failed to connect: %s\n", ssh_get_error(session));
    	return -1;
    }
    printf("Connected!\n");
    retCode = ssh_is_server_known ( session );
    if(retCode != SSH_SERVER_KNOWN_OK){
    	printf("Server Unknown: %s\n", ssh_get_error(session));
    	return -1;
    }
	printf("Server Known!\n");
    retCode = ssh_userauth_password(session, NULL, PASSWORD);
    if(retCode != SSH_OK){
    	printf("Failed to authenticate: %s\n", ssh_get_error(session));
    	return -1;
    }
    printf("Authenticated!\n");

    if(show_remote_user(session) < 0){
    	printf("Failed to get remote user: %s\n", ssh_get_error(session));
    	return -1;
    }

    if(sftp_operations(session) < 0){
    	printf("Failed to get remote user: %s\n", ssh_get_error(session));
    	return -1;
    }
    ssh_disconnect(session);
    ssh_free(session);
    printf("Closed!\n");
}