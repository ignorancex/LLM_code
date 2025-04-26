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
    ssh_session newsession = ssh_new();
}

int sftp_operations(ssh_session session)
{
    return -1;
}

int main()
{
    printf("Hello world!\n");
    ssh_session mysession = ssh_new();
    if(mysession==NULL)
    	{
    		printf("session was not created\n");
    		exit(0);
    	}

    else
    	printf("session created sucessfuly\n");

    //connection to the local host and authenticating the server
    void* value;
    ssh_options_set(mysession, SSH_OPTIONS_USER, value);
    int status = ssh_connect(mysession);
    if(status==SSH_OK)
    	printf("connection to the server succeded\n");
    else
    	printf("connection to the server failed\n");

   
   	status = ssh_userauth_password(mysession, NULL, PASSWORD);
   	if(status==4)
   		printf("%s\n", "User authentication is successful!");
   	printf("%s", "Disconnecting from the server...");
   	ssh_disconnect(mysession);
   	printf("%s\n", "sucessfuly");
}