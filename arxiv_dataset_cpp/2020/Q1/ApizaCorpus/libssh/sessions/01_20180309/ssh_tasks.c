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
	printf("dasdfsaf");
	ssh_channel sshc = ssh_channel_new(session);
	printf("dsaf");
	if (sshc!=SSH_OK){
		int res = ssh_channel_request_exec(sshc,"who");
		if (res!=SSH_OK){
			printf("Command could not be executed over the ssh channel");			
			exit(1);
		}
		char str[256];
		int count;
		int stder;
		ssh_channel_read(sshc, &str, count, stder);
		str[count] = '\0';
		printf("%s\n",str);
		
	} else {
		printf("SSH channel could not be created");
		exit(1);
	}
	return -1;
}

int sftp_operations(ssh_session session)
{
	return -1;
}

int main()
{
		ssh_session sp = ssh_new();
		if (sp==NULL){
			printf("No session created\n");
			exit(1);
		} else {
			printf("Session created\n");
			//sp->Session->ssh_connect("localhost");
			ssh_options_set(sp, SSH_OPTIONS_HOST, "localhost");
			int res = ssh_connect(sp);
			if (res==SSH_OK) {
				printf("Connected to localhost\n");
				res = ssh_is_server_known(sp);
				printf("%i %i\n",res,SSH_SERVER_KNOWN_OK);
				if (res==SSH_SERVER_KNOWN_OK) {
					printf("Server is known");


					show_remote_user(sp);




					ssh_disconnect(sp);
					ssh_free(sp);
				} else {
					printf("Server is not known (check your ./ssh/known_hosts file)\n");
					exit(1);
				}
			} else if (res==SSH_AGAIN) {
				printf("Try again\n");
				exit(1);
			} else {
				printf("Could not connect to localhost\n");
				exit(1);
			}
		}

   	return -1;
}
