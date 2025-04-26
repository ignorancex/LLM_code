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
    ssh_channel newChannel = ssh_channel_new(session);
    if (ssh_channel_request_exec(newChannel, "who")!=SSH_OK){
    	fprintf(stderr, "Error happens: %s\n",ssh_get_error(session));
        return -1;
    }
    char buffer[256];
    int numberOfbytes;
    numberOfbytes = ssh_channel_read(channel, buffer, sizeof(buffer), 0);
    while(numberOfbytes>0){
    	if (fwrite(buffer, 1, numberOfbytes, stdout) != (unsigned int) numberOfbytes) {
        	return -1;
    	}
        numberOfbytes =  ssh_channel_read(channel, buffer, sizeof(buffer), 0);
    }
    if(ssh_channel_send_eof(newChannel)!=SSH_OK){
       fprintf(stderr, "Error happens: %s\n",ssh_get_error(session));
       return -1;
    }
    if(ssh_channel_close(newChannel)!=SSH_OK){
       fprintf(stderr, "Error happens: %s\n",ssh_get_error(session));
       return -1;
    }
    ssh_channel_free(newChannel);
    return 0;
}

int sftp_operations(ssh_session session)
{
    sftp_sessiion newSftp = sftp_new(session);
    if(newSftp == NULL){
        fprintf(stderr, "Error happens: %s\n",ssh_get_error(session));
        return -1;
    }
    if(sftp_mkdir(newSftp, "Dir1", & ~umask)<0){
	if(sftp_get_error(newSftp) != 11){
	   fprintf(stderr, "Error happens: %s\n",ssh_get_error(session));
           return -1;
	}
    }
    
}

int main()
{
   ssh_session newSession = ssh_new();
   if (newSession == NULL){
        fprintf(stderr, "Error happens: %s\n",ssh_get_error(newSession));
        return -1;
   }
   if (ssh_connect(newSession) != SSH_OK){
   	fprintf(stderr, "Error happens: %s\n",ssh_get_error(newSession));
	return -1;
   }
   if (ssh_is_server_known(newSession) != SSH_SERVER_KNOWN_OK){
        fprintf(stderr, "Error happens: %s\n",ssh_get_error(newSession));
	return -1;
   }
   if (ssh_userauth_agent( newSession, NULL)!= SSH_AUTH_SUCCESS ){
        fprintf(stderr, "Error happens: %s\n",ssh_get_error(newSession));
        return -1;
   }
   ssh_disconnect(newSession);
   ssh_free(newSession);
   printf("Finish");
   return 0;
}