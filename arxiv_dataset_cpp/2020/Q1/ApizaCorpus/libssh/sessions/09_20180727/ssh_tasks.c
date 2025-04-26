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
    return -1;
}

int sftp_operations(ssh_session session)
{
    return -1;
}

int main()
{
    int rc;
    ssh_session sesh = ssh_new();
    if(sesh == NULL){
    		exit(-1);
    }
    ssh_options_set(sesh, SSH_OPTIONS_HOST, SERVER);
    rc = ssh_connect(sesh);
      state = ssh_is_server_known(sesh);
    if (rc != SSH_OK)
    {
      fprintf(stderr, "error connecting: %s\n", ssh_get_error(sesh));
      exit(-1)
    }
   	show_remote_user(sesh);
   	ssh_disconnect(mysession);
    ssh_free(sesh);
}