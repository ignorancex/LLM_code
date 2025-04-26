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
    struct ssh_buffer_struct buffer,buf = ssh_buffer_new();
    int rc = ssh_channel_request_exec(channel,"who");
    if (rc > 0) {
        return -1;
    }
    while ((rc = channel_read(channel, buffer, sizeof(buffer), 0)) > 0) {
        if (fwrite(buffer, 1, rc, stdout) != (unsigned int) rc) {
        return -1;
        }
    }
    rc = ssh_channel_send_eof(channel);
    if (rc == SSH_ERROR) {
        return -1;
    } 
    while(!ssh_channel_is_eof(channel)) {
        rc = ssh_channel_read(channel, buf, sizeof(buf), 0);
        if (rc == SSH_ERROR) {
            return -1;
        }
    }
    int error = ssh_channel_close(channel);
    if(error == SSH_ERROR)
    {
       char* temp = ssh_get_error(session);
       printf("There was an error: %s/n", temp);
       return -1;
    }
    ssh_channel_free(channel);
    return 0;
}
int sftp_operations(ssh_session session)
{
    sftp_session sftp = sftp_new(session);
    sftp_mkdir(sftp, "Dir1",(mode & ~umask));
    sftp_open(session, "File1",O_WRONLY,(mode & ~umask));
    //sftp_write(
    sftp_free();
    return -1;
}
int main()
{
    ssh_session session = ssh_new();
    if(ssh_userauth_none(session, NULL) == SSH_AUTH_SUCCESS)
    {
       ssh_connect(session);
    }
    else
    {
       char* temp = ssh_get_error(session);
       printf("There was an error: %s/n", temp);
       exit(1);
    }
    ssh_disconnect(session);
}