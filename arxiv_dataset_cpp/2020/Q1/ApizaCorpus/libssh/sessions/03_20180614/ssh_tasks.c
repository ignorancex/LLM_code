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
    ssh_channel_open_session(channel);
    ssh_channel_request_exec(channel,"who");
    char  my_buffer[512];
    int read = ssh_channel_read(channel,my_buffer, sizeof(my_buffer), 0);
    printf(read);
    ssh_channel_send_eof(channel);
    ssh_channel_close(channel);
    ssh_channel_free(channel);
    return -1;
}

int sftp_operations(ssh_session session)
{
	sftp_session sftp =  sftp_new(session);
        sftp_mkdir(session,"Dir1","S_IRWXU");
        sftp_file file = sftp_open(session, "Dir1/File1","O_WRONLY | O_CREAT | O_TRUNC","S_IRWXU");
        const char* buff = "Hello,world!";
        sftp_write(file,buff, sizeof(buff));
	sftp_close(file);
	sftp_free(session);

	return -1;
}

int main()
{
    printf("Hello world!\n");
    ssh_session session=ssh_new();
    ssh_connect(session);
    ssh_userauth_publickey_auto(session,"localhost", "password");
    ssh_disconnect(session);
    ssh_free(session);
    show_remote_user(session);
    sftp_operations(session);
    ssh_disconnect(session);


}
