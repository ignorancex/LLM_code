#define LIBSSH_STATIC 1
#include <libssh/libssh.h>
#include <stdlib.h>
#include <stdio.h>
#include <libssh/sftp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

char * SERVER = "localhost";
char * USERNAME = "participant";
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
   return -1;
}