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

// SSH_OK: 0
// SSH_ERROR: -1
// SSH_AGAIN: -2

int ConnectWithRetries(ssh_session* session, int retries) {
 int status;
 for (int i = 0; i < retries; ++i) {
   status = ssh_connect(*session);
   // Keep trying until we have run out of retries or we get
   // a terminal connection status.
   if (status != SSH_AGAIN) {
     return status;
   }
 }
 return status;
}

int main()
{
 ssh_session* session = ssh_new();

 if (session == NULL) {
   printf("Failed to create ssh connection.\n");
   return -1;
 }

 printf("session ptr is %p\n", (void*) session);
 // printf("ssh_userauth_none: %d\n", ssh_userauth_none(*session, USERNAME));
 printf("set username in options: %d\n",
   ssh_options_set(*session, SSH_OPTIONS_ADD_IDENTITY, USERNAME));

 printf("About to call ssh_userauth_password\n");
 // This is causing a segfault...
 int auth_status = ssh_userauth_password(*session, USERNAME, PASSWORD);
 printf("Auth status %d\n", auth_status);
 int final_status = ConnectWithRetries(session, 10);
 if (final_status == SSH_ERROR) {
   // const char* error_msg = ssh_get_error(*session);
   // printf("Connection to \"%s\" returned %s\n", SERVER, error_msg);
   return -1;
 }
 free(session);
}