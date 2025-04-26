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
  char * buffer = (char *)malloc(sizeof(char) * 2048); // Terrible practice but assume we wont overflow
  int rc;
  char * command = "who";
  ssh_channel channel = ssh_channel_new(session);
  if (channel == NULL) {
    printf("ssh_channel_new() failed to create new channel. \n");
    return -1;
  }
  rc = ssh_channel_open_session(channel);
  if (rc != SSH_OK) {
    printf("Error opening channel\n");
    ssh_channel_free(channel);
    return rc;
  }
  rc = ssh_channel_request_exec(channel, command);
  if (rc != SSH_OK) {
    printf("Error executing command\n");
    return rc;
  }
  while ((rc = channel_read(channel, buffer, sizeof(buffer), 0)) > 0) {
    if (fwrite(buffer, 1, rc, stdout) != (unsigned int) rc) {
      return -1;
    }
  }
  rc = ssh_channel_close(channel);
  if (rc != SSH_OK) {
    printf("Error closing channel\n");
    return -1;
  }
  ssh_channel_free(channel);
  free(buffer);
    return 0;
}

int sftp_operations(ssh_session session)
{
  sftp_session sftp = sftp_new(session);
  if (sftp == NULL) {
    printf("Error creating new sftp_session\n");
    return -1;
  }
    return 0;
}

int main()
{
  int rc;
    ssh_session session = ssh_new();
    if (session != NULL) {
      printf("New session created\n");
    }
    else {
      printf("Error creating ssh session\n");
      printf("Exiting application\n");
      return -1;
    }

    ssh_options_set(session, SSH_OPTIONS_HOST, SERVER);

    rc = ssh_connect(session);
    if (rc != SSH_OK) {
      printf("Error connecting. RC: %d\n", rc);
      return -1;
    }

    rc = ssh_userauth_password(session, NULL, PASSWORD);
    if (rc != SSH_OK) {
      printf("auth_code: %d\n", rc);
      return -1;
  }

  rc = show_remote_user(session);
  if (rc != 0) {
    printf("show_remote_user rc: %d\n", rc);
    return -1;
  }

  rc = sftp_operations(session);
  if (rc != 0) {
    printf("sftp_operations rc: %d\n", rc);
    return -1;
  }

  ssh_disconnect(session);
  ssh_free(session);

  printf("Exiting application successfully.\n");
  return 0;
}