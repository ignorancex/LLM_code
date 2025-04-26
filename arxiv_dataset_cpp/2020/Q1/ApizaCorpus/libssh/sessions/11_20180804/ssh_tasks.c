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
  ssh_channel channel;
  int connection;
  int rc;
  char buffer[256];

  channel = ssh_channel_new(session);

  if (channel == NULL)
    {
      fprintf(stderr, "Error creating the channel: %s\n", ssh_get_error(session));
      return SSH_ERROR;
    }

  connection = ssh_channel_open_session(channel);

  if (connection != SSH_OK)
    {
      fprintf(stderr, "Error connecting to the channel: %s\n", ssh_get_error(session));
      ssh_channel_free(channel);
      return SSH_ERROR;
    }

  connection = ssh_channel_request_exec(channel, "who");

  if (connection != SSH_OK)
    {
      fprintf(stderr, "Error executing the command `who' on the channel: %s\n", ssh_get_error(session));
      ssh_channel_free(channel);
      return SSH_ERROR;
    }

  rc = ssh_channel_read(channel, buffer, sizeof(buffer), 0);

  while (rc > 0)
    {
      if (write(1, buffer, rc) != (unsigned int) rc)
	{
	  fprintf(stderr, "Error retrieving the information of the command: %s\n", ssh_get_error(session));
	  ssh_channel_close(channel);
	  ssh_channel_free(channel);
	  return SSH_ERROR;
	}

      rc = ssh_channel_read(channel, buffer, sizeof(buffer), 0);
    }

  ssh_channel_send_eof(channel);
  ssh_channel_close(channel);
  ssh_channel_free(channel);
  
  return SSH_OK;
}

int sftp_operations(ssh_session session)
{
  sftp_session sftp;
  int connection, writing, length;
  sftp_file file;

  char * DIRECTORY = "Dir1";
  char * FILE = "Dir1/File1";
  const char * FILE_CONTENT = "Hello, world!\n";

  length = strlen(FILE_CONTENT);

  sftp = sftp_new(session);

  if (sftp == NULL)
    {
      fprintf(stderr, "Error creating the sftp connection.\n");
      return SSH_ERROR;
    }

  connection = sftp_init(sftp);

  if (connection != SSH_OK)
    {
      fprintf(stderr, "Error initializing SFTP session.\n");
      sftp_free(sftp);
      return SSH_ERROR;
    }

  connection = sftp_mkdir(sftp, DIRECTORY, S_IRWXU);

  /*
  if (connection != SSH_OK)
    {
      if (sftp_get_error(sftp) != )
	{
	  fprintf(stderr, "Cannot create directory %s: %s.\n", DIRECTORY, ssh_get_error(session));
	  sftp_free(sftp);
	  return SSH_ERROR;
	}
    }
  */

  file = sftp_open(sftp, FILE, O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);

    if (file == NULL)
      {
	fprintf(stderr, "Cannot open file %s: %s.\n", FILE, ssh_get_error(session));
	sftp_free(sftp);
	return SSH_ERROR;
      }

  writing = sftp_write(file, FILE_CONTENT, length);

  if (writing != length)
    {
      fprintf(stderr, "Cannot write data to file: %s.\n", ssh_get_error(session));
      sftp_close(file);
      sftp_free(sftp);
      return SSH_ERROR;
    }

  connection = sftp_close(file);

  if (connection != SSH_OK)
    {
      fprintf(stderr, "Cannot close the file: %s.\n", ssh_get_error(session));
      sftp_free(sftp);
      return SSH_ERROR;
    }

  sftp_free(sftp);

  
  return SSH_OK;
}

int main()
{
  ssh_session session;
  int connection, knowledge;

  session = ssh_new();

  if (session == NULL)
    {
      fprintf(stderr, "Error creating the session.\n");
      return SSH_ERROR;
    }

  ssh_options_set(session, SSH_OPTIONS_HOST, SERVER);
  connection = ssh_connect(session);

  if (connection != SSH_OK)
    {
      fprintf(stderr, "Error connecting to %s: %s\n", SERVER, ssh_get_error(session));
      ssh_free(session);
      exit(-1);
    }

  knowledge = ssh_is_server_known(session);

  if (knowledge != SSH_SERVER_KNOWN_OK)
    {
      fprintf(stderr, "Error autenticating the server %s: %s\n", SERVER, ssh_get_error(session));
      ssh_disconnect(session);
      ssh_free(session);
      exit(-1);
    }

  connection = ssh_userauth_password(session, NULL, PASSWORD);

  if (connection != SSH_AUTH_SUCCESS)
    {
      fprintf(stderr, "Error autenticating with password: %s\n", ssh_get_error(session));
      ssh_disconnect(session);
      ssh_free(session);
      exit(-1);
    }

  show_remote_user(session);

  sftp_operations(session);

  ssh_disconnect(session);

  ssh_free(session);

  return 1;
}