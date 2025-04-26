#define LIBSSH_STATIC 1
#include <libssh/libssh.h>
#include <stdlib.h>
#include <stdio.h>
#include <libssh/sftp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

char* SERVER = "localhost";
char* PASSWORD = "password";

int show_remote_user(ssh_session sess) {
	ssh_channel channel = ssh_channel_new(sess);
	if (!channel) {
		printf("Could not create a channel: %s\n", ssh_get_error(sess));
		return 0;
	}

	int ret = ssh_channel_open_session(channel);
	if (ret != SSH_OK) {
		printf("Could not open session: %s\n", ssh_get_error(sess));
		return 0;
	}

	ret = ssh_channel_request_exec(channel, "who");
	if (ret != SSH_OK) {
		printf("Could not exec 'who': %s\n", ssh_get_error(sess));
		return 0;
	}

	char buf[512];
	int sz = ssh_channel_read(channel, buf, sizeof(buf), 0);
	if (printf("%s", buf) != sz) {
		printf("Error reading from ssh channel: %s\n", ssh_get_error(sess));
		return 0;
	}

	ssh_channel_send_eof(channel);
	ssh_channel_close(channel);
	ssh_channel_free(channel);
	return 1;
}

int sftp_operations(ssh_session sess) {
	sftp_session ftp_sess = sftp_new(sess);
	if (!ftp_sess) {
		printf("Error creating SFTP session: %s\n", ssh_get_error(sess));
		return 0;
	}

	int ret = sftp_init(ftp_sess);
	if (ret != SSH_OK) {
		printf("Could not init SFTP session: %s\n", ssh_get_error(sess));
		sftp_free(ftp_sess);
		return 0;
	}

	ret = sftp_mkdir(ftp_sess, "Dir1", S_IRWXU);
	if (ret != SSH_OK) {
		if (sftp_get_error(ftp_sess) == SSH_FX_FILE_ALREADY_EXISTS) {
			printf("Dir1 already exists!\n");
		} else {
			printf("SFTP error: %s\n", ssh_get_error(sess));
			return 0;
		}
	}

	sftp_file file = sftp_open(ftp_sess, "Dir1/File1", O_RDWR | O_TRUNC | O_CREAT, S_IRWXU);
	if (!file) {
		printf("Could not create Dir1/File1: %s\n", ssh_get_error(sess));
		return 0;
	}

	char* write_str = "Hello, world!";
	ret = sftp_write(file, write_str, sizeof(write_str));
	if (ret != sizeof(write_str)) {
		printf("Error writing string to file: %s\n", ssh_get_error(sess));
		return 0;
	}

	ret = sftp_close(file);
	if (ret != SSH_OK) {
		printf("Error closing file: %s\n", ssh_get_error(sess));
		return 0;
	}

	sftp_free(ftp_sess);
	return 1;
}

int authenticated(ssh_session sess) {
	int ssh_state = ssh_is_server_known(sess);

	switch (ssh_state) {
		case SSH_SERVER_KNOWN_OK:
			return 1;
		case SSH_SERVER_KNOWN_CHANGED:
			printf("Host key has changed!\n");
			return 0;
		case SSH_SERVER_FOUND_OTHER:
			printf("Host key not found, but another type exists!\n");
			return 0;
		case SSH_SERVER_FILE_NOT_FOUND:
			printf("Could not find host key file!\n");
			return 0;
		case SSH_SERVER_NOT_KNOWN:
			printf("Server unknown, do you trust?\n");
			printf("Just kidding, skipping this because it requires pubkey hash\n");
			return 0;
		default:
			printf("Could not authenticate server (unknown error)!\n");
			return 0;
	}

	return 0;
}

int main() {
	// new session
	ssh_session sess = ssh_new();
	if (!sess) {
		printf("Could not create a new ssh_session!\n");
		return 1;
	}

	// set options
	int ret;
	ret = ssh_options_set(sess, SSH_OPTIONS_HOST, SERVER);
	if (ret < 0) {
		printf("Could not set options: %s!\n", ssh_get_error(sess));
		return 1;
	}

	// connect
	ret = ssh_connect(sess);
	if (ret != SSH_OK) {
		printf("Failed to connect: %s!\n", ssh_get_error(sess));
		return 1;
	}

	// auth
	if (!authenticated(sess)) {
		printf("Could not authenticate!\n");
		ssh_disconnect(sess);
		ssh_free(sess);
		return 1;
	}

	// password
	ret = ssh_userauth_password(sess, NULL, PASSWORD);
	if (ret != SSH_AUTH_SUCCESS) {
		printf("Password error: %s\n", ssh_get_error(sess));
		ssh_disconnect(sess);
		ssh_free(sess);
		return 1;
	}

	printf("Successfully connected!\n");

	printf("Calling show_remote_user...\n");
	if (!show_remote_user(sess)) {
		printf("show_remote_user(...) failed!\n");
		return 1;
	}

	printf("Creating Dir1 and File1...\n");
	if (!sftp_operations(sess)) {
		printf("sftp_operations(...) failed!\n");
		return 1;
	}

	printf("Cleanly disconnected!\n");

	ssh_disconnect(sess);
	ssh_free(sess);
}