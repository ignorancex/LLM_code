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
    ssh_channel chan;
    int exec_status;

    chan = ssh_channel_new(sess);
    if (chan == NULL) {
        printf("Sorry, but creating the SSH channel failed.\n");
        exit(1);
    }

    exec_status = ssh_channel_request_exec(chan, "who");
    if (exec_status <> SSH_OK) {
        return exec_status;
    }

    while ((rc = ssh_channel_read(chan, buffer, sizeof(buffer), 0)) > 0) {
        if (fwrite(buffer, 1, rc, stdout) != (unsigned int) rc) {
            return -1;
        }
    }

    return -1;
}

int sftp_operations(ssh_session session)
{
    return -1;
}

int main()
{
    ssh_session sess;
    ssh_channel chan;
    int connection_status, forward_status, is_server_known, userauth_status;

    printf("Hello world!\n");

    sess = ssh_new();
    if (sess == NULL) {
        printf("Sorry, but creating the SSH session failed.\n");
        exit(1);
    }

    printf("The SSH session was successfully created.\n");

    chan = ssh_channel_new(sess);
    if (chan == NULL) {
        printf("Sorry, but creating the SSH channel failed.\n");
        exit(1);
    }

    printf("The SSH channel was successfully created.\n");

    // forward_status = ssh_channel_open_forward(chan, SERVER, 22, "127.0.0.1", 22);
    // if (forward_status == SSH_ERROR) {
    //     printf("Sorry, but opening an SSH channel failed.\n");
    //     exit(1);
    // } else if (forward_status == SSH_AGAIN) {
    //     printf("Sorry, but the session is in nonblocking mode. Please try again later to open an SSH channel..\n");
    //     exit(1);
    // }

    connection_status = ssh_connect(sess);
    // printf("%d\n", connection_status);
    if (connection_status == SSH_ERROR) {
        printf("Sorry, but connecting to the host failed.\n");
        exit(1);
    } else if (connection_status == SSH_AGAIN) {
        printf("Sorry, but the session is in nonblocking mode. Please try again later to connect to the host.\n");
        exit(1);
    }

    printf("A connection to the host was successfully opened.\n");

    is_server_known = ssh_is_server_known(sess);
    switch (is_server_known) {
        case SSH_SERVER_KNOWN_CHANGED:
            printf("The server key has changed. Either you are under attack or the administrator changed the key. You HAVE to warn the user about a possible attack.\n");
            exit(1);
            break;
        case SSH_SERVER_FOUND_OTHER:
            printf("The server gave use a key of a type while we had an other type recorded. It is a possible attack.\n");
            exit(1);
            break;
        case SSH_SERVER_NOT_KNOWN:
            printf("The server is unknown. User should confirm the MD5 is correct.\n");
            exit(1);
            break;
        case SSH_SERVER_FILE_NOT_FOUND:
            printf("The known host file does not exist. The host is thus unknown. File will be created if host key is accepted.\n");
            exit(1);
            break;
        case SSH_SERVER_ERROR:
            printf("Some error happened when verifying that the server is known.\n");
            exit(1);
            break;
    }

    printf("It is confirmed that the host is known.\n");

    userauth_status = ssh_userauth_password(sess, NULL, PASSWORD);
    switch (userauth_status) {
        case SSH_AUTH_ERROR:
            printf("A serious error happened when attempting to authenticate.\n");
            exit(1);
            break;
        case SSH_AUTH_DENIED:
            printf("Authentication failed: use another method.\n");
            exit(1);
            break;
        case SSH_AUTH_PARTIAL:
            printf("You've been partially authenticated, you still have to use another method.\n");
            exit(1);
            break;
        case SSH_AUTH_AGAIN:
            printf("In nonblocking mode, you've got to call this again later.\n");
            exit(1);
            break;
    }

    printf("You have been successfully authenticated to the host.\n");

    ssh_disconnect(sess);

    printf("The SSH session was successfully disconnected.\n");

    ssh_free(sess);

    printf("The SSH session was successfully freed.\n");

    return 0;
}