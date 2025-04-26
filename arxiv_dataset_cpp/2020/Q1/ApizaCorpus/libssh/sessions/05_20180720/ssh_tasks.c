#define LIBSSH_STATIC 1
#include <libssh/libssh.h> 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libssh/sftp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
char * SERVER = "localhost";
char * USERNAME = "participant";
char * PASSWORD = "password";
int show_remote_user(ssh_session session)
{
    int nbytes;
    int EOFresult;
    int CloseChannelResult;
    int FreeingChannel;
    char buffer[256];
    printf("I am about to execute the %s command\n","who"); 
    ssh_channel MuhChannel = ssh_channel_new   (session);
    int ChannelResult = ssh_channel_open_session(MuhChannel);
    if (ChannelResult != SSH_OK)
    {
        printf("I got %d when using ssh_channel_open_session which I couldn't open the session\n",ChannelResult); 
        ssh_channel_free(MuhChannel);
        return -1;
    }
    int CommandResult = ssh_channel_request_exec(MuhChannel,"who");
    switch(CommandResult){
        case SSH_OK:
            printf("I got %d when using ssh_channel_request_exec which means the command executed\n",CommandResult);  
            break;
        case SSH_ERROR:
            printf("I got %d when using ssh_channel_request_exec which means the command errored\n",CommandResult);   
            printf("The error was %s\n",ssh_get_error(session));
            return -1;
        case SSH_AGAIN :
            printf("I got %d when using ssh_channel_request_exec which means the command has to be done again\n",CommandResult);  
            return -1;
    }
    while(nbytes = ssh_channel_read(MuhChannel, buffer, sizeof(buffer), 0)>0){
        printf("I got %d as a result and text:  %s ",nbytes,buffer);
    }
    EOFresult = ssh_channel_send_eof(MuhChannel);
    if(EOFresult == SSH_OK)
    {
        printf("The EOF was sent successfully\n");
    }else
    {
        printf("There was an error sending the EOF\n");
    }
    CloseChannelResult =ssh_channel_close(MuhChannel);
    if(CloseChannelResult == SSH_OK)
    {
        printf("The channel was closed\n");
        ssh_channel_free(MuhChannel);   
        return 0;
    }else
    {
        printf("There was an error closing the channel\n");
        ssh_channel_free(MuhChannel);
        return -1;
    }
    
}
int sftp_operations(ssh_session session)
{
    sftp_session sftp;
    int InitResult;
    const char *HelloWorld = "Hello, World!\n";
    int HelloWorldLength = strlen(HelloWorld);
    sftp_file file;
    int Access = O_WRONLY | O_CREAT | O_TRUNC;
    printf("I am about to start the sftp \n");    
    sftp = sftp_new(session);
    if(sftp == NULL){
        printf("The sftp session was null");
        return -1;
    }
    InitResult = sftp_init(sftp);
    if(InitResult <0){
        printf("The sftp session failed to init");
        return InitResult;
    }
    int MakeDirResult = sftp_mkdir(sftp, "Dir1", S_IRWXU);  
    if(MakeDirResult ==0){
        printf("The Directory was created");
    }else 
    {
        int Error = sftp_get_error(sftp);
        if(Error == SSH_FX_FILE_ALREADY_EXISTS){
            printf("The Directory already exists");
        }else
        {
            printf("The sftp session failed to create the directory: %d",Error);
            return -1;
        }
    }
    file = sftp_open(sftp, "Dir1/File1", Access, S_IRWXU);
    if (file == NULL)
    {
        printf("Can't open file for writing: %s\n",ssh_get_error(session));
        return -1;
    }
    int AmountWritten = sftp_write(file, HelloWorld, HelloWorldLength);
    if (AmountWritten != HelloWorldLength)
    {
        printf("Can't write data to file: %s\n",ssh_get_error(session));
        sftp_close(file);
        return -1;
    }
    int CloseFile  = sftp_close(file);
    sftp_free(sftp);
    return 0;
    
  
 
}
int DoConnect(ssh_session MuhSession){
    int DidConnect = ssh_connect(MuhSession);
    switch (DidConnect){
        case SSH_OK:
            printf("I got %d when using ssh_connect which means I connected\n",DidConnect);   
            break;
        case SSH_ERROR:
            printf("I got %d when using ssh_connect which means I failed\n",DidConnect);  
            break;
        case SSH_AGAIN:
            printf("I got %d when using ssh_connect which means connect again\n",DidConnect); 
            break;
    }
    return DidConnect;
}
int CheckIfKnown(ssh_session MuhSession){
    int IsKnown = ssh_is_server_known(MuhSession);
    switch(IsKnown){
        case SSH_SERVER_KNOWN_OK:
            printf("I got %d when using ssh_is_server_known which means it is a known server\n",IsKnown); 
            break;
        case SSH_SERVER_KNOWN_CHANGED:
            printf("I got %d when using ssh_is_server_known which means Server has changed \n",IsKnown);  
            break;
        case SSH_SERVER_NOT_KNOWN:
            printf("I got %d when using ssh_is_server_known which means it is NOT a known server\n",IsKnown); 
            break;
        case SSH_SERVER_FILE_NOT_FOUND:
            printf("I got %d when using ssh_is_server_known which means server not found\n",IsKnown); 
            break;
        case SSH_SERVER_ERROR:
            printf("I got %d when using ssh_is_server_known which means there was an error\n",IsKnown);   
            break;
 
    }
    return IsKnown;
}
int Auth(ssh_session MuhSession){
    int DidAuthneticate = ssh_userauth_password    (MuhSession,USERNAME,PASSWORD);
    switch(DidAuthneticate){
        case SSH_AUTH_SUCCESS:
            printf("I got %d when using ssh_userauth_password which means I connected\n",DidAuthneticate);    
            break;
        case SSH_AUTH_DENIED:
            printf("I got %d when using ssh_userauth_password which means I failed\n",DidAuthneticate);   
            break;
        case SSH_AUTH_AGAIN:
            printf("I got %d when using ssh_userauth_password which means I should auth againn",DidAuthneticate); 
            break;
        case SSH_AUTH_PARTIAL:
            printf("I got %d when using ssh_userauth_password which means I need to use another method but am partially authenticated\n",DidAuthneticate);    
            break;      
        case SSH_AUTH_ERROR:
            printf("I got %d when using ssh_userauth_password which means an error happened\n",DidAuthneticate);  
            break;
    }
    return DidAuthneticate;
}
int main()
{
    ssh_session  MuhSession = ssh_new();
    ssh_options_set(MuhSession, SSH_OPTIONS_HOST, SERVER);
    int Connected = DoConnect(MuhSession);
    if(Connected != SSH_OK)
    {
        return -1;
    }
    printf("Everything went ok for connect!");
    int IsKnown = CheckIfKnown(MuhSession);
    if(IsKnown != SSH_SERVER_KNOWN_OK){
        return -1;  
    }
    printf("Everything went if the server was known!");
    int DidAuth = Auth(MuhSession);
    if(DidAuth != SSH_AUTH_SUCCESS)
    {
        return -1;
    }
    printf("Everything went ok for authentication!");
    int ShowUserResult = show_remote_user(MuhSession);
    if(ShowUserResult != 0){
        return -1;
 
    }
    printf("Everything went ok for showing remote user!");
    int SFTPRun = sftp_operations(MuhSession);
    if(SFTPRun ==0){
        printf("Everything went ok for sftp!");
    }
    return 0;
    ssh_free(MuhSession);
}