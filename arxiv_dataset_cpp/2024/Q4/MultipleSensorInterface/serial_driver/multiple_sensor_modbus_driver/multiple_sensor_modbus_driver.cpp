/******************************************************************************************************
**                                                                                                    *
**  Copyright (C) 2021 David Nuno Quelhas. LGPL License 3.0 (GNU).                                    *
**  David Nuno Quelhas; Lisboa, Portugal.  david.quelhas@yahoo.com ,                                  *
**  https://multiple-sensor-interface.blogspot.com                                                    *
**                                                                                                    *
**  Multi-sensor interface, RTU. Modbus RS485/RS232 Device Driver.                           *
**                                                                                                    *
**  This is the source file for the Modbus RS485/RS232 Driver of the Multiple-Sensor Interface, also  *
**    referenced/mentioned here by the name 'LCR sensors' .                                           *
**                                                                                                    *
**  This library is free software; you can redistribute it and/or modify it under the terms of the    *
**  GNU Lesser General Public version 3.0 License as published by the Free Software Foundation.       *
**  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;         *
**  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.         *
**  See the GNU Lesser General Public License for more details.                                       *
**  You should have received a copy of the GNU Lesser General Public License along with this library. *
**  --------------------                                                                              *
**   FILE: multiple_sensor_modbus_driver.cpp - Here are the implementations of the functions of the   *
**                                     Modbus RS485/RS232 Device Driver of Multiple-Sensor Interface. *
******************************************************************************************************/


#include "multiple_sensor_modbus_driver.h"

#include <QTimer>
#include <QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>

#include <QCoreApplication>

#include <wchar.h>
#include <string.h>
#include <stdlib.h>

#include <stdio.h>

#include <QThread>

#define MAX_N_CHAR_USART 13  //maximum number of characters(Bytes) the USART of the PIC microprocessor can handle

//#define MAX_CHARACTERS_OF_MESSAGE 60



// ----- Multiple Sensor board I2C EEPROM ADDRS -----
#define EEPROM_I2C_ADDR 0xA0  //(1010 , chip_select: 000 )
#define EEPROM_I2C_SIZE_KBITS 512	//512Kbits <=> 64Kbytes, 24LC512 or 24FC512
#define TABLE_HEADER_SIZE_BYTES 64  //size in bytes of the header of each calibration table, where is saved the units of calibration, the mode: multi channel or single channel, etc...
const short SIZE_CALIB_TABLE_BYTES_const = (short) ( (1024.0 *  EEPROM_I2C_SIZE_KBITS ) / ( 8.0*(N_LCR_CHANNELS+N_ADC_CHANNELS) ) );
const short LINES_CALIB_TABLE_const = (short) ( (1024.0 *  EEPROM_I2C_SIZE_KBITS ) / ( 64.0*(N_LCR_CHANNELS+N_ADC_CHANNELS) ) );	//each line has 8 bytes (4 bytes the RAW_value, 4 bytes the measurement)
#define WAIT_BETWEEN_I2C_WRITE_CMD 5000	//count value for a delay time between consecutive I2C commands
// --------------------------------------------------


typedef struct ret_type1_struct {
    int ret_value;	//return value is an status code (also error code)
    char error_pos;
} ret_type1;


void RemoveChar(char *str, char garbage) {

    char *src, *dst;
    for (src = dst = str; *src != '\0'; src++) {
        *dst = *src;
        if (*dst != garbage) dst++;
    }
    *dst = '\0';
}


void print_ERROR_MSG_Serial_MultipleSensor(int error_code)
 {
    switch(error_code) {
        case SERIAL_COMMAND_ERROR:
            printf("ERROR: SERIAL_COMMAND_ERROR\n");
        break;

        case SERIAL_CONNECTION_ERROR:
            printf("ERROR: SERIAL_CONNECTION_ERROR\n");
        break;

        case SENSOR_CALIB_FAILED:
            printf("ERROR: SENSOR_CALIB_FAILED\n");
        break;

        case SENSOR_READ_FAIL_DUE_MISSING_CALIB:
            printf("ERROR: SENSOR_READ_FAIL_DUE_MISSING_CALIB\n");
        break;

        case SERIAL_SET_BAUD_RATE_ERROR:
            printf("ERROR: SERIAL_SET_BAUD_RATE_ERROR\n");
        break;

        case SERIAL_SET_DATA_BITS_ERROR:
            printf("ERROR: SERIAL_SET_DATA_BITS_ERROR\n");
        break;

        case SERIAL_SET_PARITY_ERROR:
            printf("ERROR: SERIAL_SET_PARITY_ERROR\n");
        break;

        case SERIAL_SET_STOP_BITS_ERROR:
            printf("ERROR: SERIAL_SET_STOP_BITS_ERROR\n");
        break;

        case SERIAL_SET_FLOW_CONTROL_ERROR:
            printf("ERROR: SERIAL_SET_FLOW_CONTROL_ERROR\n");
        break;

        case SERIAL_WAIT_TIMEOUT_ERROR:
            printf("ERROR: SERIAL_WAIT_TIMEOUT_ERROR\n");
        break;

        case SERIAL_REPLY_WITH_INVALID_DATA:
            printf("SERIAL_REPLY_WITH_INVALID_DATA\n");
        break;

    }
}



//int AddParenthesesAtStartEnd(char *str, short max_len, short max_size) - Function to add parentheses '(' ')' at the start and en of the string 'str', 'max_len' is the maximum length available to the user, 'max_available_size' is the maximum limit of characters possible to store on 'str' string .
int AddParenthesesAtStartEnd(char *str, short max_len, short max_available_size) {
    short i;
    short str_len = strlen(str);

    if( str_len > max_len || str_len > max_available_size ) {
        return ERROR_BOOL_FUNC_MISSING_STRING_TERMINATOR;
    }
    else {
        //shift to the right by 1 position, all characters in 'str' string
        for(i=max_len; i>=0; i--) {
            str[i+1]=str[i];
        }

        //add '(' at start of 'str'
        str[0]='(';

        //add ')' at end of 'str'
        str[str_len+1]=')';
        str[str_len+2]='\0';
    }

    return OK;
}



//ret_type1 CheckBooleanFunc(char *bool_func_ptr) -  Check the syntax of bool(logical) function
ret_type1 CheckBooleanFunc(char *bool_func_ptr) {
    char bool_func_local[BOOL_FUNC_N_BYTES];
    ret_type1 return_var;
    char i, k;	//"i" integer number between -127 and 127
    char found_on_list_bool=0;
    int ret_val;
    char found_bool_operator_outside_parentheses_bool;
    short open_parentheses_count;

    char pos_end_string=-1;	//position end of the string, '\0' character position
    char start_open_parentheses_pos=-1, start_close_parentheses_pos=-1;	//integer numbers in [-127; 127]

    const char valid_chars[20]="()0123456789/+.CKMR";

    char number_unmatched_parenthesis_pair=0;

    return_var.ret_value=1;
    return_var.error_pos=-1;    //"error_pos" integer number between -127 and 127

    //find the end of the string
    for(i=0, return_var.ret_value = ERROR_BOOL_FUNC_MISSING_STRING_TERMINATOR; i<BOOL_FUNC_N_BYTES && pos_end_string<0; i++) {
        if( bool_func_ptr[i]=='\0' ) {
            pos_end_string=i;
            return_var.ret_value = OK;
        }
    }

    strncpy(bool_func_local, bool_func_ptr, BOOL_FUNC_N_BYTES);	//the string on bool_func_ptr is always copied to bool_func_local

    //check for parentheses matching (open and close)
    for(i=0, number_unmatched_parenthesis_pair=0; i<BOOL_FUNC_N_BYTES && bool_func_local[i]!='\0' && return_var.ret_value>0; i++) {

        if( bool_func_local[i] == '(' && return_var.ret_value>0) {
            number_unmatched_parenthesis_pair++;
        }

        if( bool_func_local[i] == ')' && return_var.ret_value>0) {
            number_unmatched_parenthesis_pair--;
            if(bool_func_local[i-1] == '(' && i>0) {//check for the error "()"
                return_var.ret_value=ERROR_BOOL_FUNC_EMPTY_PARENTHESES;
                return_var.error_pos=i;
            }
            if( number_unmatched_parenthesis_pair<0 && return_var.ret_value > 0 ) {
                return_var.ret_value=ERROR_BOOL_FUNC_PARENTHESES_NOT_MATCHING;
                return_var.error_pos=i;
            }
        }
    }
    if( number_unmatched_parenthesis_pair!=0 && return_var.ret_value > 0 ) {
        return_var.ret_value=ERROR_BOOL_FUNC_PARENTHESES_NOT_MATCHING;
        return_var.error_pos=0;
    }


    for(i=0, found_bool_operator_outside_parentheses_bool=FALSE_BOOL, open_parentheses_count=0;  i<(BOOL_FUNC_N_BYTES-2) && bool_func_local[i]!='\0';  i++) {
        if(bool_func_local[i] == '(') {
            open_parentheses_count++;
        }
        if(bool_func_local[i] == ')') {
            open_parentheses_count--;
        }

        //check if found an operator ('/' - NOT, '.' - AND, '+'- OR) that is not placed inside an open close parentheses (example: (M1+M2).(/M0) , the '.' operator is not inside parentheses )
        if( open_parentheses_count==0 && ( bool_func_local[i]=='/' || bool_func_local[i]=='.' || bool_func_local[i]=='+' ) ) {
            found_bool_operator_outside_parentheses_bool=TRUE_BOOL;
        }
    }

    if( found_bool_operator_outside_parentheses_bool == TRUE_BOOL ) {
        //found bool operator outside a parentheses, so add an additional par of parentheses around the whole bool function
        ret_val = AddParenthesesAtStartEnd(bool_func_local, BOOL_FUNC_MAX_LENGTH, BOOL_FUNC_N_BYTES);
        if( ret_val < 1 ) {
            return_var.ret_value=ret_val;
        }
    }


    //remove the spaces (' ') from the string
    RemoveChar(bool_func_local, ' ');

    //find the end of the string
    for(i=0, pos_end_string=-1; i<BOOL_FUNC_N_BYTES && pos_end_string<0; i++) {
        if( bool_func_local[i]=='\0' ) {
            pos_end_string=i;
        }
    }
    if( pos_end_string<0 ) {
        return_var.ret_value=ERROR_BOOL_FUNC_MISSING_STRING_TERMINATOR;
        return_var.error_pos=i;
    }

    for(i=0; i<BOOL_FUNC_N_BYTES && bool_func_local[i]!='\0' && return_var.ret_value>0; i++) {
        //check if the character being checked is a valid character to use on the boolean function
        for(k=0, found_on_list_bool=0; k<20; k++) {
            if( bool_func_local[i] == valid_chars[k])
                found_on_list_bool=1;
        }
        if( found_on_list_bool<1 ) {
            return_var.ret_value=ERROR_BOOL_FUNC_INVALID_CHARACTER;
            return_var.error_pos=i;
        }
    }


    //check if the boolean function is only one variable (Ex: "R5" or "C1"), if this is the case make sure there is at least one operation, so save "R5.1"
    if(start_open_parentheses_pos==0 && start_close_parentheses_pos==3) {
        bool_func_local[3]='.';		bool_func_local[4]='1';		bool_func_local[5]=')';
    }

    //check for valid syntax
    for(i=0, number_unmatched_parenthesis_pair=0; i<BOOL_FUNC_N_BYTES && bool_func_local[i]!='\0' && return_var.ret_value>0; i++) {

        if(bool_func_local[i]=='/' && return_var.ret_value>0) {
            if(bool_func_local[i+1]!='M' && bool_func_local[i+1]!='R' && bool_func_local[i+1]!='C' && bool_func_local[i+1]!='K' && bool_func_local[i+1]!='/' && bool_func_local[i+1]!='0' && bool_func_local[i+1]!='1' && bool_func_local[i+1]!='(') {
                return_var.ret_value=ERROR_BOOL_FUNC_OPERATION_MISSING_1ST_VARIABLE;
                return_var.error_pos=i;
            }

        }

        if((bool_func_local[i]=='+' || bool_func_local[i]=='.') && return_var.ret_value>0) {
            if( (bool_func_local[i-1]<'0' || bool_func_local[i-1]>'9') && bool_func_local[i-1]!=')' ) {
                return_var.ret_value=ERROR_BOOL_FUNC_OPERATION_MISSING_1ST_VARIABLE;
                return_var.error_pos=i;
            }
            else {
                if( bool_func_local[i+1]!='M' && bool_func_local[i+1]!='R' && bool_func_local[i+1]!='C' && bool_func_local[i+1]!='K' && bool_func_local[i+1]!='/' && bool_func_local[i+1]!='0' && bool_func_local[i+1]!='1' && bool_func_local[i+1]!='(' ) {
                    return_var.ret_value=ERROR_BOOL_FUNC_OPERATION_MISSING_2ND_VARIABLE;
                    return_var.error_pos=i;
                }

            }
        }

        if( return_var.ret_value>0 && (bool_func_local[i]=='R' || bool_func_local[i]=='M' || bool_func_local[i]=='C' || bool_func_local[i]=='K') ) {
            if( bool_func_local[i+1]<'0' || bool_func_local[i+1]>'9' ) {
                return_var.ret_value=ERROR_BOOL_FUNC_INVALID_VARIABLE;
                return_var.error_pos=i;
            }
            else {
                if( bool_func_local[i]=='C' || bool_func_local[i]=='K' ) {	//only 6 channels available for the counters, N_LCR_CHANNELS=6
                    if( bool_func_local[i+1]<'0' || bool_func_local[i+1]>'5' )  {		//TODO: update, there was error here missing '{' '}' brackets on 'if'
                        return_var.ret_value=ERROR_BOOL_FUNC_INVALID_VARIABLE;
                        return_var.error_pos=i;
                    }
                }

                if( i < (pos_end_string-3) ) {
                    if( bool_func_local[i+2]==')' ) {
                        i=i+1;	//everything OK, move forward 1 character, only move +1 because ')' must be checked and "number_unmatched_parenthesis_pair" decreased
                    }
                    else {
                        if( bool_func_local[i+2]!='+' && bool_func_local[i+2]!='.' && i<pos_end_string-3) {
                            return_var.ret_value=ERROR_BOOL_FUNC_MISSING_OPERATOR;
                            return_var.error_pos=i;
                        }
                        else {
                            if(bool_func_local[i+3]!='M' && bool_func_local[i+3]!='R' && bool_func_local[i+3]!='C' && bool_func_local[i+3]!='K' && bool_func_local[i+3]!='/' && bool_func_local[i+3]!='0' && bool_func_local[i+3]!='1' && bool_func_local[i+3]!='(') {
                                return_var.ret_value=ERROR_BOOL_FUNC_OPERATION_MISSING_2ND_VARIABLE;
                                return_var.error_pos=i;
                            }
                            else {
                                i=i+2;	//everything OK, move forward 2 character
                            }
                        }
                    }
                }
                else {
                    i=i+1;	//everything OK, move forward 1 character
                }

            }

        }


    }


    if( return_var.ret_value > 0 ) {
        strncpy(bool_func_ptr, bool_func_local, BOOL_FUNC_N_BYTES);
    }

    return return_var;
}


unsigned char modbus_data[255];

//structure with the fields of a ModbusRTU message
typedef struct {
    unsigned char msg_dest_device_ID;
    unsigned char modbus_cmd;
    unsigned char reg_starting_address;
    unsigned char length_or_data;
    unsigned char number_data_bytes;
    unsigned char *data;
    unsigned short CRC;
    char CRC_is_OK;  //boolean value indicates the validity of CRC

}Modbus_Data_Struct;


//Calculate CRC16
#define POLY 0xA001
unsigned int crc16( unsigned char *p, unsigned char n )
{
    unsigned char  i;
    unsigned short crc = 0xFFFF;
    //unsigned short buffer;

    while (n--)
    {
        crc ^= *p++;
        for (i = 8; i != 0; i--)
        {
            if(crc & 1)
                crc = (crc >> 1) ^ POLY;
            else
                crc >>= 1;
        }
    }

    //switch byte order, from little-endian to big-endian
    //buffer = ( (crc << 8) & 0xFF00 );
    //crc = buffer | ((crc >> 8) & 0x00FF) ;

    return (crc);
}


//search for the start of the reply by Serial Port and returns the byte number(of the received buffer) of the last byte of the start of the reply
// the start of the reply is: reply[0]='a', reply[1]=Serial_Command_Code, reply[2]=Serial_Command_Code
// Last_Byte_Number_Of_Modbus_Reply
int Last_Byte_Number_Of_Modbus_Reply(unsigned char *received_buffer_modbus, int modbus_command_code, int MAX_size_reply_start) {
    int i;
    char reply_start_found_bool;

    //search for the start of the reply by Serial Port of the Multiple Sensor Interface
    // the start of the reply is: reply[0]='a', reply[1]=Serial_Command_Code, reply[2]=Serial_Command_Code
    for(i=0, reply_start_found_bool=0; reply_start_found_bool==0 && i < MAX_size_reply_start; i++) {
        if(received_buffer_modbus[i]=='a' && received_buffer_modbus[i+1]==modbus_command_code && received_buffer_modbus[i+2]==modbus_command_code) {
            reply_start_found_bool=1;
        }
    }

    if(reply_start_found_bool==0) {
        i=-1;
    }
    else {
        i=i+2; //return the byte number correspondent to the start of the arguments
    }

    return i;
}


int OpenDevice_Serial_MultipleSensor(MultipleSensor_serial_driver *session, char *serial_port_name) {

    QSerialPort *session_serial_device_qt = new QSerialPort();

    session->serial_device = (void *) session_serial_device_qt;

    QString serial_port_name_qstring = serial_port_name;

    //in case the USB connection is already opened, close it and reconnect
    if (session->isConnected == TRUE_BOOL) {
        session_serial_device_qt->close();
        session->isConnected = FALSE_BOOL;
    }

    session_serial_device_qt->setPortName(serial_port_name_qstring);
    strncpy(session->port_name, serial_port_name, SERIAL_PORT_NAME_MAX_SIZE);

    if ( !( session_serial_device_qt->open(QIODevice::ReadWrite) ) ) {
         session->error_code_recent=SERIAL_CONNECTION_ERROR;
          return SERIAL_CONNECTION_ERROR;
     }

       if ( !( session_serial_device_qt->setBaudRate(QSerialPort::Baud57600) ) ) {
           session->error_code_recent=SERIAL_SET_BAUD_RATE_ERROR;
           return SERIAL_SET_BAUD_RATE_ERROR;
       }

       if ( !( session_serial_device_qt->setDataBits(QSerialPort::Data8) ) ) {
           session->error_code_recent=SERIAL_SET_DATA_BITS_ERROR;
           return SERIAL_SET_DATA_BITS_ERROR;
       }

       if ( !( session_serial_device_qt->setParity(QSerialPort::NoParity) ) ) {
           session->error_code_recent=SERIAL_SET_PARITY_ERROR;
           return SERIAL_SET_PARITY_ERROR;
       }

       if ( !( session_serial_device_qt->setStopBits(QSerialPort::OneStop) ) ) {
           session->error_code_recent=SERIAL_SET_STOP_BITS_ERROR;
           return SERIAL_SET_STOP_BITS_ERROR;
       }

       if ( !( session_serial_device_qt->setFlowControl(QSerialPort::NoFlowControl) ) ) {
           session->error_code_recent=SERIAL_SET_FLOW_CONTROL_ERROR;
           return SERIAL_SET_FLOW_CONTROL_ERROR;
       }

    if ( session_serial_device_qt->isOpen()==1 ) {
        session->isConnected = TRUE_BOOL;
        session->error_code_recent=SERIAL_CONNECTED;
        return SERIAL_CONNECTED;
    }
    else {
        session->isConnected = FALSE_BOOL;
        session->error_code_recent=SERIAL_CONNECTION_ERROR;
        return SERIAL_CONNECTION_ERROR;
    }


}



// FUNCTION: Find_Start_Modbus_RTU - Find any start of ModbusRTU command of the type: ID|function_code
//		     Return value: Returns the index number were is the first byte of the ModbusRTU command
int Find_Start_Modbus_RTU(unsigned char ID, unsigned char function_code, unsigned char *str_msg, int str_msg_length) {

    int i, ret_value=-1;

    for( i=0, ret_value=-1; i<str_msg_length && ret_value<0; i++) {
        if(str_msg[i]==ID || str_msg[i]==255) { //search for the 1st byte that is the device_ID or in case of the broadcast_ID is 255
            if(str_msg[i+1]==function_code) {
                ret_value=i;
            }
        }
    }

    return ret_value;
}

// FUNCTION: Find_Any_Start_Modbus_RTU, Description: Find any start of ModbusRTU command of the type: ID|function_code, where function_code can be: {4; 6; 16}
//		     Return value: Returns the index number were is the first byte of the ModbusRTU command
int Find_Any_Start_Modbus_RTU(unsigned char *str_msg, int str_msg_length, unsigned char ID) {

    int ret_value;
    int i_min=9999;
    char start_of_cmd_found=0;

    //ID_temp = ID;
    i_min=9999;
    start_of_cmd_found=0;
    str_msg_length=str_msg_length+5; //some extra length to be searched

    //repeated code of Find_Start_Modbus_RTU(...), to avoid calling another function, and so increasing the call depth, reducing the risk of stack overflow

    //func_code=3;
    ret_value=Find_Start_Modbus_RTU(ID, 3, str_msg, str_msg_length);
    if(ret_value<i_min && ret_value>=0) {
        i_min=ret_value; start_of_cmd_found=1;
    }

    //func_code=4;
    ret_value=Find_Start_Modbus_RTU(ID, 4, str_msg, str_msg_length);
    if(ret_value<i_min && ret_value>=0) {
        i_min=ret_value; start_of_cmd_found=1;
    }

    //func_code=6;
    ret_value=Find_Start_Modbus_RTU(ID, 6, str_msg, str_msg_length);
    if(ret_value<i_min && ret_value>=0) {
        i_min=ret_value; start_of_cmd_found=1;
    }

    //func_code=16;
    ret_value=Find_Start_Modbus_RTU(ID, 16, str_msg, str_msg_length);
    if(ret_value<i_min && ret_value>=0) {
        i_min=ret_value; start_of_cmd_found=1;
    }

    if( start_of_cmd_found == 0)
        i_min=-1;	//wasn't found any start of a Modbus RTU command

    return i_min;
}

// FUNCTION: Process_Modbus_Reply, Description: Analyze the reply to a ModbusRTU command, save the contents in the appropriate fields of Modbus_Data_Struct, and verify the checksum
//		     Return value: Returns the a data struct with the contents of the Modbus reply
#define DATA_BUFFER_SIZE_N2 30
Modbus_Data_Struct Process_Modbus_Reply(unsigned char *InputPacketBuffer_ptr, unsigned char this_device_ID) {

    Modbus_Data_Struct received_modbus_reply;
    unsigned short expected_CRC;
    int start_of_RTU_command, last_start_of_RTU_command;
    int byte_number, i;
    //unsigned char this_device_ID;
    unsigned char byte_1, byte_2, received_CRC_byte1, received_CRC_byte2;


    //try to process all the start of commands inside the buffer in case the CRC is OK then they are valid commands
    //start_of_RTU_command=0;
    //CRC_is_OK=0;

    for(last_start_of_RTU_command = -1,  start_of_RTU_command=0, received_modbus_reply.CRC_is_OK=0;  start_of_RTU_command<MAX_CHARACTERS_OF_MESSAGE && start_of_RTU_command>=0 && received_modbus_reply.CRC_is_OK<1 ; ) {

            //search for the start of the command, search for ID and function_code, "byte_number" here is start_of_RTU_command
            byte_number = Find_Any_Start_Modbus_RTU(InputPacketBuffer_ptr+start_of_RTU_command, 64, this_device_ID );
            if( byte_number < 0 || byte_number < last_start_of_RTU_command) {
                //in case a start of the command wasn't found considering this device_ID, then try the 255(broadcast) device_ID
                 byte_number = Find_Any_Start_Modbus_RTU(InputPacketBuffer_ptr+start_of_RTU_command, 64, 255 );
                 if( byte_number >=0 && byte_number>last_start_of_RTU_command) {
                    start_of_RTU_command = byte_number;
                 }
                 else {
                    start_of_RTU_command=start_of_RTU_command+4;
                 }
            }
            else {
                start_of_RTU_command = byte_number;
            }

            //only process the command if is addressed to this device_ID or is addressed to any device (device_ID=255)
            if( start_of_RTU_command>=0 && last_start_of_RTU_command != start_of_RTU_command) {	//start_of_RTU_command>=0 means the start of command was found inside the message content

                //the msg_dest_device_ID must be device_ID or 255
                received_modbus_reply.msg_dest_device_ID = (unsigned char) InputPacketBuffer_ptr[start_of_RTU_command];

                //after the device_ID is the function_code
                received_modbus_reply.modbus_cmd = (unsigned char) InputPacketBuffer_ptr[start_of_RTU_command+1];

                if(	received_modbus_reply.modbus_cmd==0x03 || received_modbus_reply.modbus_cmd==0x04 ) {

                    //the next byte is the size in byte of the data in the reply
                    received_modbus_reply.number_data_bytes =  (unsigned char) InputPacketBuffer_ptr[start_of_RTU_command+2];

                    for(i=0, received_modbus_reply.data=modbus_data; i<received_modbus_reply.number_data_bytes; i++) {
                        received_modbus_reply.data[i] = InputPacketBuffer_ptr[start_of_RTU_command+i+3];
                    }

                    received_CRC_byte1 = InputPacketBuffer_ptr[start_of_RTU_command+received_modbus_reply.number_data_bytes+3];
                    received_CRC_byte2 = InputPacketBuffer_ptr[start_of_RTU_command+received_modbus_reply.number_data_bytes+4];
                    received_modbus_reply.CRC = ((received_CRC_byte2<<8)&0xFF00) | (received_CRC_byte1&0x00FF);

                    //check if the CRC of the message is OK
                    expected_CRC = crc16( InputPacketBuffer_ptr+start_of_RTU_command, 3+received_modbus_reply.number_data_bytes);
                }
                else {
                    if( received_modbus_reply.modbus_cmd==0x06 || received_modbus_reply.modbus_cmd==0x10 ) {
                        //
                        //the next 2 bytes are the address of the 1st register to be read (starting address)
                        byte_1 = (unsigned char) InputPacketBuffer_ptr[start_of_RTU_command+2];
                        byte_2 = (unsigned char) InputPacketBuffer_ptr[start_of_RTU_command+3];
                        received_modbus_reply.reg_starting_address = ( (((int) byte_1) << 8 ) & 0xFF00) | ( byte_2 & 0x00FF ) ;

                        //the next 2 bytes are the length of the data to be read (number of bytes to read)
                        byte_1 = (unsigned char) InputPacketBuffer_ptr[start_of_RTU_command+4];
                        byte_2 = (unsigned char) InputPacketBuffer_ptr[start_of_RTU_command+5];
                        received_modbus_reply.length_or_data = ( (((int) byte_1) << 8 ) & 0xFF00) | ( byte_2 & 0x00FF ) ;

                        received_CRC_byte1 = (unsigned char) InputPacketBuffer_ptr[start_of_RTU_command+6];
                        received_CRC_byte2 = (unsigned char) InputPacketBuffer_ptr[start_of_RTU_command+7];
                        //check if the CRC of the message is OK
                        received_modbus_reply.CRC = (0xFF00&(received_CRC_byte2<<8)) | (0x00FF&(received_CRC_byte1));

                        //check if the CRC of the message is OK
                        expected_CRC = crc16( InputPacketBuffer_ptr+start_of_RTU_command, 6);

                    }
                }

                if(received_modbus_reply.CRC==expected_CRC) {
                    received_modbus_reply.CRC_is_OK=1;
                }
                else {
                    received_modbus_reply.CRC_is_OK=0;
                }

            }

            //save the position of the search for the start of an ModbusRTU command, "last_start_of_RTU_command"
            if(start_of_RTU_command <= last_start_of_RTU_command) {
                start_of_RTU_command=last_start_of_RTU_command+4;
            }
            else {
                last_start_of_RTU_command = start_of_RTU_command;
            }
        }

    return received_modbus_reply;
}


int ReadFrequencyMeasurement_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int32_t *integer_CH_vector) {

         int ret_value;
         // int integer_CH_vector[N_LCR_CHANNELS];	//vector were are saved integer values with sensor information (for example the frequency of the sensor signal)
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned short CRC;
         unsigned char *data_ptr;
         Modbus_Data_Struct modbus_reply;
         QThread *current_thread_ptr;

         int i, serial_device_bytes_available;

         QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

         InputPacketBuffer[0] = 0;				//First byte empty

         QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
         current_thread_ptr = QThread::currentThread();

         OutputPacketBuffer[0]=session->board_ID;
         OutputPacketBuffer[1]=QUERY_FREQUENCY_MEASUREMENTS_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[2]=0x00;
         OutputPacketBuffer[3]=QUERY_FREQUENCY_MEASUREMENTS_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[4]=0x00;
         OutputPacketBuffer[5]=12;  //request to receive 12 words (1=16bits=2bytes), the size of the 6 LCR channels (6 x sizeof(long))

         CRC = crc16( OutputPacketBuffer, 6 );
         OutputPacketBuffer[6] = CRC&0x00FF;
         OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
         //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
          if (session->serial_device->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
              memcpy(FlushBuffer, session->serial_device->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
          }

          current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
          */
        //Query_Modbus_Port( session, buffer_byte_array, 7, InputPacketBuffer, 100);
         if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)  //always write only the number of bytes filled with data on buffer_byte_array to avoid error on the read operation
         {
             session->error_code_recent=SERIAL_COMMAND_ERROR;
             return SERIAL_COMMAND_ERROR;
         }

         //Now get the response packet from the firmware.
         for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_FREQUENCY_MEASUREMENTS_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
             current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
             QCoreApplication::processEvents();  //Process event in the queue
             serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
         }
         memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

              modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

             //check the reply to the sent command
            if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_FREQUENCY_MEASUREMENTS_ModbusRTU_Function) ) {
                ret_value = SERIAL_COMMAND_OK;
            }
            else {
                ret_value = SERIAL_COMMAND_ERROR;
                session->error_code_recent=SERIAL_COMMAND_ERROR;
                CloseDevice_Serial_MultipleSensor(session);
                OpenDevice_Serial_MultipleSensor(session, session->port_name);
            }

         for(i=0;i<N_LCR_CHANNELS;i++)  {       //convert the received bytes of the frequency measurements back to 'long int' variables
            data_ptr=(unsigned char *) &integer_CH_vector[i];
            *(data_ptr+0) = modbus_reply.data[(4*i)+1];
            *(data_ptr+1) = modbus_reply.data[(4*i)];
            *(data_ptr+2) = modbus_reply.data[(4*i)+3];
            *(data_ptr+3) = modbus_reply.data[(4*i)+2];
        }

   /*      // ----- DEBUG CODE -----
         printf("\nInputPacketBuffer:\n");
         for(i=0; i<MAX_SIZE_SERIAL_RECEIVE_BUFFER; i++) {
             printf("%X ", InputPacketBuffer[i]);   //DEBUG CODE
         }
         // ----------------------

         // ----- DEBUG CODE -----
         printf("\nFlushBuffer:\n");
         for(i=0; i<MAX_SIZE_SERIAL_RECEIVE_BUFFER; i++) {
             printf("%X ", FlushBuffer[i]);   //DEBUG CODE
         }
         // ----------------------
        */

    session->error_code_recent=ret_value;
    return ret_value;
}


int ReadADCValue_Serial_MultipleSensor(MultipleSensor_serial_driver *session, float *ADC_values_vector_ptr) {

         int ch_number, byte_number, ret_value;

         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         float ADC_value;	//floating point with the value red by the ADC channel, ADC(analogue to digital converter)
         unsigned char *ADC_value_data;	//pointer to the data received by USB
         // float ADC_values_vector[N_ADC_CHANNELS];	//vector used to save the ADC values(units: volt) of the ADC channels

          unsigned short CRC;
          Modbus_Data_Struct modbus_reply;
          QThread *current_thread_ptr;

          int i, serial_device_bytes_available;

          QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

          QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected          
          current_thread_ptr = QThread::currentThread();

         OutputPacketBuffer[0]=session->board_ID;
         OutputPacketBuffer[1]=QUERY_ADC_MEASUREMENTS_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[2]=0x00;
         OutputPacketBuffer[3]=QUERY_ADC_MEASUREMENTS_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[4]=0x00;
         OutputPacketBuffer[5]=8;  //request to receive 8 words (1word=16bits=2bytes), the size of the 4 ADC channels (4 x sizeof(float))
         CRC = crc16( OutputPacketBuffer, 6 );
         OutputPacketBuffer[6] = CRC&0x00FF;
         OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
         //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
          if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
              memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
          }

          current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
          */
         if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)
         {
             session->error_code_recent=SERIAL_COMMAND_ERROR;
             CloseDevice_Serial_MultipleSensor(session);
             OpenDevice_Serial_MultipleSensor(session, session->port_name);
             ret_value = SERIAL_COMMAND_ERROR;
         }

         //Now get the response packet from the firmware.
         for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_ADC_MEASUREMENTS_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
              current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
              QCoreApplication::processEvents();  //Process event in the queue
              serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
          }
          memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

              modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
             //check the reply to the sent command
            if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_ADC_MEASUREMENTS_ModbusRTU_Function) ) {
                ret_value = SERIAL_COMMAND_OK;
            }
            else {
                ret_value = SERIAL_COMMAND_ERROR;
            }

         if(modbus_reply.CRC_is_OK==1) {        //convert the received bytes of the ADC voltage back to 'float' variables
            ADC_value_data = (unsigned char *) &ADC_value;
            for(ch_number=0,byte_number=0;ch_number<4;ch_number++,byte_number=byte_number+4) {
                ADC_value_data[0]=modbus_reply.data[byte_number+1];
                ADC_value_data[1]=modbus_reply.data[byte_number];
                ADC_value_data[2]=modbus_reply.data[byte_number+3];
                ADC_value_data[3]=modbus_reply.data[byte_number+2];
                ADC_values_vector_ptr[ch_number]=ADC_value;
            }
         }

    session->error_code_recent=ret_value;
    return ret_value;
}



int ReadMeasurements_Serial_MultipleSensor(MultipleSensor_serial_driver *session, float *measurements_values_vector_ptr) {

         int ch_number, byte_number, ret_value;

         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         float measurement_value;	//floating point with the value read of the calibrated measurement, LCR and ADC(analogue to digital converter)
         unsigned char *measurement_value_data;	//pointer to the data received by USB
         // float measurement_values_vector[N_LCR_CHANNELS+N_ADC_CHANNELS];	//vector used to save the measurements values(units: defined at calibration table) of the LCR+ADC channels

          unsigned short CRC;
          Modbus_Data_Struct modbus_reply;
          QThread *current_thread_ptr;

          int i;
          int serial_device_bytes_available;

          QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

          QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
          current_thread_ptr = QThread::currentThread();

         OutputPacketBuffer[0]=session->board_ID;
         OutputPacketBuffer[1]=QUERY_MEASUREMENTS_CH_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[2]=0x00;
         OutputPacketBuffer[3]=QUERY_MEASUREMENTS_CH_CODE_ModbusRTU;  //send the Query Measurements of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[4]=0x00;
         OutputPacketBuffer[5]=20;  //request to receive 20 words (1word=16bits=2bytes), the size of the LCR+ADC channels ((6+4) x sizeof(float))
         CRC = crc16( OutputPacketBuffer, 6 );
         OutputPacketBuffer[6] = CRC&0x00FF;
         OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
         //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
          if (session->serial_device->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
              memcpy(FlushBuffer, session->serial_device->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
          }

          current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
          */
         if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)
         {
             session->error_code_recent=SERIAL_COMMAND_ERROR;
             CloseDevice_Serial_MultipleSensor(session);
             OpenDevice_Serial_MultipleSensor(session, session->port_name);
             ret_value = SERIAL_COMMAND_ERROR;
         }

            for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_MEASUREMENTS_CH_CODE_ModbusRTU && i<10; i++) {
                current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
                QCoreApplication::processEvents();  //Process event in the queue
                serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
            }
            memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

              modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
             //check the reply to the sent command
            if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_MEASUREMENTS_CH_ModbusRTU_Function) ) {
                ret_value = SERIAL_COMMAND_OK;
            }
            else {
                ret_value = SERIAL_COMMAND_ERROR;
            }

         if(modbus_reply.CRC_is_OK==1) {
            measurement_value_data = (unsigned char *) &measurement_value;
            for(ch_number=0,byte_number=0;ch_number < (N_LCR_CHANNELS+N_ADC_CHANNELS);ch_number++,byte_number=byte_number+4) {
                measurement_value_data[0]=modbus_reply.data[byte_number+1];
                measurement_value_data[1]=modbus_reply.data[byte_number];
                measurement_value_data[2]=modbus_reply.data[byte_number+3];
                measurement_value_data[3]=modbus_reply.data[byte_number+2];
                measurements_values_vector_ptr[ch_number]=measurement_value;
            }
         }

    session->error_code_recent=ret_value;
    return ret_value;
}



int ReadCountersMeasurements_Serial_MultipleSensor(MultipleSensor_serial_driver *session, float *counters_measurements_values_vector_ptr) {

         int ch_number, byte_number, ret_value;

         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         float counter_measurement_value;	//floating point with the value read of the calibrated measurement, LCR and ADC(analogue to digital converter)
         unsigned char *counter_measurement_value_data;	//pointer to the data received by USB
         // float measurement_values_vector[N_LCR_CHANNELS+N_ADC_CHANNELS];	//vector used to save the measurements values(units: defined at calibration table) of the LCR+ADC channels

          unsigned short CRC;
          Modbus_Data_Struct modbus_reply;
          QThread *current_thread_ptr;

          int i;
          int serial_device_bytes_available;

          QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

          QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
          current_thread_ptr = QThread::currentThread();

         OutputPacketBuffer[0]=session->board_ID;
         OutputPacketBuffer[1]=QUERY_COUNTERS_MEASUREMENTS_CH_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[2]=0x00;
         OutputPacketBuffer[3]=QUERY_COUNTERS_MEASUREMENTS_CH_CODE_ModbusRTU;  //send the Query Counters Measurements of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[4]=0x00;
         OutputPacketBuffer[5]=12;  //request to receive 12 words (1word=16bits=2bytes), the size of the LCR channels (6 x sizeof(float))
         CRC = crc16( OutputPacketBuffer, 6 );
         OutputPacketBuffer[6] = CRC&0x00FF;
         OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
         //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
          if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
              memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
          }

          current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
          */
         if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)
         {
             session->error_code_recent=SERIAL_COMMAND_ERROR;
             CloseDevice_Serial_MultipleSensor(session);
             OpenDevice_Serial_MultipleSensor(session, session->port_name);
             ret_value = SERIAL_COMMAND_ERROR;
         }

            for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_COUNTERS_MEASUREMENTS_CH_CODE_ModbusRTU && i<10; i++) {
                current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
                QCoreApplication::processEvents();  //Process event in the queue
                serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
            }
            memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

              modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
             //check the reply to the sent command
            if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_MEASUREMENTS_CH_ModbusRTU_Function) ) {
                ret_value = SERIAL_COMMAND_OK;
            }
            else {
                ret_value = SERIAL_COMMAND_ERROR;
            }

         if(modbus_reply.CRC_is_OK==1) {        //convert the received bytes of the counters measurements back to 'float' variables
            counter_measurement_value_data = (unsigned char *) &counter_measurement_value;
            for(ch_number=0,byte_number=0;ch_number < (N_LCR_CHANNELS+N_ADC_CHANNELS);ch_number++,byte_number=byte_number+4) {
                counter_measurement_value_data[0]=modbus_reply.data[byte_number+1];
                counter_measurement_value_data[1]=modbus_reply.data[byte_number];
                counter_measurement_value_data[2]=modbus_reply.data[byte_number+3];
                counter_measurement_value_data[3]=modbus_reply.data[byte_number+2];
                counters_measurements_values_vector_ptr[ch_number]=counter_measurement_value;
            }
         }

    session->error_code_recent=ret_value;
    return ret_value;
}




/*
int Select_Measurement_Time_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int measurement_time) {

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

     unsigned char FlushBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];

    int return_value=SERIAL_COMMAND_ERROR;

    QByteArray buffer_byte_array;

    QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

    buffer_byte_array[0]='a';
    buffer_byte_array[1]=0x80;    //0x80 is the "Select Measurement Time" command in the firmware
    buffer_byte_array[2]=0x80;    //0x80 is the "Select Measurement Time" command in the firmware
    buffer_byte_array[3]=session->board_ID;
    buffer_byte_array[4]=measurement_time;
    buffer_byte_array[5]='X';  //dummy byte necessary to avoid errors
    buffer_byte_array[6]='X';  //dummy byte necessary to avoid errors
    buffer_byte_array[7]='\r';
    buffer_byte_array[8]='\n';

     //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
      if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
          memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
      }
      QThread::usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);

    QThread::usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
    if( session_serial_device_qt->write( buffer_byte_array , 9) == -1)
    {
        session->error_code_recent=SERIAL_COMMAND_ERROR;
        return SERIAL_COMMAND_ERROR;
    }
    else {

        //Now get the response packet from the firmware.
           QThread::usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
           if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
               memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
           }

        if(InputPacketBuffer[0] != 0x80)  //the device returns the reply with the command code on the first byte, in case is different an error occurred
        {
            session->error_code_recent=SERIAL_COMMAND_ERROR;
            CloseDevice_Serial_MultipleSensor(session);
            OpenDevice_Serial_MultipleSensor(session, session->port_name);
            return_value= SERIAL_COMMAND_ERROR;
        }
        else {
            return_value= SERIAL_COMMAND_OK;
        }
         //Flush Serial Port receive buffer, it was verified that is also required to flush the receive buffer after reading the serial port to avoid erros on the next read operation
        QThread::usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
         if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
             memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
         }
    }

    session->error_code_recent=ret_value;
    return return_value;
}
*/

int SaveSensorConfigs_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int configs_for_channel_number, char make_compare_w_calibrated_sensor_value, char compare_is_greater_than, char sensor_prescaler, char sensor_used_for_OUT[3], float sensor_trigger_value) {

        unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
        unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char *data;

         int ret_value;
         unsigned short CRC;
         Modbus_Data_Struct modbus_reply;
         QThread *current_thread_ptr;

         int i, serial_device_bytes_available;

         QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

         sensor_prescaler = sensor_prescaler;   //sensor_prescaler obsolete variable

        InputPacketBuffer[0] = 0;				//First byte empty

        QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
        current_thread_ptr = QThread::currentThread();

        OutputPacketBuffer[0]=session->board_ID;
        OutputPacketBuffer[1]=SAVE_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function;     	//0x10 is the save_multiple_registers Modbus command
        OutputPacketBuffer[2]=0x00;
        OutputPacketBuffer[3]=SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU+configs_for_channel_number;  //send the Query Channel Configurations of Multiple-Sensor Modbus protocol
        OutputPacketBuffer[4]=0x00;
        OutputPacketBuffer[5]=7;  //request to send 7 words (1word=16bits=2bytes), the size of the configuration of 1 LCR channel
        OutputPacketBuffer[6]=14;  //number of data bytes of the command ( 2x 7words = 14bytes)

        OutputPacketBuffer[7]=0;
        OutputPacketBuffer[8]=make_compare_w_calibrated_sensor_value&0x01;
        OutputPacketBuffer[9]=0;
        OutputPacketBuffer[10]=compare_is_greater_than&0x01;
        OutputPacketBuffer[11]=0;
        OutputPacketBuffer[12]=sensor_used_for_OUT[0]&0x01;
        OutputPacketBuffer[13]=0;
        OutputPacketBuffer[14]=sensor_used_for_OUT[1]&0x01;
        OutputPacketBuffer[15]=0;
        OutputPacketBuffer[16]=sensor_used_for_OUT[2]&0x01;

        data = (unsigned char *) &sensor_trigger_value;
        OutputPacketBuffer[17]=data[1];   //send byte #0 of sensor_trigger_value
        OutputPacketBuffer[18]=data[0];   //send byte #1 of sensor_trigger_value
        OutputPacketBuffer[19]=data[3];   //send byte #2 of sensor_trigger_value
        OutputPacketBuffer[20]=data[2];   //send byte #3 of sensor_trigger_value

        CRC = crc16( OutputPacketBuffer, 21 );
        OutputPacketBuffer[21] = CRC&0x00FF;
        OutputPacketBuffer[22] = (CRC>>8)&0x00FF;
/*
        //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
         if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
             memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
         }

        current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
        */
        //Query_Modbus_Port( session, buffer_byte_array, 7, InputPacketBuffer, 100);
        if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 23) == -1)  //always write only the number of bytes filled with data on buffer_byte_array to avoid error on the read operation
        {
            session->error_code_recent=SERIAL_COMMAND_ERROR;
            CloseDevice_Serial_MultipleSensor(session);
            OpenDevice_Serial_MultipleSensor(session, session->port_name);
            ret_value = SERIAL_COMMAND_ERROR;
        }

        //Now get the response packet from the firmware.
        for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_SAVE_CONFIGURATIONS_SENSOR_CH_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
             current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
             QCoreApplication::processEvents();  //Process event in the queue
             serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
         }
         memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

        //check the reply to the sent command
       if( modbus_reply.CRC_is_OK==1 && modbus_reply.modbus_cmd==SAVE_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function ) {
           ret_value = SERIAL_COMMAND_OK;
       }
       else {
           ret_value = SERIAL_COMMAND_ERROR;
       }

    session->error_code_recent=ret_value;
    return ret_value;
}


int ReadSensorConfigs_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int configs_for_channel_number, char *make_compare_w_calibrated_sensor_value_ptr, char *compare_is_greater_than_ptr, char *sensor_prescaler_ptr, char *sensor_used_for_OUT_ptr, float *sensor_trigger_value_ptr) {

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
     unsigned char *data;
     float sensor_trigger_value;

     int ret_value;
     unsigned short CRC;
     Modbus_Data_Struct modbus_reply;
     QThread *current_thread_ptr;

     int i, serial_device_bytes_available;

     QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

     QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
     current_thread_ptr = QThread::currentThread();

     OutputPacketBuffer[0]=session->board_ID;
     OutputPacketBuffer[1]=QUERY_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
     OutputPacketBuffer[2]=0x00;
     OutputPacketBuffer[3]=SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU+configs_for_channel_number;  //send the Query Channel Configurations of Multiple-Sensor Modbus protocol
     OutputPacketBuffer[4]=0x00;
     OutputPacketBuffer[5]=8;  //request to receive 8 words (1word=16bits=2bytes), the size of the configuration of 1 LCR channel
     CRC = crc16( OutputPacketBuffer, 6 );
     OutputPacketBuffer[6] = CRC&0x00FF;
     OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
     //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
      if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
          memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
      }

      current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
      */
     if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)
     {
         session->error_code_recent=SERIAL_COMMAND_ERROR;
         CloseDevice_Serial_MultipleSensor(session);
         OpenDevice_Serial_MultipleSensor(session, session->port_name);
         ret_value = SERIAL_COMMAND_ERROR;
     }

     //Now get the response packet from the firmware. 
     for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
          current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
          QCoreApplication::processEvents();  //Process event in the queue
          serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
      }
      memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

          modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

         //check the reply to the sent command
        if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function)) {
            ret_value = SERIAL_COMMAND_OK;
        }
        else {
            ret_value = SERIAL_COMMAND_ERROR;
        }

        if(modbus_reply.CRC_is_OK<1)
        {
            if(modbus_reply.data[1]>1 && modbus_reply.data[3]>1) {
                session->error_code_recent=SERIAL_REPLY_WITH_INVALID_DATA;
                ret_value = SERIAL_REPLY_WITH_INVALID_DATA;
            }
            else {
                session->error_code_recent=SERIAL_COMMAND_ERROR;
                CloseDevice_Serial_MultipleSensor(session);
                OpenDevice_Serial_MultipleSensor(session, session->port_name);
                ret_value = SERIAL_COMMAND_ERROR;
            }
        }
        else {

            //data[1]: make_compare_w_calibrated_sensor, data[3]: compare_is_greater_than; these are 0 or 1, if different declare invalid data
            if(modbus_reply.data[1]>1 && modbus_reply.data[3]>1) {
                session->error_code_recent=SERIAL_REPLY_WITH_INVALID_DATA;
                ret_value = SERIAL_REPLY_WITH_INVALID_DATA;
            }
            else {

                if(modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04) {

                    //read the sensor channel configurations from the bits of sensors_configs
                    *make_compare_w_calibrated_sensor_value_ptr=modbus_reply.data[1]&0x01;
                    *compare_is_greater_than_ptr=modbus_reply.data[3]&0x01;
                    *sensor_prescaler_ptr=modbus_reply.data[5];
                    sensor_used_for_OUT_ptr[0]=modbus_reply.data[7]&0x01;
                    sensor_used_for_OUT_ptr[1]=modbus_reply.data[9]&0x01;
                    sensor_used_for_OUT_ptr[2]=modbus_reply.data[11]&0x01;

                    // read the trigger value of the sensor for the associated outputs
                    data = (unsigned char *) &sensor_trigger_value;
                    data[0]=modbus_reply.data[13];
                    data[1]=modbus_reply.data[12];
                    data[2]=modbus_reply.data[15];
                    data[3]=modbus_reply.data[14];

                    *sensor_trigger_value_ptr=sensor_trigger_value;  //save the read trigger_value on the argument variable

                    //printf("*sensor_trigger_value_ptr: %d", *sensor_trigger_value_ptr);   //DEBUG CODE
                }

            }
        }

    session->error_code_recent=ret_value;
    return ret_value;
}


int WriteOutputConfig_Serial_MultipleSensor(MultipleSensor_serial_driver *session, unsigned char output_number, char *out_bool_func_str) {

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1


     int ret_value = SERIAL_COMMAND_ERROR;
     unsigned short CRC;
     Modbus_Data_Struct modbus_reply;
     QThread *current_thread_ptr;

     ret_type1 ret_value_check_bool_func;

     int i, serial_device_bytes_available;

     QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

     if( output_number >=0 && output_number < N_DIGITAL_OUT) {

         //checks for missing start and end parenthesis '(' ')' in the bool function, and adds it if necessary
         //checks the syntax and variables of the bool function
         ret_value_check_bool_func=CheckBooleanFunc(out_bool_func_str);
         ret_value = ret_value_check_bool_func.ret_value;
         if( ret_value_check_bool_func.ret_value<0 ) {
             session->error_code_recent=ret_value;
             session->error_at_CH_number_or_position=ret_value_check_bool_func.error_pos;
         }

         if(ret_value>0) {
             QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
             current_thread_ptr = QThread::currentThread();

             OutputPacketBuffer[0]=session->board_ID;
             OutputPacketBuffer[1]=SAVE_OUTPUT_CONFIG_ModbusRTU_Function;     	//SAVE_OUTPUT_CONFIG_ModbusRTU_Function is 0x10 Modbus command
             OutputPacketBuffer[2]=0x00;
             OutputPacketBuffer[3]=SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU+output_number;  //send the save outputs active level config of Multiple-Sensor Modbus protocol
             OutputPacketBuffer[4]=0x00;
             OutputPacketBuffer[5]=BOOL_FUNC_N_BYTES/2;  //request to save 27 (BOOL_FUNC_N_BYTES/2) words (1word=16bits=2bytes)
             OutputPacketBuffer[6]=BOOL_FUNC_N_BYTES;   //number of bytes to save

             for(i=0; i < BOOL_FUNC_N_BYTES; i++) {
                OutputPacketBuffer[7+i]=out_bool_func_str[i];
             }

             CRC = crc16( OutputPacketBuffer, 7+BOOL_FUNC_N_BYTES );
             OutputPacketBuffer[7+BOOL_FUNC_N_BYTES] = CRC&0x00FF;
             OutputPacketBuffer[8+BOOL_FUNC_N_BYTES] = (CRC>>8)&0x00FF;

             if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 9+BOOL_FUNC_N_BYTES) == -1)
             {
                 session->error_code_recent=SERIAL_COMMAND_ERROR;
                 CloseDevice_Serial_MultipleSensor(session);
                 OpenDevice_Serial_MultipleSensor(session, session->port_name);
                 ret_value = SERIAL_COMMAND_ERROR;
             }
             else {

               //Now get the response packet from the firmware.
                 for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
                      current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
                      QCoreApplication::processEvents();  //Process event in the queue
                      serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
                  }
                  memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

                  modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
                 //check the reply to the sent command
                if( modbus_reply.CRC_is_OK==1 && modbus_reply.modbus_cmd==SAVE_OUTPUT_CONFIG_ModbusRTU_Function ) {
                    ret_value = SERIAL_COMMAND_OK;
                }
                else {
                    ret_value = SERIAL_COMMAND_ERROR;
                }

            }

        }

     }

    session->error_code_recent=ret_value;
    return ret_value;
}



int ReadOutputConfig_Serial_MultipleSensor(MultipleSensor_serial_driver *session, unsigned char output_number, char *out_bool_func_str) {

     unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
     unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

    int ret_value = SERIAL_COMMAND_ERROR;
    unsigned short CRC;
    Modbus_Data_Struct modbus_reply;
    QThread *current_thread_ptr;

    QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

    int i, serial_device_bytes_available;

    if( output_number >=0 && output_number < N_DIGITAL_OUT) {

        QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
        current_thread_ptr = QThread::currentThread();

        OutputPacketBuffer[0]=session->board_ID;
        OutputPacketBuffer[1]=QUERY_OUTPUT_CONFIG_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
        OutputPacketBuffer[2]=0x00;
        OutputPacketBuffer[3]=SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU+output_number;  //send the Query Outputs_Config of Multiple-Sensor Modbus protocol
        OutputPacketBuffer[4]=0x00;
        OutputPacketBuffer[5]=BOOL_FUNC_N_BYTES/2;  //request to read 27 (BOOL_FUNC_N_BYTES/2) words (1word=16bits=2bytes)

        CRC = crc16( OutputPacketBuffer, 6 );
        OutputPacketBuffer[6] = CRC&0x00FF;
        OutputPacketBuffer[7] = (CRC>>8)&0x00FF;

        if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)
        {
            session->error_code_recent=SERIAL_COMMAND_ERROR;
            CloseDevice_Serial_MultipleSensor(session);
            OpenDevice_Serial_MultipleSensor(session, session->port_name);
            ret_value = SERIAL_COMMAND_ERROR;
        }
        else {

          //Now get the response packet from the firmware.
            for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
                 current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
                 QCoreApplication::processEvents();  //Process event in the queue
                 serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
             }
             memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

             modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
             //check the reply to the sent command
            if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_OUTPUT_CONFIG_ModbusRTU_Function) ) {
                ret_value = SERIAL_COMMAND_OK;
               strncpy(out_bool_func_str, (char *) modbus_reply.data, BOOL_FUNC_N_BYTES);
            }
            else {
                ret_value = SERIAL_COMMAND_ERROR;
                session->error_code_recent=SERIAL_COMMAND_ERROR;
                CloseDevice_Serial_MultipleSensor(session);
                OpenDevice_Serial_MultipleSensor(session, session->port_name);
            }
       }

    }

    session->error_code_recent=ret_value;
    return ret_value;
}



int ReadOutputs_Serial_MultipleSensor(MultipleSensor_serial_driver *session, unsigned char *outputs_vector) {

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

    int ret_value = SERIAL_COMMAND_ERROR;
    unsigned short CRC;
    Modbus_Data_Struct modbus_reply;
    QThread *current_thread_ptr;

    int i, serial_device_bytes_available;

    QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

    QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
    current_thread_ptr = QThread::currentThread();

    OutputPacketBuffer[0]=session->board_ID;
    OutputPacketBuffer[1]=QUERY_OUTPUTS_CURRENT_VALUE_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
    OutputPacketBuffer[2]=0x00;
    OutputPacketBuffer[3]=QUERY_OUTPUTS_CURRENT_VALUE_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
    OutputPacketBuffer[4]=0x00;
    OutputPacketBuffer[5]=3;  //request to receive 3 words (1=16bits=2bytes), the size of the 3 outputs

    CRC = crc16( OutputPacketBuffer, 6 );
    OutputPacketBuffer[6] = CRC&0x00FF;
    OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
    //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
     if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
         memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
     }

     current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
     */
    if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)
    {
        session->error_code_recent=SERIAL_COMMAND_ERROR;
        CloseDevice_Serial_MultipleSensor(session);
        OpenDevice_Serial_MultipleSensor(session, session->port_name);
        ret_value = SERIAL_COMMAND_ERROR;
    }
    else {

      //Now get the response packet from the firmware.
        for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_OUTPUTS_CURRENT_VALUE_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
             current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
             QCoreApplication::processEvents();  //Process event in the queue
             serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
         }
         memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_OUTPUTS_CURRENT_VALUE_ModbusRTU_Function) ) {
             ret_value = SERIAL_COMMAND_OK;

             outputs_vector[0]=modbus_reply.data[1];
             outputs_vector[1]=modbus_reply.data[3];
             outputs_vector[2]=modbus_reply.data[5];
         }
         else {
             ret_value = SERIAL_COMMAND_ERROR;
         }
   }

    session->error_code_recent=ret_value;
    return ret_value;
}



int SaveModeCHConfig_Serial_MultipleSensor(MultipleSensor_serial_driver *session, unsigned char mode_is_single_channel_bool, int selected_channel_number) {

         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         int ret_value = SERIAL_COMMAND_ERROR;
         unsigned short CRC;
         Modbus_Data_Struct modbus_reply;
         unsigned char data_byte1, data_byte2;
         QThread *current_thread_ptr;

         int i, serial_device_bytes_available;

         QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

         data_byte1 = mode_is_single_channel_bool&0x01;
         data_byte2 = selected_channel_number;


          QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
          current_thread_ptr = QThread::currentThread();

          OutputPacketBuffer[0]=session->board_ID;
          OutputPacketBuffer[1]=SAVE_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function;     	//0x06 is the write_single_register Modbus command
          OutputPacketBuffer[2]=0x00;
          OutputPacketBuffer[3]=SAVE_OR_QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
          OutputPacketBuffer[4]=data_byte1; //send in byte1 the mode (0: multi-channel; 1:single-channel)
          OutputPacketBuffer[5]=data_byte2;  //send in byte2 the channel number (0 to 5)

          CRC = crc16( OutputPacketBuffer, 6 );
          OutputPacketBuffer[6] = CRC&0x00FF;
          OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
          //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
           if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
               memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
           }

         current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
         */
         if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 10) == -1)
         {
             session->error_code_recent=SERIAL_COMMAND_ERROR;
             CloseDevice_Serial_MultipleSensor(session);
             OpenDevice_Serial_MultipleSensor(session, session->port_name);
             ret_value = SERIAL_COMMAND_ERROR;
         }
         else {

           //Now get the response packet from the firmware.
             for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_SAVE_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
                  current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
                  QCoreApplication::processEvents();  //Process event in the queue
                  serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
              }
              memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);


              modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
             //check the reply to the sent command
            if( modbus_reply.CRC_is_OK==1 && modbus_reply.modbus_cmd==SAVE_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function ) {
                ret_value = SERIAL_COMMAND_OK;
            }
            else {
                ret_value = SERIAL_COMMAND_ERROR;
            }


        }

    session->error_code_recent=ret_value;
    return ret_value;
}


int ReadModeCHConfig_Serial_MultipleSensor(MultipleSensor_serial_driver *session, unsigned char *mode_is_single_channel_bool_ptr, int *selected_channel_number_ptr) {

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

    QByteArray buffer_byte_array;
    int ret_value = SERIAL_COMMAND_ERROR;
    unsigned short CRC;
    Modbus_Data_Struct modbus_reply;
    QThread *current_thread_ptr;

    int i, serial_device_bytes_available;

    QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

    QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
    current_thread_ptr = QThread::currentThread();

    OutputPacketBuffer[0]=session->board_ID;
    OutputPacketBuffer[1]=QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
    OutputPacketBuffer[2]=0x00;
    OutputPacketBuffer[3]=SAVE_OR_QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
    OutputPacketBuffer[4]=0x00;
    OutputPacketBuffer[5]=1;  //request to receive 1 words (1=16bits=2bytes), the size of the channel mod plus selected channel number

    CRC = crc16( OutputPacketBuffer, 6 );
    OutputPacketBuffer[6] = CRC&0x00FF;
    OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
    //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
     if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
         memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
     }

   current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
   */
    if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)
    {
        session->error_code_recent=SERIAL_COMMAND_ERROR;
        CloseDevice_Serial_MultipleSensor(session);
        OpenDevice_Serial_MultipleSensor(session, session->port_name);
        ret_value = SERIAL_COMMAND_ERROR;
    }
    else {

      //Now get the response packet from the firmware.
        for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_READ_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
             current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
             QCoreApplication::processEvents();  //Process event in the queue
             serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
         }
         memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

        modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
        //check the reply to the sent command
       if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function) ) {
           ret_value = SERIAL_COMMAND_OK;
           *mode_is_single_channel_bool_ptr = modbus_reply.data[0];
           *selected_channel_number_ptr = modbus_reply.data[1];
       }
       else {
           ret_value = SERIAL_COMMAND_ERROR;
       }
   }

    session->error_code_recent=ret_value;
    return ret_value;
}

int ReadSensorCounters_Serial_MultipleSensor(MultipleSensor_serial_driver *session, long int *integer_CH_vector) {

    unsigned char *data_ptr;

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    int ret_value;
    unsigned short CRC;
    Modbus_Data_Struct modbus_reply;
    QThread *current_thread_ptr;

    int i, serial_device_bytes_available;

    QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

    QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
    current_thread_ptr = QThread::currentThread();

    OutputPacketBuffer[0]=session->board_ID;
    OutputPacketBuffer[1]=QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
    OutputPacketBuffer[2]=0x00;
    OutputPacketBuffer[3]=QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
    OutputPacketBuffer[4]=0x00;
    OutputPacketBuffer[5]=12;  //request to receive 12 words (1=16bits=2bytes), the size of the 6 counters (of LCR channels)

    CRC = crc16( OutputPacketBuffer, 6 );
    OutputPacketBuffer[6] = CRC&0x00FF;
    OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
    //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
    if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
        memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
    }

    current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
    */
    //Query_Modbus_Port( session, buffer_byte_array, 7, InputPacketBuffer, 100);
    if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)  //always write only the number of bytes filled with data on buffer_byte_array to avoid error on the read operation
    {
        session->error_code_recent=SERIAL_COMMAND_ERROR;
        CloseDevice_Serial_MultipleSensor(session);
        OpenDevice_Serial_MultipleSensor(session, session->port_name);
        ret_value = SERIAL_COMMAND_ERROR;
    }
    else {

        //Now get the response packet from the firmware.
        for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
            current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
            QCoreApplication::processEvents();  //Process event in the queue
            serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
        }
        memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

        modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
        //check the reply to the sent command
        if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function) ) {
            ret_value = SERIAL_COMMAND_OK;
            for(i=0;i<N_LCR_CHANNELS;i++)  {
                data_ptr=(unsigned char *) &integer_CH_vector[i];
                *(data_ptr+0) = modbus_reply.data[(4*i)+1];
                *(data_ptr+1) = modbus_reply.data[(4*i)];
                *(data_ptr+2) = modbus_reply.data[(4*i)+3];
                *(data_ptr+3) = modbus_reply.data[(4*i)+2];
            }
        }
        else {
            ret_value = SERIAL_COMMAND_ERROR;
        }
    }

    session->error_code_recent=ret_value;
    return ret_value;
}


int ResetSensorCounters_Serial_MultipleSensor(MultipleSensor_serial_driver *session) {

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1


    QByteArray buffer_byte_array;
    int ret_value = SERIAL_COMMAND_ERROR;
    unsigned short CRC;
    Modbus_Data_Struct modbus_reply;
    QThread *current_thread_ptr;

    int i, serial_device_bytes_available;

    QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

    QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
    current_thread_ptr = QThread::currentThread();

    OutputPacketBuffer[0]=session->board_ID;
    OutputPacketBuffer[1]=RESET_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function;     	//0x06 is the save_single_register Modbus command
    OutputPacketBuffer[2]=0x00;
    OutputPacketBuffer[3]=RESET_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
    OutputPacketBuffer[4]=0x00;
    OutputPacketBuffer[5]=0x01;  //data sent

    CRC = crc16( OutputPacketBuffer, 6 );
    OutputPacketBuffer[6] = CRC&0x00FF;
    OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
    //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
     if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
         memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
     }

     current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
     */
    if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)
    {
        session->error_code_recent=SERIAL_COMMAND_ERROR;
        CloseDevice_Serial_MultipleSensor(session);
        OpenDevice_Serial_MultipleSensor(session, session->port_name);
        ret_value = SERIAL_COMMAND_ERROR;
    }
    else {

      //Now get the response packet from the firmware.
        for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_RESET_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
             current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
             QCoreApplication::processEvents();  //Process event in the queue
             serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
         }
         memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
         //check the reply to the sent command
        if( modbus_reply.CRC_is_OK==1 && modbus_reply.modbus_cmd==RESET_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function ) {
            ret_value = SERIAL_COMMAND_OK;
        }
        else {
            ret_value = SERIAL_COMMAND_ERROR;
        }

   }

    session->error_code_recent=ret_value;
    return ret_value;
}


int GetBoardIDCode_Serial_MultipleSensor(MultipleSensor_serial_driver *session, unsigned char *board_ID_ptr) {

         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         int ret_value = SERIAL_COMMAND_ERROR;
         unsigned short CRC;
         Modbus_Data_Struct modbus_reply;
         QThread *current_thread_ptr;

         int i, serial_device_bytes_available;

         QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

         QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
         current_thread_ptr = QThread::currentThread();

         OutputPacketBuffer[0]=session->board_ID;
         OutputPacketBuffer[1]=GET_BOARD_ID_CODE_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
         OutputPacketBuffer[2]=0x00;
         OutputPacketBuffer[3]=SET_OR_GET_BOARD_ID_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[4]=0x00;
         OutputPacketBuffer[5]=1;  //request to receive 12 words (1=16bits=2bytes), the size of the 6 LCR channels (6 x sizeof(long))

         CRC = crc16( OutputPacketBuffer, 6 );
         OutputPacketBuffer[6] = CRC&0x00FF;
         OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
         //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
          if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
              memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
          }

          current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
          */
         if( session_serial_device_qt->write( (const char *) OutputPacketBuffer  , 8) == -1)
         {
             session->error_code_recent=SERIAL_COMMAND_ERROR;
             return SERIAL_COMMAND_ERROR;
         }
         else {

           //Now get the response packet from the firmware.
             for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_GET_BOARD_ID_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
                  current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
                  QCoreApplication::processEvents();  //Process event in the queue
                  serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
              }
              memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

              modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

              //check the reply to the sent command
              if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==GET_BOARD_ID_CODE_ModbusRTU_Function) ) {
                ret_value = SERIAL_COMMAND_OK;

                *board_ID_ptr = modbus_reply.data[1];  //save the read board_ID
              }
              else {
                ret_value = SERIAL_COMMAND_ERROR;
              }

        }

    session->error_code_recent=ret_value;
    return ret_value;
}


int SetBoardIDCode_Serial_MultipleSensor(MultipleSensor_serial_driver *session, unsigned char new_board_ID) {

         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         int ret_value = SERIAL_COMMAND_ERROR;
         unsigned short CRC;
         Modbus_Data_Struct modbus_reply;
         QThread *current_thread_ptr;

         int i, serial_device_bytes_available;

         QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

         QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
         current_thread_ptr = QThread::currentThread();

         OutputPacketBuffer[0]=session->board_ID;
         OutputPacketBuffer[1]=SET_BOARD_ID_CODE_ModbusRTU_Function;     	//0x06 is the write_single_register Modbus command
         OutputPacketBuffer[2]=0x00;
         OutputPacketBuffer[3]=SET_OR_GET_BOARD_ID_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[4]=0x00;
         OutputPacketBuffer[5]=new_board_ID;  //request to receive 12 words (1=16bits=2bytes), the size of the 6 LCR channels (6 x sizeof(long))

         CRC = crc16( OutputPacketBuffer, 6 );
         OutputPacketBuffer[6] = CRC&0x00FF;
         OutputPacketBuffer[7] = (CRC>>8)&0x00FF;
/*
         //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
          if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
              memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
          }


         current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
         */
         if( session_serial_device_qt->write( (const char *) OutputPacketBuffer  , 8) == -1)
         {
             session->error_code_recent=SERIAL_COMMAND_ERROR;
             return SERIAL_COMMAND_ERROR;
         }
         else {

           //Now get the response packet from the firmware.
             for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_SET_BOARD_ID_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
                  current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
                  QCoreApplication::processEvents();  //Process event in the queue
                  serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
              }
              memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

              modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
              //check the reply to the sent command
              if( modbus_reply.CRC_is_OK==1 && modbus_reply.modbus_cmd==SET_BOARD_ID_CODE_ModbusRTU_Function ) {
                ret_value = SERIAL_COMMAND_OK;
              }
              else {
                  ret_value = SERIAL_COMMAND_ERROR;
              }

        }

    session->error_code_recent=ret_value;
    return ret_value;
}




int ReadLineCalibTable_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int channel_number, int line_number, float *RAW_value_ptr, float *measurement_ptr) {

         int ret_value;

         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         float RAW_value_local;	//floating point with the value read of RAW_value
         unsigned char *RAW_value_data;	//pointer to the data received by USB
         float measurement_local;	//floating point with the value read of measurement
         unsigned char *measurement_data;	//pointer to the data received by USB

          unsigned short CRC;
          Modbus_Data_Struct modbus_reply;
          QThread *current_thread_ptr;

          int i, serial_device_bytes_available;

          QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

          QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
          current_thread_ptr = QThread::currentThread();

         OutputPacketBuffer[0]=session->board_ID;
         OutputPacketBuffer[1]=QUERY_LINE_CALIB_TABLE_CH_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
         OutputPacketBuffer[2]=0x00;
         OutputPacketBuffer[3]=SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU+channel_number;  //send the Read calibration table line of the selected channel number command of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[4]=((line_number>>8)&0x00FF);   //DATA: request to receive the selected line number
         OutputPacketBuffer[5]=(line_number&0x00FF);  //DATA: request to receive the selected line number
         CRC = crc16( OutputPacketBuffer, 6 );
         OutputPacketBuffer[6] = CRC&0x00FF;
         OutputPacketBuffer[7] = (CRC>>8)&0x00FF;

 /*        //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
          if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
              memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
          }

         current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
         */
         if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)
         {
             session->error_code_recent=SERIAL_COMMAND_ERROR;
             CloseDevice_Serial_MultipleSensor(session);
             OpenDevice_Serial_MultipleSensor(session, session->port_name);
             ret_value = SERIAL_COMMAND_ERROR;
         }

         //Now get the response packet from the firmware.
         for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
              current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
              QCoreApplication::processEvents();  //Process event in the queue
              serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
          }
          memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

              modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);
             //check the reply to the sent command
            if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_LINE_CALIB_TABLE_CH_ModbusRTU_Function) ) {
                ret_value = SERIAL_COMMAND_OK;
            }
            else {
                ret_value = SERIAL_COMMAND_ERROR;
            }

         if(modbus_reply.CRC_is_OK==1) {
            RAW_value_data = (unsigned char *) &RAW_value_local;
            RAW_value_data[0]=modbus_reply.data[1];
            RAW_value_data[1]=modbus_reply.data[0];
            RAW_value_data[2]=modbus_reply.data[3];
            RAW_value_data[3]=modbus_reply.data[2];
            *RAW_value_ptr=RAW_value_local;

            measurement_data = (unsigned char *) &measurement_local;
            measurement_data[0]=modbus_reply.data[4+1];
            measurement_data[1]=modbus_reply.data[4];
            measurement_data[2]=modbus_reply.data[4+3];
            measurement_data[3]=modbus_reply.data[4+2];
            *measurement_ptr=measurement_local;
         }

    session->error_code_recent=ret_value;
    return ret_value;
}


int WriteLineCalibTable_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int channel_number, int line_number, float RAW_value, float measurement) {

        unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
        unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
        unsigned char *RAW_value_data, *measurement_data;

        int ret_value;
        unsigned short CRC;
        Modbus_Data_Struct modbus_reply;
        QThread *current_thread_ptr;

        int i, serial_device_bytes_available;

        QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

        InputPacketBuffer[0] = 0;				//First byte empty

        QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
        current_thread_ptr = QThread::currentThread();

        OutputPacketBuffer[0]=session->board_ID;
        OutputPacketBuffer[1]=SAVE_LINE_CALIB_TABLE_CH_ModbusRTU_Function;     	//0x10 is the write_multiple_registers Modbus command
        OutputPacketBuffer[2]=0x00;
        OutputPacketBuffer[3]=SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU+channel_number;  //send the Write calibration table line command of the selected channel number of Multiple-Sensor Modbus protocol
        OutputPacketBuffer[4]=0;   //number of registers to be written 2 x N_bytes
        OutputPacketBuffer[5]=5;   //number of registers to be written 2 x N_bytes
        OutputPacketBuffer[6]=10;  //number of data bytes of the command ( 2x 7words = 14bytes)

        OutputPacketBuffer[7]=((line_number>>8)&0x00FF);   //DATA: request to receive the selected line number
        OutputPacketBuffer[8]=(line_number&0x00FF);  //DATA: request to receive the selected line number

        RAW_value_data = (unsigned char *) &RAW_value;
        OutputPacketBuffer[9]=RAW_value_data[1];   //send byte #0 of RAW_value
        OutputPacketBuffer[10]=RAW_value_data[0];   //send byte #1 of RAW_value
        OutputPacketBuffer[11]=RAW_value_data[3];   //send byte #2 of RAW_value
        OutputPacketBuffer[12]=RAW_value_data[2];  //send byte #3 of RAW_value

        measurement_data = (unsigned char *) &measurement;
        OutputPacketBuffer[13]=measurement_data[1];   //send byte #0 of measurement
        OutputPacketBuffer[14]=measurement_data[0];   //send byte #1 of measurement
        OutputPacketBuffer[15]=measurement_data[3];   //send byte #2 of measurement
        OutputPacketBuffer[16]=measurement_data[2];   //send byte #3 of measurement

        CRC = crc16( OutputPacketBuffer, 17 );
        OutputPacketBuffer[17] = CRC&0x00FF;
        OutputPacketBuffer[18] = (CRC>>8)&0x00FF;

    /*    //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
         if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
             memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
         }

         current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
         */
        //Query_Modbus_Port( session, buffer_byte_array, 7, InputPacketBuffer, 100);
        if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 19) == -1)  //always write only the number of bytes filled with data on buffer_byte_array to avoid error on the read operation
        {
            session->error_code_recent=SERIAL_COMMAND_ERROR;
            CloseDevice_Serial_MultipleSensor(session);
            OpenDevice_Serial_MultipleSensor(session, session->port_name);
            ret_value = SERIAL_COMMAND_ERROR;
        }

        //Now get the response packet from the firmware.
        for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_SAVE_LINE_CALIB_TABLE_CH_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
             current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
             QCoreApplication::processEvents();  //Process event in the queue
             serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
         }
         memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);

             modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

            //check the reply to the sent command
           if( modbus_reply.CRC_is_OK==1 && modbus_reply.modbus_cmd==SAVE_LINE_CALIB_TABLE_CH_ModbusRTU_Function ) {
               ret_value = SERIAL_COMMAND_OK;
           }
           else {
               ret_value = SERIAL_COMMAND_ERROR;
           }

    session->error_code_recent=ret_value;
    return ret_value;
}


int WriteHeaderCalibTable_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int channel_number, unsigned char calib_table_is_active_bool, unsigned char number_table_lines, unsigned char calibration_mode_multi_or_single_ch, char jumper_selection_osc_tunning_range, char *units_str, char *counter_calib_units_str) {

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

    int ret_value;
    unsigned short CRC;
    Modbus_Data_Struct modbus_reply;
    QThread *current_thread_ptr;

    int i, serial_device_bytes_available;

    QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

    InputPacketBuffer[0] = 0;				//First byte empty

    QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
    current_thread_ptr = QThread::currentThread();

    OutputPacketBuffer[0]=session->board_ID;
    OutputPacketBuffer[1]=SAVE_HEADER_CALIB_TABLE_CH_ModbusRTU_Function;     	//0x10 is the write_multiple_registers Modbus command
    OutputPacketBuffer[2]=0x00;
    OutputPacketBuffer[3]=SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU+channel_number;  //send the Write header of calibration table command of the selected channel number of Multiple-Sensor Modbus protocol
    OutputPacketBuffer[4]=0;   //number of registers to be written 2 x N_bytes
    OutputPacketBuffer[5]=(8+(2*CALIB_UNITS_SIZE_STR))/2;   //number of registers to be written 2 x N_bytes
    OutputPacketBuffer[6]=8+(2*CALIB_UNITS_SIZE_STR);  //number of data bytes of the command

    OutputPacketBuffer[7]=0;    //send "calib_table_is_active_bool"
    OutputPacketBuffer[8]=calib_table_is_active_bool;    //send "calib_table_is_active_bool"

    OutputPacketBuffer[9]=0;   //send "number_table_lines"
    OutputPacketBuffer[10]=(number_table_lines&0x00FF);     //send "number_table_lines"

    OutputPacketBuffer[11]=0;   //send "calibration_mode_multi_or_single_ch"
    OutputPacketBuffer[12]=calibration_mode_multi_or_single_ch;     //send "calibration_mode_multi_or_single_ch"

    OutputPacketBuffer[13]=0;   //send "jumper_selection_osc_tunning_range", the jumper selection of oscillator tuning range , this parameter is the jumper position, it can be A or B ( A, B coded in ASCII)
    OutputPacketBuffer[14]=jumper_selection_osc_tunning_range;     //send "jumper_selection_osc_tunning_range", the jumper selection of oscillator tuning range , this parameter is the jumper position, it can be A or B ( A, B coded in ASCII)

    for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
       OutputPacketBuffer[15+i] = units_str[i];
    }
    for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
       OutputPacketBuffer[15+CALIB_UNITS_SIZE_STR+i] = counter_calib_units_str[i];
    }


    CRC = crc16( OutputPacketBuffer, 15+2*CALIB_UNITS_SIZE_STR );
    OutputPacketBuffer[15+2*CALIB_UNITS_SIZE_STR] = CRC&0x00FF;
    OutputPacketBuffer[16+2*CALIB_UNITS_SIZE_STR] = (CRC>>8)&0x00FF;

/*    //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
     if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
         memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
     }

     current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
     */
    //Query_Modbus_Port( session, buffer_byte_array, 7, InputPacketBuffer, 100);
    if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 17+(2*CALIB_UNITS_SIZE_STR)) == -1)  //always write only the number of bytes filled with data on buffer_byte_array to avoid error on the read operation
    {
        session->error_code_recent=SERIAL_COMMAND_ERROR;
        CloseDevice_Serial_MultipleSensor(session);
        OpenDevice_Serial_MultipleSensor(session, session->port_name);
        ret_value = SERIAL_COMMAND_ERROR;
    }

    //Now get the response packet from the firmware.
    for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_SAVE_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
         current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
         QCoreApplication::processEvents();  //Process event in the queue
         serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
    }
    memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);


     // ----- DEBUG CODE -----
 /*    printf("\nInputPacketBuffer:\n");
     for(i=0; i<MAX_SIZE_SERIAL_RECEIVE_BUFFER; i++) {
         printf("%X ", InputPacketBuffer[i]);   //DEBUG CODE
     }
     // ----------------------

     // ----- DEBUG CODE -----
     printf("\nFlushBuffer:\n");
     for(i=0; i<MAX_SIZE_SERIAL_RECEIVE_BUFFER; i++) {
         printf("%X ", FlushBuffer[i]);   //DEBUG CODE
     }   */
     // ----------------------

     modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

    //check the reply to the sent command
    if( modbus_reply.CRC_is_OK==1 && modbus_reply.modbus_cmd==SAVE_HEADER_CALIB_TABLE_CH_ModbusRTU_Function ) {
        ret_value = SERIAL_COMMAND_OK;
    }
    else {
        ret_value = SERIAL_COMMAND_ERROR;
    }

    session->error_code_recent=ret_value;
    return ret_value;
}


int ReadHeaderCalibTable_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int channel_number, unsigned char *calib_table_is_active_bool_ptr, unsigned char *number_table_lines_ptr, unsigned char *calibration_mode_multi_or_single_ch_ptr, char *jumper_selection_osc_tunning_range_ptr, char *units_str_ptr, char *counter_calib_units_str_ptr) {

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

     int ret_value;
     unsigned short CRC;
     Modbus_Data_Struct modbus_reply;
     QThread *current_thread_ptr;

     int i, serial_device_bytes_available;

     QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

    InputPacketBuffer[0] = 0;				//First byte empty

    QThread::currentThread()->setPriority(QThread::HighPriority);  //increase thread priority to make sure the timings are respected
    current_thread_ptr = QThread::currentThread();

    OutputPacketBuffer[0]=session->board_ID;
    OutputPacketBuffer[1]=QUERY_HEADER_CALIB_TABLE_CH_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
    OutputPacketBuffer[2]=0x00;
    OutputPacketBuffer[3]=SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU+channel_number;  //send the Read Header of calibration table command of the selected channel number of Multiple-Sensor Modbus protocol
    OutputPacketBuffer[4]=0;   //number of registers to be read 2 x N_bytes
    OutputPacketBuffer[5]=(8+(2*CALIB_UNITS_SIZE_STR))/2;   //number of registers to be read 2 x N_bytes

    CRC = crc16( OutputPacketBuffer, 6 );
    OutputPacketBuffer[6] = CRC&0x00FF;
    OutputPacketBuffer[7] = (CRC>>8)&0x00FF;

 /*   //Flush Serial Port receive buffer before starting a new query (send and receive data) to avoid trash being mixed with the reply
     if (session_serial_device_qt->waitForReadyRead(SERIAL_WAIT_TIMEOUT)) {
         memcpy(FlushBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER);
     }

     current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_1);  //wait some mili-seconds
     */
    //Query_Modbus_Port( session, buffer_byte_array, 7, InputPacketBuffer, 100);
    if( session_serial_device_qt->write( (const char *) OutputPacketBuffer , 8) == -1)  //always write only the number of bytes filled with data on buffer_byte_array to avoid error on the read operation
    {
        session->error_code_recent=SERIAL_COMMAND_ERROR;
        CloseDevice_Serial_MultipleSensor(session);
        OpenDevice_Serial_MultipleSensor(session, session->port_name);
        ret_value = SERIAL_COMMAND_ERROR;
    }

    //Now get the response packet from the firmware.
    for(i=0, serial_device_bytes_available=0; serial_device_bytes_available < REPLY_N_BYTES_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU && i<TIMEOUT_CYCLES_ModbusRTU; i++) {
         current_thread_ptr->usleep(WAIT_BETWEEN_SERIAL_COMMANDS_us_2);
         QCoreApplication::processEvents();  //Process event in the queue
         serial_device_bytes_available = session_serial_device_qt->bytesAvailable();
         //printf("serial_device_bytes_available: %d", serial_device_bytes_available);    //DEBUG CODE
     }
     memcpy(InputPacketBuffer, session_serial_device_qt->readAll().data(), MAX_SIZE_SERIAL_RECEIVE_BUFFER-1);


         // ----- DEBUG CODE -----
      /*   printf("\nInputPacketBuffer:\n");
         for(i=0; i<MAX_SIZE_SERIAL_RECEIVE_BUFFER; i++) {
             printf("%X ", InputPacketBuffer[i]);   //DEBUG CODE
         }
         // ----------------------

         // ----- DEBUG CODE -----
         printf("\nFlushBuffer:\n");
         for(i=0; i<MAX_SIZE_SERIAL_RECEIVE_BUFFER; i++) {
             printf("%X ", FlushBuffer[i]);   //DEBUG CODE
         }  */
         // ----------------------

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

        //check the reply to the sent command
       if( modbus_reply.CRC_is_OK==1 && modbus_reply.modbus_cmd==QUERY_HEADER_CALIB_TABLE_CH_ModbusRTU_Function ) {
           ret_value = SERIAL_COMMAND_OK;
       }
       else {
           ret_value = SERIAL_COMMAND_ERROR;
       }

       *calib_table_is_active_bool_ptr = InputPacketBuffer[4];
       *number_table_lines_ptr = InputPacketBuffer[6] & 0x00FF;
       *calibration_mode_multi_or_single_ch_ptr=InputPacketBuffer[8];
       *jumper_selection_osc_tunning_range_ptr = InputPacketBuffer[10];
       for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
          units_str_ptr[i] = InputPacketBuffer[11+i];
       }
       if(channel_number < N_LCR_CHANNELS && counter_calib_units_str_ptr!=NULL) {
           for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
              counter_calib_units_str_ptr[i] = InputPacketBuffer[11+CALIB_UNITS_SIZE_STR+i];
           }
       }

    session->error_code_recent=ret_value;
    return ret_value;
}


int WriteCountersCalib_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int channel_number, float C0, float C1, float C2, float C3) {

    int ret_value;

    ret_value = WriteLineCalibTable_Serial_MultipleSensor(session, channel_number, COUNTER_CALIB_ROW_NUMBER, C0, C1);
    if( ret_value >=0 )
        ret_value = WriteLineCalibTable_Serial_MultipleSensor(session, channel_number, COUNTER_CALIB_ROW_NUMBER+1, C2, C3);

    session->error_code_recent=ret_value;
    return ret_value;
}


int ReadCountersCalib_Serial_MultipleSensor(MultipleSensor_serial_driver *session, int channel_number, float *C0_ptr, float *C1_ptr, float *C2_ptr, float *C3_ptr) {

    int ret_value;

    ret_value = ReadLineCalibTable_Serial_MultipleSensor(session, channel_number, COUNTER_CALIB_ROW_NUMBER, C0_ptr, C1_ptr);
    if( ret_value >=0 )
        ret_value = ReadLineCalibTable_Serial_MultipleSensor(session, channel_number, COUNTER_CALIB_ROW_NUMBER+1, C2_ptr, C3_ptr);

    session->error_code_recent=ret_value;
    return ret_value;
}


void CloseDevice_Serial_MultipleSensor(MultipleSensor_serial_driver *session) {
    QSerialPort *session_serial_device_qt = ((QSerialPort *) session->serial_device);

    if (session->isConnected == TRUE_BOOL) {
        session_serial_device_qt->close();
        session->isConnected = FALSE_BOOL;
    }
}

