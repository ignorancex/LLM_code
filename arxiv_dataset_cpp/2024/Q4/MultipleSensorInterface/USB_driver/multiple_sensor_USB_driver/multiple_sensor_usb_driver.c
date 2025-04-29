/******************************************************************************************************
**                                                                                                    *
**  Copyright (C) 2021 David Nuno Quelhas. LGPL License 3.0 (GNU).                                    *
**  David Nuno Quelhas; Lisboa, Portugal.  david.quelhas@yahoo.com ,                                  *
**  https://multiple-sensor-interface.blogspot.com                                                    *
**                                                                                                    *
**  Multi-sensor interface, low-cost RTU. Modbus USB Device Driver.                                   *
**                                                                                                    *
**  This is the header file for the USB Driver of the Multiple-Sensor Interface, also                 *
**                               referenced/mentioned here by the name 'LCR sensors' .                *
**                                                                                                    *
**  This library is free software; you can redistribute it and/or modify it under the terms of the    *
**  GNU Lesser General Public version 3.0 License as published by the Free Software Foundation.       *
**  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;         *
**  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.         *
**  See the GNU Lesser General Public License for more details.                                       *
**  You should have received a copy of the GNU Lesser General Public License along with this library. *
**  --------------------                                                                              *
**   FILE: multiple_sensor_usb_driver.c - Here are the implementations of the functions of the            *
**                          USB Device Driver of Multiple-Sensor Interface. The USB driver has the    *
**                          same design/coding as the Modbus RS485/serial driver.                     *
******************************************************************************************************/


#include "../HIDAPI/hidapi.h"

#include <wchar.h>
#include <string.h>
#include <stdlib.h>

#include <stdio.h>

#include "multiple_sensor_usb_driver.h"


// ----- LCR sensors ModbusRTU register codes -----
#define SELECT_MEASUREMENT_TIME_CODE_ModbusRTU 0x01
#define QUERY_FREQUENCY_MEASUREMENTS_CODE_ModbusRTU 0x10
//#define SAVE_CALIBRATE_ZERO_LOAD_ModbusRTU 0x83
#define QUERY_ADC_MEASUREMENTS_CODE_ModbusRTU 0x16
#define SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU 0x20	//from 0x20 to 0x2A (0x20 + N_LCR_CHANNELS + N_ADC_CHANNELS)
//#define QUERY_MEASUREMENTS_LOWER_CH_CODE_ModbusRTU 0x86
//#define QUERY_MEASUREMENTS_HIGHER_CH_CODE_ModbusRTU 0x87
#define SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU 0x30	//from 0x30 to 0x3A (0x30 + N_LCR_CHANNELS + N_ADC_CHANNELS)
#define SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU 0x40	//from 0x40 to 0x4A (0x40 + N_LCR_CHANNELS + N_ADC_CHANNELS)
#define QUERY_MEASUREMENTS_CH_CODE_ModbusRTU 0x87
#define SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU 0x91
#define QUERY_OUTPUTS_CURRENT_VALUE_CODE_ModbusRTU 0x95
#define SAVE_OR_QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_CODE_ModbusRTU 0x96
#define QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU 0x50
#define RESET_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU 0x58
#define QUERY_COUNTERS_MEASUREMENTS_CH_CODE_ModbusRTU 0x60
#define SET_OR_GET_BOARD_ID_CODE_ModbusRTU 0x98

#define MULTIPLE_SENSOR_FIRMWARE_ERROR_ModbusRTU 0xFF


//#define SELECT_MEASUREMENT_TIME_ModbusRTU_Function
//#define SAVE_CALIBRATE_ZERO_LOAD_ModbusRTU_Function
#define QUERY_FREQUENCY_MEASUREMENTS_ModbusRTU_Function 0x04
#define QUERY_ADC_MEASUREMENTS_ModbusRTU_Function 0x04
#define SAVE_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function 0x10
#define QUERY_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function 0x03
//#define QUERY_MEASUREMENTS_LOWER_CH_ModbusRTU_Function
//#define QUERY_MEASUREMENTS_HIGHER_CH_ModbusRTU_Function
#define SAVE_LINE_CALIB_TABLE_CH_ModbusRTU_Function 0x10
#define QUERY_LINE_CALIB_TABLE_CH_ModbusRTU_Function 0x03
#define SAVE_HEADER_CALIB_TABLE_CH_ModbusRTU_Function 0x10
#define QUERY_HEADER_CALIB_TABLE_CH_ModbusRTU_Function 0x03
#define QUERY_MEASUREMENTS_CH_ModbusRTU_Function 0x04
#define QUERY_COUNTERS_MEASUREMENTS_CH_ModbusRTU_Function 0x04
#define SAVE_OUTPUT_CONFIG_ModbusRTU_Function 0x10
#define QUERY_OUTPUT_CONFIG_ModbusRTU_Function 0x03
#define QUERY_OUTPUTS_CURRENT_VALUE_ModbusRTU_Function 0x04
#define SAVE_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function 0x06
#define QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function 0x03
#define QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function 0x04
#define RESET_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function 0x06
#define GET_BOARD_ID_CODE_ModbusRTU_Function 0x03
#define SET_BOARD_ID_CODE_ModbusRTU_Function 0x06
// ---------------------------------------



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

typedef struct ret_type1_struct {
    int ret_value;	//return value is an status code (also error code)
    char error_pos;
} ret_type1;


// void RemoveChar(char *str, char garbage) - Function to remove all occurrences of a character 'garbage' from the string 'str'
void RemoveChar(char *str, char garbage) {

    char *src, *dst;
    for (src = dst = str; *src != '\0'; src++) {
        *dst = *src;
        if (*dst != garbage) dst++;
    }
    *dst = '\0';
}


void print_ERROR_MSG_USB_MultipleSensor(int error_code)
 {
    switch(error_code) {
        case USB_COMMAND_ERROR:
            printf("ERROR: USB_COMMAND_ERROR\n");
        break;

        case USB_CONNECTION_ERROR:
            printf("ERROR: USB_CONNECTION_ERROR\n");
        break;

        case SENSOR_CALIB_FAILED:
            printf("ERROR: SENSOR_CALIB_FAILED\n");
        break;

        case SENSOR_READ_FAIL_DUE_MISSING_CALIB:
            printf("ERROR: SENSOR_READ_FAIL_DUE_MISSING_CALIB\n");
        break;

    default:
        printf("ERROR:  error on USB driver");
        break;
    }
}


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





// FUNCTION: Process_Modbus_Reply, Description: Analyze the reply to a ModbusRTU command, save the contents in the appropriate fields of Modbus_Data_Struct, and verify the checksum
//		     Return value: Returns the a data struct with the contents of the Modbus reply
Modbus_Data_Struct Process_Modbus_Reply(unsigned char *InputPacketBuffer_ptr, unsigned char this_device_ID) {

    Modbus_Data_Struct received_modbus_reply;
    unsigned short expected_CRC;
    int i;
    //unsigned char this_device_ID;
    unsigned char byte_1, byte_2, received_CRC_byte1, received_CRC_byte2;


    //try to process all the start of commands inside the buffer in case the CRC is OK then they are valid commands
    //start_of_RTU_command=0;
    //CRC_is_OK=0;


    //the msg_dest_device_ID must be device_ID or 255
    received_modbus_reply.msg_dest_device_ID = (unsigned char) InputPacketBuffer_ptr[0];

    //after the device_ID is the function_code
    received_modbus_reply.modbus_cmd = (unsigned char) InputPacketBuffer_ptr[1];

    if(	received_modbus_reply.modbus_cmd==0x03 || received_modbus_reply.modbus_cmd==0x04 ) {

        //the next byte is the size in byte of the data in the reply
        received_modbus_reply.number_data_bytes =  (unsigned char) InputPacketBuffer_ptr[2];

        for(i=0, received_modbus_reply.data=modbus_data; i<received_modbus_reply.number_data_bytes; i++) {
            received_modbus_reply.data[i] = InputPacketBuffer_ptr[i+3];
        }

        received_CRC_byte1 = InputPacketBuffer_ptr[received_modbus_reply.number_data_bytes+3];
        received_CRC_byte2 = InputPacketBuffer_ptr[received_modbus_reply.number_data_bytes+4];
        received_modbus_reply.CRC = ((received_CRC_byte2<<8)&0xFF00) | (received_CRC_byte1&0x00FF);

        //check if the CRC of the message is OK
        expected_CRC = crc16( InputPacketBuffer_ptr, 3+received_modbus_reply.number_data_bytes);
    }
    else {
        if( received_modbus_reply.modbus_cmd==0x06 || received_modbus_reply.modbus_cmd==0x10 ) {
            //
            //the next 2 bytes are the address of the 1st register to be read (starting address)
            byte_1 = (unsigned char) InputPacketBuffer_ptr[2];
            byte_2 = (unsigned char) InputPacketBuffer_ptr[3];
            received_modbus_reply.reg_starting_address = ( (((int) byte_1) << 8 ) & 0xFF00) | ( byte_2 & 0x00FF ) ;

            //the next 2 bytes are the length of the data to be read (number of bytes to read)
            byte_1 = (unsigned char) InputPacketBuffer_ptr[4];
            byte_2 = (unsigned char) InputPacketBuffer_ptr[5];
            received_modbus_reply.length_or_data = ( (((int) byte_1) << 8 ) & 0xFF00) | ( byte_2 & 0x00FF ) ;

            received_CRC_byte1 = (unsigned char) InputPacketBuffer_ptr[6];
            received_CRC_byte2 = (unsigned char) InputPacketBuffer_ptr[7];
            //check if the CRC of the message is OK
            received_modbus_reply.CRC = (0xFF00&(received_CRC_byte2<<8)) | (0x00FF&(received_CRC_byte1));

            //check if the CRC of the message is OK
            expected_CRC = crc16( InputPacketBuffer_ptr, 6);

            //printf("received_CRC_byte1: %X, received_CRC_byte2: %X\n", received_CRC_byte1, received_CRC_byte2); //DEBUG CODE
            //printf("received_modbus_reply.CRC: %d\n", received_modbus_reply.CRC); //DEBUG CODE
            //printf("expected_CRC: %d\n", expected_CRC);   //DEBUG CODE

        }
    }

    if(received_modbus_reply.CRC==expected_CRC) {
        received_modbus_reply.CRC_is_OK=1;
    }
    else {
        received_modbus_reply.CRC_is_OK=0;
    }


    return received_modbus_reply;
}



int OpenDevice_USB_MultipleSensor(MultipleSensor_USB_driver *session) {
    //in case the USB connection is already opened, close it and reconnect
    if (session->isConnected == TRUE_BOOL) {
        hid_close(session->device);
        session->isConnected = FALSE_BOOL;
    }

     session->board_ID = 255;   //with USB interface always communicate using the broadcast board_ID(255), because USB doesn't supports 2 simultaneous device with the same PID (USB - Product ID)

    session->device = hid_open(MY_DEVICE_VID, MY_DEVICE_PID, NULL);

    if (session->device) {
        session->isConnected = TRUE_BOOL;
        // hid_set_nonblocking(session->device, TRUE_BOOL); //this is for, when working in a pool mode,                                                     //were the application will check periodically for the received data
        return USB_CONNECTED;
    }
    else {
        session->isConnected = FALSE_BOOL;
        session->error_code_recent=USB_CONNECTION_ERROR;
        return USB_CONNECTION_ERROR;
    }
}


int ReadFrequencyMeasurement_USB_MultipleSensor(MultipleSensor_USB_driver *session, long int *integer_CH_vector) {

         int i;
         // int integer_CH_vector[N_LCR_CHANNELS];	//vector were are saved integer values with sensor information (for example the frequency of the sensor signal)
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned char *data_ptr;
         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1] = session->board_ID;
         OutputPacketBuffer[2] = QUERY_FREQUENCY_MEASUREMENTS_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[3] = 0;
         OutputPacketBuffer[4] = QUERY_FREQUENCY_MEASUREMENTS_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5] = 0;
         OutputPacketBuffer[6]=12;  //request to receive 12 words (1=16bits=2bytes), the size of the 6 LCR channels (6 x sizeof(long))

         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //printf("Read_Frequency_Measurement, CALLED.\n");     //DEBUG CODE

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_FREQUENCY_MEASUREMENTS_ModbusRTU_Function) ) {
             ret_value = USB_COMMAND_OK;
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

          for(i=0; i<N_LCR_CHANNELS; i++)  {    //convert the received bytes of the frequency measurements back to 'long int' variables
             data_ptr=(unsigned char *) &integer_CH_vector[i];
             *(data_ptr+0) = modbus_reply.data[(4*i)+1];
             *(data_ptr+1) = modbus_reply.data[(4*i)];
             *(data_ptr+2) = modbus_reply.data[(4*i)+3];
             *(data_ptr+3) = modbus_reply.data[(4*i)+2];
         }

/*   //code of alternative (but worst) way to convert the received bytes back to a 'long int' variable
      //   for(i=0;i<N_LCR_CHANNELS;i++)  {
      //       data_ptr=(unsigned char *) &integer_CH_vector[i];
      //       *(data_ptr+0) = InputPacketBuffer[(4*i)+1];
      //       *(data_ptr+1) = InputPacketBuffer[(4*i)+2];
      //       *(data_ptr+2) = InputPacketBuffer[(4*i)+3];
      //       *(data_ptr+3) = InputPacketBuffer[(4*i)+4];
      //   }

      //   for(i=0;i<N_LCR_CHANNELS;i++)  {
      //       integer_CH_vector[i]=(16777216*InputPacketBuffer[(4*i)+4])+(65536*InputPacketBuffer[(4*i)+3])+(256*InputPacketBuffer[(4*i)+2])+InputPacketBuffer[(4*i)+1];
      //   }
*/

    return ret_value;
}



int SaveSensorConfigs_USB_MultipleSensor(MultipleSensor_USB_driver *session, int configs_for_channel_number, char make_compare_w_calibrated_sensor_value, char compare_is_greater_than, char sensor_prescaler, char sensor_used_for_OUT[3], float sensor_trigger_value) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned char *data_ptr;
         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.


         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=SAVE_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function;     	//0x10 is the save_multiple_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU+configs_for_channel_number;  //send the Query Channel Configurations of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=7;  //request to send 7 words (1word=16bits=2bytes), the size of the configuration of 1 LCR channel
         OutputPacketBuffer[7]=14;  //number of data bytes of the command ( 2x 7words = 14bytes)

         OutputPacketBuffer[8]=0;
         OutputPacketBuffer[9]=make_compare_w_calibrated_sensor_value&0x01;
         OutputPacketBuffer[10]=0;
         OutputPacketBuffer[11]=compare_is_greater_than&0x01;
         OutputPacketBuffer[12]=0;
         OutputPacketBuffer[13]=sensor_used_for_OUT[0]&0x01;
         OutputPacketBuffer[14]=0;
         OutputPacketBuffer[15]=sensor_used_for_OUT[1]&0x01;
         OutputPacketBuffer[16]=0;
         OutputPacketBuffer[17]=sensor_used_for_OUT[2]&0x01;

         data_ptr = (unsigned char *) &sensor_trigger_value;
         OutputPacketBuffer[18]=data_ptr[1];   //send byte #0 of sensor_trigger_value
         OutputPacketBuffer[19]=data_ptr[0];   //send byte #1 of sensor_trigger_value
         OutputPacketBuffer[20]=data_ptr[3];   //send byte #2 of sensor_trigger_value
         OutputPacketBuffer[21]=data_ptr[2];   //send byte #3 of sensor_trigger_value

         CRC = crc16( OutputPacketBuffer+1, 21 );
         OutputPacketBuffer[22] = CRC&0x00FF;
         OutputPacketBuffer[23] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }
         for(i=0; i<100000; i++);
         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==SAVE_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

    return ret_value;
}



int ReadSensorConfigs_USB_MultipleSensor(MultipleSensor_USB_driver *session, int configs_for_channel_number, char *make_compare_w_calibrated_sensor_value_ptr, char *compare_is_greater_than_ptr, char *sensor_prescaler_ptr, char *sensor_used_for_OUT_ptr, float *sensor_trigger_value_ptr) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned char *data_ptr;
         unsigned short CRC;
         float sensor_trigger_value;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=QUERY_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU+configs_for_channel_number;  //send the Query Channel Configurations of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=8;  //request to receive 8 words (1word=16bits=2bytes), the size of the configuration of 1 LCR channel
         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;


         //printf("Read_Sensor_Configs, CALLED.\n");    //DEBUG CODE

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_CONFIGURATIONS_SENSOR_CH_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;

             //data[1]: make_compare_w_calibrated_sensor, data[3]: compare_is_greater_than; these are 0 or 1, if different declare invalid data
             if(modbus_reply.data[1]>1 && modbus_reply.data[3]>1) {
                 session->error_code_recent=USB_REPLY_WITH_INVALID_DATA;
                 ret_value = USB_REPLY_WITH_INVALID_DATA;
             }
             else {

                     //read the sensor channel configurations from the bits of sensors_configs
                     *make_compare_w_calibrated_sensor_value_ptr=modbus_reply.data[1]&0x01;
                     *compare_is_greater_than_ptr=modbus_reply.data[3]&0x01;
                     *sensor_prescaler_ptr=modbus_reply.data[5];
                     sensor_used_for_OUT_ptr[0]=modbus_reply.data[7]&0x01;
                     sensor_used_for_OUT_ptr[1]=modbus_reply.data[9]&0x01;
                     sensor_used_for_OUT_ptr[2]=modbus_reply.data[11]&0x01;

                     // read the trigger value of the sensor for the associated outputs
                     data_ptr = (unsigned char *) &sensor_trigger_value;
                     data_ptr[0]=modbus_reply.data[13];
                     data_ptr[1]=modbus_reply.data[12];
                     data_ptr[2]=modbus_reply.data[15];
                     data_ptr[3]=modbus_reply.data[14];

                     *sensor_trigger_value_ptr=sensor_trigger_value;  //save the read trigger_value on the argument variable
             }

         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }


    return ret_value;
}



int ReadADCValue_USB_MultipleSensor(MultipleSensor_USB_driver *session, float *ADC_values_vector_ptr) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned char *data_ptr;   //pointer to the data received by USB
         unsigned short CRC;
         float ADC_value;	//floating point with the value red by the ADC channel, ADC(analogue to digital converter)
         int ch_number, byte_number;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=QUERY_ADC_MEASUREMENTS_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=QUERY_ADC_MEASUREMENTS_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=8;  //request to receive 8 words (1word=16bits=2bytes), the size of the 4 ADC channels (4 x sizeof(float))
         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_ADC_MEASUREMENTS_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         if(modbus_reply.CRC_is_OK==1) {        //convert the received bytes of the ADC voltage back to 'float' variables
            data_ptr = (unsigned char *) &ADC_value;
            for(ch_number=0,byte_number=0;ch_number<4;ch_number++,byte_number=byte_number+4) {
                data_ptr[0]=modbus_reply.data[byte_number+1];
                data_ptr[1]=modbus_reply.data[byte_number];
                data_ptr[2]=modbus_reply.data[byte_number+3];
                data_ptr[3]=modbus_reply.data[byte_number+2];
                ADC_values_vector_ptr[ch_number]=ADC_value;
            }
        }

    return ret_value;
}


int ReadLineCalibTable_USB_MultipleSensor(MultipleSensor_USB_driver *session, int channel_number, int line_number, float *RAW_value_ptr, float *measurement_ptr) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned char *data_ptr;   //pointer to the data received by USB
         unsigned short CRC;
         float RAW_value_local, measurement_local;	//floating point with the values of the calibration line

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=QUERY_LINE_CALIB_TABLE_CH_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU+channel_number;  //send the Read calibration table line of the selected channel number command of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=((line_number>>8)&0x00FF);   //DATA: request to receive the selected line number
         OutputPacketBuffer[6]=(line_number&0x00FF);  //DATA: request to receive the selected line number
         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_LINE_CALIB_TABLE_CH_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;
             data_ptr = (unsigned char *) &RAW_value_local;
             data_ptr[0]=modbus_reply.data[1];
             data_ptr[1]=modbus_reply.data[0];
             data_ptr[2]=modbus_reply.data[3];
             data_ptr[3]=modbus_reply.data[2];
             *RAW_value_ptr=RAW_value_local;

             data_ptr = (unsigned char *) &measurement_local;
             data_ptr[0]=modbus_reply.data[4+1];
             data_ptr[1]=modbus_reply.data[4];
             data_ptr[2]=modbus_reply.data[4+3];
             data_ptr[3]=modbus_reply.data[4+2];
             *measurement_ptr=measurement_local;
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

    return ret_value;
}


int WriteLineCalibTable_USB_MultipleSensor(MultipleSensor_USB_driver *session, int channel_number, int line_number, float RAW_value, float measurement) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned char *data_ptr;   //pointer to the data received by USB
         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=SAVE_LINE_CALIB_TABLE_CH_ModbusRTU_Function;     	//0x10 is the write_multiple_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU+channel_number;  //send the Write calibration table line command of the selected channel number of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0;   //number of registers to be written N_bytes / 2
         OutputPacketBuffer[6]=5;   //number of registers to be written N_bytes / 2
         OutputPacketBuffer[7]=10;  //number of data bytes of the command ( 2x 7words = 14bytes)

         OutputPacketBuffer[8]=((line_number>>8)&0x00FF);   //DATA: request to receive the selected line number
         OutputPacketBuffer[9]=(line_number&0x00FF);  //DATA: request to receive the selected line number

         data_ptr = (unsigned char *) &RAW_value;
         OutputPacketBuffer[10]=data_ptr[1];   //send byte #0 of RAW_value
         OutputPacketBuffer[11]=data_ptr[0];   //send byte #1 of RAW_value
         OutputPacketBuffer[12]=data_ptr[3];   //send byte #2 of RAW_value
         OutputPacketBuffer[13]=data_ptr[2];  //send byte #3 of RAW_value

         data_ptr = (unsigned char *) &measurement;
         OutputPacketBuffer[14]=data_ptr[1];   //send byte #0 of measurement
         OutputPacketBuffer[15]=data_ptr[0];   //send byte #1 of measurement
         OutputPacketBuffer[16]=data_ptr[3];   //send byte #2 of measurement
         OutputPacketBuffer[17]=data_ptr[2];   //send byte #3 of measurement

         CRC = crc16( OutputPacketBuffer+1, 17 );
         OutputPacketBuffer[18] = CRC&0x00FF;
         OutputPacketBuffer[19] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==SAVE_LINE_CALIB_TABLE_CH_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

    return ret_value;
}




int ReadHeaderCalibTable_USB_MultipleSensor(MultipleSensor_USB_driver *session, int channel_number, unsigned char *calib_table_is_active_bool_ptr, unsigned char *number_table_lines_ptr, unsigned char *calibration_mode_multi_or_single_ch_ptr, char *jumper_selection_osc_tunning_range_ptr, char *units_str_ptr, char *counter_calib_units_str_ptr) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=QUERY_HEADER_CALIB_TABLE_CH_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU+channel_number;  //send the Read Header of calibration table command of the selected channel number of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0;   //number of registers to be read N_bytes / 2
         OutputPacketBuffer[6]=(8+(2*CALIB_UNITS_SIZE_STR))/2;   //number of registers to be read N_bytes / 2

         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_HEADER_CALIB_TABLE_CH_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;
             *calib_table_is_active_bool_ptr = InputPacketBuffer[4];
             *number_table_lines_ptr = (InputPacketBuffer[6]&0x00FF);   //InputPacketBuffer[6] is filled with '0' (zero)
             *calibration_mode_multi_or_single_ch_ptr = InputPacketBuffer[8];
             *jumper_selection_osc_tunning_range_ptr = InputPacketBuffer[10];

             for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
                 units_str_ptr[i]=InputPacketBuffer[11+i];
             }
             if(channel_number < N_LCR_CHANNELS && counter_calib_units_str_ptr!=NULL) {
                 for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
                     counter_calib_units_str_ptr[i]=InputPacketBuffer[11+CALIB_UNITS_SIZE_STR+i];
                 }
             }

         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
         }

    return ret_value;
}


int WriteHeaderCalibTable_USB_MultipleSensor(MultipleSensor_USB_driver *session, int channel_number, unsigned char calib_table_is_active_bool, unsigned char number_table_lines, unsigned char calibration_mode_multi_or_single_ch, char jumper_selection_osc_tunning_range, char *units_str, char *counter_calib_units_str) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=SAVE_HEADER_CALIB_TABLE_CH_ModbusRTU_Function;     	//0x10 is the write_multiple_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU+channel_number;  //send the Write header of calibration table command of the selected channel number of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0;   //number of registers to be written N_bytes / 2
         OutputPacketBuffer[6]=(8+(2*CALIB_UNITS_SIZE_STR))/2;   //number of registers to be written N_bytes / 2
         OutputPacketBuffer[7]=8+(2*CALIB_UNITS_SIZE_STR);  //number of data bytes of the command

         OutputPacketBuffer[8]=0;    //send "calib_table_is_active_bool"
         OutputPacketBuffer[9]=calib_table_is_active_bool;    //send "calib_table_is_active_bool"

         OutputPacketBuffer[10]=0;   //send "number_table_lines"
         OutputPacketBuffer[11]=number_table_lines;     //send "number_table_lines"

         OutputPacketBuffer[12]=0;   //send "calibration_mode_multi_or_single_ch"
         OutputPacketBuffer[13]=calibration_mode_multi_or_single_ch;     //send "calibration_mode_multi_or_single_ch"

         OutputPacketBuffer[14]=0;   //send "jumper_selection_osc_tunning_range", the jumper selection of oscillator tuning range , this parameter is the jumper position, it can be A or B ( A, B coded in ASCII)
         OutputPacketBuffer[15]=jumper_selection_osc_tunning_range;     //send "jumper_selection_osc_tunning_range", the jumper selection of oscillator tuning range , this parameter is the jumper position, it can be A or B ( A, B coded in ASCII)

         for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
            OutputPacketBuffer[16+i] = units_str[i];
         }
         for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
            OutputPacketBuffer[16+CALIB_UNITS_SIZE_STR+i] = counter_calib_units_str[i];
         }

         CRC = crc16( OutputPacketBuffer+1, 15+(2*CALIB_UNITS_SIZE_STR) );
         OutputPacketBuffer[16+(2*CALIB_UNITS_SIZE_STR)] = CRC&0x00FF;
         OutputPacketBuffer[17+(2*CALIB_UNITS_SIZE_STR)] = (CRC>>8)&0x00FF;


         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==SAVE_HEADER_CALIB_TABLE_CH_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;

         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
         }

    return ret_value;
}


int ReadMeasurements_USB_MultipleSensor(MultipleSensor_USB_driver *session, float *measurements_vector_ptr) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned char *data_ptr;   //pointer to the data received by USB
         unsigned short CRC;
         float measurement_value;	//floating point with the read measurements for sensor channels.
         int ch_number, byte_number;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=QUERY_MEASUREMENTS_CH_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=QUERY_MEASUREMENTS_CH_CODE_ModbusRTU;  //send the Query Measurements of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=20;  //request to receive 20 words (1word=16bits=2bytes), the size of the LCR+ADC channels ((6+4) x sizeof(float))
         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;


         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_MEASUREMENTS_CH_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;

             data_ptr = (unsigned char *) &measurement_value;
             for(ch_number=0,byte_number=0;ch_number < (N_LCR_CHANNELS+N_ADC_CHANNELS);ch_number++,byte_number=byte_number+4) {
                data_ptr[0]=modbus_reply.data[byte_number+1];
                data_ptr[1]=modbus_reply.data[byte_number];
                data_ptr[2]=modbus_reply.data[byte_number+3];
                data_ptr[3]=modbus_reply.data[byte_number+2];
                measurements_vector_ptr[ch_number]=measurement_value;
             }
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
         }

    return ret_value;
}



int ReadOutputs_USB_MultipleSensor(MultipleSensor_USB_driver *session, unsigned char *outputs_vector) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=QUERY_OUTPUTS_CURRENT_VALUE_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=QUERY_OUTPUTS_CURRENT_VALUE_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=BOOL_FUNC_N_BYTES/2;  //request to read 27 (BOOL_FUNC_N_BYTES/2) words (1word=16bits=2bytes)

         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_OUTPUTS_CURRENT_VALUE_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;
             outputs_vector[0]=modbus_reply.data[1];
             outputs_vector[1]=modbus_reply.data[3];
             outputs_vector[2]=modbus_reply.data[5];

         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
         }

    return ret_value;
}



int WriteOutputConfig_USB_MultipleSensor(MultipleSensor_USB_driver *session, unsigned char output_number, char *out_bool_func_str) {

    unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
    unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1


     int ret_value = USB_COMMAND_ERROR;
     unsigned short CRC;
     Modbus_Data_Struct modbus_reply;

     ret_type1 ret_value_check_bool_func;

     int i;

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
             InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
             OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

             OutputPacketBuffer[1]=session->board_ID;
             OutputPacketBuffer[2]=SAVE_OUTPUT_CONFIG_ModbusRTU_Function;     	//SAVE_OUTPUT_CONFIG_ModbusRTU_Function is 0x10 Modbus command
             OutputPacketBuffer[3]=0x00;
             OutputPacketBuffer[4]=SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU+output_number;  //send the save outputs active level config of Multiple-Sensor Modbus protocol
             OutputPacketBuffer[5]=0x00;
             OutputPacketBuffer[6]=BOOL_FUNC_N_BYTES/2;  //request to save 27 (BOOL_FUNC_N_BYTES/2) words (1word=16bits=2bytes)
             OutputPacketBuffer[7]=BOOL_FUNC_N_BYTES;   //number of bytes to save

             for(i=0; i < BOOL_FUNC_N_BYTES; i++) {
                OutputPacketBuffer[8+i]=out_bool_func_str[i];
             }

             CRC = crc16( OutputPacketBuffer+1, 7+BOOL_FUNC_N_BYTES );
             OutputPacketBuffer[8+BOOL_FUNC_N_BYTES] = CRC&0x00FF;
             OutputPacketBuffer[9+BOOL_FUNC_N_BYTES] = (CRC>>8)&0x00FF;

             /*
             // ----- DEBUG CODE -----
             printf("\nOutputPacketBuffer:\n");
             for(i=0; i<MAX_SIZE_SERIAL_RECEIVE_BUFFER; i++) {
                 printf("%X ", OutputPacketBuffer[i]);   //DEBUG CODE
             }
             // ----------------------  */

             //flush the USB buffer to remove trash packets
            for(i=0; i<10; i++) {
               hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
            }

             if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
             {
                printf("ERROR: hid_write error @ WriteOutputConfig; LCR_sensor_USB_driver\n");
                CloseDevice_USB_MultipleSensor(session);
                session->error_code_recent=USB_CONNECTION_ERROR;
                return USB_CONNECTION_ERROR;
             }

              //Now get the response packet from the firmware.
              if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
              {
                  CloseDevice_USB_MultipleSensor(session);
                  session->error_code_recent=USB_CONNECTION_ERROR;
                  return USB_CONNECTION_ERROR;
              }

              modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

               //check the reply to the sent command
              if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x10 || modbus_reply.modbus_cmd==SAVE_OUTPUT_CONFIG_ModbusRTU_Function ) ) {
                  ret_value = USB_COMMAND_OK;

              }
              else {
                  ret_value = USB_COMMAND_ERROR;
                  session->error_code_recent=USB_CONNECTION_ERROR;
              }

        }

     }

    session->error_code_recent=ret_value;
    return ret_value;
}



int ReadOutputConfig_USB_MultipleSensor(MultipleSensor_USB_driver *session, unsigned char output_number, char *out_bool_func_str) {

     unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
     unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

    int ret_value = USB_COMMAND_ERROR;
    unsigned short CRC;
    Modbus_Data_Struct modbus_reply;

    int i;

    if( output_number >=0 && output_number < N_DIGITAL_OUT) {

        InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
        OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

        OutputPacketBuffer[1]=session->board_ID;
        OutputPacketBuffer[2]=QUERY_OUTPUT_CONFIG_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
        OutputPacketBuffer[3]=0x00;
        OutputPacketBuffer[4]=SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU+output_number;  //send the Query Outputs_Config of Multiple-Sensor Modbus protocol
        OutputPacketBuffer[5]=0x00;
        OutputPacketBuffer[6]=3;  //request to receive 3 words (1word=16bits=2bytes), 1 word for each output active level config

        CRC = crc16( OutputPacketBuffer+1, 6 );
        OutputPacketBuffer[7] = CRC&0x00FF;
        OutputPacketBuffer[8] = (CRC>>8)&0x00FF;


        //flush the USB buffer to remove trash packets
        for(i=0; i<10; i++) {
           hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
        }

        if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
        {
            CloseDevice_USB_MultipleSensor(session);
            session->error_code_recent=USB_CONNECTION_ERROR;
            return USB_CONNECTION_ERROR;
        }

        //Now get the response packet from the firmware.
        if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
        {
            CloseDevice_USB_MultipleSensor(session);
            session->error_code_recent=USB_CONNECTION_ERROR;
            return USB_CONNECTION_ERROR;
        }

        modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

         //check the reply to the sent command
        if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_OUTPUT_CONFIG_ModbusRTU_Function ) ) {
            ret_value = USB_COMMAND_OK;
            strncpy(out_bool_func_str, (char *) modbus_reply.data, BOOL_FUNC_N_BYTES);
        }
        else {
            ret_value = USB_COMMAND_ERROR;
            session->error_code_recent=USB_CONNECTION_ERROR;
        }


    }

    session->error_code_recent=ret_value;
    return ret_value;
}



int SaveModeCHConfig_USB_MultipleSensor(MultipleSensor_USB_driver *session, unsigned char mode_is_single_channel_bool, int selected_channel_number) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         unsigned char data_byte1, data_byte2;

         data_byte1 = mode_is_single_channel_bool&0x01;
         data_byte2 = selected_channel_number;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=SAVE_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function;     	//0x06 is the write_single_register Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SAVE_OR_QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=data_byte1; //send in byte1 the mode (0: multi-channel; 1:single-channel)
         OutputPacketBuffer[6]=data_byte2;  //send in byte2 the channel number (0 to 5)

         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;


         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==SAVE_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;

         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
         }

    return ret_value;
}


int ReadModeCHConfig_USB_MultipleSensor(MultipleSensor_USB_driver *session, unsigned char *mode_is_single_channel_bool_ptr, int *selected_channel_number_ptr) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SAVE_OR_QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=1;  //request to receive 1 words (1=16bits=2bytes), the size of the channel mod plus selected channel number

         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;
             *mode_is_single_channel_bool_ptr = modbus_reply.data[0];
             *selected_channel_number_ptr = modbus_reply.data[1];
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
         }

    return ret_value;
}


int ReadSensorCounters_USB_MultipleSensor(MultipleSensor_USB_driver *session, long int *integer_CH_vector) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned char *data_ptr;
         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=12;  //request to receive 12 words (1=16bits=2bytes), the size of the 6 counters (of LCR channels)

         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function) ) {
             ret_value = USB_COMMAND_OK;
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         for(i=0;i<N_LCR_CHANNELS;i++)  {       //convert the received bytes of the counters values back to 'long int' variables
           data_ptr=(unsigned char *) &integer_CH_vector[i];
           *(data_ptr+0) = modbus_reply.data[(4*i)+1];
           *(data_ptr+1) = modbus_reply.data[(4*i)];
           *(data_ptr+2) = modbus_reply.data[(4*i)+3];
           *(data_ptr+3) = modbus_reply.data[(4*i)+2];
        }

    return ret_value;
}


int ResetSensorCounters_USB_MultipleSensor(MultipleSensor_USB_driver *session) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=RESET_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function;     	//0x06 is the save_single_register Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=RESET_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=0x01;  //data sent

         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==RESET_MULTIPLE_SENSOR_COUNTER_VALUES_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;

         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
         }

    return ret_value;
}


int GetBoardIDCode_USB_MultipleSensor(MultipleSensor_USB_driver *session, unsigned char *board_ID_ptr) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=GET_BOARD_ID_CODE_ModbusRTU_Function;     	//0x03 is the read_holding_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SET_OR_GET_BOARD_ID_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=1;  //request to receive 12 words (1=16bits=2bytes), the size of the 6 LCR channels (6 x sizeof(long))

         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==GET_BOARD_ID_CODE_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;
             *board_ID_ptr = modbus_reply.data[1];  //save the read board_ID
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
         }

    return ret_value;
}


int SetBoardIDCode_USB_MultipleSensor(MultipleSensor_USB_driver *session, unsigned char new_board_ID) {

         int i;
         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1

         unsigned short CRC;

         int ret_value;

         Modbus_Data_Struct modbus_reply;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=SET_BOARD_ID_CODE_ModbusRTU_Function;     	//0x06 is the write_single_register Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=SET_OR_GET_BOARD_ID_CODE_ModbusRTU;  //send the Query Frequency Measurement of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=new_board_ID;  //request to receive 12 words (1=16bits=2bytes), the size of the 6 LCR channels (6 x sizeof(long))

         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
         for(i=0; i<10; i++) {
            hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
         }

         if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

          //check the reply to the sent command
         if( modbus_reply.CRC_is_OK==1 && ( modbus_reply.modbus_cmd==SET_BOARD_ID_CODE_ModbusRTU_Function ) ) {
             ret_value = USB_COMMAND_OK;
         }
         else {
             ret_value = USB_COMMAND_ERROR;
             session->error_code_recent=USB_CONNECTION_ERROR;
         }

    return ret_value;
}


int ReadCountersCalib_USB_MultipleSensor(MultipleSensor_USB_driver *session, int channel_number, float *C0_ptr, float *C1_ptr, float *C2_ptr, float *C3_ptr) {

    int ret_value;

    ret_value = ReadLineCalibTable_USB_MultipleSensor(session, channel_number, COUNTER_CALIB_ROW_NUMBER, C0_ptr, C1_ptr);
    if( ret_value >=0 )
        ret_value = ReadLineCalibTable_USB_MultipleSensor(session, channel_number, COUNTER_CALIB_ROW_NUMBER+1, C2_ptr, C3_ptr);

    return ret_value;

}




int WriteCountersCalib_USB_MultipleSensor(MultipleSensor_USB_driver *session, int channel_number, float C0, float C1, float C2, float C3) {

    int ret_value;

    ret_value = WriteLineCalibTable_USB_MultipleSensor(session, channel_number, COUNTER_CALIB_ROW_NUMBER, C0, C1);
    if( ret_value >=0 )
        ret_value = WriteLineCalibTable_USB_MultipleSensor(session, channel_number, COUNTER_CALIB_ROW_NUMBER+1, C2, C3);

return ret_value;
}


int ReadCountersMeasurements_USB_MultipleSensor(MultipleSensor_USB_driver *session, float *counters_measurements_values_vector_ptr) {

         int ch_number, byte_number, ret_value;

         unsigned char OutputPacketBuffer[MAX_SIZE_SERIAL_SEND_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         unsigned char InputPacketBuffer[MAX_SIZE_SERIAL_RECEIVE_BUFFER];	//Allocate a memory buffer equal to our endpoint size + 1
         float counter_measurement_value;	//floating point with the value read of the calibrated measurement, LCR and ADC(analogue to digital converter)
         unsigned char *counter_measurement_value_data;	//pointer to the data received by USB
         // float measurement_values_vector[N_LCR_CHANNELS+N_ADC_CHANNELS];	//vector used to save the measurements values(units: defined at calibration table) of the LCR+ADC channels

          unsigned short CRC;

          Modbus_Data_Struct modbus_reply;

          int i;

         InputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0.
         OutputPacketBuffer[0] = 0;				//The first byte is the "Report ID" and does not get transmitted over the USB bus.  Always set = 0

         OutputPacketBuffer[1]=session->board_ID;
         OutputPacketBuffer[2]=QUERY_COUNTERS_MEASUREMENTS_CH_ModbusRTU_Function;     	//0x04 is the read_input_registers Modbus command
         OutputPacketBuffer[3]=0x00;
         OutputPacketBuffer[4]=QUERY_COUNTERS_MEASUREMENTS_CH_CODE_ModbusRTU;  //send the Query Counters Measurements of Multiple-Sensor Modbus protocol
         OutputPacketBuffer[5]=0x00;
         OutputPacketBuffer[6]=12;  //request to receive 12 words (1word=16bits=2bytes), the size of the LCR channels (6 x sizeof(float))
         CRC = crc16( OutputPacketBuffer+1, 6 );
         OutputPacketBuffer[7] = CRC&0x00FF;
         OutputPacketBuffer[8] = (CRC>>8)&0x00FF;

         //flush the USB buffer to remove trash packets
          for(i=0; i<10; i++) {
             hid_read_timeout(session->device, InputPacketBuffer, sizeof(InputPacketBuffer), 2);
          }

          if(hid_write(session->device, OutputPacketBuffer, sizeof(OutputPacketBuffer)) == -1)
          {
              CloseDevice_USB_MultipleSensor(session);
              session->error_code_recent=USB_CONNECTION_ERROR;
              return USB_CONNECTION_ERROR;
          }

         //Now get the response packet from the firmware.
         if(hid_read(session->device, InputPacketBuffer, sizeof(InputPacketBuffer)) == -1)
         {
             CloseDevice_USB_MultipleSensor(session);
             session->error_code_recent=USB_CONNECTION_ERROR;
             return USB_CONNECTION_ERROR;
         }

         modbus_reply = Process_Modbus_Reply(InputPacketBuffer, session->board_ID);

             //check the reply to the sent command
            if( modbus_reply.CRC_is_OK==1 && (modbus_reply.modbus_cmd==0x03 || modbus_reply.modbus_cmd==0x04 || modbus_reply.modbus_cmd==QUERY_MEASUREMENTS_CH_ModbusRTU_Function) ) {
                ret_value = USB_COMMAND_OK;
            }
            else {
                ret_value = USB_COMMAND_ERROR;
            }

         if(modbus_reply.CRC_is_OK==1 && ret_value >= 0) {
            counter_measurement_value_data = (unsigned char *) &counter_measurement_value;
            for(ch_number=0,byte_number=0;ch_number < (N_LCR_CHANNELS+N_ADC_CHANNELS);ch_number++,byte_number=byte_number+4) {
                counter_measurement_value_data[0]=modbus_reply.data[byte_number+1];
                counter_measurement_value_data[1]=modbus_reply.data[byte_number];
                counter_measurement_value_data[2]=modbus_reply.data[byte_number+3];
                counter_measurement_value_data[3]=modbus_reply.data[byte_number+2];
                counters_measurements_values_vector_ptr[ch_number]=counter_measurement_value;
            }
         }

    return ret_value;
}


void CloseDevice_USB_MultipleSensor(MultipleSensor_USB_driver *session) {
    if (session->isConnected == TRUE_BOOL) {
        hid_close(session->device);
        session->isConnected = FALSE_BOOL;
    }
}
