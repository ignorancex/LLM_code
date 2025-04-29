/*********************************************************************************
 *   Multiple Sensor Interface - Connect to device, measure sensors,             *
 *   read and write configurations and calibration tables                        *
 *                                                                               *
 *   Copyright (C) 2021 by David Nuno Quelhas; Lisboa, Portugal;                 *
 *    david.quelhas@yahoo.com ,  https://multiple-sensor-interface.blogspot.com  *
 *    In accordance with 7-b) and 7-c) of GNU-GPLv3, this license requires       *
 *     to preserve/include prior copyright notices with author names, and also   *
 *     add new additional copyright notices, specifically on start/top of source *
 *     code files and on 'About'(or similar) dialog/windows at 'user interfaces',*
 *     on the software and on any 'modified version' or 'derivative work'.       *
 *                                                                               *
 *   This program is free software; you can redistribute it and/or modify it     *
 *   under the terms of the GNU General Public License version 3 as published by *
 *   the Free Software Foundation.                                               *
 *                                                                               *
 *   This program is distributed in the hope that it will be useful, but         *
 *   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY  *
 *   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License    *
 *   for more details.                                                           *
 *                                                                               *
 *   You should have received a copy of the GNU General Public License along     *
 *   with this program; if not, write to the                                     *
 *   Free Software Foundation, Inc.,                                             *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.                   *
 *  --------------------                                                         *
 *   FILE: worker.cpp - Here are implementations of the classes and functions    *
 *              related to data transfer(upload/download) of calibration data    *
 *              between a PC and Multiple-Sensor Interface. These instruction are*
 *              executed as a separated thread to avoid locking the program GUI. *
 ********************************************************************************/

#include <QStringList>
#include <QTime>
#include <QCoreApplication>
#include <QThread>
 
#include "worker.h"

#include "multiple_sensor_modbus/multiple_sensor_modbus_driver.h"
#include "multiple_sensor_USB/multiple_sensor_usb_driver.h"

#define MAX_N_ERROR_RETRIES 3
 
Worker::Worker(QObject *parent) :
    QObject(parent)
{
    m_pause = false;
    m_stop = false; 
    m_pause_cond = 0;   
}
 
void Worker::Cancel()
{
    //emit progress_text_changed_signal(tr("@ Worker::Cancel()")); //DEBUG CODE
    m_sync.lock();
    m_stop = true;
	m_sync.unlock();
}
 
void Worker::Pause()
{
    //emit progress_text_changed_signal(tr("@ Worker::Pause()")); //DEBUG CODE
    m_sync.lock();
    m_pause = !m_pause;
    m_sync.unlock();
    //qDebug() << "Pause toggled to " << (m_pause ? "true" : "false" )<< " from thread : " << QThread::currentThreadId();   //DEBUG CODE
}
 
void Worker::Set_Pause_Condition(QWaitCondition &pause_condition)
{
    if(&pause_condition != 0){
        m_pause_cond = &pause_condition;
        //qDebug() << "Pause condition set from thread : " << QThread::currentThreadId();   //DEBUG CODE
    }
}

const char RET_MSG_IN_CASE_SUCCESS_FOR_READ_TABLE_HEADER[200] = "OK: the calibration tables headers were received from the device.\n";
const char RET_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_RS485[200] = "ERROR: an error occurred when downloading the calibration tables headers from the device by serial (RS485) connection.\n";
const char LOG_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_RS485[200] = "ERROR: an error occurred when downloading the headers of calibration tables from the device by serial (RS485) connection.\n";
const char RET_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_USB[200] = "ERROR: an error occurred when downloading the calibration tables headers from the device by USB connection.\n";
const char LOG_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_USB[200] = "ERROR: an error occurred when downloading the headers of calibration tables from the device by USB connection.\n";

const char RET_MSG_IN_CASE_SUCCESS_FOR_WRITE_TABLE_HEADER[200] = "OK: the calibration tables headers were sent to the device.\n";
const char RET_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_HEADER_RS485[200] = "ERROR: an error occurred when uploading the calibration tables header to the device by serial (RS485) connection.\n";
const char LOG_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_HEADER_RS485[200] = "ERROR: an error occurred when uploading the calibration tables header to the device by serial (RS485) connection.\n";
const char RET_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_HEADER_USB[200] = "ERROR: an error occurred when uploading the calibration tables header to the device by USB connection.\n";
const char LOG_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_HEADER_USB[200] = "ERROR: an error occurred when uploading the calibration tables header to the device by USB connection.\n";

const char RET_MSG_IN_CASE_SUCCESS_FOR_READ_TABLE_ROW[200] = "OK: the calibration tables were received from the device.\n";
const char RET_MSG_IN_CASE_ERROR_FOR_READ_TABLE_ROW_RS485[200] = "ERROR: an error occurred when downloading the calibration tables from the device by serial (RS485) connection.\n";
const char LOG_MSG_IN_CASE_ERROR_FOR_READ_TABLE_ROW_RS485[200] = "ERROR: an error occurred when downloading the calibration tables from the device by serial (RS485) connection.\n";
const char RET_MSG_IN_CASE_ERROR_FOR_READ_TABLE_ROW_USB[200] = "ERROR: an error occurred when downloading the calibration tables from the device by USB connection.\n";
const char LOG_MSG_IN_CASE_ERROR_FOR_READ_TABLE_ROW_USB[200] = "ERROR: an error occurred when downloading the calibration tables from the device by USB connection.\n";

const char RET_MSG_IN_CASE_SUCCESS_FOR_WRITE_TABLE_ROW[200] = "OK: the calibration tables were saved on the device.\n";
const char RET_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_ROW_RS485[200] = "ERROR: an error occurred when uploading the calibration tables to the device by serial (RS485) connection.\n";
const char LOG_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_ROW_RS485[200] = "ERROR: an error occurred when uploading the calibration tables to the device by serial (RS485) connection.\n";
const char RET_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_ROW_USB[200] = "ERROR: an error occurred when uploading the calibration tables to the device by USB connection.\n";
const char LOG_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_ROW_USB[200] = "ERROR: an error occurred when uploading the calibration tables to the device by USB connection.\n";


void Worker::Process_Return_Value(int return_value_of_operation, char is_operation_on_header_bool, const char *ret_message_str_in_case_of_success, const char *ret_message_str_in_case_of_error, const char *log_msg_str_ptr_in_case_of_error, int *current_value_progress_bar_ptr, Worker_Arguments *item_args_ptr , Data_Struct *ret_data_ptr, int *error_retry_counter_ptr, int *channel_number_ptr, int *row_number_ptr, float RAW_value, float measurement) {
    if( return_value_of_operation < 0) {
        //report that an error occurred and save it to the log file, show a message box indicating error
        sprintf(item_args_ptr->log_msg_str_ptr, log_msg_str_ptr_in_case_of_error);
        sprintf(ret_data_ptr->ret_message_str, ret_message_str_in_case_of_error);
        ret_data_ptr->ret_value = THREAD_WORKER_RET_VAL_FAIL;
        (*error_retry_counter_ptr)++;
        if(*error_retry_counter_ptr > MAX_N_ERROR_RETRIES) {
            (ret_data_ptr->error_counter)++;
            *error_retry_counter_ptr=0;
            emit status_text_changed_signal(tr("STATUS: ERROR, (error_count:%1)").arg(ret_data_ptr->error_counter));
            //save the error messages in case of error
            if(ret_data_ptr->error_counter==1) {
                strncpy(ret_data_ptr->first_error_message, ret_data_ptr->ret_message_str, MAX_N_CHARS_ERROR_MSG);
            }
            strncpy(ret_data_ptr->last_error_message, ret_data_ptr->ret_message_str, MAX_N_CHARS_ERROR_MSG);
        }
        else {
            *error_retry_counter_ptr++;
            if( is_operation_on_header_bool == 1 ) {
                (*channel_number_ptr--);    //in case of reading header, instruct to retry read on header by changing 'channel_number'
            }
            else {
                (*row_number_ptr)--;    //in case of reading row, instruct to retry read on row by changing 'row_number'
            }
        }
    }
    else {
        // OK sent calibration tables, show a message box indicating success
        sprintf(ret_data_ptr->ret_message_str, ret_message_str_in_case_of_success);
        ret_data_ptr->ret_value = THREAD_WORKER_RET_VAL_OK;
        *error_retry_counter_ptr=0;
        (*current_value_progress_bar_ptr)++;
        emit progress_value_changed_signal(*current_value_progress_bar_ptr);    //update the dialog with the progress bar
        if( is_operation_on_header_bool == 1 ) {
        emit progress_text_changed_signal(tr("Processing...\n \"CH.%1,HEADER\"").arg(*channel_number_ptr));
        }
        else {
            emit progress_text_changed_signal(tr("Processing...\n CH.#,row.#:(RAW_value; measurement):\n \"CH.%1,row.%2:(%3;%4)\"").arg(*channel_number_ptr).arg(*row_number_ptr).arg(RAW_value).arg(measurement));
        }
        emit status_text_changed_signal(tr("STATUS: OK, (error_count:%1)").arg(ret_data_ptr->error_counter));
    }

    //qDebug() << "DEBUG_POS @ Process_Return_Value(...), error_counter=" << ret_data_ptr->error_counter;   // DEBUG CODE

}
 
void Worker::Start_Work(const QList<Worker_Arguments> &list)
{
    int N_steps, plus_N_steps;
    int channel_number;
    Data ret_data;
    float RAW_value, measurement;
    int row_number, filled_row_count;
    int current_value_progress_bar;
    int error_retry_counter, last_error_counter;
    int operation_ret_value;

    int connect_status;

    //int k;  //DEBUG CODE

    ret_data.error_counter = 0;
    last_error_counter = 0;
    error_retry_counter = 0;

     //qDebug() << "DEBUG_POS_A, error_counter=" << ret_data.error_counter;   // DEBUG CODE

    m_sync.lock();
    m_pause = false;
    m_stop = false;    
    m_sync.unlock();
    //int counter = 0;
 
    QList<Worker_Arguments> internalList;
    internalList = list;

    emit progress_text_changed_signal(tr("Starting...."));
    qDebug() << "Start process from thread : " << QThread::currentThreadId();

    //emit progress_text_changed_signal(tr("DEBUG POS1")); //DEBUG CODE
    //emit progress_text_changed_signal(tr("Processing \"%1\"").arg(args.device_calib_header_data_ptr->device_calib_units_vector[0]));

    foreach(Worker_Arguments item_args, internalList){

        // ----- reopen the serial port so is available to the current thread -----
        if(item_args.communication_is_SERIAL_bool==1) {
            // CloseDevice_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr);
            connect_status = OpenDevice_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr,  item_args.serial_driver_LCR_sensors_session_ptr->port_name);
        }
        // -----------------------

        ret_data.error_counter=0;
        error_retry_counter=-1;        

        if( item_args.flag_transfer_mode_bool == 1) {
            //in case of upload, calculate the number of rows to be uploaded
            for(channel_number=0, N_steps=10; channel_number < (N_LCR_CHANNELS+N_ADC_CHANNELS); channel_number++ ) {
               plus_N_steps = item_args.device_calib_header_data_ptr->number_table_lines[channel_number];
                if( plus_N_steps < 0 && plus_N_steps > item_args.calib_table_max_rows_to_process_on_single_mode )
                   plus_N_steps = item_args.calib_table_max_rows_to_process_on_single_mode;
                N_steps = N_steps + plus_N_steps;
            }
        }
        else {
            //in case of download, the numbers of steps is the download of the full useful table
            N_steps = item_args.calib_table_max_rows_to_process_on_single_mode*item_args.N_channels ;
        }

        emit max_progress_range_changed_signal(N_steps);
        emit progress_value_changed_signal(0);

        for(channel_number=0; channel_number < item_args.N_channels; channel_number++) {

            if(item_args.flag_transfer_mode_bool == 0) {
                //download the calibration tables from the device
                //1st read the headers of the calibration tables
                  if(item_args.communication_is_SERIAL_bool==1) {                 

                      if(channel_number<N_LCR_CHANNELS) {    //the sensor counters are only available on channels: 0,1,2,3,4,5 (the LCR channels)
                          operation_ret_value = ReadHeaderCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->number_table_lines)[channel_number]), &((item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number]), (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], (item_args.device_calib_header_data_ptr->counters_calib_units_vector)[channel_number]);
                          if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                              QThread::usleep(500000);  //wait 500000 us
                              operation_ret_value = ReadHeaderCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->number_table_lines)[channel_number]), &((item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number]), (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], (item_args.device_calib_header_data_ptr->counters_calib_units_vector)[channel_number]);
                          }

                          if(operation_ret_value > 0) {
                            operation_ret_value = ReadCountersCalib_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][0]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][1]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][2]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][3]));
                            if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                                QThread::usleep(500000);  //wait 500000 us
                                operation_ret_value = ReadCountersCalib_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][0]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][1]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][2]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][3]));
                            }
                          }

                      }
                      else {

                          operation_ret_value = ReadHeaderCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->number_table_lines)[channel_number]), &((item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number]), (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], NULL);
                          if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                              QThread::usleep(500000);  //wait 500000 us
                              operation_ret_value = ReadHeaderCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->number_table_lines)[channel_number]), &((item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number]), (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], NULL);
                          }

                      }
                      Process_Return_Value(operation_ret_value, 1, RET_MSG_IN_CASE_SUCCESS_FOR_READ_TABLE_HEADER, RET_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_RS485, LOG_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_RS485, &current_value_progress_bar, &item_args , &ret_data, &error_retry_counter, &channel_number, &row_number, RAW_value, measurement);

                  }
                  else {

                    if(channel_number<N_LCR_CHANNELS) {   //the sensor counters are only available on channels: 0,1,2,3,4,5 (the LCR channels)
                        operation_ret_value = ReadHeaderCalibTable_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->number_table_lines)[channel_number]), &((item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number]), (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], (item_args.device_calib_header_data_ptr->counters_calib_units_vector)[channel_number]);
                        if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                            QThread::usleep(500000);  //wait 500000 us
                            operation_ret_value = ReadHeaderCalibTable_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->number_table_lines)[channel_number]), &((item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number]), (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], (item_args.device_calib_header_data_ptr->counters_calib_units_vector)[channel_number]);
                        }

                        if(operation_ret_value>0) {
                            operation_ret_value = ReadCountersCalib_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][0]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][1]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][2]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][3]));
                            if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                                QThread::usleep(500000);  //wait 500000 us
                                operation_ret_value = ReadCountersCalib_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][0]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][1]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][2]), &((item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][3]));
                            }
                        }

                    }
                    else {
                        operation_ret_value = ReadHeaderCalibTable_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->number_table_lines)[channel_number]), &((item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number]), (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], NULL);
                        if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                            QThread::usleep(500000);  //wait 500000 us
                            operation_ret_value = ReadHeaderCalibTable_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, &((item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->number_table_lines)[channel_number]), &((item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number]), &((item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number]), (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], NULL);
                        }
                    }
                    Process_Return_Value(operation_ret_value, 1, RET_MSG_IN_CASE_SUCCESS_FOR_READ_TABLE_HEADER, RET_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_USB, LOG_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_USB, &current_value_progress_bar, &item_args , &ret_data, &error_retry_counter, &channel_number, &row_number, RAW_value, measurement);
                  }
            }


            if( item_args.flag_transfer_mode_bool == 1) {
                //upload the calibration tables to the device

                //1st read the header of the calibration table
                       if(item_args.communication_is_SERIAL_bool==1) {
                           operation_ret_value = WriteHeaderCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, (item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number], (item_args.device_calib_header_data_ptr->number_table_lines)[channel_number], (item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number], (item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number], (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], (item_args.device_calib_header_data_ptr->counters_calib_units_vector)[channel_number]);
                           if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                               QThread::usleep(500000);  //wait 500000 us
                               operation_ret_value = WriteHeaderCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, (item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number], (item_args.device_calib_header_data_ptr->number_table_lines)[channel_number], (item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number], (item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number], (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], (item_args.device_calib_header_data_ptr->counters_calib_units_vector)[channel_number]);
                           }

                           if(operation_ret_value > 0 && channel_number<N_LCR_CHANNELS) {  //the sensor counters are only available on channels: 0,1,2,3,4,5 (the LCR channels)
                               operation_ret_value = WriteCountersCalib_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][0], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][1], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][2], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][3]);
                               if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                                   QThread::usleep(500000);  //wait 500000 us
                                   operation_ret_value = WriteCountersCalib_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][0], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][1], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][2], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][3]);
                               }
                           }

                           Process_Return_Value(operation_ret_value, 1, RET_MSG_IN_CASE_SUCCESS_FOR_WRITE_TABLE_HEADER, RET_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_HEADER_RS485, LOG_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_HEADER_RS485, &current_value_progress_bar, &item_args , &ret_data, &error_retry_counter, &channel_number, &row_number, RAW_value, measurement);
                       }
                       else {
                            operation_ret_value = WriteHeaderCalibTable_USB_MultipleSensor( item_args.driver_LCR_sensors_session_ptr, channel_number, (item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number], (item_args.device_calib_header_data_ptr->number_table_lines)[channel_number], (item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number], (item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number], (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], (item_args.device_calib_header_data_ptr->counters_calib_units_vector)[channel_number] );
                            if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                                QThread::usleep(500000);  //wait 500000 us
                                operation_ret_value = WriteHeaderCalibTable_USB_MultipleSensor( item_args.driver_LCR_sensors_session_ptr, channel_number, (item_args.device_calib_header_data_ptr->calib_table_is_active_bool)[channel_number], (item_args.device_calib_header_data_ptr->number_table_lines)[channel_number], (item_args.device_calib_header_data_ptr->calib_channel_mode_bool)[channel_number], (item_args.device_calib_header_data_ptr->jumper_selection_osc_tuning_range)[channel_number], (item_args.device_calib_header_data_ptr->device_calib_units_vector)[channel_number], (item_args.device_calib_header_data_ptr->counters_calib_units_vector)[channel_number] );
                            }

                            if(operation_ret_value > 0 && channel_number<N_LCR_CHANNELS) { //the sensor counters are only available on channels: 0,1,2,3,4,5 (the LCR channels)
                                operation_ret_value = WriteCountersCalib_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][0], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][1], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][2], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][3]);
                                if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                                    QThread::usleep(500000);  //wait 500000 us
                                    operation_ret_value = WriteCountersCalib_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][0], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][1], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][2], (item_args.device_calib_header_data_ptr->counters_calib_constants_vector)[channel_number][3]);
                                }
                            }

                            Process_Return_Value(operation_ret_value, 1, RET_MSG_IN_CASE_SUCCESS_FOR_WRITE_TABLE_HEADER, RET_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_HEADER_USB, LOG_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_HEADER_USB, &current_value_progress_bar, &item_args , &ret_data, &error_retry_counter, &channel_number, &row_number, RAW_value, measurement);
                       }


            }

              if(channel_number >= 0 && error_retry_counter<=0) {                  

                  filled_row_count = (item_args.device_calib_header_data_ptr->number_table_lines)[channel_number];   //this is 'number_of_table_lines', reuse of variable
                  if(item_args.flag_transfer_mode_bool<1 || 0 > filled_row_count || filled_row_count > item_args.calib_table_max_rows_to_process_on_single_mode ) {
                      //in this case is download mode, or if is upload the 'number_table_lines' wasn't in the valid range [ 0; CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE ]
                      filled_row_count = item_args.calib_table_max_rows_to_process_on_single_mode;   // CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE;
                  }

                  for(row_number=0; row_number<filled_row_count; row_number++) {

                      if( item_args.flag_transfer_mode_bool == 0) {

                          if(item_args.communication_is_SERIAL_bool==1) {
                              operation_ret_value = ReadLineCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, row_number, &RAW_value, &measurement);            
                              if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                                  QThread::usleep(500000);  //wait 500000 us
                                  operation_ret_value = ReadLineCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, row_number, &RAW_value, &measurement);
                              }
                              Process_Return_Value(operation_ret_value, 0, RET_MSG_IN_CASE_SUCCESS_FOR_READ_TABLE_ROW, RET_MSG_IN_CASE_ERROR_FOR_READ_TABLE_ROW_RS485, LOG_MSG_IN_CASE_ERROR_FOR_READ_TABLE_ROW_RS485, &current_value_progress_bar, &item_args , &ret_data, &error_retry_counter, &channel_number, &row_number, RAW_value, measurement);
                          }
                          else {
                              operation_ret_value = ReadLineCalibTable_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, row_number, &RAW_value, &measurement);
                              if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                                  QThread::usleep(500000);  //wait 500000 us
                                  operation_ret_value = ReadLineCalibTable_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, row_number, &RAW_value, &measurement);
                              }
                              Process_Return_Value(operation_ret_value, 0, RET_MSG_IN_CASE_SUCCESS_FOR_READ_TABLE_HEADER, RET_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_USB, LOG_MSG_IN_CASE_ERROR_FOR_READ_TABLE_HEADER_USB, &current_value_progress_bar, &item_args , &ret_data, &error_retry_counter, &channel_number, &row_number, RAW_value, measurement);
                          }

                          //save the read value to the application window showing the device calib table
                          if(ret_data.ret_value == THREAD_WORKER_RET_VAL_OK && RAW_value==RAW_value && measurement==measurement) {
                              item_args.model_tables_device_calib_vector[channel_number]->setItem(row_number, 0, new QStandardItem());
                              item_args.model_tables_device_calib_vector[channel_number]->setItem(row_number, 1, new QStandardItem());
                              item_args.model_tables_device_calib_vector[channel_number]->setData( item_args.model_tables_device_calib_vector[channel_number]->index(row_number,0), RAW_value);
                              item_args.model_tables_device_calib_vector[channel_number]->setData( item_args.model_tables_device_calib_vector[channel_number]->index(row_number,1), measurement);
                          }
                      }


                      if( item_args.flag_transfer_mode_bool == 1) {
                        //upload the calibration tables to the device
                         RAW_value = (item_args.model_tables_device_calib_vector[channel_number]->index(row_number,0)).data().toFloat();  //RAW_value on left column of calib table
                         measurement = (item_args.model_tables_device_calib_vector[channel_number]->index(row_number,1)).data().toFloat();    //measurement on right column of calib table

                        if(item_args.communication_is_SERIAL_bool==1) {
                             operation_ret_value = WriteLineCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, row_number, RAW_value, measurement);
                             if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                                 QThread::usleep(500000);  //wait 500000 us
                                 operation_ret_value = WriteLineCalibTable_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr, channel_number, row_number, RAW_value, measurement);
                             }

                             Process_Return_Value(operation_ret_value, 0, RET_MSG_IN_CASE_SUCCESS_FOR_WRITE_TABLE_ROW, RET_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_ROW_RS485, LOG_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_ROW_RS485, &current_value_progress_bar, &item_args , &ret_data, &error_retry_counter, &channel_number, &row_number, RAW_value, measurement);
                         }
                         else {
                            operation_ret_value = WriteLineCalibTable_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, row_number, RAW_value, measurement);
                            if( operation_ret_value < 0 ) {   //in case an error occurs on the transfer of the current line of the calibration table, then retry to transfer the current line.
                                QThread::usleep(500000);  //wait 500000 us
                                operation_ret_value = WriteLineCalibTable_USB_MultipleSensor(item_args.driver_LCR_sensors_session_ptr, channel_number, row_number, RAW_value, measurement);
                            }
                            Process_Return_Value(operation_ret_value, 0, RET_MSG_IN_CASE_SUCCESS_FOR_WRITE_TABLE_ROW, RET_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_ROW_USB, LOG_MSG_IN_CASE_ERROR_FOR_WRITE_TABLE_ROW_USB, &current_value_progress_bar, &item_args , &ret_data, &error_retry_counter, &channel_number, &row_number, RAW_value, measurement);
                         }
                      }


                      last_error_counter = ret_data.error_counter;
                      //emit progress_value_changed_signal(++current_value_progress_bar);    //update the dialog with the progress bar
                      //emit progress_text_changed_signal(tr("Processing...\n CH.#,row.#:(RAW_value; measurement):\n \"CH.%1,row.%2:(%3;%4)\"").arg(channel_number).arg(row_number).arg(RAW_value).arg(measurement));
                      //emit progress_text_changed_signal(tr("m_stop \"%1\"").arg(m_stop)); //DEBUG CODE

                      //Check pausing or canceling
                      //==========================
                      m_sync.lock();
                      if(m_pause && m_pause_cond){
                          //emit progress_text_changed_signal(tr("ON PAUSE")); //DEBUG CODE
                          //qDebug() << "Waiting on pause condition from thread : " << QThread::currentThreadId();    //DEBUG CODE
                          m_pause_cond->wait(&m_sync);
                          //Reset pause flag
                        m_pause = false;
                      }
                      if(m_stop) {
                          //emit progress_text_changed_signal(tr("ON STOP"));  //DEBUG CODE
                          m_sync.unlock();
                          //Leave the loop
                          channel_number = 10;  //set 'channel_number' to 10 to finish now
                          break;
                      }
                      m_sync.unlock();

                      //send the data result
                      emit result_ready_signal(ret_data);

                      //Process event in the queue
                      QCoreApplication::processEvents();

                  }


              }

          }

        CloseDevice_Serial_MultipleSensor(item_args.serial_driver_LCR_sensors_session_ptr);    //make the current thread disconnect from the serial port

    }

    m_sync.lock();
    m_pause = false;
    m_stop = false;
    m_sync.unlock();
    emit finished_signal();

}
