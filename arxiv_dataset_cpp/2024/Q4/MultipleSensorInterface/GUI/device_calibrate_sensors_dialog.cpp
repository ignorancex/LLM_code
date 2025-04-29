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
 *   FILE: device_calibrate_sensors_dialog.cpp - Here is the implementation of   *
 *      the class 'Device_Calibrate_Sensors_Dialog', used for view/edit of       *
 *      Multiple-Sensor device calibration tables of the various sensor channels.*
 ********************************************************************************/


#include "device_calibrate_sensors_dialog.h"

// -------- CLASS  Device_Calibrate_Sensors_Dialog  ---------
Device_Calibrate_Sensors_Dialog::Device_Calibrate_Sensors_Dialog(int *previous_ch_number_for_saved_device_calib_dialog_ptr, DeviceCalibHeaderData *device_calib_header_ptr, QStandardItemModel **model_tables_device_calib_vector_ptr, counters_calib *counters_calib_ptr) : Calibrate_Sensors_Dialog(previous_ch_number_for_saved_device_calib_dialog_ptr, device_calib_header_ptr->device_calib_units_vector, model_tables_device_calib_vector_ptr, counters_calib_ptr) {
    int selected_calib_sensor_number;
    this->setWindowTitle("Calibrate Sensors (saved to device EEPROM)");

    int k;

    this->device_calib_header_ptr = device_calib_header_ptr;
    //create another reference to the pointer to the vector of strings with the units of the calibration of the sensor channels
    this->m_sensors_calib_units_vector_ptr = device_calib_header_ptr->device_calib_units_vector;

    calib_table_is_active_widget = new QWidget(this);
    calib_table_is_active_widget_layout = new QGridLayout();
    calib_table_is_active_widget->setLayout(calib_table_is_active_widget_layout);
    calib_table_is_active_label = new QLabel("Calibration table is active:");
    calib_table_is_active_widget_layout->addWidget(calib_table_is_active_label,0,0);
    calib_table_is_active_checkbox = new QCheckBox();
    calib_table_is_active_widget_layout->addWidget(calib_table_is_active_checkbox, 0, 1);
    top_select_widget_layout->addWidget(calib_table_is_active_widget, 1, 0);

    number_valid_table_lines_widget = new QWidget(this);
    number_valid_table_lines_widget_layout = new QGridLayout();
    number_valid_table_lines_widget->setLayout(number_valid_table_lines_widget_layout);
    number_valid_table_lines_label = new QLabel("Number valid rows:");
    number_valid_table_lines_widget_layout->addWidget(number_valid_table_lines_label,0,0);
    number_valid_table_lines_box = new QSpinBox();
    number_valid_table_lines_widget_layout->addWidget(number_valid_table_lines_box, 0, 1);
    top_select_widget_layout->addWidget(number_valid_table_lines_widget, 1, 2);
    number_valid_table_lines_box->setMinimum(2);
    number_valid_table_lines_box->setMaximum(CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE);

    //reprogram the function of the OK_button for executing the "Save_Device_Sensors_Calibrations" that is the appropriate function to save all the data of "Device_Calibrate_Sensors_Dialog"
    QObject::connect(OK_button, SIGNAL( clicked(bool) ), this, SLOT(Save_Device_Sensors_Calibrations()) );

    QObject::connect(calib_channel_mode_comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(Calib_Channel_Mode_Changed()) );

    selected_calib_sensor_number = select_channel_number_spinBox->value();
    //fill selected channel mode
    calib_channel_mode_comboBox->setCurrentIndex( device_calib_header_ptr->calib_channel_mode_bool[selected_calib_sensor_number] );
    //fill the table is set to active
    calib_table_is_active_checkbox->setChecked( (bool) device_calib_header_ptr->calib_table_is_active_bool[selected_calib_sensor_number]);
    //fill number of valid lines
    number_valid_table_lines_box->setValue(device_calib_header_ptr->number_table_lines[selected_calib_sensor_number]);

    //fill the jumper selection (JPX)
    for(k=0, calib_channel_jumper_selection_osc_tuning_range_comboBox->setCurrentIndex(-1); k<(calib_channel_jumper_selection_osc_tuning_range_comboBox->count()); k++ ) {
        if( device_calib_header_ptr->jumper_selection_osc_tuning_range[selected_calib_sensor_number] == calib_channel_jumper_selection_osc_tuning_range_comboBox->itemText(k).toUtf8().data()[0] )  //only compare the first character of each option on the ComboBox for the 'jumper_selection_osc_tuning_range' ('A', 'B', '+', '-')
           calib_channel_jumper_selection_osc_tuning_range_comboBox->setCurrentIndex(k);    //discover and select the index of the ComboBox where is the letter that matches the passed value for "jumper_selection_osc_tuning_range" ('A', 'B', '+', '-')
    }

    //fill the calibration constants and units

    //reprogram the function of the select_channel_number_spinBox for executing the "Update_Device_Calibrate_Sensors_Dialog()" that is the appropriate function to update all the data of "Device_Calibrate_Sensors_Dialog"
    QObject::connect(select_channel_number_spinBox, SIGNAL(valueChanged(int)), this, SLOT( Update_Device_Calibrate_Sensors_Dialog() ) );

    QObject::connect(add_new_row_calib_button, SIGNAL(clicked()), this, SLOT(Increase_Number_Valid_Lines()) );

    QObject::connect(number_valid_table_lines_box, SIGNAL(valueChanged(int)), this, SLOT( Number_Valid_Table_Lines_Changed() ) );


    //disconnect the button 'add_new_row_calib_button' from the function 'Calibrate_Sensors_Dialog::Add_New_Row()'
    //then connect the button 'add_new_row_calib_button' to the function 'Device_Calib_Add_New_Row()', that is the appropriate function of class 'Device_Calibrate_Sensors_Dialog' to add a new row on device tables
    disconnect(add_new_row_calib_button, 0, 0, 0);
    QObject::connect(add_new_row_calib_button, SIGNAL(clicked()), this, SLOT(Device_Calib_Add_New_Row()) );

    this->Update_Device_Calibrate_Sensors_Dialog();

}

Device_Calibrate_Sensors_Dialog::~Device_Calibrate_Sensors_Dialog() {
    this->close();
}

void Device_Calibrate_Sensors_Dialog::Save_Device_Sensors_Calibrations() {
    int selected_calib_sensor_number;

    selected_calib_sensor_number = select_channel_number_spinBox->value();

    this->Save_Sensors_Calibrations();

    //save the current selected channel mode
    this->device_calib_header_ptr->calib_channel_mode_bool[selected_calib_sensor_number] = (int) calib_channel_mode_comboBox->currentIndex();
    //save if the table is set to active
    this->device_calib_header_ptr->calib_table_is_active_bool[selected_calib_sensor_number] = (int) calib_table_is_active_checkbox->isChecked();
    //save the number of valid lines
    this->device_calib_header_ptr->number_table_lines[selected_calib_sensor_number] = number_valid_table_lines_box->value();

    //save the jumper selection of the oscillator tuning range of the selected sensor channel, only use the first character of each option on the ComboBox ('A', 'B', '+', '-')
    this->device_calib_header_ptr->jumper_selection_osc_tuning_range[selected_calib_sensor_number] = calib_channel_jumper_selection_osc_tuning_range_comboBox->currentText().toUtf8().data()[0];

}

void Device_Calibrate_Sensors_Dialog::Increase_Number_Valid_Lines() {
    this->number_valid_table_lines_box->stepUp();   //increase the 'number_valid_table_lines' each time a row is added to the table
}

void Device_Calibrate_Sensors_Dialog::Update_Device_Calibrate_Sensors_Dialog() {
    int k;

    int selected_channel_number = select_channel_number_spinBox->value();

    this->Update_Calibrate_Sensors_Dialog();

    //fill selected channel mode
    calib_channel_mode_comboBox->setCurrentIndex( this->device_calib_header_ptr->calib_channel_mode_bool[selected_channel_number] );
    //fill the table is set to active
    calib_table_is_active_checkbox->setChecked( (bool) this->device_calib_header_ptr->calib_table_is_active_bool[selected_channel_number]);
    //fill number of valid lines
    number_valid_table_lines_box->setValue(device_calib_header_ptr->number_table_lines[selected_channel_number]);

    //fill the jumper selection of the oscillator tuning range
    for(k=0, calib_channel_jumper_selection_osc_tuning_range_comboBox->setCurrentIndex(-1); k<(calib_channel_jumper_selection_osc_tuning_range_comboBox->count()); k++ ) {
        if( device_calib_header_ptr->jumper_selection_osc_tuning_range[selected_channel_number] == calib_channel_jumper_selection_osc_tuning_range_comboBox->itemText(k).toUtf8().data()[0] )   //only compare the first character of each option on the ComboBox for the 'jumper_selection_osc_tuning_range' ('A', 'B', '+', '-')
           calib_channel_jumper_selection_osc_tuning_range_comboBox->setCurrentIndex(k);    //discover and select the index of the ComboBox where is the letter that matches the passed value for "jumper_selection_osc_tuning_range" ('A', 'B', '+', '-')
    }

    Number_Valid_Table_Lines_Changed();    //update the color of table rows that should be painted gray because they are outside of the selected valid lines of calibration table

}

void Device_Calibrate_Sensors_Dialog::Device_Calib_Add_New_Row() {
    if(this->Calib_Table_Row_Count()<CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE) {
        this->Add_New_Row();
        Number_Valid_Table_Lines_Changed();    //update the color of table rows that should be painted gray because they are outside of the selected valid lines of calibration table
    }
}

void Device_Calibrate_Sensors_Dialog::Calib_Channel_Mode_Changed() {

    if(this->calib_channel_mode_comboBox->currentIndex()==0) {
        number_valid_table_lines_box->setMaximum(CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_MULTI_MODE);       //configure the maximum limit of the number of valid table lines SpinBox, so it doesn't exceeds the maximum of calibration table lines that the device can process when in multiple channel mode.
    }
    else {
        number_valid_table_lines_box->setMaximum(CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE);       //configure the maximum limit of the number of valid table lines SpinBox, so it doesn't exceeds the maximum of calibration table lines that the device can process when in single channel mode.
    }
}


//'void Device_Calibrate_Sensors_Dialog::Number_Valid_Table_Lines_Changed()' - Updates the color (to gray) of table cells that are outside the selected valid lines of calibration table
void Device_Calibrate_Sensors_Dialog::Number_Valid_Table_Lines_Changed() {

    int start_row_color_as_not_valid, end_row_color_as_not_valid;

    start_row_color_as_not_valid = number_valid_table_lines_box->value();

    end_row_color_as_not_valid = this->Calib_Table_Row_Count();

    //check if current mode is single channel or multiple channel,
    if(this->calib_channel_mode_comboBox->currentIndex()==0) {  //in case is multiple channel then maximum number of rows is 'CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_MULTI_MODE'
        if(end_row_color_as_not_valid>CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_MULTI_MODE) {
            end_row_color_as_not_valid=CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_MULTI_MODE;
        }
    }
    else {      //in case is single channel then maximum number of rows is 'CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE'
        if(end_row_color_as_not_valid>CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE) {
            end_row_color_as_not_valid=CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE;
        }
    }

    this->Set_Rows_Color(0, end_row_color_as_not_valid, QColor(Qt::white));     //clear any previous color on the table cells by painting them all to white
    this->Set_Rows_Color(start_row_color_as_not_valid, end_row_color_as_not_valid, QColor(Qt::gray));   //set to gray color the table cells that are NOT ACTIVE/available by painting those in yelloh color.
}

// ----------------------------------------------------------------------------------------------
