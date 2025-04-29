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
 *   FILE: virtual_calibrate_sensors_dialog.cpp - Here is the implementation of  *
 *     the class 'Virtual_Calibrate_Sensors_Dialog', used for view/edit of       *
 *     Multiple-Sensor virtual calibration tables of the various sensor channels.*
 ********************************************************************************/

#include "virtual_calibrate_sensors_dialog.h"

// -------- CLASS  Virtual_Calibrate_Sensors_Dialog  ---------
Virtual_Calibrate_Sensors_Dialog::Virtual_Calibrate_Sensors_Dialog(int *previous_ch_number_for_saved_virtual_calib_dialog_ptr,char (*virtual_calib_units_vector_ptr)[N_CHARS_CALIB_UNITS], QStandardItemModel **model_tables_virtual_calib_vector_ptr, counters_calib *counters_virtual_calib_ptr, unsigned char *virtual_calib_channel_mode_bool_vector_ptr, char *virtual_calib_jumper_selection_osc_tuning_range_vector_ptr) : Calibrate_Sensors_Dialog(previous_ch_number_for_saved_virtual_calib_dialog_ptr, virtual_calib_units_vector_ptr, model_tables_virtual_calib_vector_ptr, counters_virtual_calib_ptr) {
    this->setWindowTitle("Calibrate LCR Sensors (Virtual, not saved on the device)");

    this->m_virtual_calib_channel_mode_bool_vector_ptr = virtual_calib_channel_mode_bool_vector_ptr;
    this->m_virtual_calib_jumper_selection_osc_tuning_range_vector_ptr = virtual_calib_jumper_selection_osc_tuning_range_vector_ptr;

    //reprogram the function of the OK_button for executing the "Save_Virtual_Sensors_Calibrations" that is the appropriate function to save all the data of "Virtual_Calibrate_Sensors_Dialog"
    QObject::connect(OK_button, SIGNAL( clicked(bool) ), this, SLOT(Save_Virtual_Sensors_Calibrations()) );

    //reprogram the function of the select_channel_number_spinBox for executing the "Update_Virtual_Calibrate_Sensors_Dialog()" that is the appropriate function to update all the data of "Virtual_Calibrate_Sensors_Dialog"
    QObject::connect(select_channel_number_spinBox, SIGNAL(valueChanged(int)), this, SLOT( Update_Virtual_Calibrate_Sensors_Dialog() ) );
    this->Update_Virtual_Calibrate_Sensors_Dialog();
}

Virtual_Calibrate_Sensors_Dialog::~Virtual_Calibrate_Sensors_Dialog() {
    this->close();
}


void Virtual_Calibrate_Sensors_Dialog::Save_Virtual_Sensors_Calibrations() {
    int selected_calib_sensor_number;

    selected_calib_sensor_number = select_channel_number_spinBox->value();

    this->Save_Sensors_Calibrations();

    //save the current selected channel mode
    m_virtual_calib_channel_mode_bool_vector_ptr[selected_calib_sensor_number] = (int) calib_channel_mode_comboBox->currentIndex();
    //save the jumper selection of the oscillator tuning range of the selected sensor channel, only use the first character of each option on the ComboBox ('A', 'B', '+', '-')
    m_virtual_calib_jumper_selection_osc_tuning_range_vector_ptr[selected_calib_sensor_number] = calib_channel_jumper_selection_osc_tuning_range_comboBox->currentText().toUtf8().data()[0];
}



void Virtual_Calibrate_Sensors_Dialog::Update_Virtual_Calibrate_Sensors_Dialog() {
    int k;

    int selected_channel_number = select_channel_number_spinBox->value();

    this->Update_Calibrate_Sensors_Dialog();

    //fill selected channel mode
    calib_channel_mode_comboBox->setCurrentIndex( m_virtual_calib_channel_mode_bool_vector_ptr[selected_channel_number] );

    //fill the jumper selection of the oscillator tuning range
    for(k=0, calib_channel_jumper_selection_osc_tuning_range_comboBox->setCurrentIndex(-1); k<(calib_channel_jumper_selection_osc_tuning_range_comboBox->count()); k++ ) {
        if( m_virtual_calib_jumper_selection_osc_tuning_range_vector_ptr[selected_channel_number] == calib_channel_jumper_selection_osc_tuning_range_comboBox->itemText(k).toUtf8().data()[0] )    //only compare the first character of each option on the ComboBox for the 'jumper_selection_osc_tuning_range' ('A', 'B', '+', '-')
           calib_channel_jumper_selection_osc_tuning_range_comboBox->setCurrentIndex(k);    //discover and select the index of the ComboBox where is the letter that matches the passed value for "jumper_selection_osc_tuning_range" ('A', 'B', '+', '-')
    }

}

// ----------------------------------------------------------------------------------------------
