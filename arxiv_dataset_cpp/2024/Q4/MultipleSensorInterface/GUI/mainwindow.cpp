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
 *   FILE: mainwindow.cpp - Here are declared most classes and functions that    *
 *                     implement the GUI software of Multiple-Sensor Interface.  *
 ********************************************************************************/

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QString>
#include <QTextStream>
#include <QMessageBox>

#include <QSignalMapper>

#include <QThread>

#include <iostream> // useful or text debugging

#include <math.h>

#include <QTextEdit>
#include <QDialogButtonBox>
#include <QDesktopServices>

#include <QtWidgets>

#include "about_dialog.h"

#include "device_calibrate_sensors_dialog.h"
#include "virtual_calibrate_sensors_dialog.h"

#include "worker_handler.h"

#define SENSOR_PRESCALER 4


bool Convert_Char_To_Bool(char value) {
    if(value>0) {
        return true;
    }
    else {
        return false;
    }
}

char Convert_Bool_To_Char(bool value) {
    if(value) {
        return 1;
    }
    else {
        return 0;
    }
}

// -------------------------------------------------------------------------------------



// ----- Functions of the class Config_Sensor_Channels_Dialog  -  START -----

Config_Sensor_Channels_Dialog::Config_Sensor_Channels_Dialog(MultipleSensor_USB_driver *session, MultipleSensor_serial_driver *session_serial , int *previous_ch_number_for_saved_config_dialog_ptr, char communication_is_SERIAL_bool) {

    this->m_USB_driver_LCR_sensors_session=session;
    this->m_serial_driver_LCR_sensors_session=session_serial;
    this->m_previous_ch_number_for_saved_config_dialog_ptr = previous_ch_number_for_saved_config_dialog_ptr;
    this->m_communication_is_SERIAL_bool = communication_is_SERIAL_bool;

    const char *space_label_xl = "          ";
    const char *space_label_small = "  ";

    QString text_display;

    config_output_sensors_dialog_layout = new QGridLayout();
    this->setLayout(config_output_sensors_dialog_layout);

    this->setWindowTitle("Config Device Sensor Channels");

    this->default_style_sheet = this->styleSheet();    //save the default styleSheet(and so the color) of the dialog box

    config_output_sensors_widget = new QWidget(this);
    config_output_sensors_widget_layout = new QGridLayout();
    config_output_sensors_widget->setLayout(config_output_sensors_widget_layout);

    config_output_sensors_dialog_layout->addWidget(config_output_sensors_widget);

    select_channel_number_label = new QLabel("Select Channel #:", this);

    // select the channel number to calibrate
    select_channel_number_spinBox = new QSpinBox(this);
    select_channel_number_spinBox->setMinimum(0);
    select_channel_number_spinBox->setMaximum(N_LCR_CHANNELS+N_ADC_CHANNELS-1);

    select_channel_widget = new QWidget(this);
    select_channel_widget_layout = new QGridLayout();
    select_channel_widget->setLayout(select_channel_widget_layout);

    select_channel_widget_layout->addWidget(select_channel_number_label, 0, 0);
    select_channel_widget_layout->addWidget(select_channel_number_spinBox, 0, 1);

    config_output_sensors_widget_layout->addWidget(select_channel_widget, 0, 0);

    select_channel_number_spinBox->setValue(*m_previous_ch_number_for_saved_config_dialog_ptr); //restore the previously selected channel number


    select_comparison_activate_sensor_for_outputs_label = new QLabel("Comparison used to activate\nthe channel for board outputs :", this);
    select_comparison_activate_sensor_for_outputs_comboBox = new QComboBox(this);
    select_comparison_activate_sensor_for_outputs_comboBox->insertItem(0,"< Smaller than");
    select_comparison_activate_sensor_for_outputs_comboBox->insertItem(1,"> Bigger than");
    select_comparison_activate_sensor_for_outputs_comboBox->setMinimumWidth(100);

    config_output_sensors_widget_layout->addWidget(select_comparison_activate_sensor_for_outputs_label, 2,0);
    config_output_sensors_widget_layout->addWidget(select_comparison_activate_sensor_for_outputs_comboBox, 2, 1);

    trigger_value_label = new QLabel("Trigger value :", this);
    trigger_value_doubleSpinBox = new QDoubleSpinBox(this);
    trigger_value_doubleSpinBox->setMaximum(9999999999);
    trigger_value_doubleSpinBox->setMinimum(-9999999999);
    trigger_value_doubleSpinBox->setLocale(QLocale::C);    //configure the QDoubleSpinBox to use '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations

    config_output_sensors_widget_layout->addWidget(trigger_value_label, 6, 0);
    config_output_sensors_widget_layout->addWidget(trigger_value_doubleSpinBox, 6, 1);

    button_widget = new QWidget(this);
    button_widget_layout = new QGridLayout();
    button_widget->setLayout(button_widget_layout);

    OK_button = new QPushButton("OK", this);
    cancel_button = new QPushButton("Cancel", this);

    status_label = new QLabel(this);

    button_widget_layout->addWidget(new QLabel(space_label_small),0,0);
    button_widget_layout->addWidget(OK_button,0,1);
    button_widget_layout->addWidget(new QLabel(space_label_small),0,2);
    button_widget_layout->addWidget(cancel_button,0,3);
    button_widget_layout->addWidget(new QLabel(space_label_small),0,4);

    config_output_sensors_dialog_layout->addWidget(button_widget);

    config_output_sensors_dialog_layout->addWidget(status_label);

    QObject::connect(cancel_button, SIGNAL( clicked(bool) ), this, SLOT( Cancel_Dialog() ) );
    QObject::connect(OK_button, SIGNAL( clicked(bool) ), this, SLOT(Send_Config_Sensor_Channels()) );

    this->show();

    select_channel_number_spinBox->setValue(*m_previous_ch_number_for_saved_config_dialog_ptr); //restore the previously selected channel number


    QObject::connect(select_channel_number_spinBox, SIGNAL(valueChanged(int)), this, SLOT(Update_Config_Sensor_Channels_Dialog()));

    QObject::connect(select_comparison_activate_sensor_for_outputs_comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(Set_Status_Label_Invalid()) );

    QObject::connect(trigger_value_doubleSpinBox, SIGNAL(valueChanged(double)), this, SLOT(Set_Status_Label_Invalid()) );

    this->show();

    //update the configs of the selected sensor channel
    Update_Config_Sensor_Channels_Dialog();

}

Config_Sensor_Channels_Dialog::~Config_Sensor_Channels_Dialog() {
    this->close();
}

void Config_Sensor_Channels_Dialog::Set_Status_Label_Valid() {
    status_label->setText("Current configs are valid.");
}

void Config_Sensor_Channels_Dialog::Set_Status_Label_Invalid() {
    status_label->setText("Current configs are invalid.");
}

void Config_Sensor_Channels_Dialog::Set_Status_Label_Error() {
    status_label->setText("ERROR: Error when updating current configs.");
}

void Config_Sensor_Channels_Dialog::Cancel_Dialog() {
    *m_previous_ch_number_for_saved_config_dialog_ptr = this->select_channel_number_spinBox->value();  //save the current selected sensor channel
    this->reject();
}

void Config_Sensor_Channels_Dialog::Update_Config_Sensor_Channels_Dialog() {
    char make_compare_w_calibrated_sensor_value;
    char compare_is_greater_than;
    char sensor_prescaler;
    char sensor_used_for_OUT[3];
    float sensor_trigger_value;

    int ret_value;

    this->Set_Status_Label_Invalid();
    this->show();
    this->config_output_sensors_widget->setDisabled(true);   //set disabled the dialog window while the saved configurations are being read from the sensors board

    // read the previous saved calibration for the selected channel number

     if(m_communication_is_SERIAL_bool==1) {
         ret_value = ReadSensorConfigs_Serial_MultipleSensor(m_serial_driver_LCR_sensors_session, select_channel_number_spinBox->value(), &make_compare_w_calibrated_sensor_value, &compare_is_greater_than, &sensor_prescaler, sensor_used_for_OUT, &sensor_trigger_value);
     }
     else {
         ret_value = ReadSensorConfigs_USB_MultipleSensor(m_USB_driver_LCR_sensors_session, select_channel_number_spinBox->value(), &make_compare_w_calibrated_sensor_value, &compare_is_greater_than, &sensor_prescaler, sensor_used_for_OUT, &sensor_trigger_value);
     }
    if( ret_value  < 0 ) {
        if(ret_value == USB_REPLY_WITH_INVALID_DATA || ret_value == SERIAL_REPLY_WITH_INVALID_DATA) {
            this->Set_Status_Label_Invalid();
            this->setStyleSheet("background-color:yellow;");
        }
        else {
            //print error dialog
            this->Set_Status_Label_Error();
        }
    }
    else {
        select_comparison_activate_sensor_for_outputs_comboBox->setCurrentIndex(compare_is_greater_than);
        trigger_value_doubleSpinBox->setValue(sensor_trigger_value);
        this->Set_Status_Label_Valid();
        this->setStyleSheet(default_style_sheet);
    }

    // ----- DEBUG CODE -----
 /*  char msg_str[200];
    sprintf(msg_str, "@ Update_Config_Sensor_Channels_Dialog(), ret_value:%d", ret_value);
    QMessageBox msgBoxK;
    msgBoxK.setText(msg_str);
    msgBoxK.exec();  */
    // ----------------------

    this->config_output_sensors_widget->setDisabled(false);   //set enabled the dialog window while the saved configurations are being read from the sensors board
}

void Config_Sensor_Channels_Dialog::Send_Config_Sensor_Channels() {
    char make_compare_w_calibrated_sensor_value;
    char compare_is_greater_than;
    char sensor_used_for_OUT[3];
    float sensor_trigger_value;

    int selected_channel_number;
    int ret_value;

    // read the values from the configs dialog and send them to the sensors board
    selected_channel_number=select_channel_number_spinBox->value();
    compare_is_greater_than=select_comparison_activate_sensor_for_outputs_comboBox->currentIndex();
    sensor_trigger_value=trigger_value_doubleSpinBox->value();
    make_compare_w_calibrated_sensor_value = 0; //INFO: deprecated parameter, now the Multiple-Sensor Interface device will compare the selected threshold value to the RAW sensor, sensor measurement, counter, or counter measurement ('R?', 'M?', 'C?', 'K?') as specified on the boolean function of each output (OUT0, OUT1, OUT2).

    //save the configurations on the sensors board
    if(m_communication_is_SERIAL_bool==1) {
        ret_value = SaveSensorConfigs_Serial_MultipleSensor(m_serial_driver_LCR_sensors_session, selected_channel_number, make_compare_w_calibrated_sensor_value, compare_is_greater_than, SENSOR_PRESCALER, sensor_used_for_OUT, sensor_trigger_value);
    }
    else {
        ret_value = SaveSensorConfigs_USB_MultipleSensor(m_USB_driver_LCR_sensors_session, selected_channel_number, make_compare_w_calibrated_sensor_value, compare_is_greater_than, SENSOR_PRESCALER, sensor_used_for_OUT, sensor_trigger_value);
    }

    *m_previous_ch_number_for_saved_config_dialog_ptr = selected_channel_number;  //save the current selected sensor channel

    this->accept();

}
// ----- Functions of the class Config_Sensor_Channels_Dialog  -  END -----


// ----- Functions of the class Config_Outputs_Dialog  -  START -----
Config_Outputs_Dialog::Config_Outputs_Dialog(MultipleSensor_USB_driver *session, MultipleSensor_serial_driver *session_serial, int *previous_output_number_for_saved_config_dialog_ptr, char communication_is_SERIAL_bool) {

    int i;

    this->m_USB_driver_LCR_sensors_session=session;
    this->m_serial_driver_LCR_sensors_session=session_serial;
    this->m_previous_output_number_for_saved_config_dialog_ptr = previous_output_number_for_saved_config_dialog_ptr;
    this->m_communication_is_SERIAL_bool=communication_is_SERIAL_bool;

    const char *space_label_xl = "          ";
    const char *space_label_small = "  ";

    QString text_display;

    config_outputs_dialog_layout = new QGridLayout();
    this->setLayout(config_outputs_dialog_layout);

    this->setWindowTitle("Config Outputs of Board");

    config_outputs_widget = new QWidget(this);
    config_outputs_widget_layout = new QGridLayout();
    config_outputs_widget->setLayout(config_outputs_widget_layout);

    config_outputs_dialog_layout->addWidget(config_outputs_widget);

    for(i=0;i<N_DIGITAL_OUT;i++) {
        text_display.clear();
        QTextStream( &text_display ) << "Insert a boolean function for output N." << i << ":" << "  .\n";
        select_output_active_level_label_vector[i] =  new QLabel(text_display, this);
        select_output_active_level_bool_func_vector[i] = new QLineEdit(this);
        select_output_active_level_bool_func_vector[i]->setMinimumWidth(200);

        config_outputs_widget_layout->addWidget(select_output_active_level_label_vector[i], i,0);
        config_outputs_widget_layout->addWidget(select_output_active_level_bool_func_vector[i], i, 1);

        QObject::connect(select_output_active_level_bool_func_vector[i], SIGNAL(textChanged(QString)), this, SLOT(Set_Status_Label_Invalid()) );
    }

    button_widget = new QWidget(this);
    button_widget_layout = new QGridLayout();
    button_widget->setLayout(button_widget_layout);

    OK_button = new QPushButton("OK", this);
    cancel_button = new QPushButton("Cancel", this);

    status_label = new QLabel(this);

    button_widget_layout->addWidget(new QLabel(space_label_small),0,0);
    button_widget_layout->addWidget(OK_button,0,1);
    button_widget_layout->addWidget(new QLabel(space_label_small),0,2);
    button_widget_layout->addWidget(cancel_button,0,3);
    button_widget_layout->addWidget(new QLabel(space_label_small),0,4);

    config_outputs_dialog_layout->addWidget(button_widget);

    config_outputs_dialog_layout->addWidget(status_label);

    QObject::connect(cancel_button, SIGNAL( clicked(bool) ), this, SLOT( Cancel_Dialog() ) );
    QObject::connect(OK_button, SIGNAL( clicked(bool) ), this, SLOT(Send_Config_Outputs()) );

    this->show();

    //update the configs of the outputs
    Update_Config_Outputs_Dialog();

    this->show();

}


Config_Outputs_Dialog::~Config_Outputs_Dialog() {
    this->close();
}

void Config_Outputs_Dialog::Set_Status_Label_Valid() {
    status_label->setText("Current configs are valid.");
}

void Config_Outputs_Dialog::Set_Status_Label_Invalid() {
    status_label->setText("Current configs are invalid.");
}

void Config_Outputs_Dialog::Set_Status_Label_Error() {
    status_label->setText("ERROR: Error when updating current configs.");
}

void Config_Outputs_Dialog::Cancel_Dialog() {
    this->reject();
}


void Config_Outputs_Dialog::Update_Config_Outputs_Dialog() {
    char OUTPUTS_bool_func[N_DIGITAL_OUT][BOOL_FUNC_N_BYTES];
    int i;

    if(m_communication_is_SERIAL_bool==1) {

         for(i=0;i<N_DIGITAL_OUT;i++) {
            if( ReadOutputConfig_Serial_MultipleSensor(m_serial_driver_LCR_sensors_session, i, OUTPUTS_bool_func[i])  < 0) {
                //print error dialog
                this->Set_Status_Label_Error();
            }
            else {
                this->select_output_active_level_bool_func_vector[i]->setText( QString(OUTPUTS_bool_func[i]) );
                this->Set_Status_Label_Valid();
            }
         }
    }
    else {

        for(i=0;i<N_DIGITAL_OUT;i++) {
           if( ReadOutputConfig_USB_MultipleSensor(m_USB_driver_LCR_sensors_session, i, OUTPUTS_bool_func[i])  < 0) {
               //print error dialog
               this->Set_Status_Label_Error();
           }
           else {
               this->select_output_active_level_bool_func_vector[i]->setText( QString(OUTPUTS_bool_func[i]) );
               this->Set_Status_Label_Valid();
           }
        }


    }
}


void Config_Outputs_Dialog::Send_Config_Outputs() {
    char OUTPUTS_bool_func[N_DIGITAL_OUT][BOOL_FUNC_N_BYTES];
    int i;

    for(i=0;i<N_DIGITAL_OUT;i++) {
        strncpy( OUTPUTS_bool_func[i], select_output_active_level_bool_func_vector[i]->text().toUtf8().data(), BOOL_FUNC_N_BYTES);

        if(this->m_communication_is_SERIAL_bool==1) {
            WriteOutputConfig_Serial_MultipleSensor(m_serial_driver_LCR_sensors_session, i, OUTPUTS_bool_func[i]);
        }
        else {
            WriteOutputConfig_USB_MultipleSensor(m_USB_driver_LCR_sensors_session, i, OUTPUTS_bool_func[i]);
        }

    }

    this->accept(); //close the dialog box after reading the user selections
}

// ----- Functions of the class Config_Outputs_Dialog  -  END -----



// ----- Functions of the class Config_BoardID_Dialog  -  START -----

Config_BoardID_Dialog::Config_BoardID_Dialog(MultipleSensor_USB_driver *session, MultipleSensor_serial_driver *session_serial, char communication_is_SERIAL_bool) {

    this->m_USB_driver_LCR_sensors_session=session;
    this->m_serial_driver_LCR_sensors_session=session_serial;
    this->m_communication_is_SERIAL_bool=communication_is_SERIAL_bool;

    //const char *space_label_xl = "          ";
    const char *space_label_small = "  ";

    //QString text_display;

    boardID_name_label = new QLabel("Board ID:");
    boardID_spinbox = new QSpinBox();
    boardID_spinbox->setMinimum(0);
    boardID_spinbox->setMaximum(255);

    config_boardID_dialog_layout = new QGridLayout();
    this->setLayout(config_boardID_dialog_layout);

    this->setWindowTitle("Config BoardID of Multiple-Sensor Interface");

    config_boardID_widget = new QWidget(this);
    config_boardID_widget_layout = new QGridLayout();
    config_boardID_widget->setLayout(config_boardID_widget_layout);

    config_boardID_dialog_layout->addWidget(config_boardID_widget);

    config_boardID_widget_layout->addWidget(boardID_name_label, 0, 0);
    config_boardID_widget_layout->addWidget(boardID_spinbox, 0, 1);

    button_widget = new QWidget(this);
    button_widget_layout = new QGridLayout();
    button_widget->setLayout(button_widget_layout);

    set_boardID_button = new QPushButton("Set BoardID", this);
    read_boardID_button = new QPushButton("Read BoardID", this);
    close_dialog_button = new QPushButton("Close", this);

    status_label = new QLabel(this);

    button_widget_layout->addWidget(new QLabel(space_label_small),0,0);
    button_widget_layout->addWidget(set_boardID_button,0,1);
    button_widget_layout->addWidget(new QLabel(space_label_small),0,2);
    button_widget_layout->addWidget(read_boardID_button,0,3);
    button_widget_layout->addWidget(new QLabel(space_label_small),0,4);
    button_widget_layout->addWidget(close_dialog_button,0,5);
    button_widget_layout->addWidget(new QLabel(space_label_small),0,6);

    config_boardID_dialog_layout->addWidget(button_widget);

    config_boardID_dialog_layout->addWidget(status_label);

    QObject::connect(set_boardID_button, SIGNAL( clicked(bool) ), this, SLOT( Send_New_BoardID() ) );
    QObject::connect(read_boardID_button, SIGNAL( clicked(bool) ), this, SLOT(Read_BoardID()) );
    QObject::connect(close_dialog_button, SIGNAL( clicked(bool) ), this, SLOT(Close_Dialog()) );


    this->show();

}

Config_BoardID_Dialog::~Config_BoardID_Dialog() {
    this->close();
}


void Config_BoardID_Dialog::Close_Dialog() {
    this->reject();
}

void Config_BoardID_Dialog::Set_Status_Label_Error() {
    status_label->setText("ERROR: Error when updating current boardID.");
}


void Config_BoardID_Dialog::Set_Status_Label_OK() {
    status_label->setText("OK: Current configurations updated.");
}

void Config_BoardID_Dialog::Read_BoardID() {
    unsigned char board_ID;

    if(m_communication_is_SERIAL_bool==1) {

        if( GetBoardIDCode_Serial_MultipleSensor(m_serial_driver_LCR_sensors_session, &board_ID) < 0) {
            //print error dialog
            this->Set_Status_Label_Error();
        }
        else {
            boardID_spinbox->setValue((unsigned int) board_ID);
            this->Set_Status_Label_OK();
        }

    }
    else {

        if( GetBoardIDCode_USB_MultipleSensor(m_USB_driver_LCR_sensors_session, &board_ID) < 0) {
            //print error dialog
            this->Set_Status_Label_Error();
        }
        else {
           boardID_spinbox->setValue((unsigned int) board_ID);
           this->Set_Status_Label_OK();
        }

    }

}


void Config_BoardID_Dialog::Send_New_BoardID() {

    int board_ID;

    board_ID = this->boardID_spinbox->value();

    if(m_communication_is_SERIAL_bool==1) {

        if( SetBoardIDCode_Serial_MultipleSensor(m_serial_driver_LCR_sensors_session, board_ID) < 0) {
            //print error dialog
            this->Set_Status_Label_Error();
        }
        else {
            // OK sent BoardID
            this->Set_Status_Label_OK();
        }

    }
    else {


        if(  SetBoardIDCode_USB_MultipleSensor(m_USB_driver_LCR_sensors_session, board_ID) < 0) {
            //print error dialog
            this->Set_Status_Label_Error();
        }
        else {
            // OK sent BoardID
            this->Set_Status_Label_OK();
        }

    }


}


// ----- Functions of the class Config_BoardID_Dialog  -  END -----



// ----- Functions of the class Transfer_Calib_Table_Dialog  -  START -----

Transfer_Calib_Table_Dialog::Transfer_Calib_Table_Dialog(QWidget *parent, Worker_Arguments *worker_args_ptr, char flag_transfer_mode_bool) : QDialog(parent) {

    this->m_worker_args_ptr = worker_args_ptr;
    this->m_flag_transfer_mode_bool = flag_transfer_mode_bool;

    // ----- DEBUG CODE -----
  /*  char msg_str[200];
    sprintf(msg_str, "@ Transfer_Calib_Table_Dialog(...)");
    QMessageBox msgBox3;
    msgBox3.setText(msg_str);
    msgBox3.exec(); */
    // ----------------------

    qRegisterMetaType< QList<Worker_Arguments> >( "QList<Worker_Arguments>" );
    qRegisterMetaType<Data>( "Data" );

    worker = new Worker_Handler(this);

    // Progress Dialog bar variables
    m_max_N_steps_progress_bar = CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE*(N_LCR_CHANNELS+N_ADC_CHANNELS);
    m_current_progress_bar = 0;

    //progress_dialog = new QDialog(this);
    progress_dialog_layout = new QGridLayout();
    this->setLayout(progress_dialog_layout);

    this->setMinimumSize(400, 150);

    title_widget = new QWidget(this);
    title_widget_layout = new QGridLayout();
    title_widget->setLayout(title_widget_layout);

    title_label = new QLabel(this);
    title_label->setMinimumSize(300, 20);
    title_widget_layout->addWidget(title_label, 0,0);

    progress_dialog_layout->addWidget(title_widget, 0, 0);

    status_widget = new QWidget(this);
    status_widget_layout = new QGridLayout();
    status_widget->setLayout(status_widget_layout);

    status_widget->setMinimumSize(390, 85);

    progress_bar = new QProgressBar(this);
    process_label = new QLabel(this);
    status_label = new QLabel(this);
    QLabel *space_label = new QLabel(this);

    space_label->setText("              ");

    progress_bar->setMinimumSize(175, 20);
    process_label->setMinimumSize(210, 40);
    status_label->setMinimumSize(200, 20);

    status_widget_layout->addWidget(process_label, 0,0);
    status_widget_layout->addWidget(space_label, 0, 1);
    status_widget_layout->addWidget(progress_bar, 0, 2);
    status_widget_layout->addWidget(status_label, 1, 0);

    progress_dialog_layout->addWidget(status_widget, 1, 0);

    button_widget = new QWidget(this);
    button_widget_layout = new QGridLayout();
    button_widget->setLayout(button_widget_layout);

    if(this->m_flag_transfer_mode_bool==0) {
        title_label->setText("<b> - Download the calibration tables from device.</&b>");
        start_button = new QPushButton("Download", this);
    }
    if(this->m_flag_transfer_mode_bool==1) {
        title_label->setText("<b> - Upload the calibration tables to device.</&b>");
        start_button = new QPushButton("Upload", this);
    }

    pause_button = new QPushButton("Pause", this);
    stop_button = new QPushButton("Stop", this);

    button_widget_layout->addWidget(start_button, 0, 0);
    button_widget_layout->addWidget(pause_button, 0, 1);
    button_widget_layout->addWidget(stop_button, 0, 2);

    progress_dialog_layout->addWidget(button_widget, 2, 0);

    progress_bar->setVisible(false);
    process_label->setVisible(false);
    pause_button->setEnabled(false);
    stop_button->setEnabled(false);
    start_button->setEnabled(true);
    status_label->setText("STATUS:OK");
    status_label->setVisible(true);

    connect(worker, SIGNAL(max_progress_range_changed_signal(int)), progress_bar, SLOT(setMaximum(int)));
    connect(worker, SIGNAL(progress_text_changed_signal(QString)), process_label, SLOT(setText(QString)));
    connect(worker, SIGNAL(progress_value_changed_signal(int)), progress_bar, SLOT(setValue(int)));
    connect(worker, SIGNAL(status_text_changed_signal(QString)), status_label, SLOT(setText(QString)));
    connect(worker, SIGNAL(result_ready_signal(Data)), this, SLOT(Data_Ready(Data)));
    connect(worker, SIGNAL(finished_signal()), this, SLOT(Call_On_Process_Finished()));
    connect(pause_button, SIGNAL(clicked()), worker, SLOT(Toggle_Paused_Thread()));
    connect(stop_button, SIGNAL(clicked()), worker, SLOT(Cancel_Thread()));
    connect(start_button, SIGNAL(clicked()), this, SLOT(Start_Transfer_Calib_Table()));

}

Transfer_Calib_Table_Dialog::~Transfer_Calib_Table_Dialog() {
    this->close();
}

void Transfer_Calib_Table_Dialog::Data_Copy(Data *data_dest,  Data const &data_src) {

    data_dest->ret_value = data_src.ret_value;
    data_dest->error_counter = data_src.error_counter;
    strncpy( data_dest->first_error_message, data_src.first_error_message, MAX_N_CHARS_ERROR_MSG);
    strncpy( data_dest->last_error_message, data_src.last_error_message, MAX_N_CHARS_ERROR_MSG);
    strncpy( data_dest->ret_message_str, data_src.ret_message_str, MAX_N_CHARS_ERROR_MSG);

}

void Transfer_Calib_Table_Dialog::Start_Transfer_Calib_Table() {

    QList<Worker_Arguments> list;

    progress_bar->setVisible(true);
    process_label->setVisible(true);
    status_label->setVisible(true);
    list << *(this->m_worker_args_ptr);

    CloseDevice_Serial_MultipleSensor(m_worker_args_ptr->serial_driver_LCR_sensors_session_ptr);    //make this thread disconnect from serial port

    //Start a new Worker Thread for performing the upload/download between the PC and the device
    worker->Start_Work_Thread(list);
    pause_button->setEnabled(true);
    stop_button->setEnabled(true);
    start_button->setEnabled(false);

    status_label->setMinimumWidth(250);
    status_label->setWordWrap(true);
    status_label->setText("...");

}

void Transfer_Calib_Table_Dialog::Data_Ready(const Data data)
{
    Data_Copy(&(this->ret_data_worker), data);  //save the returned data of the worker thread (with the messages and error counter)
}

void Transfer_Calib_Table_Dialog::Call_On_Process_Finished()
{
    QString error_details;
    QString success_msg_str, error_msg_str;


    // ----- reopen the serial port so is available to the current thread -----
    if(this->m_worker_args_ptr->communication_is_SERIAL_bool==1) {
        OpenDevice_Serial_MultipleSensor(this->m_worker_args_ptr->serial_driver_LCR_sensors_session_ptr,  this->m_worker_args_ptr->serial_driver_LCR_sensors_session_ptr->port_name);  //reconnect to the serial port after the worker thread has finished the transfer
    }
    // -----------------------

    if( m_flag_transfer_mode_bool == 0) {
        success_msg_str = "SUCCESS: the calibrations tables were downloaded completely.\n";
        error_msg_str = "ERROR: (the following errors occurred during the\n              download of the calibration tables).\n";
    }
    else {
        success_msg_str = "SUCCESS: the calibrations tables were uploaded completely.\n";
        error_msg_str = "ERROR: (the following errors occurred during the\n              upload of the calibration tables).\n";
    }

   // ui->edLog->append("\nFinished ! ");
    progress_bar->setVisible(false);
    process_label->setVisible(false);
    pause_button->setEnabled(false);
    stop_button->setEnabled(false);
    start_button->setEnabled(true);

    if( ret_data_worker.error_counter == 0 && ret_data_worker.ret_value == THREAD_WORKER_RET_VAL_OK ) {
        status_label->setText(success_msg_str);
    }
    else {
        error_details.append(QString("ERROR_COUNT#: %1\n").arg(this->ret_data_worker.error_counter, 0, 10));
        error_details.append(QString("ERROR_MSG: %1\n").arg(this->ret_data_worker.first_error_message));
        error_details.append(QString("ERROR_MSG: %1\n").arg(this->ret_data_worker.last_error_message));
        error_msg_str.append(error_details);
        status_label->setText(error_msg_str);
        this->setMinimumHeight(350);
    }

    //remove the "Start" button and replace it by a "Close" button (close dialog box)
    start_button->deleteLater();
    this->layout()->removeWidget(start_button);

    close_button = new QPushButton("Close", this);
    button_widget_layout->addWidget(close_button, 0, 0);
    connect(close_button, SIGNAL(clicked()), this, SLOT(close()));

}

// ----- Functions of the class Transfer_Calib_Table_Dialog  -  END -----



// ----- Functions of the class MainWindow  -  START -----

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    int i;

    QString text_display;

    ui->setupUi(this);

    m_USB_driver_LCR_sensors_session.isConnected = FALSE_BOOL;    //the sensors board starts has not connected (USB)
    m_serial_driver_LCR_sensors_session.isConnected = FALSE_BOOL; //the sensors board starts has not connected (serial)

    previous_ch_number_for_saved_calib_dialog=0;  //set the initial selected channel on the calibration constants dialog

    error_log.append("ERROR LOG:\n");

    // ----- Toolbar for the LCR sensors application -----
    calibrate_device_sensors_action = new QAction(QIcon(":/images/balance_icon.png"), "Calibrate Device Sensors", this);
    QObject::connect(calibrate_device_sensors_action, SIGNAL( triggered(bool) ), this, SLOT( Create_Calibrate_Device_Sensors_Dialog() ) );
    ui->mainToolBar->addAction(calibrate_device_sensors_action);

    config_sensors_action = new QAction(QIcon(":/images/config_icon.png"), "Config Sensors Channels", this);
    QObject::connect(config_sensors_action, SIGNAL( triggered(bool) ), this, SLOT(Create_Config_Sensors_Dialog()) );
    ui->mainToolBar->addAction(config_sensors_action);

    config_outputs_action = new QAction(QIcon(":/images/square_wave_icon.png"), "Config Outputs", this);
    QObject::connect(config_outputs_action, SIGNAL( triggered(bool) ), this, SLOT(Create_Config_Outputs_Dialog()) );
    ui->mainToolBar->addAction(config_outputs_action);

    virtual_calibrate_sensors_action = new QAction(QIcon(":/images/virtual_calib_icon.png"), "Virtual Calibrate LCR Sensors", this);
    QObject::connect(virtual_calibrate_sensors_action, SIGNAL( triggered(bool) ), this, SLOT( Create_Virtual_Calibrate_Sensors_Dialog() ) );
    ui->mainToolBar->addAction(virtual_calibrate_sensors_action);

    config_boardID_action = new QAction(QIcon(":/images/ID_icon.png"), "Config BoardID", this);
    QObject::connect(config_boardID_action, SIGNAL( triggered(bool) ), this, SLOT(Create_Config_BoardID_Dialog()) );
    ui->mainToolBar->addAction(config_boardID_action);
    //----------------------------------------------------

    //initialize the pointers of the model_tables_device_calib_vector
    for(i=0;i<N_LCR_CHANNELS+N_ADC_CHANNELS;i++) {
        model_tables_device_calib_vector[i]=NULL;
    }
    //initialize the pointers of the model_tables_virtual_calib_vector
    for(i=0;i<N_LCR_CHANNELS+N_ADC_CHANNELS;i++) {
        model_tables_virtual_calib_vector[i]=NULL;
    }

    // ----- Menu for the LCR sensors application -----

    //connect the action DownloadDeviceCalibrations
    QObject::connect(ui->action_Download_Device_Calibrations, SIGNAL(triggered()), this, SLOT(Download_Device_Calib_Table()));

    //connect the action UploadDeviceCalibrations
    QObject::connect(ui->action_Upload_Device_Calibrations, SIGNAL(triggered()), this, SLOT(Upload_Device_Calib_Table()) );

    //connect the action Import_Device_Calibrations
    QObject::connect(ui->action_Import_Device_Calibrations, SIGNAL(triggered()), this, SLOT(Import_Device_Calibrations()));

    //connect the action Export_Device_Calibrations
    QObject::connect(ui->action_Export_Device_Calibrations, SIGNAL(triggered()), this, SLOT(Export_Device_Calibrations()) );

    //connect the action Import_Virtual_Calibrations
    QObject::connect(ui->action_Import_Virtual_Calibrations, SIGNAL(triggered()), this, SLOT(Import_Virtual_Calibrations()));

    //connect the action Export_Virtual_Calibrations
    QObject::connect(ui->action_Export_Virtual_Calibrations, SIGNAL(triggered()), this, SLOT(Export_Virtual_Calibrations()) );

    //show the about dialog of the LCR Sensors App
     QObject::connect(ui->action_About, SIGNAL(triggered()), this, SLOT(About()) );

    //show the Error Log dialog of the LCR Sensors App
    QObject::connect(ui->action_Show_Error_Log, SIGNAL(triggered()), this, SLOT(Show_Error_Log()) );

    //show the User Guide dialog of the LCR Sensors App
    QObject::connect(ui->action_Show_User_Guide, SIGNAL(triggered()), this, SLOT(Show_User_Guide()) );

    //quit the LCR Sensors App
    QObject::connect(ui->action_Quit, SIGNAL(triggered()), this, SLOT(close()) );
    // ------------------------------------------------

    Sensors_Values_groupBox_layout = new QGridLayout();
    ui->Sensors_Values_groupBox->setLayout(Sensors_Values_groupBox_layout);

    //layout for the Measurements_2 tab
    virtual_calib_Sensors_Values_groupBox_layout = new QGridLayout();
    ui->virtual_calib_measurements_groupBox->setLayout(virtual_calib_Sensors_Values_groupBox_layout);

    Outputs_Values_groupBox_layout = new QGridLayout();
    ui->Outputs_Values_groupBox->setLayout(Outputs_Values_groupBox_layout);

    //set the size of the text display sensors values
    font_display_sensors= new QFont("Times");
    font_display_sensors->setPointSize(8);
    font_display_sensors->setBold(true);

    sensors_graphic_view_groupBox_layout = new QGridLayout();
    ui->Sensors_Graphic_View_groupBox->setLayout(sensors_graphic_view_groupBox_layout);

    //layout for the Measurements_3 tab
    counters_virtual_calib_Sensors_Values_groupBox_layout = new QGridLayout();
    ui->counters_virtual_calibs_groupBox->setLayout(counters_virtual_calib_Sensors_Values_groupBox_layout);

    //fill the select_CH_mode_comboBox, place the default selection of multiple/single CHs on multiple
    this->ui->select_CH_mode_comboBox->addItem("Multiple Simultaneous CHs");
    this->ui->select_CH_mode_comboBox->addItem("Single CH (better precision)");
    this->ui->select_CH_mode_comboBox->setCurrentIndex(0);

    QObject::connect(ui->select_CH_mode_comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(ModeMultipleCH_Changed()) );
    ModeMultipleCH_Changed();   //update the enable state of the channel number selection box

    //save the channel mode configuration when the save button is clicked
    QObject::connect(ui->save_mode_CH_pushButton, SIGNAL(clicked()), this, SLOT(SaveModeMultipleCH()));

    //use of the measured values reading isn't compatible with the Virtual Calibration Measurements, so deactivate the Virtual Calibration Measurements when read_measurements is selected
    QObject::connect( ui->measured_radioButton, SIGNAL(pressed()), this, SLOT(Disable_Virtual_Calib_Measurements()) );

    QObject::connect( ui->enable_virtual_calib_checkBox, SIGNAL(pressed()), this, SLOT(EnableRAWValues()) );

    //update the sensors display when is clicked the Update_Values_Now_Button
    QObject::connect( ui->Update_Values_Now_Button, SIGNAL(clicked(bool)), this, SLOT(Call_Update_Sensors_Display()) );

    //reset the sensor counters when is clicked the Reset_Counters_Now_Button
    QObject::connect( ui->Reset_Counters_Now_Button, SIGNAL(clicked(bool)), this, SLOT(Reset_Counters()) );

    //place the default selection of sensor value type  on RAW values (frequency in Hz)
    ui->RAW_radioButton->setChecked(true);
    ui->measured_radioButton->setChecked(false);

    ui->USB_radioButton->setChecked(true);  //set the default choice for the communication protocol
    //ui->serial_radioButton->setChecked(true);  //if working in linux set Serial Port the default choice for the communication protocol, because USB driver of Multiple Sensor wasn't implemented for Linux

    timer_read_sensors = new QTimer(this);
    timer_read_sensors->setInterval(300);
    //every time the read sensors timer expires call the Update_Sensors_Display
    QObject::connect(timer_read_sensors, SIGNAL( timeout() ), this, SLOT(Call_Update_Sensors_Display()));

    //every time the Select_Continuous_Read_CheckBox is changed update the read mode of the sensors
    QObject::connect(ui->Select_Continuous_Read_CheckBox, SIGNAL( stateChanged(int)), this, SLOT(Change_Sensors_Read_Mode()));

    // QStandardItemModel excel table used to show the virtual calibration data through the QTableView QT table
    for(i=0; i<N_LCR_CHANNELS+N_ADC_CHANNELS; i++) {
        model_tables_device_calib_vector[i] = new QStandardItemModel(1, 2);
    }
    for(i=0; i<N_LCR_CHANNELS+N_ADC_CHANNELS; i++) {
        model_tables_virtual_calib_vector[i] = new QStandardItemModel(1, 2);
    }


    space_label = new QLabel("  ", this);
    long_space_label = new QLabel("                    .", this);

    for(i=0;i<N_LCR_CHANNELS+N_ADC_CHANNELS+2;i++) {
        sensors_names_display_vector[i] = new QLabel(this);
        sensors_values_display_vector[i] = new QLabel(this);
        sensors_names_display_vector[i]->setMaximumWidth(90);
        Sensors_Values_groupBox_layout->addWidget(sensors_names_display_vector[i], i,0);
        Sensors_Values_groupBox_layout->addWidget(sensors_values_display_vector[i], i,1);
        Sensors_Values_groupBox_layout->addWidget(long_space_label, i,2);
        Sensors_Values_groupBox_layout->addWidget(long_space_label, i,3);

        sensors_values_display_vector[i]->setMinimumWidth(400);
    }


    for(i=0;i<N_DIGITAL_OUT;i++) {
        outputs_names_display_vector[i] = new QLabel(this);
        outputs_values_display_vector[i] = new QLabel(this);
        Outputs_Values_groupBox_layout->addWidget(outputs_names_display_vector[i], i,0);
        Outputs_Values_groupBox_layout->addWidget(outputs_values_display_vector[i], i,1);
        Outputs_Values_groupBox_layout->addWidget(space_label, i,2);
        Outputs_Values_groupBox_layout->addWidget(space_label, i,3);
    }



    for(i=0;i<N_LCR_CHANNELS+N_ADC_CHANNELS+2;i++) {
        virtual_calib_sensors_names_display_vector[i] = new QLabel(this);
        virtual_calib_sensors_values_display_vector[i] = new QLabel(this);
        virtual_calib_Sensors_Values_groupBox_layout->addWidget(virtual_calib_sensors_names_display_vector[i], i,0);
        virtual_calib_Sensors_Values_groupBox_layout->addWidget(virtual_calib_sensors_values_display_vector[i], i,1);
        virtual_calib_Sensors_Values_groupBox_layout->addWidget(space_label, i,2);
        virtual_calib_Sensors_Values_groupBox_layout->addWidget(space_label, i,3);
    }



    for(i=0;i<N_LCR_CHANNELS+1;i++) {
        counters_virtual_calib_sensors_names_display_vector[i] = new QLabel(this);
        counters_virtual_calib_sensors_values_display_vector[i] = new QLabel(this);
        counters_virtual_calib_Sensors_Values_groupBox_layout->addWidget(counters_virtual_calib_sensors_names_display_vector[i], i,0);
        counters_virtual_calib_Sensors_Values_groupBox_layout->addWidget(counters_virtual_calib_sensors_values_display_vector[i], i,1);
        counters_virtual_calib_Sensors_Values_groupBox_layout->addWidget(space_label, i,2);
        counters_virtual_calib_Sensors_Values_groupBox_layout->addWidget(space_label, i,3);
    }


    //initialize the vector were the virtual calib units are saved
    for(i=0;i<(N_LCR_CHANNELS+N_ADC_CHANNELS);i++) {
        strcpy(virtual_calib_units_vector[i],"#");
    }

    //declare/construct the variables for the Sensors Graphic View
    for(i=0;i<N_LCR_CHANNELS+N_ADC_CHANNELS;i++) {
        sensors_graphic_view_names_display_vector[i] = new QLabel(this);
        sensors_graphic_view_select_min_vector[i] = new QDoubleSpinBox(this);
        sensors_graphic_view_select_max_vector[i] = new QDoubleSpinBox(this);
        sensors_progress_bar_vector[i] = new QProgressBar(this);
    }

    //declare the signalMappers for the minimum and maximum values of the sensors graph display
    QSignalMapper *signalMapper_minimum_sensor_graph = new QSignalMapper(this);
    QSignalMapper *signalMapper_maximum_sensor_graph = new QSignalMapper(this);

    //fill the title for each channel graph
    for(i=0;i<N_LCR_CHANNELS;i++) {
        sensors_graphic_view_names_display_vector[i]->setFont(*font_display_sensors);
        text_display.clear();
        QTextStream( &text_display ) << "CH." << i << ": ";
        sensors_graphic_view_names_display_vector[i]->setText(text_display);
    }

    //fill the title for each ADC channel graph
    for(i=0;i<N_ADC_CHANNELS;i++) {
        sensors_graphic_view_names_display_vector[N_LCR_CHANNELS+i]->setFont(*font_display_sensors);
        text_display.clear();
        QTextStream( &text_display ) << "ADC CH." << i << ": ";
        sensors_graphic_view_names_display_vector[N_LCR_CHANNELS+i]->setText(text_display);
    }

    //fill the select input list for each channel graph
    for(i=0;i<N_LCR_CHANNELS+N_ADC_CHANNELS;i++) {

        sensors_progress_bar_vector[i]->setMinimumWidth(140);  //set the size of the graph view of the sensor

        sensors_graphic_view_groupBox_layout->addWidget(sensors_graphic_view_names_display_vector[i], i,0);
        sensors_graphic_view_groupBox_layout->addWidget(sensors_progress_bar_vector[i], i, 1);
        sensors_graphic_view_groupBox_layout->addWidget(space_label, i, 2);
        sensors_graphic_view_groupBox_layout->addWidget(space_label, i, 3);
        sensors_graphic_view_groupBox_layout->addWidget(sensors_graphic_view_select_min_vector[i], i, 4);
        sensors_graphic_view_groupBox_layout->addWidget(sensors_graphic_view_select_max_vector[i], i, 5);

        sensors_graphic_view_select_max_vector[i]->setMaximum(9999999999);
        sensors_graphic_view_select_min_vector[i]->setMaximum(9999999999);
        sensors_graphic_view_select_max_vector[i]->setMinimum(-9999999999);
        sensors_graphic_view_select_min_vector[i]->setMinimum(-9999999999);

        sensors_graphic_view_select_min_vector[i]->setLocale(QLocale::C);    //configure the QDoubleSpinBox to use '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations
        sensors_graphic_view_select_max_vector[i]->setLocale(QLocale::C);    //configure the QDoubleSpinBox to use '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations


        sensors_graphic_view_select_min_vector[i]->setMaximumWidth(70);
        sensors_graphic_view_select_max_vector[i]->setMaximumWidth(70);

        signalMapper_minimum_sensor_graph->setMapping(sensors_graphic_view_select_min_vector[i], i);
        signalMapper_maximum_sensor_graph->setMapping(sensors_graphic_view_select_max_vector[i], i);

        QObject::connect(sensors_graphic_view_select_min_vector[i], SIGNAL(valueChanged(double)), signalMapper_minimum_sensor_graph, SLOT (map()));
        QObject::connect(sensors_graphic_view_select_max_vector[i], SIGNAL(valueChanged(double)), signalMapper_maximum_sensor_graph, SLOT (map()));

        QObject::connect(signalMapper_minimum_sensor_graph, SIGNAL(mapped(int)), this, SLOT(Check_Max_Graph_View(int)));
        QObject::connect(signalMapper_maximum_sensor_graph, SIGNAL(mapped(int)), this, SLOT(Check_Min_Graph_View(int)));

    }

    //clear the contents of the sensor counters device calibration
    for(i=0;i<N_LCR_CHANNELS+N_ADC_CHANNELS;i++) {
        counters_device_calib.counters_calib_C3_vector[i]=0;
        counters_device_calib.counters_calib_C2_vector[i]=0;
        counters_device_calib.counters_calib_C1_vector[i]=0;
        counters_device_calib.counters_calib_C0_vector[i]=0;
        counters_device_calib.counters_calib_units[i][0]='\0';
    }

    //clear the contents of the sensor counters virtual calibration
    for(i=0;i<N_LCR_CHANNELS;i++) {
        counters_virtual_calib.counters_calib_C3_vector[i]=0;
        counters_virtual_calib.counters_calib_C2_vector[i]=0;
        counters_virtual_calib.counters_calib_C1_vector[i]=0;
        counters_virtual_calib.counters_calib_C0_vector[i]=0;
        counters_virtual_calib.counters_calib_units[i][0]='\0';
    }

    ui->select_CH_spinBox->setMaximum(N_LCR_CHANNELS+N_ADC_CHANNELS-1);     //only allow to select a number correspondent to an available sensor channel (0, 1, 2, ..., 9)

    //disable the buttons of various functions that are only possible after connecting to the sensors board
    ui->Reading_Mode->setDisabled(true);
    ui->select_multiple_CH_groupBox->setDisabled(true);
    ui->Reset_Counters_Now_Button->setDisabled(true);

    // ...
    // ---------------------------------------------------

}

MainWindow::~MainWindow()
{
//    int i;

    if(ui->serial_radioButton->isChecked()==true) {
        CloseDevice_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session);
    }
    else {
        CloseDevice_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session);
    }

    delete ui;
}

void MainWindow::SetConnectioStatusLabel_OK() {
    ui->connect_status_label->setText("<b>"+(tr("OK"))+"</b>");
    ui->connect_status_label->setStyleSheet("QLabel { background-color : green; color : black; }");
}

void MainWindow::SetConnectioStatusLabel_ERROR() {
    ui->connect_status_label->setText("<b>"+(tr("ERROR"))+"</b>");
    ui->connect_status_label->setStyleSheet("QLabel { background-color : red; color : black; }");
}


void MainWindow::About()
{

    About_Dialog_Impl *about_dialog_ptr = new About_Dialog_Impl(this, VERSION_MULTIPLE_SENSOR_GUI);

    about_dialog_ptr->show();
}

void MainWindow::Show_Error_Log() {

    //QScrollArea scroll_area;
    QTextEdit *text_view_error_log = new QTextEdit(this);

    //sprintf(msg_str2,"SERIAL_connect_status: %s", msg_str);
    QDialog *show_error_log_dialog = new QDialog(this);
    QPushButton *OK_button = new QPushButton(this);
    OK_button->setText("Close");

    QGridLayout *show_error_log_dialog_layout = new QGridLayout();

    show_error_log_dialog->setLayout(show_error_log_dialog_layout);
    show_error_log_dialog_layout->addWidget(text_view_error_log, 0, 0);

    show_error_log_dialog->setWindowTitle("ERROR LOG - Multiple Sensor Interface");

    QWidget *button_widget = new QWidget(this);
    QGridLayout *button_widget_layout = new QGridLayout();
    button_widget->setLayout(button_widget_layout);

    button_widget_layout->addWidget(new QLabel("    "),0,0);
    button_widget_layout->addWidget(OK_button,0,1);

    show_error_log_dialog_layout->addWidget(button_widget, 1,0);

    QObject::connect(OK_button, SIGNAL(clicked()), show_error_log_dialog, SLOT(accept()) );

    text_view_error_log->setText(error_log);
    text_view_error_log->setReadOnly(true);

    //show_error_log_Box.setText( error_log );
    show_error_log_dialog->exec();
}

void MainWindow::Show_User_Guide()
{
   QString path= "user_guide/user_guide.html";
   QUrl pathUrl = QUrl::fromLocalFile(path);
   QDesktopServices::openUrl(pathUrl);
}

void MainWindow::Change_Sensors_Read_Mode() {

    if( ui->Select_Continuous_Read_CheckBox->isChecked() == true ) {
        timer_read_sensors->start(INTERVAL_MSEC_TIMER_READ_SENSOR); //activate the timer to read the sensors
        ui->Update_Values_Now_Button->setDisabled(true);

        //disable the select read channel mode widget when reading automatically the sensor measurements
        ui->select_multiple_CH_groupBox->setDisabled(true);
    }
    else {
        timer_read_sensors->stop(); //deactivate the timer to read the sensors
        ui->Update_Values_Now_Button->setDisabled(false);

        //re-enable the select read channel mode widget
        ui->select_multiple_CH_groupBox->setDisabled(false);
    }
}

void MainWindow::Disable_Virtual_Calib_Measurements() {
    ui->enable_virtual_calib_checkBox->setChecked(false);
}

void MainWindow::EnableRAWValues() {
    //the Measurements with the Virtual Calibration can only work when reading the RAW values, so RAW_values Reading GroupBox
    this->ui->measured_radioButton->setChecked(false);
    this->ui->RAW_radioButton->setChecked(true);
}

//update the minimum allowed value of the select maximum for graph view
void MainWindow::Check_Max_Graph_View(int CH_number) {
    sensors_graphic_view_select_max_vector[CH_number]->setMinimum(sensors_graphic_view_select_min_vector[CH_number]->value());
}

//update the maximum allowed value of the select minimum for graph view
void MainWindow::Check_Min_Graph_View(int CH_number) {
    sensors_graphic_view_select_min_vector[CH_number]->setMaximum(sensors_graphic_view_select_max_vector[CH_number]->value());
}

void MainWindow::Connect_LCR_Board()
{
    int connect_status, operation_ret_value, channel_number;
    char msg_str[100], COM_port_name[100];

    //check the select communication type
    if(ui->serial_radioButton->isChecked()==true) {
        strcpy(COM_port_name,ui->serial_port_name_lineEdit->text().toUtf8().data());
        m_serial_driver_LCR_sensors_session.board_ID = ui->board_ID_lineEdit->text().toInt();

        if(m_serial_driver_LCR_sensors_session.isConnected==FALSE_BOOL) {
            connect_status = OpenDevice_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, COM_port_name);
            if(connect_status == SERIAL_CONNECTED) {
                ui->Connect_LCR_Board_Button->setText("Disconnect");
                sprintf(msg_str, "%s Opened", COM_port_name);
                ui->USB_status_display->setText(msg_str);
                ui->Reading_Mode->setEnabled(true);  //Enable the reading of the sensors if connected by serial(RS232) to the device
                MainWindow::SetConnectioStatusLabel_OK();

                //also save the headers of calibration tables from device to be able to show read measurements with units
                for(channel_number=0; channel_number < N_LCR_CHANNELS+N_ADC_CHANNELS; channel_number++) {
                    operation_ret_value = ReadHeaderCalibTable_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, channel_number, &((downloaded_device_calib_header_data.calib_table_is_active_bool)[channel_number]), &((downloaded_device_calib_header_data.number_table_lines)[channel_number]), &((downloaded_device_calib_header_data.calib_channel_mode_bool)[channel_number]) , &((downloaded_device_calib_header_data.jumper_selection_osc_tuning_range)[channel_number]), (downloaded_device_calib_header_data.device_calib_units_vector)[channel_number], (downloaded_device_calib_header_data.counters_calib_units_vector)[channel_number]);
                    if( operation_ret_value > 0 ) {
                        MainWindow::SetConnectioStatusLabel_OK();
                    }
                    else {
                        sprintf(msg_str, "ERROR: error downloading headers of calib tables by serial RS485.\n");
                        error_log.append(msg_str); //ERROR occurred, post to error log
                        MainWindow::SetConnectioStatusLabel_ERROR();
                    }
                }

                //enable the buttons of various functions that are now possible after connecting to the sensors board
                ui->Reading_Mode->setEnabled(true);
                ui->select_multiple_CH_groupBox->setEnabled(true);
                ui->Reset_Counters_Now_Button->setEnabled(true);

            }
            else {                
                sprintf(msg_str, "ERROR opening %s.\n", COM_port_name);
                error_log.append(msg_str); //ERROR occurred, post to error log
                MainWindow::SetConnectioStatusLabel_ERROR();
            }
        }
        else {
            //disconnect from Serial board Action
            ui->Select_Continuous_Read_CheckBox->setChecked(false); //before disconnecting uncheck the "Continuous Read" box
            CloseDevice_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session);
            ui->Connect_LCR_Board_Button->setText("Connect");
            ui->USB_status_display->setText("Not Connected");

            //disable the buttons of various functions that are only possible after connecting to the sensors board
            ui->Reading_Mode->setDisabled(true);
            ui->select_multiple_CH_groupBox->setDisabled(true);
            ui->Reset_Counters_Now_Button->setDisabled(true);
        }

    }
    else {

        if(m_USB_driver_LCR_sensors_session.isConnected==FALSE_BOOL) {
            m_USB_driver_LCR_sensors_session.board_ID=255;  //always use the broadcast board_ID ('255') when using USB interface, because isn't possible to use 2 Multiple-Sensor devices simultaniously on the same PC by USB,
                                                      //this is because the USB driver always connects to the device identified by its PID (USB - product ID), and so always receives a reply from the devices connected by USB driver with corresponding PID (if the Multiple-Sensor board_ID is correct receives the reply, otherwise receives a buffer with trash data)
            connect_status = OpenDevice_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session);
            if(connect_status == USB_CONNECTED) {
                ui->Connect_LCR_Board_Button->setText("Disconnect");
                ui->USB_status_display->setText("Connected");
                ui->Reading_Mode->setEnabled(true);  //Enable the reading of the sensors if connected by USB to the device
                MainWindow::SetConnectioStatusLabel_OK();

                //also save the headers of calibration tables from device to be able to show read measurements with units
                for(channel_number=0; channel_number < N_LCR_CHANNELS+N_ADC_CHANNELS; channel_number++) {
                    operation_ret_value = ReadHeaderCalibTable_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, channel_number, &((downloaded_device_calib_header_data.calib_table_is_active_bool)[channel_number]), &((downloaded_device_calib_header_data.number_table_lines)[channel_number]), &((downloaded_device_calib_header_data.calib_channel_mode_bool)[channel_number]), &((downloaded_device_calib_header_data.jumper_selection_osc_tuning_range)[channel_number]), (downloaded_device_calib_header_data.device_calib_units_vector)[channel_number], (downloaded_device_calib_header_data.counters_calib_units_vector)[channel_number]);
                    if( operation_ret_value > 0 ) {
                        MainWindow::SetConnectioStatusLabel_OK();
                    }
                    else {
                        sprintf(msg_str, "ERROR: error downloading headers of calib tables by USB.\n");
                        error_log.append(msg_str); //ERROR occurred, post to error log
                        MainWindow::SetConnectioStatusLabel_ERROR();
                    }
                }

                //enable the buttons of various functions that are now possible after connecting to the sensors board
                ui->Reading_Mode->setEnabled(true);
                ui->select_multiple_CH_groupBox->setEnabled(true);
                ui->Reset_Counters_Now_Button->setEnabled(true);

            }
            else {
                sprintf(msg_str, "ERROR 'OpenDevice' by USB.\n");
                error_log.append(msg_str); //ERROR occured, post to error log
                MainWindow::SetConnectioStatusLabel_ERROR();
            }
        }
        else {
            //disconnect from USB board Action
            ui->Select_Continuous_Read_CheckBox->setChecked(false); //before disconnecting uncheck the "Continuous Read" box
            CloseDevice_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session);
            ui->Connect_LCR_Board_Button->setText("Connect");
            ui->USB_status_display->setText("Not Connected");

            //disable the buttons of various functions that are only possible after connecting to the sensors board
            ui->Reading_Mode->setDisabled(true);
            ui->select_multiple_CH_groupBox->setDisabled(true);
            ui->Reset_Counters_Now_Button->setDisabled(true);
        }

        sprintf(msg_str,"USB_connect_status: %d.\n", connect_status);
        QMessageBox msgBox;
        msgBox.setText(msg_str);
        msgBox.exec();

    }

    //set previous select_CH_mode to assure the select_CH_mode will be updated
    select_CH_mode_comboBox_previous_value=-1;
    select_CH_spinBox_previous_value=-1;

}


void MainWindow::ModeMultipleCH_Changed() {

    if( (this->ui->select_CH_mode_comboBox->currentIndex())==0 ) {
        //is selected the mode multiple simultaneous channels
        this->ui->select_CH_spinBox->setDisabled(true); 
    }
    else {
        //is selected the mode single channels
        this->ui->select_CH_spinBox->setDisabled(false);
    }

}

void MainWindow::SaveModeMultipleCH() {
    int i;

    //clear the current values shown on the LCR sensors groupBox
    for(i=0;i<N_LCR_CHANNELS;i++) {
        sensors_values_display_vector[i+1]->clear();
    }

    //clear the current values shown on the ADC sensors groupBox
    for(i=0;i<N_ADC_CHANNELS;i++) {
        sensors_values_display_vector[i+N_LCR_CHANNELS+2]->clear();
    }

    if(ui->serial_radioButton->isChecked()==true) {
        //Communicating with the Multiple-Sensor Interface by Serial Port RS232
        if( (this->ui->select_CH_mode_comboBox->currentIndex())==0 ) {
            //is selected the mode multiple simultaneous channels
             SaveModeCHConfig_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, 0, 0);
        }
        else {
            //is selected the mode single channels
            SaveModeCHConfig_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, 1, this->ui->select_CH_spinBox->value());
         }

    }
    else {
        if( (this->ui->select_CH_mode_comboBox->currentIndex())==0 ) {
            //is selected the mode multiple simultaneous channels
             SaveModeCHConfig_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, 0, 0);
        }
        else {
            //is selected the mode single channels
             SaveModeCHConfig_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, 1, this->ui->select_CH_spinBox->value());
        }
    }

    //save the previous values of select CH mode
    select_CH_mode_comboBox_previous_value=this->ui->select_CH_mode_comboBox->currentIndex();
    select_CH_spinBox_previous_value=this->ui->select_CH_spinBox->value();

}

void MainWindow::Reset_Counters() {
    int return_value;
    QString error_msg;

    //update all the outputs values
    if(ui->serial_radioButton->isChecked()==true) {
        return_value = ResetSensorCounters_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session);
         if( return_value < 0 ) {
             error_msg = "ERROR: resetting the sensor counters. (Reset_Sensor_Counters_Serial_LCR by Serial Port).\n";
             MainWindow::SetConnectioStatusLabel_ERROR();
         }
         else {
             MainWindow::SetConnectioStatusLabel_OK();
         }
    }
    else {
        return_value = ResetSensorCounters_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session);
        if( return_value < 0 ) {
            error_msg = "ERROR: resetting the sensor counters. (Reset_Sensor_Counters by USB).\n";
            MainWindow::SetConnectioStatusLabel_ERROR();
        }
        else {
            MainWindow::SetConnectioStatusLabel_OK();
        }
    }

    if( return_value < 0 ) {
        //an error occurred while resetting the sensor counters (set to zero)
        error_log.append(error_msg); //ERROR occurred, post to error log
        MainWindow::SetConnectioStatusLabel_ERROR();
    }

}


void MainWindow::Call_Update_Sensors_Display() {

    unsigned char mode_is_single_channel_bool;

    int selected_channel_number;

    //update the current mode selected for the LCR sensors channels, before reading the sensors
    if(this->ui->select_CH_mode_comboBox->currentIndex()!=select_CH_mode_comboBox_previous_value || this->ui->select_CH_spinBox->value()!=select_CH_spinBox_previous_value) {

        if(ui->serial_radioButton->isChecked()==true) {
            ReadModeCHConfig_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, &mode_is_single_channel_bool, &selected_channel_number);
        }
        else {           
            ReadModeCHConfig_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, &mode_is_single_channel_bool, &selected_channel_number);
        }
        this->ui->select_CH_mode_comboBox->setCurrentIndex(mode_is_single_channel_bool);
        this->ui->select_CH_spinBox->setValue(selected_channel_number);


        //save the previous values of select CH mode
        select_CH_mode_comboBox_previous_value=this->ui->select_CH_mode_comboBox->currentIndex();
        select_CH_spinBox_previous_value=this->ui->select_CH_spinBox->value();


        //place a little delay, because calling Read_Frequency_Measurement(...) very close to ReadModeCHConfig(...) will cause an error
        QTimer::singleShot(100, this, SLOT(Update_Sensors_Display()));
    }
    else {
        this->Update_Sensors_Display();
    }

}

// ----- DEBUG CODE -----
// sprintf(msg_str, "DEBUG POS 3. return_value: %d\n", return_value);
// QMessageBox msgBox3;
// msgBox3.setText(msg_str);
// //msgBox3.setText("DEBUG POS 3.\n");
// msgBox3.exec();
// ----------------------


void MainWindow::Update_Sensors_Display()
{
    int i, return_value, return_value_2;
    QString text_display, msg_str;

    unsigned char outputs_vector[N_DIGITAL_OUT];

    int selected_channel_number;

    char units_sensors_vector[N_LCR_CHANNELS+N_ADC_CHANNELS][N_CHARS_CALIB_UNITS];
    char counters_units_sensors_vector[N_LCR_CHANNELS][N_CHARS_CALIB_UNITS];

    QLabel *LCR_sensors_label = new QLabel("LCR Sensors:");

    selected_channel_number = this->ui->select_CH_spinBox->value();

    sensors_names_display_vector[0]->setText("LCR Sensors:");
    sensors_names_display_vector[1]->setText("CH.0:");
    sensors_names_display_vector[2]->setText("CH.1:");
    sensors_names_display_vector[3]->setText("CH.2:");
    sensors_names_display_vector[4]->setText("CH.3:");
    sensors_names_display_vector[5]->setText("CH.4:");
    sensors_names_display_vector[6]->setText("CH.5:");

    sensors_names_display_vector[7]->setText("ADC Sensors:");
    sensors_names_display_vector[8]->setText("CH.6 (ADC.0 +-):");
    sensors_names_display_vector[9]->setText("CH.7 (ADC.1 +-):");
    sensors_names_display_vector[10]->setText("CH.8 (ADC.2 +):");
    sensors_names_display_vector[11]->setText("CH.9 (ADC.3 +):");


     // -----IMPORTANT - DON'T REMOVE - This code line avoids the application to crash in some PCs !! -----
     QMessageBox msgBoxB;     msgBoxB.setText("This code line avoids the application to crash in some PCs !!!.\n");
     // ----------------------

     QThread::usleep(10000);  //wait some mili-seconds between serial commands

     if(ui->RAW_radioButton->isChecked()==true) {   //check if is to read RAW values or measurements from the device

         if(ui->serial_radioButton->isChecked()==true) {
             // Read using serial port the counter values of the LCR sensor channels, counters are useful for sensor with output as frquency of a digital square wave
             ReadSensorCounters_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, counters_sensors_vector);
             //don't check for errors in the read of Sensors Counters, usually only mistaken the error status of "Connection State"
         }
         else {
            // Read using USB the counter values of the LCR sensor channels, counters are useful for sensor with output as frequency of a digital square wave
            ReadSensorCounters_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, counters_sensors_vector);
            //don't check for errors in the read of Sensors Counters, usually only mistaken the error status of "Connection State"
         }
         for(i=0; i < N_LCR_CHANNELS; i++) {    //convert the counters RAW values array to a float array to be displayed on the GUI
            counters_sensors_measurements_vector[i] = (float) counters_sensors_vector[i];
            strncpy(counters_units_sensors_vector[i], "events", 7);
         }

           if(ui->serial_radioButton->isChecked()==true) {
                return_value=ReadFrequencyMeasurement_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, frequency_sensors_vector);
                if(return_value < 0) {
                    msg_str = "ERROR: reading the LCR sensors. (Read_Frequency_Measurement_Serial_LCR by Serial Port).\n";
                    MainWindow::SetConnectioStatusLabel_ERROR();
                }
                else {
                    MainWindow::SetConnectioStatusLabel_OK();
                }
            }
            else {
                return_value=ReadFrequencyMeasurement_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, (long *) frequency_sensors_vector);
                if(return_value < 0) {
                    msg_str = "ERROR: reading the LCR sensors. (Read_Frequency_Measurement by USB).\n";
                    MainWindow::SetConnectioStatusLabel_ERROR();
                }
                else {
                    MainWindow::SetConnectioStatusLabel_OK();
                }
            }
            if( return_value < 0 ) {
                //an error occurred while reading the frequency from the sensors
                error_log.append(msg_str); //ERROR occurred, post to error log
                MainWindow::SetConnectioStatusLabel_ERROR();

            }
           else {

                if(this->ui->select_CH_mode_comboBox->currentIndex()==0) {

                    for(i=0;i<N_LCR_CHANNELS;i++) {
                        LCR_sensors_values_vector[i]=(float) frequency_sensors_vector[i];
                        strncpy(units_sensors_vector[i], "Hz", 3);
                        //printf("LCR_sensors_values_vector[%d]=%f", i, (float) frequency_sensors_vector[i]);   //DEBUG CODE
                    }
                }
                else {
                    selected_channel_number = this->ui->select_CH_spinBox->value();
                    LCR_sensors_values_vector[selected_channel_number]=(float) frequency_sensors_vector[selected_channel_number];
                    strncpy(units_sensors_vector[selected_channel_number], "Hz", 3);
                }
            }

     }
     else {

         if(ui->serial_radioButton->isChecked()==true) {
             // Read using serial port the counter values of the LCR sensor channels, counters are useful for sensor with output as frequency of a digital square wave
             ReadCountersMeasurements_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, counters_sensors_measurements_vector);
             //don't check for errors in the read of Sensors Counters, usually only mistaken the error status of "Connection State"
         }
         else {
            // Read using USB the counter values of the LCR sensor channels, counters are useful for sensor with output as frequency of a digital square wave
            ReadCountersMeasurements_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, counters_sensors_measurements_vector);
            //don't check for errors in the read of Sensors Counters, usually only mistaken the error status of "Connection State"
         }
         for(i=0; i < N_LCR_CHANNELS; i++) {    //copy the measurements units to display on the GUI
            strncpy(counters_units_sensors_vector[i], this->downloaded_device_calib_header_data.counters_calib_units_vector[i], N_CHARS_CALIB_UNITS-1);
         }

           if(ui->serial_radioButton->isChecked()==true) {
               return_value = ReadMeasurements_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, measurements_sensor_vector);
                if(return_value < 0) {
                    msg_str = "ERROR: reading the sensors measurements. (ReadMeasurements_Modbus_LCR by Serial Port).\n";
                    MainWindow::SetConnectioStatusLabel_ERROR();
                }
                else {
                    MainWindow::SetConnectioStatusLabel_OK();
                }
            }
            else {
                return_value = ReadMeasurements_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, measurements_sensor_vector);
                if(return_value < 0) {
                    msg_str = "ERROR: reading the sensors measurements. (ReadMeasurements by USB).\n";
                    MainWindow::SetConnectioStatusLabel_ERROR();
                }
                else {
                    MainWindow::SetConnectioStatusLabel_OK();
                }
            }
            if( return_value < 0 ) {
                //an error occurred while reading the frequency from the sensors
                error_log.append(msg_str); //ERROR occurred, post to error log
                MainWindow::SetConnectioStatusLabel_ERROR();

            }
            else {

                if(this->ui->select_CH_mode_comboBox->currentIndex()==0) {
                    for(i=0; i<N_LCR_CHANNELS; i++) {
                        LCR_sensors_values_vector[i]=(float) measurements_sensor_vector[i];
                        strncpy(units_sensors_vector[i], this->downloaded_device_calib_header_data.device_calib_units_vector[i], N_CHARS_CALIB_UNITS-1);
                    }                    
                }
                else {
                    selected_channel_number = this->ui->select_CH_spinBox->value();
                    LCR_sensors_values_vector[selected_channel_number]=(float) measurements_sensor_vector[selected_channel_number];
                    strncpy(units_sensors_vector[selected_channel_number], this->downloaded_device_calib_header_data.device_calib_units_vector[selected_channel_number], N_CHARS_CALIB_UNITS-1);
                }
                for(i=0; i<N_ADC_CHANNELS; i++) {
                    ADC_sensors_values_vector[i]=(float) measurements_sensor_vector[i+N_LCR_CHANNELS];
                    strncpy(units_sensors_vector[i+N_LCR_CHANNELS], this->downloaded_device_calib_header_data.device_calib_units_vector[i+N_LCR_CHANNELS], N_CHARS_CALIB_UNITS-1);
                }
            }

     }


     if( return_value >= 0 ) {

        //fill the data tab with the readings from the LCR sensors
        Sensors_Values_groupBox_layout->addWidget(LCR_sensors_label, 0,0);
        if(this->ui->select_CH_mode_comboBox->currentIndex()==0) {
            //multiple sensors channel mode selected
            for(i=0;i<N_LCR_CHANNELS;i++) {
                sensors_values_display_vector[i+1]->setFont(*font_display_sensors);
                text_display.clear();
                QTextStream( &text_display ) << LCR_sensors_values_vector[i] << " " << units_sensors_vector[i] << "               Counter: " << counters_sensors_measurements_vector[i] << " " << counters_units_sensors_vector[i];
                sensors_values_display_vector[i+1]->setText(text_display);
            }
        }
        else {
            //single sensor channel mode selected
            if(selected_channel_number < N_LCR_CHANNELS) {
                sensors_values_display_vector[selected_channel_number+1]->setFont(*font_display_sensors);
                selected_channel_number=this->ui->select_CH_spinBox->value();
                text_display.clear();
                QTextStream( &text_display ) << LCR_sensors_values_vector[selected_channel_number] << " " << units_sensors_vector[selected_channel_number] << "               Counter: " << counters_sensors_measurements_vector[selected_channel_number] << " " << counters_units_sensors_vector[selected_channel_number];
                sensors_values_display_vector[selected_channel_number+1]->setText(text_display);
            }
        }

    }

    //fill the data tab with the readings from the ADC channels in case of RAW_values mode
     if(ui->RAW_radioButton->isChecked()==true) {
         for(i=0; i<N_ADC_CHANNELS; i++) {
             strncpy(units_sensors_vector[i+N_LCR_CHANNELS], "V", 2);
         }
        if(ui->serial_radioButton->isChecked()==true) {
            return_value_2=ReadADCValue_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, ADC_sensors_values_vector);
        }
        else {
            return_value_2=ReadADCValue_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, ADC_sensors_values_vector);
        }
     }

    if( return_value_2 < 0 ) {
        if( return_value == USB_COMMAND_ERROR ) {
            //an error occurred while reading the frequency from the sensors
            error_log.append("ERROR: reading the LCR sensors (ReadADC by USB).\n"); //ERROR occurred, post to error log
        }
        if( return_value == SERIAL_COMMAND_ERROR ) {
            //an error occurred while reading the frequency from the sensors
            error_log.append("ERROR: reading the LCR sensors (ReadADC by Serial Port).\n"); //ERROR occurred, post to error log
        }
        MainWindow::SetConnectioStatusLabel_ERROR();
    }
    else {
        if(return_value >= 0)   //set label OK if read frequency also completed with no errors
            MainWindow::SetConnectioStatusLabel_OK();
    }

    if(return_value_2 >= 0) {

        if(this->ui->select_CH_mode_comboBox->currentIndex()==0) {
            //multiple sensors channel mode selected
            for(i=0;i<N_ADC_CHANNELS;i++) {
                sensors_values_display_vector[i+N_LCR_CHANNELS+2]->setFont(*font_display_sensors);
                text_display.clear();
                QTextStream( &text_display ) << ADC_sensors_values_vector[i] << " " << units_sensors_vector[i+N_LCR_CHANNELS];
                sensors_values_display_vector[i+N_LCR_CHANNELS+2]->setText(text_display);
            }
        }
        else {
            //single sensors channel mode selected
            if(selected_channel_number >= N_LCR_CHANNELS) {
                sensors_values_display_vector[selected_channel_number+2]->setFont(*font_display_sensors);
                selected_channel_number=this->ui->select_CH_spinBox->value();
                text_display.clear();
                QTextStream( &text_display ) << ADC_sensors_values_vector[selected_channel_number-N_LCR_CHANNELS] << " " << units_sensors_vector[selected_channel_number];
                sensors_values_display_vector[selected_channel_number+2]->setText(text_display);

            }

        }


        //update all the outputs values
        if(ui->serial_radioButton->isChecked()==true) {
            ReadOutputs_Serial_MultipleSensor(&m_serial_driver_LCR_sensors_session, outputs_vector);
        }
        else {
            ReadOutputs_USB_MultipleSensor(&m_USB_driver_LCR_sensors_session, outputs_vector);
        }
        for(i=0;i<N_DIGITAL_OUT;i++) {
            text_display.clear();
            QTextStream( &text_display ) << "OUT." << i << ": ";
            outputs_names_display_vector[i]->setText(text_display);

            if(outputs_vector[i]==1) {
                outputs_values_display_vector[i]->setText("ACTIVE");
            }
            else {
                outputs_values_display_vector[i]->setText("INACTIVE");
            }
        }


    }

    // ----- DEBUG CODE -----
    //QMessageBox msgBox6;
    //msgBox6.setText("DEBUG POS 6.\n");
    //msgBox6.exec();
    // ----------------------

    //call also the Update_Virtual_Calibration_Measurements_Display to update the measurement values obtained from the virtual calibration
    if(ui->enable_virtual_calib_checkBox->isChecked()==true) {

        this->Update_Virtual_Calibration_Measurements_Display();
        //the Measurements with the Virtual Calibration can only work when reading the RAW values, so RAW_values Reading GroupBox
        this->ui->measured_radioButton->setChecked(false);
        this->ui->RAW_radioButton->setChecked(true);

    }

    //call also the Update_Counters_Virtual_Calibration_Measurements_Display to update the measurement values obtained from the counters virtual calibration
    if(ui->enable_counters_virtual_calibs_checkBox->isChecked()==true) {
        this->Update_Counters_Virtual_Calibration_Measurements_Display();
    }


    //update the graphic view representation of the sensor measurements
    Update_Sensors_Graphic_View();
}



void MainWindow::Update_Virtual_Calibration_Measurements_Display() {
    int i, selected_channel_number;
    int return_value;
    QString text_display;
    float measurement;
    float sensors_measurement_vector[N_LCR_CHANNELS+N_ADC_CHANNELS];

    QLabel *LCR_sensors_label = new QLabel("LCR Sensors:");

    virtual_calib_sensors_names_display_vector[0]->setText("LCR Sensors:");
    virtual_calib_sensors_names_display_vector[1]->setText("CH.0:");
    virtual_calib_sensors_names_display_vector[2]->setText("CH.1:");
    virtual_calib_sensors_names_display_vector[3]->setText("CH.2:");
    virtual_calib_sensors_names_display_vector[4]->setText("CH.3:");
    virtual_calib_sensors_names_display_vector[5]->setText("CH.4:");
    virtual_calib_sensors_names_display_vector[6]->setText("CH.5:");

    virtual_calib_sensors_names_display_vector[7]->setText("ADC Sensors:");
    virtual_calib_sensors_names_display_vector[8]->setText("ADC(+-) CH.0:");
    virtual_calib_sensors_names_display_vector[9]->setText("ADC(+-) CH.1:");
    virtual_calib_sensors_names_display_vector[10]->setText("ADC(+) CH.0:");
    virtual_calib_sensors_names_display_vector[11]->setText("ADC(+) CH.1:");

        Sensors_Values_groupBox_layout->addWidget(LCR_sensors_label, 0,0);
        if(this->ui->select_CH_mode_comboBox->currentIndex()==0) {
            //multiple sensors selected

            //update all the LCR sensors virtual calibration display
            for(i=0;i<N_LCR_CHANNELS;i++) {
                if( frequency_sensors_vector[i] < 0 ) {
                    //an error occurred while reading the frequency from the sensors
                    text_display.clear();
                    QTextStream( &text_display ) << "ERROR: a negative value was read for the frequency of CH" << i << ", so probably an error occurred while reading the frequency from the sensors.\n";
                    error_log.append(text_display); //ERROR occurred, post to error log
                    MainWindow::SetConnectioStatusLabel_ERROR();
                }

                virtual_calib_LCR_sensors_measurement_vector[i] = Calibrate_Sensors_Dialog::Read_Measurement_From_Calib_Table(model_tables_virtual_calib_vector, i, frequency_sensors_vector[i]);

                //printf("CH.%d: %ld , ", i, frequency_sensors_vector[i]);  //DEBUG CODE
                virtual_calib_sensors_values_display_vector[i+1]->setFont(*font_display_sensors);
                text_display.clear();

                if(virtual_calib_channel_mode_bool_vector[i]==0) {
                    QTextStream( &text_display ) << virtual_calib_LCR_sensors_measurement_vector[i] << " " << virtual_calib_units_vector[i];
                }
                if(virtual_calib_channel_mode_bool_vector[i]==1) {
                    QTextStream( &text_display ) << "N/A (the available virtual calibration is for single CH mode.";
                }
                virtual_calib_sensors_values_display_vector[i+1]->setText(text_display);
            }

        }
        else {

            selected_channel_number=this->ui->select_CH_spinBox->value();

            if(  frequency_sensors_vector[selected_channel_number] < 0 ) {
                //an error occurred while reading the frequency from the sensors
                text_display.clear();
                QTextStream( &text_display ) << "ERROR: a negative value was read for the frequency of CH" << selected_channel_number << ", so probably an error occurred while reading the frequency from the sensors.\n";
                error_log.append(text_display); //ERROR occurred, post to error log
                MainWindow::SetConnectioStatusLabel_ERROR();

            }

            measurement = Calibrate_Sensors_Dialog::Read_Measurement_From_Calib_Table(model_tables_virtual_calib_vector, selected_channel_number, (float) frequency_sensors_vector[selected_channel_number]);

            text_display.clear();
            if(virtual_calib_channel_mode_bool_vector[selected_channel_number]==1) {
               QTextStream( &text_display ) << measurement << " " <<  virtual_calib_units_vector[selected_channel_number];
            }
            if(virtual_calib_channel_mode_bool_vector[selected_channel_number]==0) {
                QTextStream( &text_display ) << "N/A (the available virtual calibration is for multiple CH mode.";
            }

            virtual_calib_sensors_values_display_vector[selected_channel_number+1]->setFont(*font_display_sensors);
            virtual_calib_sensors_values_display_vector[selected_channel_number+1]->setText(text_display);

        }

        //update all the ADC sensors virtual calibration display
        for(i=0;i<N_ADC_CHANNELS;i++) {
            if( ADC_sensors_values_vector[i] < -0.2 ) {
                //an error occurred while reading the ADC values from the sensors
                text_display.clear();
                QTextStream( &text_display ) << "ERROR: negative value read on ADC_CH" << i << ", so probably an error occurred while reading the ADC values from the sensors.\n";
                error_log.append(text_display); //ERROR occured, post to error log
                MainWindow::SetConnectioStatusLabel_ERROR();
            }

            virtual_calib_ADC_sensors_measurement_vector[i] = Calibrate_Sensors_Dialog::Read_Measurement_From_Calib_Table(model_tables_virtual_calib_vector, N_LCR_CHANNELS+i, ADC_sensors_values_vector[i]);

            //printf("CH.%d: %ld , ", i, frequency_sensors_vector[i]);  //DEBUG CODE
            virtual_calib_sensors_values_display_vector[i+N_LCR_CHANNELS+2]->setFont(*font_display_sensors);
            text_display.clear();
            QTextStream( &text_display ) << virtual_calib_ADC_sensors_measurement_vector[i] << " " << virtual_calib_units_vector[N_LCR_CHANNELS+i];
            virtual_calib_sensors_values_display_vector[i+N_LCR_CHANNELS+2]->setText(text_display);
        }

}



// Updates the Measurements tab for showing the values of the sensor counters using calibrated scale and units
void MainWindow::Update_Counters_Virtual_Calibration_Measurements_Display() {
    int i, selected_channel_number;
    QString text_display;
    float sensor_counters_measurement_vector[N_LCR_CHANNELS];

    counters_virtual_calib_sensors_names_display_vector[0]->setText("Measurements (Sensor Counters):");
    counters_virtual_calib_sensors_names_display_vector[1]->setText("CH.0:");
    counters_virtual_calib_sensors_names_display_vector[2]->setText("CH.1:");
    counters_virtual_calib_sensors_names_display_vector[3]->setText("CH.2:");
    counters_virtual_calib_sensors_names_display_vector[4]->setText("CH.3:");
    counters_virtual_calib_sensors_names_display_vector[5]->setText("CH.4:");
    counters_virtual_calib_sensors_names_display_vector[6]->setText("CH.5:");

        if(this->ui->select_CH_mode_comboBox->currentIndex()==0) {
            //multiple sensors selected


            //update all the LCR sensors virtual calibration display
            for(i=0;i<N_LCR_CHANNELS;i++) {

                if( counters_sensors_vector[i] < 0 ) {
                    //an error occurred while reading the frequency from the sensors
                    error_log.append("ERROR: an error occurred while reading the frequency from the sensors.\n"); //ERROR occurred, post to error log
                    MainWindow::SetConnectioStatusLabel_ERROR();
                }
                else {

                    sensor_counters_measurement_vector[i] = counters_virtual_calib.counters_calib_C3_vector[i]*(counters_sensors_vector[i]*counters_sensors_vector[i]*counters_sensors_vector[i])+counters_virtual_calib.counters_calib_C2_vector[i]*(counters_sensors_vector[i]*counters_sensors_vector[i])+counters_virtual_calib.counters_calib_C1_vector[i]*counters_sensors_vector[i]+counters_virtual_calib.counters_calib_C0_vector[i];

                    counters_virtual_calib_sensors_values_display_vector[i+1]->setFont(*font_display_sensors);
                    text_display.clear();
                    if(counters_virtual_calib.calib_channel_mode_bool==0) {
                        QTextStream( &text_display ) << sensor_counters_measurement_vector[i] << " " << counters_virtual_calib.counters_calib_units[i];
                    }
                    if(counters_virtual_calib.calib_channel_mode_bool==1) {
                        QTextStream( &text_display ) << "N/A (the available virtual calibration is for single CH mode.";
                    }

                    counters_virtual_calib_sensors_values_display_vector[i+1]->setText(text_display);
                }

            }

        }

        else {
            selected_channel_number=this->ui->select_CH_spinBox->value();

            if(  counters_sensors_vector[selected_channel_number] < 0 ) {
                //an error occurred while reading the counters of the sensors input
                error_log.append("ERROR: an error occurred while reading the counters of the sensors input.\n"); //ERROR occurred, post to error log
                MainWindow::SetConnectioStatusLabel_ERROR();
            }

            sensor_counters_measurement_vector[selected_channel_number] = counters_virtual_calib.counters_calib_C3_vector[selected_channel_number]*(counters_sensors_vector[selected_channel_number]*counters_sensors_vector[selected_channel_number]*counters_sensors_vector[selected_channel_number])+counters_virtual_calib.counters_calib_C2_vector[selected_channel_number]*(counters_sensors_vector[selected_channel_number]*counters_sensors_vector[selected_channel_number])+counters_virtual_calib.counters_calib_C1_vector[selected_channel_number]*counters_sensors_vector[selected_channel_number]+counters_virtual_calib.counters_calib_C0_vector[selected_channel_number];

            text_display.clear();
            if(counters_virtual_calib.calib_channel_mode_bool==1) {
                QTextStream( &text_display ) << "CH." << selected_channel_number << ": " << sensor_counters_measurement_vector[selected_channel_number] << " " << counters_virtual_calib.counters_calib_units[selected_channel_number];
            }
            if(counters_virtual_calib.calib_channel_mode_bool==0) {
                QTextStream( &text_display ) << "N/A (the available virtual calibration is for multiple CH mode.";
            }

            counters_virtual_calib_sensors_values_display_vector[selected_channel_number+1]->setFont(*font_display_sensors);
            counters_virtual_calib_sensors_values_display_vector[selected_channel_number+1]->setText(text_display);

        }

}


//update the graph view of the sensors inputs
void MainWindow::Update_Sensors_Graphic_View() {
    int i;
    double sensor_value, min_value, max_value;
    double graph_view_value;

    QString text_display;


    for(i=0;i<N_LCR_CHANNELS+N_ADC_CHANNELS;i++) {

        if(i<N_LCR_CHANNELS) {
            sensor_value = LCR_sensors_values_vector[i];
        }
        else {
            sensor_value = ADC_sensors_values_vector[i-N_LCR_CHANNELS];
        }

        min_value = sensors_graphic_view_select_min_vector[i]->value();
        max_value = sensors_graphic_view_select_max_vector[i]->value();


        if( sensor_value <= min_value || min_value==max_value) {
            graph_view_value = 0;
        }
        else {
            if( sensor_value >= max_value) {
                graph_view_value = 100;
            }
            else {//this is sensors_graphic_view_select_min_vector[i] < sensor_value < sensors_graphic_view_select_max_vector[i]
                graph_view_value = 100 * ((sensor_value - min_value) / ( max_value - min_value ));
            }
        }

        sensors_progress_bar_vector[i]->setValue(graph_view_value);
    }

}



//reads from a text file the Device Calibrations for the sensor channels
void MainWindow::Import_Device_Calibrations() {
    QString import_file_name;
    QFile *import_device_calib;
    QString line;
    QString text_display;
    QString device_calib_CH_header;
    char line_str[MAX_SIZE_OF_CHARACTERS_PER_CALIB_TABLE+1], table_value[200];
    int ch_number, line_char_count;
    char *str_ptr, *row_start_ptr, *row_end_ptr, *col_separator_ptr;
    char *units_ptr, *table_ptr, *semicolon_ptr, *ch_mode_ptr, *is_active_ptr, *N_valid_lines_ptr, *jumper_selection_osc_tuning_range_ptr;
    char *C3_ptr, *C2_ptr, *C1_ptr, *C0_ptr;

    char calib_constant_str[40];

    QModelIndex model_index;

    int row_number;

    QString device_calib_units_Qstr, table_Qstr;

    import_file_name = QFileDialog::getOpenFileName(this, tr("Import Virtual Calibrations from File"), "", tr("All Files (*);;Text Files (*.txt);;LCR Sensors Calib Files (*.LCR_calib)") );

    if (!import_file_name.isEmpty()) {

        import_device_calib = new QFile(import_file_name);

        if (!import_device_calib->open(QIODevice::ReadOnly | QIODevice::Text))
        {
            //an error occurred while reading the import file name
            text_display.clear();
            QTextStream( &text_display ) << "ERROR: Error opening" << import_file_name;
            QMessageBox msgBox;
            msgBox.setText(text_display);
            msgBox.exec();
            error_log.append(text_display.toUtf8().data()); //ERROR occurred, post to error log
            MainWindow::SetConnectioStatusLabel_ERROR();

            return;
        }

        //read the next line of the import_device_calib file
        line_char_count = import_device_calib->readLine(line_str,MAX_SIZE_OF_CHARACTERS_PER_CALIB_TABLE);
        line.clear();
        line.append(line_str);

        for(;line_char_count>=0;) {

            //import the virtual calibration tables for the sensor channels
            for(ch_number=0;ch_number<N_LCR_CHANNELS+N_ADC_CHANNELS;ch_number++) {

                device_calib_CH_header.clear();
                QTextStream( &device_calib_CH_header ) << "DEVICE_CALIB_CHANNEL_N." << ch_number << ":";

                str_ptr = strstr(line_str, device_calib_CH_header.toUtf8());
                if( str_ptr != NULL ) {     //it was found on the current text line the reference 'DEVICE_CALIB_CHANNEL_N.<ch_number>'

                    //search and copy the UNITS and TABLE  and the header fields of current channel device calibration
                    units_ptr = strstr(line_str, "UNITS:");                    
                    ch_mode_ptr = strstr(line_str, "CH_MODE:");
                    jumper_selection_osc_tuning_range_ptr = strstr(line_str, "JUMPER_SELECT_OSC_TUNING_RANGE:");
                    N_valid_lines_ptr = strstr(line_str, "N_VALID_LINES:");
                    is_active_ptr = strstr(line_str, "IS_ACTIVE:");
                    table_ptr = strstr(line_str, "TABLE:");

                    //save the units of the calib table
                    semicolon_ptr = strstr(units_ptr+1, ";");
                    strncpy( table_value,units_ptr+strlen("UNITS:") , sizeof(table_value)-10);  //use strncpy to avoid writing outside the allocated space of the variable
                    table_value[semicolon_ptr-(units_ptr+strlen("UNITS:"))]='\0';
                    device_calib_units_Qstr.clear();
                    device_calib_units_Qstr.append(table_value);                    

                    table_Qstr.clear();
                    table_Qstr.append(table_ptr+strlen("TABLE"));

                    //save the device calibration units
                    strncpy(device_calib_header_data.device_calib_units_vector[ch_number], device_calib_units_Qstr.toUtf8().data(), sizeof(device_calib_header_data.device_calib_units_vector[ch_number])-10);  //use strncpy to avoid writing outside the allocated space of the variable
                    //clear the model_table before filling with the values read from the text file
                    this->model_tables_device_calib_vector[ch_number]->clear();

                    //save the channel mode of the calib table
                    strncpy(table_value, ch_mode_ptr+strlen("CH_MODE:"), 1); table_value[1]='\0';
                    this->device_calib_header_data.calib_channel_mode_bool[ch_number] = (atoi(table_value))&0x01;

                    //save the jumper selection of the oscillator tuning range of the selected sensor channel
                    this->device_calib_header_data.jumper_selection_osc_tuning_range[ch_number] = *(jumper_selection_osc_tuning_range_ptr+strlen("JUMPER_SELECT_OSC_TUNING_RANGE:"));

                    //save the number of valid lines of the calib table
                    semicolon_ptr = strstr(N_valid_lines_ptr+1, ";");
                    strncpy(table_value, N_valid_lines_ptr+strlen("N_VALID_LINES:"), semicolon_ptr-N_valid_lines_ptr-strlen("N_VALID_LINES:"));
                    table_value[semicolon_ptr-(N_valid_lines_ptr+strlen("N_VALID_LINES:"))]='\0';
                    this->device_calib_header_data.number_table_lines[ch_number] = (unsigned char) atoi(table_value)&0x00FF;

                    //save the is_active flag of the calib table
                    strncpy(table_value, is_active_ptr+strlen("IS_ACTIVE:"), 1); table_value[1]='\0';
                    this->device_calib_header_data.calib_table_is_active_bool[ch_number] = (atoi(table_value))&0x01;

                    //make a copy to char[] of the string with table data
                    strcpy(line_str, table_Qstr.toUtf8().data());
                    row_start_ptr = strstr(line_str, ":");
                    row_end_ptr = strstr(line_str, ";");

                    //save the device calibration on the model_table
                    for(row_number=0;row_end_ptr!=NULL;row_number++) {
                        col_separator_ptr = strstr(row_start_ptr,",");

                        //check if the col_separator found was of the current row of the table
                        if(col_separator_ptr<row_end_ptr) {
                            //add a new row to the model_table
                            //QList<QStandardItem *> preparedRow =prepareRow(0,0);
                            QList<QStandardItem *> new_row_items;
                            new_row_items.append(new QStandardItem(0));
                            new_row_items.append(new QStandardItem(0));
                            this->model_tables_device_calib_vector[ch_number]->appendRow(new_row_items);

                            strncpy(table_value,row_start_ptr+1, sizeof(table_value));  //use strncpy to avoid writing outside the allocated space of the variable
                            table_value[col_separator_ptr-row_start_ptr-1]='\0';
                            model_index = this->model_tables_device_calib_vector[ch_number]->index(row_number, 0);

                            this->model_tables_device_calib_vector[ch_number]->setData(model_index,atof(table_value));
                            strncpy(table_value,col_separator_ptr+1, sizeof(table_value));  //use strncpy to avoid writing outside the allocated space of the variable
                            table_value[row_end_ptr-col_separator_ptr-1]='\0';

                            model_index = model_tables_device_calib_vector[ch_number]->index(row_number, 1);
                            this->model_tables_device_calib_vector[ch_number]->setData(model_index,atof(table_value));
                        }

                        //update the row_start_ptr, row_end_ptr to read the next row of the virtual calib table
                        row_start_ptr=row_end_ptr;
                        row_end_ptr = strstr(row_end_ptr+1,";");
                    }

                    //this line is already processed, cause over count on ch_number to read immediately the next line
                    ch_number=9999;
                }




                //search for the sensor counters device calibrations
                device_calib_CH_header.clear();
                QTextStream( &device_calib_CH_header ) << "COUNTER_CALIB_CHANNEL_N." << ch_number << ":";

                str_ptr = strstr(line_str, device_calib_CH_header.toUtf8());

                if( str_ptr != NULL ) {    //it was found on the current text line the reference 'COUNTER_CALIB_CHANNEL_N.<ch_number>'

                    if( ch_number<N_LCR_CHANNELS ) {  //the counters only exist for the channels 0 to 5 ( [0;5] )

                        //search and copy the UNITS and TABLE of current channel virtual calibration
                        units_ptr = strstr(line_str, "UNITS:");
                        C3_ptr = strstr(line_str, "; C3:");
                        C2_ptr = strstr(line_str, "; C2:");
                        C1_ptr = strstr(line_str, "; C1:");
                        C0_ptr = strstr(line_str, "; C0:");

                        strncpy( table_value,units_ptr+strlen("UNITS:") , C3_ptr-units_ptr-6);  //use strncpy to avoid writing outside the allocated space of the variable
                        //strncpy( table_value,units_ptr+strlen("UNITS:") , A_ptr-units_ptr-6);  //use strncpy to avoid writing outside the allocated space of the variable
                        table_value[(C3_ptr-units_ptr-6)]='\0';
                        device_calib_units_Qstr.clear();
                        device_calib_units_Qstr.append(table_value);

                        table_Qstr.clear();
                        table_Qstr.append(table_ptr+strlen("; C3"));

                        //save the counter device calibration units
                         strncpy(this->counters_device_calib.counters_calib_units[ch_number], table_value, N_CHARS_CALIB_UNITS-1);  //use strncpy to avoid writing outside the allocated space of the variable
                         strncpy(this->device_calib_header_data.counters_calib_units_vector[ch_number], table_value, N_CHARS_CALIB_UNITS-1);

                        //save the value of the C3 calibration constant for the current ch_number
                        row_end_ptr = strstr(C3_ptr+2, ";");
                        strncpy(calib_constant_str, C3_ptr+5, row_end_ptr-C3_ptr-5);
                        calib_constant_str[row_end_ptr-C3_ptr-5]='\0';
                        this->counters_device_calib.counters_calib_C3_vector[ch_number]=atof(calib_constant_str);


                        //save the value of the C2 calibration constant for the current ch_number
                        row_end_ptr = strstr(C2_ptr+2, ";");
                        strncpy(calib_constant_str, C2_ptr+5, row_end_ptr-C2_ptr-5);
                        calib_constant_str[row_end_ptr-C2_ptr-5]='\0';
                        this->counters_device_calib.counters_calib_C2_vector[ch_number]=atof(calib_constant_str);

                        //save the value of the C1 calibration constant for the current ch_number
                        row_end_ptr = strstr(C1_ptr+2, ";");
                        strncpy(calib_constant_str, C1_ptr+5, row_end_ptr-C1_ptr-5);
                        calib_constant_str[row_end_ptr-C1_ptr-5]='\0';
                        this->counters_device_calib.counters_calib_C1_vector[ch_number]=atof(calib_constant_str);

                        //save the value of the C0 calibration constant for the current ch_number
                        row_end_ptr = strstr(C0_ptr+2, ";");
                        strncpy(calib_constant_str, C0_ptr+5, row_end_ptr-C0_ptr-5);
                        calib_constant_str[row_end_ptr-C0_ptr-5]='\0';
                        this->counters_device_calib.counters_calib_C0_vector[ch_number]=atof(calib_constant_str);

                    }
                    else {
                        //the ADC channels (6,7,8,9) don't have counters, so for the ADC channels fill the counter calibration with zero(0) or blank
                        counters_device_calib.counters_calib_units[ch_number][0]='\0';
                        counters_device_calib.counters_calib_C3_vector[ch_number]=0.0;
                        counters_device_calib.counters_calib_C2_vector[ch_number]=0.0;
                        counters_device_calib.counters_calib_C1_vector[ch_number]=0.0;
                        counters_device_calib.counters_calib_C0_vector[ch_number]=0.0;
                    }


                    //this line is already processed, cause over count on ch_number to read immediately the next line
                    ch_number=9999;

                }

            }

            //read the next line of the import_device_calib file
            line_char_count = import_device_calib->readLine(line_str,MAX_SIZE_OF_CHARACTERS_PER_CALIB_TABLE);
            line.clear();
            line.append(line_str);
        }

        //Show a message dialog indicating the Import of the Device Calibration was successful
        QMessageBox msgBox_Device_Calib;
        msgBox_Device_Calib.setText("Sensor Channels Device Calibrations of the Program were imported from selected file.");
        msgBox_Device_Calib.exec();
    }

}


//writes to a text file the Device Calibrations for the sensor channels
void MainWindow::Export_Device_Calibrations() {
    QString export_file_name;
    QFile *export_device_calib;
    QString line;
    QString text_display;
    //char line_str[MAX_SIZE_OF_CHARACTERS_PER_CALIB_TABLE+1];
    int ch_number;
    int n_rows_model_table, row_number;

    float RAW_value, device_calib_measurement;

    char msg_str[500];

    QModelIndex model_index;

    QString table_Qstr;


    export_file_name = QFileDialog::getSaveFileName(this,
                                    tr("Export Device Calibrations to File"),
                                                            "exported_device_calibs_sensors",
                                    tr("All Files (*);;Text Files (*.txt);;LCR Sensors Calib Files (*.LCR_calib)"));

    if (!export_file_name.isEmpty()) {

        export_device_calib = new QFile(export_file_name);

          if (!export_device_calib->open(QIODevice::WriteOnly | QIODevice::Text))
          {
              //an error occurred while reading the import file name
              text_display.clear();
              QTextStream( &text_display ) << "ERROR: Error opening" << export_file_name;
              QMessageBox msgBox;
              msgBox.setText(text_display);
              msgBox.exec();
              sprintf(msg_str, "ERROR: Error opening %s.\n", export_file_name);
              error_log.append(msg_str); //ERROR occurred, post to error log
              MainWindow::SetConnectioStatusLabel_ERROR();
              return;
          }

          for(ch_number=0;ch_number<N_LCR_CHANNELS+N_ADC_CHANNELS;ch_number++) {

              //write to QString the current device calibrations
              n_rows_model_table = model_tables_device_calib_vector[ch_number]->rowCount();

              table_Qstr.clear();
              for(row_number=0; row_number<n_rows_model_table; row_number++) {
                  model_index = model_tables_device_calib_vector[ch_number]->index(row_number, 0);
                  RAW_value = model_tables_device_calib_vector[ch_number]->data(model_index).toFloat();

                  model_index = model_tables_device_calib_vector[ch_number]->index(row_number, 1);
                  device_calib_measurement = model_tables_device_calib_vector[ch_number]->data(model_index).toFloat();

                  QTextStream( &table_Qstr ) << RAW_value << "," << device_calib_measurement << ";";
               }


              //write to the file the device calibration for this channel number
              char jumper_selection_osc_tuning_range_char = device_calib_header_data.jumper_selection_osc_tuning_range[ch_number];
              if( jumper_selection_osc_tuning_range_char != 'A' && jumper_selection_osc_tuning_range_char != 'B' && jumper_selection_osc_tuning_range_char != '+' && jumper_selection_osc_tuning_range_char != '-') {
                  jumper_selection_osc_tuning_range_char = ' ';  }

              short calib_table_is_active_bool_value = (short) device_calib_header_data.calib_table_is_active_bool[ch_number];
              if( calib_table_is_active_bool_value!=0  && calib_table_is_active_bool_value!=1  )  {
                  calib_table_is_active_bool_value = 0; //in case the value of "calib_table_is_active_bool" is invalid, then set it to the default value, that is inactive state (zero-0)
              }

              line.clear();
              QTextStream( &line ) << "DEVICE_CALIB_CHANNEL_N." << ch_number << ": " << "UNITS:" << device_calib_header_data.device_calib_units_vector[ch_number]  << "; CH_MODE:" << device_calib_header_data.calib_channel_mode_bool[ch_number] << "; JUMPER_SELECT_OSC_TUNING_RANGE:" << jumper_selection_osc_tuning_range_char << "; N_VALID_LINES:" << device_calib_header_data.number_table_lines[ch_number] << "; IS_ACTIVE:" << calib_table_is_active_bool_value << "; TABLE:" << table_Qstr << "\n";

              export_device_calib->write(line.toUtf8().data());

              //write to the file the counters device calibration for this channel number
              line.clear();
              if( ch_number<N_LCR_CHANNELS ) {    //the counters only exist for the channels 0 to 5 ( [0;5] )
                QTextStream( &line ) << "COUNTER_CALIB_CHANNEL_N." << ch_number << ": " << "UNITS:" << counters_device_calib.counters_calib_units[ch_number] << "; C3:" << counters_device_calib.counters_calib_C3_vector[ch_number] << "; C2:" << counters_device_calib.counters_calib_C2_vector[ch_number] << "; C1:" << counters_device_calib.counters_calib_C1_vector[ch_number] << "; C0:" << counters_device_calib.counters_calib_C0_vector[ch_number] << ";\n";
              }
              else {
                  QTextStream( &line ) << "COUNTER_CALIB_CHANNEL_N." << ch_number << ": " << "UNITS:" << "------" << "; C3:" << "------" << "; C2:" << "------" << "; C1:" << "------" << "; C0:" << "------" << ";\n";
              }

              export_device_calib->write(line.toUtf8().data());
            }

          export_device_calib->close();
    }

}



//reads from a text file the Virtual Calibrations for the sensor channels
void MainWindow::Import_Virtual_Calibrations() {
    QString import_file_name;
    QFile *import_virtual_calib;
    QString text_line, line;
    QString text_display;
    QString virtual_calib_CH_header;
    char line_str[MAX_SIZE_OF_CHARACTERS_PER_CALIB_TABLE+1], table_value[200];
    int ch_number, line_char_count;
    char *str_ptr, *row_start_ptr, *row_end_ptr, *col_separator_ptr;
    char *units_ptr, *table_ptr, *ch_mode_ptr, *jumper_selection_osc_tuning_range_ptr;
    char *C3_ptr, *C2_ptr, *C1_ptr, *C0_ptr;

    char calib_constant_str[40];
    char msg_str[500];

    QModelIndex model_index;

    int row_number;

    QString *empty_Qstr=new QString("");
    QString virtual_calib_units_Qstr, table_Qstr;

    import_file_name = QFileDialog::getOpenFileName(this, tr("Import Virtual Calibrations from File"), "", tr("All Files (*);;Text Files (*.txt);;LCR Sensors Calib Files (*.LCR_calib)") );

    if (!import_file_name.isEmpty()) {

        import_virtual_calib = new QFile(import_file_name);

        if (!import_virtual_calib->open(QIODevice::ReadOnly | QIODevice::Text))
        {
            //an error occurred while reading the import file name
            text_display.clear();
            QTextStream( &text_display ) << "ERROR: Error opening" << import_file_name;
            QMessageBox msgBox;
            msgBox.setText(text_display);
            msgBox.exec();
            error_log.append(text_display.toUtf8().data()); //ERROR occurred, post to error log
            MainWindow::SetConnectioStatusLabel_ERROR();

            return;
        } 

        //read the next line of the import_virtual_calib file
        line_char_count = import_virtual_calib->readLine(line_str,MAX_SIZE_OF_CHARACTERS_PER_CALIB_TABLE);
        line.clear();
        line.append(line_str);

        for(;line_char_count>=0;) {

            //search for the "VIRTUAL_CALIB_CHANNEL_MODE:"
            str_ptr = strstr(line_str, "VIRTUAL_CALIB_CHANNEL_MODE:");
            if( str_ptr != NULL ) {
                counters_virtual_calib.calib_channel_mode_bool=atoi(str_ptr+strlen("VIRTUAL_CALIB_CHANNEL_MODE:"));
            }


            //import the virtual calibration tables for the LCR sensor channels
            for(ch_number=0;ch_number<N_LCR_CHANNELS+N_ADC_CHANNELS;ch_number++) {

                virtual_calib_CH_header.clear();
                QTextStream( &virtual_calib_CH_header ) << "VIRTUAL_CALIB_CHANNEL_N." << ch_number << ":";

                str_ptr = strstr(line_str, virtual_calib_CH_header.toUtf8());
                if( str_ptr != NULL ) {     //it was found on the current text line the reference 'VIRTUAL_CALIB_CHANNEL_N.<ch_number>'

                    //search and copy the UNITS and TABLE of current channel virtual calibration
                    units_ptr = strstr(line_str, "UNITS:");
                    ch_mode_ptr = strstr(line_str, "; CH_MODE:");
                    jumper_selection_osc_tuning_range_ptr = strstr(line_str, "; JUMPER_SELECT_OSC_TUNING_RANGE:");
                    table_ptr = strstr(line_str, "; TABLE:");
                    strncpy( table_value,units_ptr+strlen("UNITS:") , sizeof(table_value)-10);  //use strncpy to avoid writing outside the allocated space of the variable
                    table_value[ch_mode_ptr-(units_ptr+strlen("UNITS:"))]='\0';
                    virtual_calib_units_Qstr.clear();
                    virtual_calib_units_Qstr.append(table_value);

                    table_Qstr.clear();
                    table_Qstr.append(table_ptr+strlen("; TABLE"));

                    //save the virtual calibration units
                    strncpy(virtual_calib_units_vector[ch_number], virtual_calib_units_Qstr.toUtf8().data(), sizeof(virtual_calib_units_vector[ch_number])-10);  //use strncpy to avoid writing outside the allocated space of the variable

                    //save the channel mode of the calib table
                    strncpy(table_value, ch_mode_ptr+strlen("; CH_MODE:"), 1); table_value[1]='\0';
                    virtual_calib_channel_mode_bool_vector[ch_number] = (atoi(table_value))&0x01;

                    //save the jumper selection of the oscillator tuning range of the selected sensor channel
                    virtual_calib_jumper_selection_osc_tuning_range_vector[ch_number] = *(jumper_selection_osc_tuning_range_ptr+strlen("; JUMPER_SELECT_OSC_TUNING_RANGE:"));


                    //clear the model_table before filling with the values read from the text file
                    model_tables_virtual_calib_vector[ch_number]->clear();

                    //make a copy to char[] of the string with table data
                    strcpy(line_str, table_Qstr.toUtf8().data());
                    row_start_ptr = strstr(line_str, ":");
                    row_end_ptr = strstr(line_str, ";");

                    //save the virtual calibration on the model_table
                    for(row_number=0;row_end_ptr!=NULL;row_number++) {
                        col_separator_ptr = strstr(row_start_ptr,",");

                        //check if the col_separator found was of the current row of the table
                        if(col_separator_ptr<row_end_ptr) {
                            //add a new row to the model_table
                            //QList<QStandardItem *> preparedRow =prepareRow(0,0);
                            QList<QStandardItem *> new_row_items;
                            new_row_items.append(new QStandardItem(0));
                            new_row_items.append(new QStandardItem(0));
                            model_tables_virtual_calib_vector[ch_number]->appendRow(new_row_items);

                            strncpy(table_value,row_start_ptr+1, sizeof(table_value));  //use strncpy to avoid writing outside the allocated space of the variable
                            table_value[col_separator_ptr-row_start_ptr-1]='\0';
                            model_index = model_tables_virtual_calib_vector[ch_number]->index(row_number, 0);

                            model_tables_virtual_calib_vector[ch_number]->setData(model_index,atof(table_value));
                            strncpy(table_value,col_separator_ptr+1, sizeof(table_value));  //use strncpy to avoid writing outside the allocated space of the variable
                            table_value[row_end_ptr-col_separator_ptr-1]='\0';

                            model_index = model_tables_virtual_calib_vector[ch_number]->index(row_number, 1);
                            model_tables_virtual_calib_vector[ch_number]->setData(model_index,atof(table_value));
                        }

                        //update the row_start_ptr, row_end_ptr to read the next row of the virtual calib table
                        row_start_ptr=row_end_ptr;
                        row_end_ptr = strstr(row_end_ptr+1,";");
                    }


                    //this line is already processed, cause over count on ch_number to read immediately the next line
                    ch_number=9999;
                }         

            }

            //import the virtual counter calibration for the LCR sensor channels
            for(ch_number=0;ch_number<N_LCR_CHANNELS+N_ADC_CHANNELS;ch_number++) {

                //search for the sensor counters virtual calibrations
                virtual_calib_CH_header.clear();
                QTextStream( &virtual_calib_CH_header ) << "COUNTER_CALIB_CHANNEL_N." << ch_number << ":";

                str_ptr = strstr(line_str, virtual_calib_CH_header.toUtf8());
                if( str_ptr != NULL ) {     //it was found on the current text line the reference 'COUNTER_CALIB_CHANNEL_N.<ch_number>'

                    if( ch_number<N_LCR_CHANNELS ) { //the counters only exist for the channels 0 to 5 ( [0;5] )

                    //search and copy the UNITS and TABLE of current channel virtual calibration
                    units_ptr = strstr(line_str, "UNITS:");
                    C3_ptr = strstr(line_str, "; C3:");
                    C2_ptr = strstr(line_str, "; C2:");
                    C1_ptr = strstr(line_str, "; C1:");
                    C0_ptr = strstr(line_str, "; C0:");
                    strncpy( table_value,units_ptr+strlen("UNITS:") , C3_ptr-units_ptr-6);  //use strncpy to avoid writing outside the allocated space of the variable
                    table_value[(C3_ptr-units_ptr-6)]='\0';
                    virtual_calib_units_Qstr.clear();
                    virtual_calib_units_Qstr.append(table_value);

                    table_Qstr.clear();
                    table_Qstr.append(table_ptr+strlen("; C3"));

                    //save the counter virtual calibration units
                    strncpy(counters_virtual_calib.counters_calib_units[ch_number], table_value, 30);  //use strncpy to avoid writing outside the allocated space of the variable

                    //save the value of the C3 calibration constant for the current ch_number
                    row_end_ptr = strstr(C3_ptr+2, ";");
                    strncpy(calib_constant_str, C3_ptr+5, row_end_ptr-C3_ptr-5);
                    calib_constant_str[row_end_ptr-C3_ptr-5]='\0';
                    counters_virtual_calib.counters_calib_C3_vector[ch_number]=atof(calib_constant_str);

                    //save the value of the C2 calibration constant for the current ch_number
                    row_end_ptr = strstr(C2_ptr+2, ";");
                    strncpy(calib_constant_str, C2_ptr+5, row_end_ptr-C2_ptr-5);
                    calib_constant_str[row_end_ptr-C2_ptr-5]='\0';
                    counters_virtual_calib.counters_calib_C2_vector[ch_number]=atof(calib_constant_str);

                    //save the value of the C1 calibration constant for the current ch_number
                    row_end_ptr = strstr(C1_ptr+2, ";");
                    strncpy(calib_constant_str, C1_ptr+5, row_end_ptr-C1_ptr-5);
                    calib_constant_str[row_end_ptr-C1_ptr-5]='\0';
                    counters_virtual_calib.counters_calib_C1_vector[ch_number]=atof(calib_constant_str);

                    //save the value of the C0 calibration constant for the current ch_number
                    row_end_ptr = strstr(C0_ptr+2, ";");
                    strncpy(calib_constant_str, C0_ptr+5, row_end_ptr-C0_ptr-5);
                    calib_constant_str[row_end_ptr-C0_ptr-5]='\0';
                    counters_virtual_calib.counters_calib_C0_vector[ch_number]=atof(calib_constant_str);
                    }
                    else {
                        //the ADC channels (6,7,8,9) don't ahve counters, so for the ADC channels fill the counter calibration with zero(0) or blank
                        counters_virtual_calib.counters_calib_units[ch_number][0]='\0';
                        counters_virtual_calib.counters_calib_C3_vector[ch_number]=0.0;
                        counters_virtual_calib.counters_calib_C2_vector[ch_number]=0.0;
                        counters_virtual_calib.counters_calib_C1_vector[ch_number]=0.0;
                        counters_virtual_calib.counters_calib_C0_vector[ch_number]=0.0;
                    }


                    //this line is already processed, cause over count on ch_number to read immediately the next line
                    ch_number=9999;
                }

            }

            //read the next line of the import_virtual_calib file
            line_char_count = import_virtual_calib->readLine(line_str,MAX_SIZE_OF_CHARACTERS_PER_CALIB_TABLE);
            line.clear();
            line.append(line_str);
        }

        //Show a message dialog indicating the Import of the Virtual Calibration was successful
        QMessageBox msgBox;
        msgBox.setText("Sensor Channels Virtual Calibrations of the Program were imported from selected file.");
        msgBox.exec();
    }

    delete empty_Qstr;
}

//writes to a text file the Virtual Calibrations for the sensor channels
void MainWindow::Export_Virtual_Calibrations() {
    QString export_file_name;
    QFile *export_virtual_calib;
    QString line;
    QString text_display;
    int ch_number;
    int n_rows_model_table, row_number;
    char jumper_selection_osc_tuning_range_char;

    float RAW_value, virtual_calib_measurement;

    char msg_str[500];

    QModelIndex model_index;

    QString virtual_calib_units_Qstr, table_Qstr;

    export_file_name = QFileDialog::getSaveFileName(this,
                                    tr("Export Virtual Calibrations to File"),
                                                            "exported_virtual_calibs_sensors",
                                    tr("All Files (*);;Text Files (*.txt);;LCR Sensors Calib Files (*.LCR_calib)"));

    if (!export_file_name.isEmpty()) {


        export_virtual_calib = new QFile(export_file_name);

          if (!export_virtual_calib->open(QIODevice::WriteOnly | QIODevice::Text))
          {
              //an error occurred while reading the import file name
              text_display.clear();
              QTextStream( &text_display ) << "ERROR: Error opening" << export_file_name;
              QMessageBox msgBox;
              msgBox.setText(text_display);
              msgBox.exec();
              sprintf(msg_str, "ERROR: Error opening %s.\n", export_file_name);
              error_log.append(msg_str); //ERROR occurred, post to error log
              MainWindow::SetConnectioStatusLabel_ERROR();
              return;
          }


          for(ch_number=0;ch_number<N_LCR_CHANNELS+N_ADC_CHANNELS;ch_number++) {

              //write to QString the current virtual calibrations
              n_rows_model_table = model_tables_virtual_calib_vector[ch_number]->rowCount();

              table_Qstr.clear();
              for(row_number=0; row_number<n_rows_model_table; row_number++) {
                  model_index = model_tables_virtual_calib_vector[ch_number]->index(row_number, 0);
                  RAW_value = model_tables_virtual_calib_vector[ch_number]->data(model_index).toFloat();

                  model_index = model_tables_virtual_calib_vector[ch_number]->index(row_number, 1);
                  virtual_calib_measurement = model_tables_virtual_calib_vector[ch_number]->data(model_index).toFloat();

                  QTextStream( &table_Qstr ) << RAW_value << "," << virtual_calib_measurement << ";";
               }



              if( ch_number<N_LCR_CHANNELS ) {    //the counters only exist for the channels 0 to 5 ( [0;5] )

                jumper_selection_osc_tuning_range_char = virtual_calib_jumper_selection_osc_tuning_range_vector[ch_number];
                if( jumper_selection_osc_tuning_range_char != 'A' && jumper_selection_osc_tuning_range_char != 'B' && jumper_selection_osc_tuning_range_char != '+' && jumper_selection_osc_tuning_range_char != '-') {
                    jumper_selection_osc_tuning_range_char = ' ';  }
              }
              else {
                  jumper_selection_osc_tuning_range_char = ' ';
              }

              //write to the file the virtual calibration for this channel number
              line.clear();
              QTextStream( &line ) << "VIRTUAL_CALIB_CHANNEL_N." << ch_number << ": " << "UNITS:" << virtual_calib_units_vector[ch_number] << "; CH_MODE:" << virtual_calib_channel_mode_bool_vector[ch_number] << "; JUMPER_SELECT_OSC_TUNING_RANGE:" << jumper_selection_osc_tuning_range_char << "; TABLE:" << table_Qstr << "\n";

              export_virtual_calib->write(line.toUtf8().data());


              //write to the file the counters virtual calibration for this channel number
              line.clear();
              if( ch_number<N_LCR_CHANNELS ) {    //the counters only exist for the channels 0 to 5 ( [0;5] )
                QTextStream( &line ) << "COUNTER_CALIB_CHANNEL_N." << ch_number << ": " << "UNITS:" << counters_virtual_calib.counters_calib_units[ch_number] << "; C3:" << counters_virtual_calib.counters_calib_C3_vector[ch_number] << "; C2:" << counters_virtual_calib.counters_calib_C2_vector[ch_number] <<  "; C1:" << counters_virtual_calib.counters_calib_C1_vector[ch_number] << "; C0:" << counters_virtual_calib.counters_calib_C0_vector[ch_number] << ";\n";
              }
              else {
                  QTextStream( &line ) << "COUNTER_CALIB_CHANNEL_N." << ch_number << ": " << "UNITS:" << "------" << "; C3:" << "------" << "; C2:" << "------" <<  "; C1:" << "------" << "; C0:" << "------" << ";\n";
              }

              export_virtual_calib->write(line.toUtf8().data());
            }


          export_virtual_calib->close();
    }

}


void MainWindow::Create_Calibrate_Device_Sensors_Dialog() {
    if( m_USB_driver_LCR_sensors_session.isConnected==TRUE_BOOL || m_serial_driver_LCR_sensors_session.isConnected==TRUE_BOOL) {

        previous_ch_number_for_saved_device_calib_dialog = 0;   //set the initial value to "0" to be sure its valid, this is smaller than N_LCR_CHANNELS+N_ADC_CHANNELS

        calibrate_device_sensors_dialog_obj = new Device_Calibrate_Sensors_Dialog(&previous_ch_number_for_saved_device_calib_dialog, &(this->device_calib_header_data), this->model_tables_device_calib_vector, &(this->counters_device_calib));
        calibrate_device_sensors_dialog_obj->select_channel_number_spinBox->setMaximum(N_LCR_CHANNELS+N_ADC_CHANNELS-1);
        calibrate_device_sensors_dialog_obj->Set_Calib_Headers("RAW (Hz or V)", "Measurement");
        //update the contents of the displayed model table to match the selected sensor number virtual calibration
        calibrate_device_sensors_dialog_obj->Call_Create_Copy_Model_Table(N_LCR_CHANNELS+N_ADC_CHANNELS);
    }
    else {
        //the PC isn't connected to the sensors board, first connect to the sensors board
        QMessageBox msgBox;
        msgBox.setText("ERROR: Sensors board not connected, first connect to the sensors board.");
        msgBox.exec();
    }
}

void MainWindow::Create_Config_Sensors_Dialog() {
    char communication_is_SERIAL;

    if( m_USB_driver_LCR_sensors_session.isConnected==TRUE_BOOL || m_serial_driver_LCR_sensors_session.isConnected==TRUE_BOOL ) {
        if(ui->serial_radioButton->isChecked()==true) {
            communication_is_SERIAL=1;
        }
        else {
            communication_is_SERIAL=0;
        }
        config_sensors_dialog_obj = new Config_Sensor_Channels_Dialog(&(this->m_USB_driver_LCR_sensors_session), &(this->m_serial_driver_LCR_sensors_session), &previous_ch_number_for_saved_config_dialog, communication_is_SERIAL);
    }
    else {
        //the PC isn't connected to the sensors board, first connect to the sensors board
        QMessageBox msgBox;
        msgBox.setText("ERROR: Sensors board not connected, first connect to the sensors board.");
        msgBox.exec();
    }
}

void MainWindow::Create_Config_Outputs_Dialog() {
    char communication_is_SERIAL_bool;

    if( m_USB_driver_LCR_sensors_session.isConnected==TRUE_BOOL || m_serial_driver_LCR_sensors_session.isConnected==TRUE_BOOL ) {
       if(ui->serial_radioButton->isChecked()==true) {
            communication_is_SERIAL_bool=1;
        }
        else {
            communication_is_SERIAL_bool=0;
        }
        config_outputs_dialog_obj = new Config_Outputs_Dialog(&(this->m_USB_driver_LCR_sensors_session), &(this->m_serial_driver_LCR_sensors_session), &previous_output_number_for_saved_config_dialog, communication_is_SERIAL_bool);
    }
    else {
        //the PC isn't connected to the sensors board, first connect to the sensors board
        QMessageBox msgBox;
        msgBox.setText("ERROR: Sensors board not connected, first connect to the sensors board.");
        msgBox.exec();
    }
}

void MainWindow::Create_Virtual_Calibrate_Sensors_Dialog() {

    previous_ch_number_for_saved_virtual_calib_dialog = 0;   //set the initial value to "0" to be sure its valid, this is smaller than N_LCR_CHANNELS

    virtual_calibrate_sensors_dialog_obj = new Virtual_Calibrate_Sensors_Dialog(&previous_ch_number_for_saved_virtual_calib_dialog, virtual_calib_units_vector, model_tables_virtual_calib_vector, &counters_virtual_calib, virtual_calib_channel_mode_bool_vector, virtual_calib_jumper_selection_osc_tuning_range_vector);

    virtual_calibrate_sensors_dialog_obj->select_channel_number_spinBox->setMaximum(N_LCR_CHANNELS+N_ADC_CHANNELS-1);

    virtual_calibrate_sensors_dialog_obj->select_channel_number_spinBox->setValue(0);
    virtual_calibrate_sensors_dialog_obj->Set_Calib_Headers("Freq(Hz)", "Measurement");

    //update the contents of the displayed model table to match the selected sensor number virtual calibration
    virtual_calibrate_sensors_dialog_obj->Call_Create_Copy_Model_Table(N_LCR_CHANNELS+N_ADC_CHANNELS);
}


void MainWindow::Create_Config_BoardID_Dialog() {
    char communication_is_SERIAL_bool;

    if( m_USB_driver_LCR_sensors_session.isConnected==TRUE_BOOL || m_serial_driver_LCR_sensors_session.isConnected==TRUE_BOOL ) {
       if(ui->serial_radioButton->isChecked()==true) {
            communication_is_SERIAL_bool=1;
        }
        else {
            communication_is_SERIAL_bool=0;
        }
        config_boardID_dialog_obj = new Config_BoardID_Dialog(&(this->m_USB_driver_LCR_sensors_session), &(this->m_serial_driver_LCR_sensors_session), communication_is_SERIAL_bool);
    }
    else {
        //the PC isn't connected to the sensors board, first connect to the sensors board
        QMessageBox msgBox;
        msgBox.setText("ERROR: Sensors board not connected, first connect to the sensors board.");
        msgBox.exec();
    }
}



void MainWindow::Upload_Device_Calib_Table() {

    int i;

    Transfer_Calib_Table_Dialog *dialog_transfer_calib_tables;

    char log_msg_str[1000]; // dialogbox_msg_str[200];

    Worker_Arguments worker_args;

    if( m_USB_driver_LCR_sensors_session.isConnected==FALSE_BOOL && m_serial_driver_LCR_sensors_session.isConnected==FALSE_BOOL ) {
        //the PC isn't connected to the sensors board, first connect to the sensors board
        QMessageBox msgBox;
        msgBox.setText("ERROR: Sensors board not connected, first connect to the sensors board.");
        msgBox.exec();

        return ;
    }

    worker_args.log_msg_str_ptr = (log_msg_str);
    worker_args.N_channels = N_LCR_CHANNELS+N_ADC_CHANNELS;
    worker_args.calib_table_max_rows_to_process_on_single_mode = CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE;
    worker_args.serial_driver_LCR_sensors_session_ptr = &(this->m_serial_driver_LCR_sensors_session);
    worker_args.driver_LCR_sensors_session_ptr = &(this->m_USB_driver_LCR_sensors_session);
    worker_args.device_calib_header_data_ptr = &device_calib_header_data;
    worker_args.flag_transfer_mode_bool = 1;    //transfer_mode set to '1' means upload
    for(i=0; i<N_LCR_CHANNELS+N_ADC_CHANNELS; i++) {    //make a clone of the 'model_tables_device_calib_vector' on the on the structure with the arguments for the worker thread
        worker_args.model_tables_device_calib_vector[i] = model_tables_device_calib_vector[i];
    }

    for(i=0; i < N_LCR_CHANNELS; i++) {
         strncpy(worker_args.device_calib_header_data_ptr->counters_calib_units_vector[i], counters_device_calib.counters_calib_units[i], N_CHARS_CALIB_UNITS-1);
         worker_args.device_calib_header_data_ptr->counters_calib_constants_vector[i][0] = counters_device_calib.counters_calib_C0_vector[i];
         worker_args.device_calib_header_data_ptr->counters_calib_constants_vector[i][1] = counters_device_calib.counters_calib_C1_vector[i];
         worker_args.device_calib_header_data_ptr->counters_calib_constants_vector[i][2] = counters_device_calib.counters_calib_C2_vector[i];
         worker_args.device_calib_header_data_ptr->counters_calib_constants_vector[i][3] = counters_device_calib.counters_calib_C3_vector[i];
    }


    worker_args.flag_transfer_mode_bool = 1;    //transfer_mode set to '1' means upload

    if(ui->serial_radioButton->isChecked()==true) {
        worker_args.communication_is_SERIAL_bool=1;   //select serial connection
    }
    else {
        worker_args.communication_is_SERIAL_bool=0;   //select USB connection
    }


    dialog_transfer_calib_tables = new Transfer_Calib_Table_Dialog(this, &worker_args, 1);
    dialog_transfer_calib_tables->exec();


    if(dialog_transfer_calib_tables->ret_data_worker.error_counter > 0) {
        sprintf(log_msg_str, "ERROR: @ Upload Device Calibrations , error_counter=%d\n", dialog_transfer_calib_tables->ret_data_worker.error_counter);
        this->error_log.append(log_msg_str);
    }

}



void MainWindow::Download_Device_Calib_Table() {

    int i;
    char log_msg_str[1000];

    Transfer_Calib_Table_Dialog *dialog_transfer_calib_tables;
    Worker_Arguments worker_args;

    if( m_USB_driver_LCR_sensors_session.isConnected==FALSE_BOOL && m_serial_driver_LCR_sensors_session.isConnected==FALSE_BOOL ) {
        //the PC isn't connected to the sensors board, first connect to the sensors board
        QMessageBox msgBox;
        msgBox.setText("ERROR: Sensors board not connected, first connect to the sensors board.");
        msgBox.exec();

        return ;
    }

    worker_args.log_msg_str_ptr = (log_msg_str);
    worker_args.N_channels = N_LCR_CHANNELS+N_ADC_CHANNELS;
    worker_args.calib_table_max_rows_to_process_on_single_mode = CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE;
    worker_args.serial_driver_LCR_sensors_session_ptr = &(this->m_serial_driver_LCR_sensors_session);
    worker_args.driver_LCR_sensors_session_ptr = &(this->m_USB_driver_LCR_sensors_session);
    worker_args.device_calib_header_data_ptr = &device_calib_header_data;
    worker_args.flag_transfer_mode_bool = 0;    //transfer_mode set to '0' means download
    for(i=0; i<N_LCR_CHANNELS+N_ADC_CHANNELS; i++) {    //make a clone of the 'model_tables_device_calib_vector' on the on the structure with the arguments for the worker thread
        worker_args.model_tables_device_calib_vector[i] = model_tables_device_calib_vector[i];
    }

    worker_args.flag_transfer_mode_bool = 0;    //transfer_mode set to '0' means download

    if(ui->serial_radioButton->isChecked()==true) {
        worker_args.communication_is_SERIAL_bool=1;   //select serial connection
    }
    else {
        worker_args.communication_is_SERIAL_bool=0;   //select USB connection
    }

    dialog_transfer_calib_tables = new Transfer_Calib_Table_Dialog(this, &worker_args, 0);
    dialog_transfer_calib_tables->exec();

    this->error_log.append(log_msg_str);

    for(i=0; i < N_LCR_CHANNELS; i++) {
        strncpy(counters_device_calib.counters_calib_units[i], worker_args.device_calib_header_data_ptr->counters_calib_units_vector[i], N_CHARS_CALIB_UNITS-1);
        counters_device_calib.counters_calib_C0_vector[i] = worker_args.device_calib_header_data_ptr->counters_calib_constants_vector[i][0];
        counters_device_calib.counters_calib_C1_vector[i] = worker_args.device_calib_header_data_ptr->counters_calib_constants_vector[i][1];
        counters_device_calib.counters_calib_C2_vector[i] = worker_args.device_calib_header_data_ptr->counters_calib_constants_vector[i][2];
        counters_device_calib.counters_calib_C3_vector[i] = worker_args.device_calib_header_data_ptr->counters_calib_constants_vector[i][3];
    }

}

// ----- Functions of the class MainWindow  -  END -----
