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
 *   FILE: calibrate_sensors_dialog.cpp - Here is the implementation of the      *
 *     class 'Calibrate_Sensors_Dialog', that is the base class for              *
 *     'Device_Calibrate_Sensors_Dialog' and 'Virtual_Calibrate_Sensors_Dialog'. *
 ********************************************************************************/

#include "calibrate_sensors_dialog.h"

#include <QMessageBox>
#include <QtWidgets>


// ----- CLASS Calibrate_Sensors_Dialog -----

Calibrate_Sensors_Dialog::Calibrate_Sensors_Dialog(int *previous_ch_number_for_saved_calib_dialog_ptr, char (*calib_units_vector_ptr)[N_CHARS_CALIB_UNITS], QStandardItemModel **model_tables_calib_vector_ptr, counters_calib *counters_calib_ptr) {

    this->m_previous_ch_number_for_saved_calib_dialog_ptr = previous_ch_number_for_saved_calib_dialog_ptr;
    this->model_tables_calib_vector_ptr = model_tables_calib_vector_ptr;
    this->m_calib_units_vector_ptr = calib_units_vector_ptr;
    this->m_counters_calib_ptr = counters_calib_ptr;

    const char *space_label_xl = "          ";
    const char *space_label_small = "  ";

    //save the name of the headers of the Calibration Table
    sprintf(m_headers_list[0],"Freq(Hz)");
    sprintf(m_headers_list[1],"Measurement");

    this->setWindowTitle("Calibrate Sensors");

    this->setLocale(QLocale::C);    //configure the class 'Calibrate_Sensors_Dialog' to use Locale 'QLocale::C' so the QDoubleSpinBox uses '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations


    current_model_table_calib = new QStandardItemModel(1, 2);

    calibrate_sensors_dialog_layout = new QGridLayout();
    this->setLayout(calibrate_sensors_dialog_layout);

    calibrate_sensors_widget = new QWidget(this);
    calibrate_sensors_widget_layout = new QGridLayout();
    calibrate_sensors_widget->setLayout(calibrate_sensors_widget_layout);
    select_channel_number_label = new QLabel("Select Channel #:");

    // select the channel number to calibrate
    select_channel_number_spinBox = new QSpinBox();
    select_channel_number_spinBox->setMinimum(0);
    select_channel_number_spinBox->setMaximum(N_LCR_CHANNELS-1);

    select_channel_widget = new QWidget(this);
    select_channel_widget_layout = new QGridLayout();

    info_label_for_current_sensor_channel = new QLabel();
    info_label_for_current_sensor_channel->setMinimumWidth(200);

    select_channel_widget_layout->addWidget(select_channel_number_label, 0, 0);
    select_channel_widget_layout->addWidget(select_channel_number_spinBox, 0, 1);
    select_channel_widget_layout->addWidget(new QLabel(space_label_small), 0, 2);
    select_channel_widget_layout->addWidget(info_label_for_current_sensor_channel, 0, 3);
    select_channel_widget->setLayout(select_channel_widget_layout);    

    select_channel_widget->setMinimumWidth(400);


    calibrate_sensors_dialog_layout->addWidget(select_channel_widget);


    //QObject::connect(select_channel_number_spinBox, SIGNAL(valueChanged(int)), this, SLOT(Update_Virtual_Calibrate_Sensors_Dialog()) );

    select_channel_number_spinBox->setValue(*previous_ch_number_for_saved_calib_dialog_ptr); //restore the previously selected channel number

    select_channel_widget->setMaximumWidth(150);

    select_units_using_calibration_widget = new QWidget(this);
    select_units_using_calibration_widget_layout = new QGridLayout();
    select_units_using_calibration_widget->setLayout(select_units_using_calibration_widget_layout);
    select_units_using_calibration_label = new QLabel("Units of the Calibration:");
    select_units_using_calibration_widget_layout->addWidget(select_units_using_calibration_label,0,0);
    select_units_using_calibration_box = new QLineEdit();
    select_units_using_calibration_widget_layout->addWidget(select_units_using_calibration_box, 0, 1);

    top_select_widget = new QWidget(this);
    top_select_widget_layout = new QGridLayout();
    top_select_widget->setLayout(top_select_widget_layout);

    //fill the units edit box with the units used on the calibration of the selected sensor
    select_units_using_calibration_box->setText(calib_units_vector_ptr[select_channel_number_spinBox->value()]);

    top_select_widget_layout->addWidget(select_units_using_calibration_widget, 0, 2);

    calibrate_sensors_widget_layout->addWidget(top_select_widget, 0, 0);


    calib_channel_mode_groupBox = new QGroupBox(tr("Indicate the settings required for this calibration to be used:"), this);
    calib_channel_mode_groupBox_layout = new QGridLayout();
    calib_channel_mode_groupBox->setLayout(calib_channel_mode_groupBox_layout);

    calib_channel_mode_label = new QLabel("Channel mode:",this);
    calib_channel_mode_comboBox = new QComboBox(this);
    //fill calibration_channel_mode, place the default selection of multiple/single CHs on multiple
    calib_channel_mode_comboBox->addItem("Multiple Simultaneous CHs");  //multiple channel mode tag/name must be on index=0 of the ComboBox to match with variable calib_channel_mode_bool(=0 Multiple Channel; =1 Single Channel)
    calib_channel_mode_comboBox->addItem("Single CH (better precision)");   //single channel mode tag/name must be on index=1 of the ComboBox to match with variable calib_channel_mode_bool(=0 Multiple Channel; =1 Single Channel)

    if(counters_calib_ptr->calib_channel_mode_bool==0 || counters_calib_ptr->calib_channel_mode_bool==1) {
        calib_channel_mode_comboBox->setCurrentIndex(counters_calib_ptr->calib_channel_mode_bool);
    }
    else {
        calib_channel_mode_comboBox->setCurrentIndex(0);
    }

    calib_channel_mode_groupBox_layout->addWidget(calib_channel_mode_label, 0, 0);
    calib_channel_mode_groupBox_layout->addWidget(calib_channel_mode_comboBox, 0, 1);
    calib_channel_mode_groupBox_layout->addWidget(new QLabel(space_label_small), 0, 2);

    calib_channel_jumper_selection_osc_tuning_range_label = new QLabel("Jumper selection (JPX connector):", this);
    calib_channel_jumper_selection_osc_tuning_range_comboBox = new QComboBox(this);
    //fill calib_channel_jumper_selection_osc_tuning_range, fill the ComboBox with the option: A, B, +, - , for the position of the jumper on the JPX connector
    calib_channel_jumper_selection_osc_tuning_range_comboBox->addItem("+ (AB)");
    calib_channel_jumper_selection_osc_tuning_range_comboBox->addItem("- ( )");
    calib_channel_jumper_selection_osc_tuning_range_comboBox->addItem("A");
    calib_channel_jumper_selection_osc_tuning_range_comboBox->addItem("B");
    calib_channel_jumper_selection_osc_tuning_range_comboBox->setMaximumWidth(150);

    calib_channel_mode_groupBox_layout->addWidget(calib_channel_jumper_selection_osc_tuning_range_label, 0, 3);
    calib_channel_mode_groupBox_layout->addWidget(calib_channel_jumper_selection_osc_tuning_range_comboBox, 0, 4);

    calibrate_sensors_dialog_layout->addWidget(calib_channel_mode_groupBox);


    calibrate_sensors_dialog_layout->addWidget(calibrate_sensors_widget);

    calibrate_sensors_widget->setMaximumWidth(500);


    //excel table used to visualize the values of the calibration for the selected channel
    table_view_calib = new QTableView(this);

    calib_table_groupBox = new QGroupBox("Calibration:", this);
    calib_table_groupBox_layout = new QGridLayout();
    calib_table_groupBox->setLayout(calib_table_groupBox_layout);

    table_button_widget = new QWidget(this);
    table_button_widget_layout = new QGridLayout();
    table_button_widget->setLayout(table_button_widget_layout);

    //declare the buttons for the calibrations table
    add_new_row_calib_button = new QPushButton("Add Row");
    remove_row_calib_button = new QPushButton("Remove Row");
    adapt_table_calib_button = new QPushButton("Adapt Table");

    add_new_row_calib_button->setMaximumWidth(100);
    remove_row_calib_button->setMaximumWidth(100);
    adapt_table_calib_button->setMaximumWidth(100);

    QObject::connect(add_new_row_calib_button, SIGNAL(clicked()), this, SLOT(Add_New_Row()) );
    QObject::connect(remove_row_calib_button, SIGNAL(clicked()), this, SLOT(Remove_Selected_Row()) );
    QObject::connect(adapt_table_calib_button, SIGNAL(clicked()), this, SLOT(Adapt_Calib_Table()) );

    table_button_widget_layout->addWidget(add_new_row_calib_button, 0, 0);
    table_button_widget_layout->addWidget(remove_row_calib_button, 0, 1);
    table_button_widget_layout->addWidget(adapt_table_calib_button, 0, 2);
    table_button_widget_layout->addWidget(new QLabel(space_label_xl), 0, 3);
    calib_table_groupBox_layout->addWidget(table_button_widget);

    calib_table_groupBox_layout->addWidget(table_view_calib, 1, 0);


    // ---------- GroupBox for the Counters Calibration ---------
    calib_counters_groupBox = new QGroupBox("Counters Calibration:", this);
    calib_counters_groupBox_layout = new QGridLayout();
    calib_counters_groupBox->setLayout(calib_counters_groupBox_layout);

    // ---------- C0_C1_C2_C3_constants --------------
    // DoubleSpin boxes used to save the calibration constants of the sensor counters
    C0_counter_calib = new QDoubleSpinBox();
    C1_counter_calib = new QDoubleSpinBox();
    C2_counter_calib = new QDoubleSpinBox();
    C3_counter_calib = new QDoubleSpinBox();

    C0_counter_calib->setLocale(QLocale::C);    //configure the QDoubleSpinBox to use '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations
    C0_counter_calib->setMinimum(-HUGE_VAL);
    C0_counter_calib->setMaximum(HUGE_VAL);
    C1_counter_calib->setLocale(QLocale::C);    //configure the QDoubleSpinBox to use '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations
    C1_counter_calib->setMinimum(-HUGE_VAL);
    C1_counter_calib->setMaximum(HUGE_VAL);
    C2_counter_calib->setLocale(QLocale::C);    //configure the QDoubleSpinBox to use '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations
    C2_counter_calib->setMinimum(-HUGE_VAL);
    C2_counter_calib->setMaximum(HUGE_VAL);
    C3_counter_calib->setLocale(QLocale::C);    //configure the QDoubleSpinBox to use '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations
    C3_counter_calib->setMinimum(-HUGE_VAL);
    C3_counter_calib->setMaximum(HUGE_VAL);

    C0_counter_calib->setMinimumWidth(100);
    C1_counter_calib->setMinimumWidth(100);
    C2_counter_calib->setMinimumWidth(100);
    C3_counter_calib->setMinimumWidth(100);

    C0_counter_calib->setDecimals(10);
    C1_counter_calib->setDecimals(10);
    C2_counter_calib->setDecimals(10);
    C3_counter_calib->setDecimals(10);
    // ---------------------------------------

    calib_counters_groupBox_layout->addWidget(new QLabel("calibration for the counters are\n Measured_total = C3*(COUNTER^3) + C2*(COUNTER^2) + C1*COUNTER) + C0 ."), 0, 0);

    calib_counters_widget = new QWidget(this);
    calib_counters_widget_layout = new QGridLayout();
    calib_counters_widget->setLayout(calib_counters_widget_layout);

    calib_counters_widget_layout->addWidget(new QLabel("Measurement_total ="), 1, 0);
    calib_counters_widget_layout->addWidget(C3_counter_calib, 2, 0);
    calib_counters_widget_layout->addWidget(new QLabel("*(COUNTER^3) + "), 2, 1);
    calib_counters_widget_layout->addWidget(C2_counter_calib, 2, 2);
    calib_counters_widget_layout->addWidget(new QLabel("*(COUNTER^2) + "), 2, 3);

    //calib_counters_widget_layout->addWidget(new QLabel(space_label_small), 2, 3);
    //calib_counters_widget_layout->addWidget(new QLabel(space_label_small), 2, 4);

    calib_counters_widget_layout->addWidget(C1_counter_calib, 3, 0);
    calib_counters_widget_layout->addWidget(new QLabel("*COUNTER + "), 3, 1);
    calib_counters_widget_layout->addWidget(C0_counter_calib, 3, 2);

    calib_counters_groupBox_layout->addWidget(calib_counters_widget);

    units_measurement_total_widget = new QWidget();
    units_measurement_total_widget_layout = new QGridLayout();
    units_measurement_total_widget->setLayout(units_measurement_total_widget_layout);
    units_measurement_total_lineEdit = new QLineEdit();
    units_measurement_total_widget_layout->addWidget(new QLabel("Units of Measurement_total:"), 0, 0);
    units_measurement_total_widget_layout->addWidget( units_measurement_total_lineEdit, 0, 1 );

    calib_counters_groupBox_layout->addWidget(units_measurement_total_widget);


    calib_counters_groupBox_layout->addWidget(new QLabel(space_label_small));
    calib_counters_groupBox_layout->addWidget(new QLabel(space_label_small));

    // -----------------------------------------------------------

    // ---------- Calibration Tab Widget ---------
    calib_tabWidget = new QTabWidget;
    calib_tabWidget->addTab(calib_table_groupBox, tr("Calibration"));
    calib_tabWidget->addTab(calib_counters_groupBox, tr("Counters_Calibration"));

    calibrate_sensors_widget_layout->addWidget(calib_tabWidget, 1, 0);
    //-----------------------------------------------------

    zero_calibration_groupBox = new QGroupBox("Zero Calibration", this);
    zero_calibration_groupBox_layout = new QGridLayout();
    zero_calibration_groupBox->setLayout(zero_calibration_groupBox_layout);
    zero_calibration_button = new QPushButton("Zero Calibration");
    zero_calibration_groupBox_layout->addWidget(zero_calibration_button);

    button_widget = new QWidget(this);
    button_widget_layout = new QGridLayout();
    button_widget->setLayout(button_widget_layout);

    OK_button = new QPushButton("OK");
    cancel_button = new QPushButton("Cancel");

    button_widget_layout->addWidget(new QLabel(space_label_small),0,0);
    button_widget_layout->addWidget(OK_button,0,1);
    button_widget_layout->addWidget(new QLabel(space_label_small),0,2);
    button_widget_layout->addWidget(cancel_button,0,3);
    button_widget_layout->addWidget(new QLabel(space_label_small),0,4);
    button_widget_layout->addWidget( zero_calibration_groupBox, 0, 5 );

    calibrate_sensors_widget_layout->addWidget(button_widget, 4, 0);

    QObject::connect(cancel_button, SIGNAL( clicked(bool) ), this, SLOT( Cancel_Dialog() ) );
    QObject::connect(OK_button, SIGNAL( clicked(bool) ), this, SLOT(Save_Sensors_Calibrations()) );

    QObject::connect(zero_calibration_button, SIGNAL( clicked(bool) ), this, SLOT( Calibrate_Zero_Offset() ) );

    this->show();

    select_channel_number_spinBox->setValue(*previous_ch_number_for_saved_calib_dialog_ptr); //restore the previously selected channel number

    //update the values of the Calibrate_Sensors_Dialog, with data read from the sensors board
    this->Update_Calibrate_Sensors_Dialog();

    QObject::connect(select_channel_number_spinBox, SIGNAL(valueChanged(int)), this, SLOT( Update_Calibrate_Sensors_Dialog() ) );

    this->show();

}

Calibrate_Sensors_Dialog::~Calibrate_Sensors_Dialog() {
    this->close();
}

void Calibrate_Sensors_Dialog::Cancel_Dialog() {
    //save the current selected sensor channel number so next time the dialog is opened the same sensor channel is selected
    *m_previous_ch_number_for_saved_calib_dialog_ptr = this->select_channel_number_spinBox->value();
    this->reject();
}

void Calibrate_Sensors_Dialog::Set_Calib_Headers(char *header_name_col0, char *header_name_col1) {

    //don't allow table header names longer than 25 characters
    if(strlen(header_name_col0)>25) {
        header_name_col0[26]='\0';
    }
    if(strlen(header_name_col1)>25) {
        header_name_col1[26]='\0';
    }

    //save the name of the headers of the Calibration Table
    sprintf(m_headers_list[0], header_name_col0);
    sprintf(m_headers_list[1], header_name_col1);
}

QStandardItemModel *Calibrate_Sensors_Dialog::Create_Copy_Model_Table(QStandardItemModel *src_model_table) {
    int src_row_count;
    int src_col_count;
    int i, k;
    QString text_display;

    //calculate the number of rows and number of columns of the source model table
    src_row_count = src_model_table->rowCount();
    src_col_count = src_model_table->columnCount();

    //create a destination table with the same format as the source table
    QStandardItemModel *dest_model_table = new QStandardItemModel(src_row_count, src_col_count);

    for(i=0;i<src_row_count;i++) {

        for(k=0;k<src_col_count;k++) {
            dest_model_table->setItem(i, k, new QStandardItem());
            //dest_model_table->item(i,k)->setData(100, Qt::DisplayRole);

            if( src_model_table->item(i,k) != 0 ) {
                dest_model_table->setData(dest_model_table->index(i,k), src_model_table->data(src_model_table->index(i,k)));
            }

        }
    }

    return dest_model_table;
}


void Calibrate_Sensors_Dialog::Call_Create_Copy_Model_Table(int N_channels) {

    int selected_channel_number = select_channel_number_spinBox->value();

    if(selected_channel_number<N_channels) {
        if(model_tables_calib_vector_ptr[selected_channel_number]!=NULL) {
            //update the contents of the displayed model table to match the selected sensor number calibration
            current_model_table_calib=this->Create_Copy_Model_Table( (model_tables_calib_vector_ptr[selected_channel_number]) );
        }
     }

    this->Update_Calibrate_Sensors_Dialog();
}

void Calibrate_Sensors_Dialog::Add_New_Row() {
    current_model_table_calib->appendRow(new QStandardItem());
}

void Calibrate_Sensors_Dialog::Remove_Selected_Row() {
    int current_selected_row;
    int remove_row_number;

    current_selected_row=select_channel_number_spinBox->value();

    selectionM = table_view_calib->selectionModel();
    selectionL = selectionM->selectedIndexes();

    if(selectionL.count()>0) {
        remove_row_number = selectionL.at(0).row();
        current_model_table_calib->removeRow( remove_row_number );
    }
}




float Calibrate_Sensors_Dialog::Read_Measurement_From_Calib_Table(QStandardItemModel **model_tables_calib_vector, int sensor_number, float RAW_value) {
    int n_rows_current_model_table;
    int k;
    float last_table_RAW_value, current_table_RAW_value;
    float last_table_measurement_value, current_table_measurement_value;
    float a, b, measurement;

    QModelIndex model_index;

    n_rows_current_model_table = model_tables_calib_vector[sensor_number]->rowCount();

    model_index = model_tables_calib_vector[sensor_number]->index(0,0);
    if(model_index.isValid()==true) {
        last_table_RAW_value =  model_tables_calib_vector[sensor_number]->data( model_index ).toFloat();
    }
    else {
        last_table_RAW_value = HUGE_VAL;
    }

    //search the sensor calibration table until is found the RAW_value range of the calibration
    for(k=1, measurement=-HUGE_VAL;k<n_rows_current_model_table && measurement==-HUGE_VAL;k++) {

        model_index = model_tables_calib_vector[sensor_number]->index(k,0);

        if(model_index.isValid()==true) {
            current_table_RAW_value =   model_tables_calib_vector[sensor_number]->data(model_index).toFloat(); //reads the value from the table from the RAW_values column, at row k
        }
        else {
            current_table_RAW_value = HUGE_VAL;
        }


        //check if the read RAW_value is on the current range read from the frequency table
        if( (RAW_value>=last_table_RAW_value && RAW_value<=current_table_RAW_value) || (RAW_value>=current_table_RAW_value && RAW_value<=last_table_RAW_value ) ) {

            //in this case the value belongs to the current table interval, so calculate it's associated measurement value


            model_index = model_tables_calib_vector[sensor_number]->index(k-1,1);
            if(model_index.isValid()==true) {
                last_table_measurement_value = model_tables_calib_vector[sensor_number]->data(model_index).toFloat(); //reads the value from the table from the measurement_values column, at row k
            }
            else {
                last_table_measurement_value = HUGE_VAL;
            }

            model_index = model_tables_calib_vector[sensor_number]->index(k,1);
            if(model_index.isValid()==true) {
                current_table_measurement_value = model_tables_calib_vector[sensor_number]->data(model_index).toFloat(); //reads the value from the table from the measurement_values column, at row k
            }
            else {
                current_table_measurement_value = HUGE_VAL;
            }

            //calculate the line segment that approximates the sensor calibration behavior on the range [last_table_RAW_value; current_table_RAW_value]
            // measurement=a*RAW_value + b ; a = (current_table_measurement_value-last_table_measurement_value)/(current_table_RAW_value-last_table_RAW_value)
            // b = current_table_measurement_value - a*last_table_measurement_value
            a = (current_table_measurement_value-last_table_measurement_value)/(current_table_RAW_value-last_table_RAW_value);
            b = current_table_measurement_value - a*current_table_RAW_value;

            measurement=a*RAW_value+b;

        }

        last_table_RAW_value=current_table_RAW_value;
    }

    if(measurement <= -HUGE_VAL) {
        measurement = NAN;
    }

    return measurement;
}


void Calibrate_Sensors_Dialog::Adapt_Calib_Table() {
    int minimum_1st_RAW_value_index, minimum_2nd_RAW_value_index, maximum_1st_RAW_value_index, maximum_2nd_RAW_value_index;
    float minimum_1st_value[2], minimum_2nd_value[2], maximum_1st_value[2], maximum_2nd_value[2];
    int channel_number, model_table_row_count;
    int index, row_count;
    float RAW_value;
    float a_min, a_max;
    float min_EXT_RAW_value, max_EXT_RAW_value, min_EXT_measurement_value, max_EXT_measurement_value;

    QString text_display;

    QList<QStandardItem *> new_row_items;

    minimum_1st_RAW_value_index=0;
    maximum_1st_RAW_value_index=0;

    min_EXT_RAW_value=-999999999.9F;
    max_EXT_RAW_value=999999999.9F;

    channel_number=this->select_channel_number_spinBox->value();
    model_table_row_count = current_model_table_calib->rowCount();

    QModelIndex model_index;

    minimum_1st_value[0]=HUGE_VAL;
    minimum_2nd_value[0]=HUGE_VAL;
    maximum_1st_value[0]=-HUGE_VAL;
    maximum_2nd_value[0]=-HUGE_VAL;
    //search for the minimum and maximum RAW values on the calibration table
    for(index=0;index<model_table_row_count;index++) {
        model_index = current_model_table_calib->index(index,0);
        RAW_value = (float) current_model_table_calib->data(model_index).toInt();

        //check if RAW_value is the minimum
        if(RAW_value<minimum_1st_value[0]) {
            minimum_2nd_value[0]=minimum_1st_value[0];
            minimum_2nd_RAW_value_index=minimum_1st_RAW_value_index;
            minimum_1st_value[0]=RAW_value;
            minimum_1st_RAW_value_index=index;
        }
        else {
            if(RAW_value<minimum_2nd_value[0]) {
                minimum_2nd_value[0]=RAW_value;
                minimum_2nd_RAW_value_index=index;
            }
        }

        //check if RAW_value is the maximum
        if(RAW_value>maximum_1st_value[0]) {
            maximum_2nd_value[0]=maximum_1st_value[0];
            maximum_2nd_RAW_value_index=maximum_1st_RAW_value_index;
            maximum_1st_value[0]=RAW_value;
            maximum_1st_RAW_value_index=index;
        }
        else {
            if(RAW_value>maximum_2nd_value[0]) {
                maximum_2nd_value[0]=RAW_value;
                maximum_2nd_RAW_value_index=index;
            }
        }
    }

    //save the calibration measurement for the maximum and minimum RAW values
    model_index = current_model_table_calib->index(minimum_1st_RAW_value_index,1);
    minimum_1st_value[1] = current_model_table_calib->data(model_index).toFloat();
    model_index = current_model_table_calib->index(minimum_2nd_RAW_value_index,1);
    minimum_2nd_value[1] = current_model_table_calib->data(model_index).toFloat();

    model_index = current_model_table_calib->index(maximum_1st_RAW_value_index,1);
    maximum_1st_value[1] = current_model_table_calib->data(model_index).toFloat();
    model_index = current_model_table_calib->index(maximum_2nd_RAW_value_index,1);
    maximum_2nd_value[1] = current_model_table_calib->data(model_index).toFloat();

    //calculate the slope on the minimum RAW_value range, (a?, measurement=a*RAW_value+b)
    a_min=(minimum_2nd_value[1]-minimum_1st_value[1])/(minimum_2nd_value[0]-minimum_1st_value[0]);
    a_max=(maximum_1st_value[1]-maximum_2nd_value[1])/(maximum_1st_value[0]-maximum_2nd_value[0]);

    //caculate the 2 samples to make the extended calibration table
    min_EXT_measurement_value = a_min*min_EXT_RAW_value + minimum_1st_value[1] - a_min*minimum_1st_value[0];
    max_EXT_measurement_value = a_max*max_EXT_RAW_value + maximum_1st_value[1] - a_max*maximum_1st_value[0];

    //add to the current_model_table_calib the new low extension for calibration values
    new_row_items.clear();
    new_row_items.append(new QStandardItem());
    new_row_items.append(new QStandardItem());
    current_model_table_calib->appendRow(new_row_items);
    row_count=current_model_table_calib->rowCount();

    model_index = current_model_table_calib->index(row_count-1,0);
    current_model_table_calib->setData(model_index, min_EXT_RAW_value);
    model_index = current_model_table_calib->index(row_count-1,1);
    current_model_table_calib->setData(model_index, min_EXT_measurement_value);

    //add to the current_model_table_calib the new high extension for calibration values
    new_row_items.clear();
    new_row_items.append(new QStandardItem());
    new_row_items.append(new QStandardItem());
    current_model_table_calib->appendRow(new_row_items);

    row_count=current_model_table_calib->rowCount();

    model_index = current_model_table_calib->index(row_count-1,0);
    current_model_table_calib->setData(model_index, max_EXT_RAW_value);
    model_index = current_model_table_calib->index(row_count-1,1);
    current_model_table_calib->setData(model_index, max_EXT_measurement_value);

    //sort the table by ascending value of the RAW_value
    current_model_table_calib->sort(0,Qt::AscendingOrder);

}


void Calibrate_Sensors_Dialog::Calibrate_Zero_Offset() {

    QString msg_text;
    int channel_number;
    const char *space_label_small = "  ";

    channel_number=this->select_channel_number_spinBox->value();

    calib_zero_dialog = new QDialog(this);
    QGridLayout *calib_zero_dialog_layout = new QGridLayout();
    calib_zero_dialog->setLayout(calib_zero_dialog_layout);

    QWidget *manual_calib_insert_value_widget = new QWidget(this);
    QGridLayout *manual_calib_insert_value_widget_layout = new QGridLayout();
    manual_calib_insert_value_widget->setLayout(manual_calib_insert_value_widget_layout);

    QTextStream(&msg_text) << "Correct offset of the sensor calibration:";
    QLabel *msg_title_calib_correct_offset = new QLabel(msg_text);

    QLabel *msg_raw_value_to_correct_offset = new QLabel("RAW_value_(to_correct_offset):");
    raw_value_to_correct_offset_spinBox = new QDoubleSpinBox(this);

    raw_value_to_correct_offset_spinBox->setMinimum(-999999999999);
    raw_value_to_correct_offset_spinBox->setMaximum(999999999999);
    raw_value_to_correct_offset_spinBox->setLocale(QLocale::C);    //configure the QDoubleSpinBox to use '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations

    QLabel *msg_measurement_to_correct_offset = new QLabel("Measurement_(to_correct_offset):");
    measurement_to_correct_offset_spinBox = new QDoubleSpinBox(this);

    measurement_to_correct_offset_spinBox->setMinimum(-999999999999);
    measurement_to_correct_offset_spinBox->setMaximum(999999999999);
    measurement_to_correct_offset_spinBox->setLocale(QLocale::C);    //configure the QDoubleSpinBox to use '.'(dot) as the decimal separator in accordance with the format used on the files where is saved the Multiple Sensor Interface calibrations

    calib_zero_dialog_layout->addWidget(msg_title_calib_correct_offset, 0, 0);

    //widget to insert the manually the RAW value and measurement to correct the offset of the calibration
    manual_calib_insert_value_widget_layout->addWidget(msg_raw_value_to_correct_offset, 0, 0);
    manual_calib_insert_value_widget_layout->addWidget(raw_value_to_correct_offset_spinBox, 0, 1);
    manual_calib_insert_value_widget_layout->addWidget(msg_measurement_to_correct_offset, 1, 0);
    manual_calib_insert_value_widget_layout->addWidget(measurement_to_correct_offset_spinBox, 1, 1);
    calib_zero_dialog_layout->addWidget(manual_calib_insert_value_widget, 1, 0);

    QWidget *calib_zero_button_widget = new QWidget();
    QGridLayout *calib_zero_button_widget_layout = new QGridLayout();
    calib_zero_button_widget->setLayout(calib_zero_button_widget_layout);
    QPushButton *calib_zero_OK_button = new QPushButton("OK", this);
    QPushButton *calib_zero_cancel_button = new QPushButton("Cancel", this);
    calib_zero_button_widget_layout->addWidget(new QLabel(space_label_small), 0, 0);
    calib_zero_button_widget_layout->addWidget(calib_zero_OK_button, 0, 1);
    calib_zero_button_widget_layout->addWidget(new QLabel(space_label_small), 0, 2);
    calib_zero_button_widget_layout->addWidget(calib_zero_cancel_button, 0, 3);

    calib_zero_dialog_layout->addWidget(calib_zero_button_widget, 2, 0);

    QObject::connect(calib_zero_OK_button, SIGNAL(clicked()), this, SLOT( Calibrate_Zero_Offset_part2() ) );
    QObject::connect(calib_zero_cancel_button, SIGNAL(clicked()), calib_zero_dialog, SLOT(reject()) );

    calib_zero_dialog->exec();

}

void Calibrate_Sensors_Dialog::Calibrate_Zero_Offset_part2() {
    float raw_value_to_correct_offset=-HUGE_VAL, measurement_to_correct_offset=-HUGE_VAL;
    int return_value;
    float offset_correction;
    int channel_number, model_table_row_count, i;
    float calib_table_measurement, measurement_from_table_at_the_offset;
    QModelIndex model_index;

    channel_number = select_channel_number_spinBox->value();

    raw_value_to_correct_offset = raw_value_to_correct_offset_spinBox->value();
    measurement_to_correct_offset = measurement_to_correct_offset_spinBox->value();

    if(raw_value_to_correct_offset!=-HUGE_VAL) {
        measurement_from_table_at_the_offset = Calibrate_Sensors_Dialog::Read_Measurement_From_Calib_Table(model_tables_calib_vector_ptr, channel_number, raw_value_to_correct_offset);

        //check if the reading was sucessful (this is different from NAN)
        if( measurement_from_table_at_the_offset == measurement_from_table_at_the_offset ) {

            offset_correction = measurement_from_table_at_the_offset -  measurement_to_correct_offset;

            //rewrite the table with the correction of the zero offset calibration
            model_table_row_count = current_model_table_calib->rowCount();

            //DEBUG CODE
            char msg_str[600];
            sprintf(msg_str, "@ Calibrate_Sensors_Dialog::Calibrate_Zero_Offset_part2(...)");
            QMessageBox msgBox3;
            sprintf(msg_str, "CH#:%d; raw_value_to_correct_offset:%f ; measurement_from_table_at_the_offset: %f ; offset_correction: %f ; model_table_row_count: %d", channel_number, raw_value_to_correct_offset, measurement_from_table_at_the_offset, offset_correction, model_table_row_count);
            msgBox3.setText(msg_str);
            msgBox3.exec();
            //DEBUG CODE

            for(i=0;i<model_table_row_count; i++) {
                model_index = current_model_table_calib->index(i,1);
                calib_table_measurement = current_model_table_calib->data(model_index).toFloat();

                current_model_table_calib->setData(model_index,calib_table_measurement - offset_correction);
            }

        }

    }

    calib_zero_dialog->accept();  //close the Calibrate_Zero_Offset Dialog
}



void Calibrate_Sensors_Dialog::Update_Calibrate_Sensors_Dialog() {
    int i;
    char units_text[N_CHARS_CALIB_UNITS];

    int selected_channel_number = select_channel_number_spinBox->value();

    if(model_tables_calib_vector_ptr[selected_channel_number]!=NULL) {

        //fill the units edit box with the units used on the calibration of the selected sensor
        select_units_using_calibration_box->setText(m_calib_units_vector_ptr[select_channel_number_spinBox->value()]);

        //update the contents of the displayed model table to match the selected sensor number calibration
        current_model_table_calib = this->Create_Copy_Model_Table( (model_tables_calib_vector_ptr[selected_channel_number]) );

        table_view_calib->setModel( current_model_table_calib );

        //set the names of the header of the table
        for(i=0;i<2; i++) {
            table_view_calib->horizontalHeader()->model()->setHeaderData(i, Qt::Horizontal, m_headers_list[i]);
        }

    }

    //check if the currently selected channel is and ADC channel (this corresponds to channel # 6,7,8,9 )
    if( selected_channel_number >= N_LCR_CHANNELS ) {
        this->calib_channel_mode_groupBox->setDisabled(true);    //the ADC sensor channels don't have any jumpers (to select the oscillation tuning range) and the calibration table should be the same whether in single channel or multiple channel mode
        this->calib_counters_groupBox->setDisabled(true);           //the ADC sensor channels don't have any counters, so the counter calibration isn't available

        this->info_label_for_current_sensor_channel->setText( tr("<b>ADC_%1 (ADC sensor channel)</b>").arg(selected_channel_number-N_LCR_CHANNELS) );
        this->Set_Calib_Headers("Voltage(V)", "Measurement");
    }
    else {
        this->calib_channel_mode_groupBox->setDisabled(false);    //the LCR sensor channels must have the jumper selection (to select the oscillation tuning range) and the channel mode selection(single or multiple) available
        this->calib_counters_groupBox->setDisabled(false);           //the LCR sensor channels must have the counter calibration available

        this->info_label_for_current_sensor_channel->setText(  tr("<b>CH_%1 (LCR sensor channel)</b>").arg(selected_channel_number) );
        this->Set_Calib_Headers("Freq(Hz)", "Measurement");

        C3_counter_calib->setValue(m_counters_calib_ptr->counters_calib_C3_vector[selected_channel_number]);
        C2_counter_calib->setValue(m_counters_calib_ptr->counters_calib_C2_vector[selected_channel_number]);
        C1_counter_calib->setValue(m_counters_calib_ptr->counters_calib_C1_vector[selected_channel_number]);
        C0_counter_calib->setValue(m_counters_calib_ptr->counters_calib_C0_vector[selected_channel_number]);

        strncpy(units_text, m_counters_calib_ptr->counters_calib_units[selected_channel_number], 30);
        units_measurement_total_lineEdit->setText(units_text);
    }

}

void Calibrate_Sensors_Dialog::Save_Sensors_Calibrations() {
    int selected_calib_sensor_number;

    selected_calib_sensor_number = select_channel_number_spinBox->value();
    //update the contents of the displayed model table to the correspondent sensor number calibration
    (model_tables_calib_vector_ptr[selected_calib_sensor_number]) = this->Create_Copy_Model_Table(current_model_table_calib);

    //save the units of the calibration
    strncpy(m_calib_units_vector_ptr[selected_calib_sensor_number], select_units_using_calibration_box->text().toUtf8(), sizeof(m_calib_units_vector_ptr[selected_calib_sensor_number])-1 );  //use strncpy to avoid writing outside the allocated space of the variable
    m_calib_units_vector_ptr[selected_calib_sensor_number][30]='\0';

    //save the current selected sensor channel number so next time the dialog is opened the same sensor channel is selected
    *m_previous_ch_number_for_saved_calib_dialog_ptr = this->select_channel_number_spinBox->value();

    //save calib_channel_mode_comboBox
    m_counters_calib_ptr->calib_channel_mode_bool = calib_channel_mode_comboBox->currentIndex();

    //save the calibration constants and units for the calibration of sensor counter of the selected channel
    m_counters_calib_ptr->counters_calib_C3_vector[selected_calib_sensor_number] = (float) C3_counter_calib->value();
    m_counters_calib_ptr->counters_calib_C2_vector[selected_calib_sensor_number] = (float) C2_counter_calib->value();
    m_counters_calib_ptr->counters_calib_C1_vector[selected_calib_sensor_number] = (float) C1_counter_calib->value();
    m_counters_calib_ptr->counters_calib_C0_vector[selected_calib_sensor_number] = (float) C0_counter_calib->value();
    strncpy(m_counters_calib_ptr->counters_calib_units[selected_calib_sensor_number], units_measurement_total_lineEdit->text().toUtf8(), 30);

    this->accept();
}


void Calibrate_Sensors_Dialog::Set_Rows_Color(int start_row, int end_row, QColor color) {

    int row,col;

    int row_count = table_view_calib->model()->rowCount();
    int column_count = table_view_calib->model()->columnCount();

    for(row=start_row; row<row_count && row<=end_row; row++) {
        QModelIndex index2 = table_view_calib->model()->index(row,0,QModelIndex());
        // set the background color to all siblings (includes the index itself..)
        for (col = 0; col < column_count; ++col)
        {
            QModelIndex sibling = index2.sibling(row,col);
            table_view_calib->model()->setData(sibling, color, Qt::BackgroundColorRole);
        }
    }
}


int Calibrate_Sensors_Dialog::Calib_Table_Row_Count() {
    return this->table_view_calib->model()->rowCount();
}

// --------------------------------------------------------------------------------------------
