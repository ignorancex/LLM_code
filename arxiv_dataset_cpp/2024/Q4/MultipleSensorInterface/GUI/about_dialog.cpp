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
 *   FILE: about_dialog.cpp - Here are declared the classes and functions related*
 *                  to the dialog for displaying 'About'/'License' information.  *
 ********************************************************************************/

#include  <QFile>

#include  "about_dialog.h"


About_Dialog_Impl::About_Dialog_Impl(QWidget* parent_widget /* = 0*/, QString app_version) : QDialog(parent_widget, 0), Ui::About_Dialog()
{
    setupUi(this);

    QPalette pal (m_main_text_textBrowser->palette());
    pal.setColor(QPalette::Base, pal.color(QPalette::Disabled, QPalette::Window));

    m_main_text_textBrowser->setPalette(pal);

    this->m_app_version_label->setText("Multiple-Sensor Interface GUI,   version: " + app_version);  //write the title and application version on the top of the about window

    m_main_text_textBrowser->setHtml(
        "<p style=\"margin-bottom:8px; margin-top:1px; \">" + About_Dialog_Impl::tr("Developed by %1 %2").arg("<a href=\"mailto:david.quelhas@yahoo.com?subject=000 s\">David Nuno Quelhas</a>").arg(" © 2021 , Lisboa, Portugal") + "</p>"
        "<p style=\"margin-bottom:8px; margin-top:1px; \">" + About_Dialog_Impl::tr("Drivers and Firmware developed by %1, %2").arg("David Nuno Quelhas").arg("2012-2021") + "</p>"
        "<p style=\"margin-bottom:8px; margin-top:1px; \">" + About_Dialog_Impl::tr("Software distributed under %1 ;").arg("<a href=\"http://www.gnu.org/licenses/gpl-3.0.html\">GNU GPL 3.0</a>") + "&nbsp; &nbsp;&nbsp;&nbsp;  " + About_Dialog_Impl::tr("Using %1, released under %2 ;").arg("<a href=\"http://qt-project.org\">Qt</a>").arg("<a href=\"http://www.gnu.org/licenses/lgpl-3.0.html\">GNU-LGPL 3.0</a>") + "&nbsp; &nbsp;&nbsp;&nbsp;  " + About_Dialog_Impl::tr("USB and serial drivers released under %1 ;").arg("<a href=\"http://www.gnu.org/licenses/lgpl-3.0.html\">GNU-LGPL 3.0</a>") + "</p>"
        "<p style=\"margin-bottom:8px; margin-top:1px; \">" + About_Dialog_Impl::tr("In accordance with 7-b) and 7-c) of GNU-GPLv3, this license requires to preserve/include prior copyright notices with author names, and also add new aditional copyright notices, specifically on start/top of source code files and on 'About'(or similar) dialog/windows at 'user interfaces', on the software and on any 'modified version' or 'derivative work'.") + "</p>"
        "<p style=\"margin-bottom:8px; margin-top:1px; \">" + About_Dialog_Impl::tr("Multiple-Sensor device electronic hardware invented and developed by %1, %2, hardware design and schematics published under %3 ;").arg("David Nuno Quelhas").arg("2012-2021").arg("<a href=\"http://cern-ohl.web.cern.ch/\">CERN Open Hardware Licence Version 2 - Weakly Reciprocal</a>") + "</p>"
        "<p style=\"margin-bottom:8px; margin-top:1px; \">" + About_Dialog_Impl::tr("The user-guide published under %1 ;").arg("<a href=\"https://creativecommons.org/licenses/by-sa/4.0/legalcode\">Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)</a>") + "</p>"
        "<p style=\"margin-bottom:8px; margin-top:1px; \">" + About_Dialog_Impl::tr("Home page and documentation: %1").arg("<a href=\"https://multiple-sensor-interface.blogspot.com" + QString(" ") + "/\">https://multiple-sensor-interface.blogspot.com" + QString(" ") + "/</a>") + "</p>"
        "<br>"
        "<img src=\":/images/about.png\" alt=\"multiple-sensor\" width=\"200\" height=\"100\">" + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"  + "<img src=\":/images/logo-gnu-linux-GPL-3-0.png\" alt=\"multiple-sensor\" width=\"250\" height=\"80\">"
        );

    Init_Text(m_about_device_textBrowser, ":/licenses/about_device.txt");
    Init_Text(m_GPLv30_textBrowser, ":/licenses/gplv3.txt");
    Init_Text(m_LGPLv30_textBrowser, ":/licenses/lgplv3.txt");
    Init_Text(m_CERN_OHL_W_v2_textBrowser, ":/licenses/cern_ohl_w_v2.txt");
    Init_Text(m_CC_BY_SA_textBrowser, ":/licenses/CC-by-sa-4_legalcode.txt");

    m_main_text_textBrowser->setFocus();

    this->setMaximumSize(850, 500);

    m_about_device_textBrowser->setMinimumSize (800, 350); //expand the widget size do display all its content
    this->adjustSize (); // resize the window
    m_about_device_textBrowser->setMinimumSize (400, 250); // allow shrinking afterwards

    this->tabs_tabWidget->setCurrentIndex(0);
}



void About_Dialog_Impl::Init_Text(QTextBrowser* text_display_box_ptr, const char* text_file_path_and_name)
{
    QFile text_file(text_file_path_and_name);
    //QFile::FileError err (text_file.error());
    //qDebug("file: %d", (int)err);
    //qDebug("size : %d", (int)text_file.size());
    text_file.open(QIODevice::ReadOnly);
    text_display_box_ptr->setText(QString::fromUtf8(text_file.readAll()));
}


About_Dialog_Impl::~About_Dialog_Impl()
{
}


