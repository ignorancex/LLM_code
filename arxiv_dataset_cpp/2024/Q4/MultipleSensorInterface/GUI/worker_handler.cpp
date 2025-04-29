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
 *     on the software and any on 'modified version' or 'derivative work'.       *
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
 *   FILE: worker_handler.cpp - Here is the implementation of the class           *
 *             'Worker_Handler', used to allow the main process(MainWindow) to    *
 *             interact with the 'Worker' thread   (for example: getting % of    *
 *             progress status; or canceling current data transfer)              *
 ********************************************************************************/

#include "worker_handler.h"

#include <QMessageBox>  //DEBUG CODE

Worker_Handler::Worker_Handler(QObject *parent) :
    QObject(parent)
{
    m_pause = false;
}

Worker_Handler::~Worker_Handler()
{
 
}

void Worker_Handler::Cancel_Thread()
{
    // ----- DEBUG CODE -----
  //  char msg_str[200];
  //  sprintf(msg_str, "@ Worker_Handler::cancel()");
  //  QMessageBox msgBox3;
  //  msgBox3.setText(msg_str);
  //  msgBox3.exec();
    // ----------------------
    if(m_pause) m_pause_cond.wakeAll();
    emit internal__cancel_signal();
}
 
void Worker_Handler::Toggle_Paused_Thread()
{
    // ----- DEBUG CODE -----
 //   char msg_str[200];
 //    sprintf(msg_str, "@ Worker_Handler::togglePaused()");
 //    QMessageBox msgBox4;
 //    msgBox4.setText(msg_str);
 //    msgBox4.exec();
    // ----------------------

    m_pause = !m_pause;
    if(!m_pause) {
		m_pause_cond.wakeAll();
    }
    else {
		emit internal__pause_signal();
    }
}
 
void Worker_Handler::Start_Work_Thread(const QList<Worker_Arguments> &list)
{
    //Create objects and do connections
    worker = new Worker();
    worker_thread = new QThread(this);
    connect(worker, SIGNAL(max_progress_range_changed_signal(int)), this, SIGNAL(max_progress_range_changed_signal(int)));
    connect(worker, SIGNAL(progress_text_changed_signal(QString)), this, SIGNAL(progress_text_changed_signal(QString)));
    connect(worker, SIGNAL(status_text_changed_signal(QString)), this, SIGNAL(status_text_changed_signal(QString)));
    connect(worker, SIGNAL(progress_value_changed_signal(int)), this, SIGNAL(progress_value_changed_signal(int)));
    connect(worker, SIGNAL(finished_signal()), this, SIGNAL(finished_signal()));
    connect(worker, SIGNAL(finished_signal()), this, SLOT(Call_On_Finish()));
    connect(worker, SIGNAL(result_ready_signal(Data)), this, SIGNAL(result_ready_signal(Data)));
 
    connect(this, SIGNAL(internal__cancel_signal()), worker, SLOT(Cancel()));
    connect(this, SIGNAL(internal__pause_signal()), worker, SLOT(Pause()));
    connect(this, SIGNAL(internal__start_work_signal(QList<Worker_Arguments>)), worker, SLOT(Start_Work(QList<Worker_Arguments>)));
 
    connect(worker_thread, SIGNAL(finished()), worker, SLOT(deleteLater()));
    connect(worker_thread, SIGNAL(finished()), worker_thread, SLOT(deleteLater()));
 
    worker->Set_Pause_Condition(m_pause_cond);
    worker->moveToThread(worker_thread);
    //Start thread used for the upload/download of calibration tables between PC and device.
    worker_thread->start();
 
    //Start the process of the list.
    m_is_running = true;
    emit internal__start_work_signal(list);
}
 
bool Worker_Handler::Is_Paused()
{
    return m_pause;
}
 
bool Worker_Handler::Is_Running()
{
    return m_is_running;
}
 
void Worker_Handler::Call_On_Finish()
{
    m_is_running = false;
    /*
    Exit the thread event loop as we don't need it until new process has to be done.
    The thread is not deleted, as it is done automatically thanks to connection
    of signal finished_signal() to slot deleteLater().
    */
    worker_thread->exit();
    worker_thread->wait();
}
