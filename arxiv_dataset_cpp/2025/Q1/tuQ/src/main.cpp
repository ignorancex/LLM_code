#include "gui/mainwindow.hpp"

#include <QApplication>

int main(int argc, char *argv[])
{
   QApplication tuq(argc, argv);
   MainWindow window;

   window.show();

   return tuq.exec();
}