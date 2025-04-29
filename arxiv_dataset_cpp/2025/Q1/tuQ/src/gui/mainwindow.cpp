//
// framework for graph state interaction
//
#include "mainwindow.hpp"

#include <algorithm>
#include <QDebug>
#include <QFile>
#include <QFileDialog>
#include <QMenuBar>
#include <QRegExpValidator>
#include <QTextStream>


// public:
MainWindow::MainWindow(QWidget * parent)
   : QMainWindow(parent)
{
   setActions();
   createMenus();

   settings= new GraphSelect();
   // tuQ settings, 'Modeller'
   connect(settings->buttons[0],&QPushButton::clicked, this, &MainWindow::setModeller);
   // tuQ settings, 'Simulator'
   connect(settings->buttons[1],&QPushButton::clicked,this, &MainWindow::setSimulator);
   // tuQ settings, 'Compiler'
   //connect(settings->buttons[2],&QPushButton::clicked,this, &MainWindow::setCompiler);
   // tuQ settings, 'Exit tuQ'
   connect(settings->buttons[3],&QPushButton::clicked, this, &MainWindow::noSession);

   settings->setModal(true);   // prevent a user from bypassing 'settings'
   settings->show();

}

// private:
void MainWindow::addLattice() {
   // create input Dialog, positive integers only
   // pre-condition: call via MainWindow menu option
   // post-condition: populate menu function, Lattice > Add Lattice

   // TO DO: vanilla Linux includes title bar on dialog?
   auto * dialog= new InputDialog("Add Lattice");

   // "Rows" and "Columns" QLineEdits
   QRegExp positive_integer {"(0|[1-9][0-9]+)"};

   auto * rowsLineEdit= new QLineEdit(dialog);
   rowsLineEdit->setValidator(new QRegExpValidator(positive_integer));
   rowsLineEdit->setPlaceholderText("0");
   dialog->form->insertRow(1,"&Rows", rowsLineEdit);

   auto * columnsLineEdit= new QLineEdit(dialog);
   columnsLineEdit->setValidator(new QRegExpValidator(positive_integer));
   columnsLineEdit->setPlaceholderText("0");
   dialog->form->insertRow(2,"&Columns", columnsLineEdit);

   if (dialog->exec() == QDialog::Accepted){
      // QLineEdit values set at QDialog::Accepted
      unsigned long rows= rowsLineEdit->text().toULong();
      unsigned long columns= columnsLineEdit->text().toULong();

      // edge case: one input value = 0 but other input value > 0
      if ((rows == 0 && columns > 0) || (rows > 0 && columns == 0))
         view_modeller->setLattice(0, 0);
      else
         view_modeller->setLattice(rows, columns);
   }
   delete rowsLineEdit;
   delete columnsLineEdit;
}

void MainWindow::createMenus() {
   fileMenu= menuBar()->addMenu(tr("&File"));
   fileMenu->addAction(a_openGraph);
   fileMenu->addAction(a_saveGraph);

   fileMenu->addSeparator();
   fileMenu->addAction(a_openAlgorithm);
   fileMenu->addAction(a_saveAlgorithm);

   fileMenu->addSeparator();
   fileMenu->addAction(tr("E&xit"), [this](){
      QWidget::close();
      if (*p_view_setting == tuQ_mode::simulator)
         view_simulator->s_scene->p_operators->close();
   }, tr("Ctrl+q"));

   editMenu= menuBar()->addMenu(tr("&Edit"));
   // TO DO: *** function; shortcut key(s) Ctrl + Z ***
   //editMenu->addAction(tr("&Undo"));
   editMenu->addAction(tr("Cle&ar Screen"), [this]() {
      if (*p_view_setting == tuQ_mode::modeller)
         view_modeller->clear_scene();
      else if (*p_view_setting == tuQ_mode::simulator){
         delete view_simulator;
         setSimulator();
      }
      show();
   }, tr("Shift+0"));

   circuitMenu= menuBar()->addMenu(tr("&Circuit"));
   circuitMenu->addAction(a_readCircuit);
   circuitMenu->addAction(a_compile);

   graphMenu= menuBar()->addMenu(tr("&Graph"));
   graphMenu->addAction(a_addLattice);
}

void MainWindow::dialogOpen(const QString * openfile) {
   QString file_name= QFileDialog::getOpenFileName(this, tr("Open File"),""
         ,tr("Text files (*.txt)"));
   openfile= &file_name;

   // handle 'cancel' at Dialog box
   if (openfile->isEmpty())
      return ;

   if (*p_view_setting == tuQ_mode::modeller){
         QFile file(*openfile);

         // read conditions: read-only, text
         if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return ;

         QTextStream inStream(&file);
         QString marker;
         inStream >> marker;
         file.close();

         // route .txt handle to either 'openAlgorithm' or 'openGraph'; first
         // character of [algorithm].txt is always 'Greek small letter psi'
         if (modellerOpen == open_mode::algorithm && marker == "south")
            view_modeller->openAlgorithm(*openfile);
         else if (modellerOpen == open_mode::graph && marker != "south")
            view_modeller->openGraph(*openfile);
   }
   else if (*p_view_setting == tuQ_mode::simulator)
         view_simulator->openAlgorithm(*openfile);
}

void MainWindow::dialogSave(const QString * savefile) {
   QString file_name= QFileDialog::getSaveFileName(this, tr("Save File"),""
         ,tr("Text files (*.txt)"));
   // QFileDialog.setDefaultSuffix() does not work
   if (!file_name.endsWith(".txt"))
      file_name.append(".txt");

   savefile= &file_name;

   // handle 'cancel' at Dialog box
   if (savefile->isEmpty())
      return ;
   else {
      if (*p_view_setting == tuQ_mode::modeller)
         view_modeller->saveGraph(*savefile);
      else if (*p_view_setting == tuQ_mode::simulator)
         view_simulator->saveAlgorithm(*savefile
                                       , view_simulator->s_scene->columnAtRow);
   }
}

void MainWindow::noSession() {
   settings->close();
   this->close();
}

void  MainWindow::readCircuitDialog(const QString * circuitFile) {
   QString file_name= QFileDialog::getOpenFileName(this, tr("Read Circuit"),""
         ,tr("Json files (*.json)"));

   circuitFile= &file_name;

   // handle 'cancel' at Dialog box
   if (circuitFile->isEmpty())
      return ;
   else {
      if (*p_view_setting == tuQ_mode::modeller)
         view_modeller->readCircuit(*circuitFile);
      else if (*p_view_setting == tuQ_mode::simulator)
         view_simulator->readCircuit(*circuitFile);
   }
}

void MainWindow::setActions() {
   a_addLattice= new QAction(tr("A&dd Lattice"),this);
   a_addLattice->setShortcut(tr("Ctrl+l"));
   connect(a_addLattice,&QAction::triggered,[this](){ addLattice(); });

   a_compile= new QAction(tr("Co&mpile"),this);
   a_compile->setShortcut(tr("Ctrl+m"));
   connect(a_compile,&QAction::triggered,[this](){ view_simulator->toQASM(theta); });

   a_openAlgorithm= new QAction(tr("Open Al&gorithm"),this);
   a_openAlgorithm->setShortcut(tr("Ctrl+Alt+o"));
   connect(a_openAlgorithm,&QAction::triggered
           ,[this](){ modellerOpen= open_mode::algorithm;
      dialogOpen(algorithmopenfile);
   });

   a_openGraph= new QAction(tr("&Open Graph"),this);
   a_openGraph->setShortcut(tr("Ctrl+o"));
   connect(a_openGraph,&QAction::triggered
           ,[this](){ modellerOpen= open_mode::graph;
      dialogOpen(graphopenfile);
   });

   a_readCircuit= new QAction(tr("&Read Circuit"),this);
   a_readCircuit->setShortcut(tr("Ctrl+r"));
   connect(a_readCircuit,&QAction::triggered,this
           ,[this](){ readCircuitDialog(graphreadcircuit); });

   a_saveAlgorithm= new QAction(tr("Sa&ve Algorithm"),this);
   a_saveAlgorithm->setShortcut(tr("Ctrl+Alt+s"));
   connect(a_saveAlgorithm,&QAction::triggered
           ,[this](){ dialogSave(algorithmsavefile); });

   a_saveGraph= new QAction(tr("&Save Graph"),this);
   a_saveGraph->setShortcut(tr("Ctrl+s"));
   connect(a_saveGraph,&QAction::triggered,this
           ,[this](){ dialogSave(graphsavefile); });
}
/*
void MainWindow::setCompiler() {
   // 'grey out' menu items
   a_addLattice->setEnabled(false);
   a_openGraph->setEnabled(false);
   a_readCircuit->setEnabled(false);
   a_saveAlgorithm->setEnabled(false);
   a_saveGraph->setEnabled(false);

   *p_view_setting= tuQ_mode::compiler;
   setView();

   settings->close();   // close dialog
}
*/
void MainWindow::setModeller() {
   // 'grey out' menu items
   a_compile->setEnabled(false);
   a_saveAlgorithm->setEnabled(false);

   setView();   // tuQ_mode::mode_modeller is default view_setting

   settings->close();   // close dialog
}

void MainWindow::setSimulator() {
   // 'grey out' menu items
   a_addLattice->setEnabled(false);
   a_openGraph->setEnabled(false);
   a_saveGraph->setEnabled(false);

   *p_view_setting= tuQ_mode::simulator;
   setView();

   settings->close();   // close dialog
}

// set the QGraphicsView to instantiate as central widget
void MainWindow::setView() {
   if /*(view_setting == tuQ_mode::compiler){
      view_compiler= new CompilerView(this);
      view_compiler->setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
      setCentralWidget(view_compiler);

      setWindowTitle(tr("tuQ: mode_compiler"));
   }
   else if */(view_setting == tuQ_mode::modeller){
      view_modeller= new GraphView(this);
      view_modeller->setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
      setCentralWidget(view_modeller);

      setWindowTitle(tr("tuQ: mode_modeller"));
   }
   else if (view_setting == tuQ_mode::simulator){
      view_simulator= new SimulatorView(this);
      view_simulator->setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
      setCentralWidget(view_simulator);

      setWindowTitle(tr("tuQ: mode_simulator"));
   }
}

double MainWindow::theta(QString tile) {
   // obtain theta of 0-1 for X/Y/Z-rotation through an input Dialog
   // pre-condition: an X/Y/Z-rotation component of (Simulator) algorithm
   // post-condition: specified X/Y/Z-rotation component of openQASM output

   QString rt {"rotation " % QChar(0x03B8)};
   auto * dialog= new InputDialog(rt);

   // tile format: rotation axis-row-column
   QString column, rotation, row;

   QStringList rotationSpec= tile.split(QChar(' '));
   rotation= rotationSpec.at(0);
   row= rotationSpec.at(1);
   column= rotationSpec.at(2);

   // rotation theta QLineEdit
   QRegExp positive_decimal {"^(0(.[0-9]{1,2})|1.0)"};

   auto * thetaLineEdit= new QLineEdit(dialog);
   thetaLineEdit->setValidator(new QRegExpValidator(positive_decimal));
   thetaLineEdit->setPlaceholderText("0.25");

   auto label= rotation % "-rotation, tile [" % row % "," % column % "]";
   dialog->form->insertRow(1,label, thetaLineEdit);

   if (dialog->exec() == QDialog::Accepted){
      // QLineEdit values set at QDialog::Accepted
      auto thetaValue= thetaLineEdit->text().toDouble();

      delete thetaLineEdit;

      return thetaValue;
   }
   else {
      delete thetaLineEdit;

      return 0.0;
   }
}