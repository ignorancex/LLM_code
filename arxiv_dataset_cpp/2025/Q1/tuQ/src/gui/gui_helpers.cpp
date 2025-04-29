//
// assorted helper functions & classes
//
#include "gui_helpers.hpp"

#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>


//public:
GraphSelect::GraphSelect(QWidget * parent) : QDialog(parent)
{
   createVerticalGroupBox();

   auto * dialogFrame= new QVBoxLayout;
   dialogFrame->addWidget(verticalGroupBox);

   setLayout(dialogFrame);
}

InputDialog::InputDialog(QString menu_function, QWidget * parent)
      : QDialog(parent), dialog_type(menu_function), form(new QFormLayout(this))
{
   form->addRow(new QLabel(dialog_type));

   auto * buttonBox= new QDialogButtonBox(QDialogButtonBox::Ok
                                          | QDialogButtonBox::Cancel,Qt::Horizontal, this);
   form->addRow(buttonBox);

   connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
   connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

//private:
void GraphSelect::createVerticalGroupBox() {
   verticalGroupBox= new QGroupBox(tr("which tuQ setting do you require?"));
   auto * layout= new QVBoxLayout;

   for (int i= 0; i < 3; ++i) {
      buttons[i]= new QPushButton;
      layout->addWidget(buttons[i]);
   }

   buttons[0]->setText("Modeller");
   buttons[1]->setText("Simulator");
   buttons[2]->setText("Compiler");

   buttons[3]= new QPushButton(tr("Exit tuQ"));
   buttons[3]->setStyleSheet("QPushButton { color: red; }");
   layout->addWidget(buttons[3]);

   verticalGroupBox->setLayout(layout);
}
