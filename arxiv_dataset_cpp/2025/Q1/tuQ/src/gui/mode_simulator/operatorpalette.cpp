//
// specify object, 'operator palette'
//

#include "operatorpalette.hpp"

#include <QFormLayout>
#include <QGridLayout>
#include <QLabel>
#include <QSpacerItem>


// public:
OperatorPalette::OperatorPalette(QWidget * parent) : QDialog(parent)
{
   createMeasurementsGroupBox();
   createPatternsGroupBox();
   createRowsGroupBox();

   auto * dialogLayout= new QVBoxLayout;
   auto * dialogSpacer= new QSpacerItem(1,15);

   dialogLayout->addWidget(p_measurements_groupBox);
   dialogLayout->addSpacerItem(dialogSpacer);
   dialogLayout->addWidget(p_patterns_groupBox);
   dialogLayout->addSpacerItem(dialogSpacer);
   dialogLayout->addWidget(p_rows_groupBox);

   setLayout(dialogLayout);
}

// private:
void OperatorPalette::createMeasurementsGroupBox() {
   p_measurements_groupBox= new QGroupBox(tr("measurement bases"));
   p_measurements_groupBox->setAlignment(Qt::AlignCenter);

   auto * p_hboxLayout= new QHBoxLayout;

   measurement_buttons= new QButtonGroup;
   for (unsigned short i= 0; i < measurement_nr; ++i) {
      QPushButton * measurement_button= h_createOperatorButton(measurements[i]);
      measurement_buttons->addButton(measurement_button,i);

      p_hboxLayout->addWidget(measurement_button);
   }
   p_measurements_groupBox->setLayout(p_hboxLayout);
}

void OperatorPalette::createPatternsGroupBox() {
   p_patterns_groupBox= new QGroupBox(tr("measurement patterns"));
   p_patterns_groupBox->setAlignment(Qt::AlignHCenter);

   auto * p_patternsGridLayout= new QGridLayout;

   pattern_buttons= new QButtonGroup;
   // add patterns to p_patternsGridLayout; group as pattern_buttons
   unsigned short row {0};
   unsigned short column {0};
   for (unsigned short i= 0; i < pattern_nr; ++i) {
      QPushButton * pattern_button= h_createOperatorButton(patterns[i]);
      // formatting the 'readout' button
      if (i == 8)
         pattern_button->setStyleSheet("QPushButton { color: red; }");

      pattern_buttons->addButton(pattern_button,i);

      if (i > 0){
         if ((i % 3) == 0){
            row += 1;
            column= 0;
         }
         else if ((i % 3) == 1)
            column= 1;
         else
            column= 2;
      }

      p_patternsGridLayout->addWidget(pattern_button, row, column, 1, 1);
   }
   p_patterns_groupBox->setLayout(p_patternsGridLayout);
}

void OperatorPalette::createRowsGroupBox() {
   p_rows_groupBox= new QGroupBox(tr("row operations"));
   p_rows_groupBox->setAlignment(Qt::AlignCenter);

   auto * p_addRowFormLayout= new QFormLayout;

   p_addRow= h_createOperatorButton("add row");
   p_addRowFormLayout->addWidget(p_addRow);

   auto * changeRowLabel= new QLabel;
   changeRowLabel->setStyleSheet("QLabel { color: blue; }");
   changeRowLabel->setFont(QFont{"Times New Roman", 14});
   changeRowLabel->setText("switch rows");
   possibleRows= new QComboBox;
   possibleRows->insertItem(0,"0");
   p_addRowFormLayout->addRow(changeRowLabel, possibleRows);

   p_rows_groupBox->setLayout(p_addRowFormLayout);
}
