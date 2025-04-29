//
// proxy lattice for (OperatorPalette) patterns
//

#include "algorithmlattice.hpp"


// public
AlgorithmLattice::AlgorithmLattice(QWidget * parent)
      : QGraphicsScene(parent), p_stats(new LatticeStats(1,1))
{
   p_operatorType= nullptr;
   p_operators= new OperatorPalette();

   p_initialiseRow= new SignMeasure(ket0);
   p_initialiseRow->setPos(nodeAddress[0][0]);
   addItem(p_initialiseRow);
   // column counter reflects |0> at nodeAddress[0][0]
   columnAtRow[*rowMarker] += 1;

   p_stats->setPos(statsPos);

   // method by (p_operators) button id, excluding button 'add row'
   connect(p_operators->measurement_buttons,&QButtonGroup::idClicked
           ,[this](const int id){
      placeOperator(p_operators->measurements[id], columnAtRow[*rowMarker]);
   });
   connect(p_operators->pattern_buttons,&QButtonGroup::idClicked
           ,[this](const int id){ placeOperator(p_operators->patterns[id]
           , columnAtRow[*rowMarker]); });
   // 'add row' button
   connect(p_operators->p_addRow, &QPushButton::clicked, [this](){ addRow(); });
   // 'switch rows' form
   connect(p_operators->possibleRows
           ,QOverload<int>::of(&QComboBox::highlighted)
           ,[this](int index){ *rowMarker= index; });

   p_operators->show();
   addItem(p_stats);
}

void AlgorithmLattice::addRow() {
//   add a row in response to pre-conditions
//   pre-condition: either
//      - user clicks OperatorPalette button, 'Add Row'; OR
//      - user clicks OperatorPalette button, 'CNOT t downwards arrow',
//   but not both.
//   post-condition: N/A

   // cap, number of rows at latticeDim
   if (*maxRowMarker < latticeDim)
      *maxRowMarker += 1;
   else
      return ;

   // update the QComboBox, 'switch rows'
   auto rowID= (int) *maxRowMarker;
   auto rowValue= QString::number(*maxRowMarker);
   p_operators->possibleRows->insertItem(rowID,rowValue);
   // reset 'switch rows' display to the added row
   p_operators->possibleRows->setCurrentIndex(rowID);

   // align nodeAddress[row][...] with added row
   unsigned long nowRowMarker= *maxRowMarker;
   *rowMarker= nowRowMarker;

   // initialise the new row
   p_initialiseRow= new SignMeasure(ket0);
   p_initialiseRow->setPos(nodeAddress[*rowMarker][0]);
   addItem(p_initialiseRow);

   // advance column counter at inserted row
   columnAtRow[*rowMarker] += 1;
}

void AlgorithmLattice::prepareOperator(SignMeasure & graphOperator
      , unsigned long row
      , unsigned long column) {
// meet placement requirements common to CNOT/non-CNOT operators
// pre-condition: subsidiary to method, placeOperator()
//   post-condition: N/A

   // (set)Pos = parent coordinates else, scene coordinates
   graphOperator.setPos(nodeAddress[row][column]);
   // retain nodeAddress[...][column] of current row
   columnAtRow[row] += 1;
}


// private
void AlgorithmLattice::alignColumns(unsigned long control
                                    , unsigned long target) {
   if (columnAtRow[control] < columnAtRow[target])
      columnAtRow[control]= columnAtRow[target];
   else if (columnAtRow[control] > columnAtRow[target])
      columnAtRow[target]= columnAtRow[control];
}

void AlgorithmLattice::placeOperator(QString sign, unsigned long column) {
//   instantiate and place graph operators at (simulator_helpers) nodeAddress
//   coordinates
//   pre-condition: user clicks any of the 'measurement bases' or 'measurement
//   patterns' buttons
//   post-condition: ordered, traversable path of 1+ SignMeasure objects

   // provision to identify previous, adjacent-by-column operator
   QGraphicsItem * signMeasureAtPreviousColumn= itemAt(
         nodeAddress[*rowMarker][column - 1], QTransform());

   // edge case!
   //             signMeasureAtPreviousColumn: catch nullptr arising from the
   // previous operator being 'CNOT downwards arrow' and its control row not
   // being (switch row) active
   QString previousOperator {};
   if (signMeasureAtPreviousColumn == nullptr){
      QGraphicsItem * operatorAtRowMinusOnePreviousColumn= itemAt(
            nodeAddress[*rowMarker - 1][column - 1], QTransform());
      auto * p_cnotAtPreviousColumn= qgraphicsitem_cast<SignMeasure *>(
            operatorAtRowMinusOnePreviousColumn);

      if (p_cnotAtPreviousColumn != nullptr)
         previousOperator= p_cnotAtPreviousColumn->showOperator();
   }
   else {
      auto * p_operatorAtPreviousColumn= qgraphicsitem_cast<SignMeasure *>(
            signMeasureAtPreviousColumn);
      previousOperator= p_operatorAtPreviousColumn->showOperator();
   }

   // caps on number of columns
   // cap 1: 'readout' operator marks the end of this row
   if (previousOperator == "+")
      return ;
   // cap 2: number of columns at latticeDim
   if (column == (latticeDim - 1)){
      p_endOfRow= new SignMeasure(ketPlus);
      prepareOperator(*p_endOfRow,*rowMarker, column);
      addItem(p_endOfRow);

      return ;
   }

   // format the lattice marker of argument 'sign'
   if (sign == "readout")
      p_operatorType= new SignMeasure(ketPlus);
   else
      p_operatorType= new SignMeasure(sign);

   // place the operator
   if (sign == "CNOT t" % QChar(0x2191)){   // operator, CNOT t upwards arrow
      if (*rowMarker > 0){
         unsigned long rowCNOTUpwardsArrow= *rowMarker - 1;
         // level the columns
         alignColumns(*rowMarker, rowCNOTUpwardsArrow);

         prepareOperator(*p_operatorType,rowCNOTUpwardsArrow
                         , columnAtRow[rowCNOTUpwardsArrow]);
         // advance column count at adjacent, up row
         columnAtRow[*rowMarker] += 1;
      }
      else
         // CNOT t upwards arrow does not insert a row!!
         return ;
   }
   else if (sign == "CNOT t" % QChar(0x2193)){   // operator, CNOT t downwards arrow
      // edge case!
      //             three consecutive CNOTs as proxy for Swap gate
      unsigned long activeRow= p_operators->possibleRows->
            currentText().toULong();   // the (switch) row presently active for the user
      if (previousOperator == "CNOT t" % QChar(0x2191)
      && activeRow != *rowMarker){
         prepareOperator(*p_operatorType, activeRow, column);
         // align column counts of control and target rows
         columnAtRow[*maxRowMarker]= column + 1;
      }
      else {
         unsigned long rowCNOTDownwardsArrow= *rowMarker + 1;
         // level the columns
         alignColumns(*rowMarker, rowCNOTDownwardsArrow);

         prepareOperator(*p_operatorType, *rowMarker
                         ,columnAtRow[rowCNOTDownwardsArrow]);

         // IFF the target row does not exist, add a new row
         if (maxRow == nodeRow)
            addRow();
         else   // edge case!  if addRow() has not fired, *rowmarker != target row
            columnAtRow[*rowMarker + 1] += 1;
      }
   }
   else   // non-CNOT operators
      prepareOperator(*p_operatorType, *rowMarker, column);

   addItem(p_operatorType);
}
