//
// specify modeller view/scene objects
//
# include "graphview.hpp"

#include <QDebug>
#include <QFile>
#include <QKeyEvent>
#include <QScrollBar>
#include <QStringBuilder>
#include <QTextStream>
#include <QVector>


// public:
GraphView::GraphView(QWidget * parent)
   : QGraphicsView(parent), scene(new QGraphicsScene(this))
{
   // '... one unit on the scene is represented by one pixel on the screen.'
   // *** ~[121, 121] accessible at this sceneRect ***
/* tuQ v0.2:
 * setSceneRect has to be consistent between this and setLattice otherwise
 *    lattice created in one set of coordinates do not show up in the other set
*/
   scene->setSceneRect(-2500,-2500,11000,11000);

   setScene(scene);
   setMouseTracking(true);

   clabel= new QLabel;
   clabel->setWindowFlag(Qt::ToolTip);
//   clabel->setWindowOpacity(0);

   createElementMenus(scene);
}

// create the lattice specified by method, SimulatorView::latticeFromPatterns
// then repaint each vertex including its ID to reflect the individual
// measurements abstracted as Simulator tiles
void GraphView::openAlgorithm(const QString & afile) {
   // Note in every case except as output of this method, a GraphVertex ID is
   // type unsigned long

   QFile loadfile(afile);
   // read conditions: read-only, text
   if (!loadfile.open(QIODevice::ReadOnly | QIODevice::Text)){
      qDebug() << "file is not open";
      return ;
   }

   QTextStream inStream(&loadfile);

// part 1 of [algorithm].txt read: instantiate a base lattice with dimensions
// set by the source algorithm; see SimulatorView method, latticeFromPatterns
   QString direction, magnitude;
   unsigned long row {0};
   unsigned long column {0};
   inStream >> direction >> magnitude;

   if (direction == "south")
      row= magnitude.toULong();

   direction.clear();
   magnitude.clear();

   inStream >> direction >> magnitude;

   if (direction == "east")
      column= magnitude.toULong();

   // Note: column 0 of the (lattice) output of this function will always
   // represent the |G> initialisation column whereas the easternmost column
   // of the (lattice) output will represent the |G> readout column iff each
   // row of the source algorithm is closed with a readout tile
   setLattice(row, column);

   // move inStream pointer to next character (the inStream.readline()
   // of pass 1b, below, requires it)
   qint64 spos= inStream.pos();
   if (spos != -1)
      inStream.seek(spos + 1);
   else
      return ;

// part 2 of [algorithm].txt read:
//   - space the rows of the lattice as dictated by CNOT tiles of the
//     algorithm; and
//   - obtain dimensions for a 2D matrix of 'first qbit' coordinates; and
//   - initialise each 'wire' row of the lattice.
   QString cnots= inStream.readLine();

   // [algorithm].txt row addresses
   int tileRow {0};
   int rowsTileToLattice[row];
   // dimensions for patternQbit1, a 2D matrix of 'first qbit' positions
   int maxTile {0};
   int columnDimension {0};

   // initialise rowsTileToLattice
   for (int i= 0; i < row; ++i) {
      rowsTileToLattice[i]= i;
   }

   // collect tile data for populating 2D matrix of 'first qbit' coordinates
   QVector<QString> tiles;

   // space rows of the lattice by position of CNOT tiles in [algorithm].txt
   QString downCNOT= "CNOT t" % QChar(0x2193);
   QString upCNOT= "CNOT t" % QChar(0x2191);
   while (!cnots.isNull()) {
      tiles.push_back(cnots);

      int tileCounter {0};
      int matrixCol {0};
      QString tile;

      QStringList instanceCNOT= cnots.split(QChar(' '));
      if (instanceCNOT.length() == 4){
         tile= instanceCNOT.at(0) % " " % instanceCNOT.at(1);
         tileCounter= instanceCNOT.at(2).toInt();
         matrixCol= instanceCNOT.at(3).toInt();
      }
      else if (instanceCNOT.length() == 3){
         tile= instanceCNOT.at(0);
         tileCounter= instanceCNOT.at(1).toInt();
         matrixCol= instanceCNOT.at(2).toInt();
      }

      // obtain data for 2D matrix, patternQbit1, declared below
      if (tileCounter > maxTile)
         maxTile= tileCounter;
      if (matrixCol > columnDimension)
         columnDimension= matrixCol;

      // set the lattice row of both CNOT downwards arrow and CNOT upwards
      // arrow
      if (tile == downCNOT || tile == upCNOT){
         // row address of algorithm tile
         tileRow= instanceCNOT.at(2).toInt();

         int adjRowIndex;
         adjRowIndex= rowsTileToLattice[tileRow] + 2;

         int base= tileRow + 1;
         if (rowsTileToLattice[base] < adjRowIndex){
            rowsTileToLattice[base]= adjRowIndex;

            // reset all successive LATTICE row addresses
            if (base + 1 < row)
               for (int i= base + 1; i < row ; ++i) {
                  rowsTileToLattice[i] += 1;
               }
         }
      }

      cnots= inStream.readLine();
   }

   loadfile.close();

   // generic pattern variables
   QPointF pattern_pos;
   QGraphicsItem * ptr_pattern {nullptr};

   // initialise each 'wire' row of the lattice
   for (int i= 0; i < maxTile + 1; ++i) {
      int wireRow= rowsTileToLattice[i];

      pattern_pos= {0,tileToRow[wireRow]};

      // map pattern_pos to (scene) then, locate initialisation vertex within
      // scene
      ptr_pattern= scene->itemAt(
            mapToParent(pattern_pos.toPoint()), QTransform());
      auto initialisation_vertex= qgraphicsitem_cast<GraphVertex *>(ptr_pattern);

      // change vertex id to measurement prompt
      initialisation_vertex->resetVertexID(GraphVertex::measure_char::X);
   }

   // GraphView::openAlgorithm looks up x, y coordinates, preset as
   // modeller_helpers, tileToColumn and tileToRow; it follows the same
   // approach as used in methods of AlgorithmLattice and SimulatorView.
   // patternQbit1 is a 2D matrix of first qbits of patterns as lattice
   // positions
   unsigned long patternQbit1[maxTile + 1][columnDimension + 1];

   // 'populate patternQbit1', iteration 1: set patternQbit1 to default value
   // to zero and replicate the matrix as 'eachPatternSize'
   for (int r= 0; r < maxTile + 1; ++r) {
      for (int c= 0; c < columnDimension + 1; ++c) {
         patternQbit1[r][c]= 0;
      }
   }

   unsigned long eachPatternSize[maxTile + 1][columnDimension + 1];
   for (int r= 0; r < maxTile + 1; ++r) {
      for (int c= 0; c < columnDimension + 1; ++c) {
         eachPatternSize[r][c]= 0;
      }
   }

   // (QVector) 'tiles' variables
   QString p;
   int plusColumn, plusRow;

   // 'populate patternQbit1', iteration 2: set matrix, eachPatternSize with
   // count of qbits in individual patterns
   for (auto prc : tiles) {
      QStringList pPlusRowPlusColumn= prc.split(QChar(' '));

      if (pPlusRowPlusColumn.length() == 4){
         p= pPlusRowPlusColumn.at(0) % " " % pPlusRowPlusColumn.at(1);
         plusRow= pPlusRowPlusColumn.at(2).toInt();
         plusColumn= pPlusRowPlusColumn.at(3).toInt();
      }
      else if (pPlusRowPlusColumn.length() == 3){
         p= pPlusRowPlusColumn.at(0);
         plusRow= pPlusRowPlusColumn.at(1).toInt();
         plusColumn= pPlusRowPlusColumn.at(2).toInt();
      }

      // in every case, 'psi' initialisation qbit begins a row and '+'
      // readout qbit closes a row
      if (p.startsWith(QChar(0x03C8)) || p == "+")
         eachPatternSize[plusRow][plusColumn]= 0;
      else if (p.startsWith(QChar(0x03C3)))
         eachPatternSize[plusRow][plusColumn]= 1;   // tile = measurement, basis X/Y/Z
      else if (p.startsWith("CNOT"))
         eachPatternSize[plusRow][plusColumn]= 6;   // ** Note: includes tile 'CNOT_marker' **
      else
         eachPatternSize[plusRow][plusColumn]= 4;   // all other patterns
   }

   // 'populate patternQbit1', iteration 3: write cumulative count of qbits by
   // pattern to matrix patternQbit1 from individual pattern counts of matrix
   // eachPatternSize
   for (int r= 0; r < maxTile + 1; ++r) {
      unsigned long accumulator= 0;
      for (int c= 0; c < columnDimension + 1; ++c) {
         accumulator += eachPatternSize[r][c];
         patternQbit1[r][c]= accumulator;
      }
   }

   // 'populate patternQbit1', iteration 4: overwrite the cumulative
   // patternQbit1 value at any CNOT column with the greater of the cumulative
   // counts of qbits at row m and row m + 1
   QVector<QString> cnotsPos;
   for (auto prc : tiles) {
      if (prc.startsWith("CNOT "))
         cnotsPos.push_back(prc);
   }

   if (!cnotsPos.isEmpty()){
      int cnotAtRow, cnotAtColumn;

      for (QString ppos : cnotsPos) {
         QStringList cnotrc= ppos.split(QChar(' '));
         cnotAtRow= cnotrc.at(2).toInt();
         cnotAtColumn= cnotrc.at(3).toInt();

         unsigned long northQbit1, southQbit1;
         northQbit1= patternQbit1[cnotAtRow][cnotAtColumn];
         southQbit1= patternQbit1[cnotAtRow + 1][cnotAtColumn];

         unsigned long accumulator {0};

         if (northQbit1 > southQbit1){
            patternQbit1[cnotAtRow + 1][cnotAtColumn]= northQbit1;

            accumulator= northQbit1;

            for (int c= cnotAtColumn + 1; c < columnDimension + 1; ++c) {
               accumulator += eachPatternSize[cnotAtRow + 1][c];
               patternQbit1[cnotAtRow + 1][c]= accumulator;
            }
         }
         else if (southQbit1 > northQbit1){
            patternQbit1[cnotAtRow][cnotAtColumn]= southQbit1;

            accumulator= southQbit1;

            for (int c= cnotAtColumn + 1; c < columnDimension + 1; ++c) {
               accumulator += eachPatternSize[cnotAtRow][c];
               patternQbit1[cnotAtRow][c]= accumulator;
            }
         }
      }
   }

   // mapping [algorithm].txt row to lattice row
   int latticeRow;

   // set vertex ID(s) to the pattern's required individual measurement(s)
   for (auto patternRowColumn : tiles) {
      QStringList prc= patternRowColumn.split(QChar(' '));
      if (prc.length() == 4){
         p= prc.at(0) % " " % prc.at(1);
         plusRow= prc.at(2).toInt();
         plusColumn= prc.at(3).toInt();
      }
      else if (prc.length() == 3){
         p= prc.at(0);
         plusRow= prc.at(1).toInt();
         plusColumn= prc.at(2).toInt();
      }

      // Lattice row address
      latticeRow= rowsTileToLattice[plusRow];

      // set vertex id of readout pattern
      if (p == "+"){   // 'readout'
         // these are lattice readout qbits so place them at rightmost column
         QPointF readout_pos= {tileToColumn[column - 1], tileToRow[latticeRow]};
         QGraphicsItem * ptr_readout= scene->itemAt(mapToParent(readout_pos.toPoint())
               , QTransform());
         auto readout_vertex= qgraphicsitem_cast<GraphVertex *>(ptr_readout);
         // set vertex ID to required measurement
         readout_vertex->resetVertexID(GraphVertex::measure_char::Z);
      }

      // set vertex id of all other patterns except initialisation and readout
      if (p.startsWith(QChar(0x03C3))){   // individual measurement
         // locate qbit 1 of p on the lattice
         unsigned long qbitAddress= patternQbit1[plusRow][plusColumn];
         pattern_pos= {tileToColumn[qbitAddress], tileToRow[latticeRow]};
         ptr_pattern= scene->itemAt(mapToParent(pattern_pos.toPoint())
               , QTransform());
         auto v_measure= qgraphicsitem_cast<GraphVertex *>(ptr_pattern);

         // set vertex ID to required measurement
         if (p.endsWith("x"))
            v_measure->resetVertexID(GraphVertex::measure_char::X);
         else if (p.endsWith("y"))
            v_measure->resetVertexID(GraphVertex::measure_char::Y);
         else if (p.endsWith("z"))
            v_measure->resetVertexID(GraphVertex::measure_char::Z);
      }
      else if (p.startsWith("CNOT ")){
         // collect all vertices of p
         GraphVertex * p_CNOT[6];

         // extrapolate qbit 1 of the northern row of the CNOT from its
         // terminating qbit on this row.  Recall that qbit 1 of control and
         // target rows are aligned as positions in patternQbit1
         unsigned long qN= patternQbit1[plusRow][plusColumn] - 5;

         // place northern row of CNOT
         for (int i= 0; i < 6; ++i) {
            pattern_pos= {tileToColumn[qN], tileToRow[latticeRow]};
            ptr_pattern= scene->itemAt(
                  mapToParent(pattern_pos.toPoint()), QTransform());
            p_CNOT[i]= qgraphicsitem_cast<GraphVertex *>(ptr_pattern);

            qN += 1;
         }
         if (p == downCNOT){   // 'control' row
            for (int i= 0; i < 5; ++i) {
               p_CNOT[i]->resetVertexID(GraphVertex::measure_char::Y);
            }
            p_CNOT[5]->resetVertexID(GraphVertex::measure_char::X);
         }
         else if (p == upCNOT){   // 'target' row
            p_CNOT[0]->resetVertexID(GraphVertex::measure_char::X);
            p_CNOT[1]->resetVertexID(GraphVertex::measure_char::X);
            p_CNOT[2]->resetVertexID(GraphVertex::measure_char::Y);
            p_CNOT[3]->resetVertexID(GraphVertex::measure_char::X);
            p_CNOT[4]->resetVertexID(GraphVertex::measure_char::X);
            p_CNOT[5]->resetVertexID(GraphVertex::measure_char::X);
         }

         // place southern row of CNOT
         auto southRow= rowsTileToLattice[plusRow + 1];
         qN= qN - 6;

         for (int i= 0; i < 6; ++i) {
            pattern_pos= {tileToColumn[qN], tileToRow[southRow]};
            ptr_pattern= scene->itemAt(
                  mapToParent(pattern_pos.toPoint()), QTransform());
            p_CNOT[i]= qgraphicsitem_cast<GraphVertex *>(ptr_pattern);

            qN += 1;
         }
         if (p == upCNOT){   // 'control' row
            for (int i= 0; i < 5; ++i) {
               p_CNOT[i]->resetVertexID(GraphVertex::measure_char::Y);
            }
            p_CNOT[5]->resetVertexID(GraphVertex::measure_char::X);
         }
         else if (p == downCNOT){   // 'target' row
            p_CNOT[0]->resetVertexID(GraphVertex::measure_char::X);
            p_CNOT[1]->resetVertexID(GraphVertex::measure_char::X);
            p_CNOT[2]->resetVertexID(GraphVertex::measure_char::Y);
            p_CNOT[3]->resetVertexID(GraphVertex::measure_char::X);
            p_CNOT[4]->resetVertexID(GraphVertex::measure_char::X);
            p_CNOT[5]->resetVertexID(GraphVertex::measure_char::X);
         }

         // 'nexus' row
         pattern_pos= {tileToColumn[qN - 4], tileToRow[latticeRow + 1]};
         ptr_pattern= scene->itemAt(
               mapToParent(pattern_pos.toPoint()), QTransform());
         auto nexus= qgraphicsitem_cast<GraphVertex *>(ptr_pattern);
         nexus->resetVertexID(GraphVertex::measure_char::Y);
      }
      else if (!p.startsWith("CNOT") && p != QChar(0x03C8) && p != "+"){
         // all other patterns excluding CNOT_marker and (initialisation) psi:
         // assemble qbits of the pattern

         // collect all vertices of p
         GraphVertex * p_rotate_HST[4];

         // locate terminating qbit of preceding pattern on the row
         unsigned long endQbitAddress= patternQbit1[plusRow][plusColumn - 1];
         // derive first qbit of p from terminating qbit of previous pattern
         auto qN {endQbitAddress + 1};

         for (int i= 0; i < 4; ++i) {
            pattern_pos= {tileToColumn[qN], tileToRow[latticeRow]};
            ptr_pattern= scene->itemAt(
                  mapToParent(pattern_pos.toPoint()), QTransform());
            p_rotate_HST[i]= qgraphicsitem_cast<GraphVertex *>(ptr_pattern);

            qN += 1;
         }

         if (p == "Hadamard"){   // Clifford pattern: Hadamard
            p_rotate_HST[0]->resetVertexID(GraphVertex::measure_char::Y);
            p_rotate_HST[1]->resetVertexID(GraphVertex::measure_char::Y);
            p_rotate_HST[2]->resetVertexID(GraphVertex::measure_char::Y);
            p_rotate_HST[3]->resetVertexID(GraphVertex::measure_char::X);
         }
         else if (p == "S"){   // Clifford pattern: 'S' phase
            p_rotate_HST[0]->resetVertexID(GraphVertex::measure_char::X);
            p_rotate_HST[1]->resetVertexID(GraphVertex::measure_char::Y);
            p_rotate_HST[2]->resetVertexID(GraphVertex::measure_char::X);
            p_rotate_HST[3]->resetVertexID(GraphVertex::measure_char::X);
         }
         else if (p.startsWith("X")){   // '_-rotation' patterns
            p_rotate_HST[0]->resetVertexID(GraphVertex::measure_char::l_xi);

            p_rotate_HST[1]->resetVertexID(GraphVertex::measure_char::X);
            p_rotate_HST[2]->resetVertexID(GraphVertex::measure_char::X);
            p_rotate_HST[3]->resetVertexID(GraphVertex::measure_char::X);
         }
         else if (p.startsWith("Y")){
            p_rotate_HST[0]->resetVertexID(GraphVertex::measure_char::X);
            p_rotate_HST[1]->resetVertexID(GraphVertex::measure_char::X);

            p_rotate_HST[2]->resetVertexID(GraphVertex::measure_char::l_zeta);
            p_rotate_HST[3]->resetVertexID(GraphVertex::measure_char::X);
         }
         else if (p.startsWith("Z")){
            p_rotate_HST[0]->resetVertexID(GraphVertex::measure_char::X);

            p_rotate_HST[1]->resetVertexID(GraphVertex::measure_char::l_eta);

            p_rotate_HST[2]->resetVertexID(GraphVertex::measure_char::X);
            p_rotate_HST[3]->resetVertexID(GraphVertex::measure_char::X);
         }
         else if (p == "T"){
            // the vertex transformations of Raussendorf and Briegel (2002)
            // 'general rotation' will stand-in for a T-pattern
            p_rotate_HST[0]->resetVertexID(GraphVertex::measure_char::l_xi, Qt::blue);
            p_rotate_HST[1]->resetVertexID(GraphVertex::measure_char::l_eta, Qt::blue);
            p_rotate_HST[2]->resetVertexID(GraphVertex::measure_char::l_zeta, Qt::blue);
            p_rotate_HST[3]->resetVertexID(GraphVertex::measure_char::X, Qt::blue);
         }
      }
   }
}

// read instructions, format .txt
void GraphView::openGraph(const QString & rfile) {
   QFile loadfile(rfile);
   // read conditions: read-only, text
   if (!loadfile.open(QIODevice::ReadOnly | QIODevice::Text)){
      qDebug() << "file is not open";
      return ;
   }

   QTextStream in_stream(&loadfile);

   // 1. place all vertices
   QString vertex_data{in_stream.readLine()};

   while (!vertex_data.isNull()) {
      QStringList vertex_stats= vertex_data.split(QChar(' '));

// tuQ v0.2: populate nested class, GraphVertex::IDs; call
// GraphVertex::getVertexMeasure()
      // reproduce vertex
      unsigned long vid= vertex_stats.at(0).toULong();
      auto *v= new GraphVertex(vertexmenu, vid);

      // set vertex position
      double x= vertex_stats.at(1).toDouble();
      double y= vertex_stats.at(2).toDouble();
      v->setPos(x, y);

      // add vertex to scene
      scene->addItem(v);

      // set scrollbar slider positions by NW apex of lattice
      if (vid == 1){
         auto hbar= new QScrollBar(Qt::Horizontal, this);
         setHorizontalScrollBar(hbar);
         hbar->setSliderPosition((int) x - 25);

         auto vbar= new QScrollBar(Qt::Vertical, this);
         setVerticalScrollBar(vbar);
         vbar->setSliderPosition((int) y - 25);
      }

      vertex_data= in_stream.readLine();
   }
   // return pointer to beginning of in_stream
   in_stream.seek(0);

   // 2. place all edges
   QString edges_data{in_stream.readLine()};

   while (!edges_data.isNull()) {
      QStringList edge_stats= edges_data.split(QChar(' '));
      // reproduce edge
      if (edge_stats.size() > 3) {
         // details of 'reference' vertex
         double x= edge_stats.at(1).toDouble();
         double y= edge_stats.at(2).toDouble();
         QPointF v_pos {x, y};
         QGraphicsItem * ptr_ref_item= scene->itemAt(
               mapToParent(v_pos.toPoint()),QTransform());
         auto ref_v= qgraphicsitem_cast<GraphVertex *>(ptr_ref_item);

         // edge_i coordinates
         double epos_x;
         double epos_y;
         unsigned long edge_coordinates{1};

         for (int counter= 3; counter < edge_stats.size(); ++counter) {
            if (edge_coordinates % 2 == 0){
               epos_y= edge_stats.at(counter).toDouble();
               QPointF end_pos {epos_x,epos_y};

               // map end_pos to (scene) then, locate (vertex) within scene
               QGraphicsItem * ptr_other_v= scene->itemAt(
                     mapToParent(end_pos.toPoint()), QTransform());
               auto other_v= qgraphicsitem_cast<GraphVertex *>(ptr_other_v);

               // other_v is type GraphVertex?
               if (other_v->type() == 4){
                  auto * e= new GraphEdge(ref_v, other_v, edgemenu);
                  ref_v->add_edge(e);
                  other_v->add_edge(e);

                  scene->addItem(e);
               }
               epos_x= 0;
               epos_y= 0;
            }
            else
               epos_x= edge_stats.at(counter).toDouble();

            edge_coordinates += 1;
         }
      }

      edges_data= in_stream.readLine();
   }

   loadfile.close();
}

// read a quantum circuit file, format .json (schemata: 'native', which is
// adapted from ionQ; and Google Cirq, which is reformatted to the native
// schema). See comments of src/layout/io_circuit.cpp.
void GraphView::readCircuit(const QString & cjson) {
   // convert QString to utf8 string
   std::string cjson_utf8= cjson.toUtf8().constData();
   std::ifstream jsonCircuit {cjson_utf8};

   if (jsonCircuit.is_open()){
      using json= nlohmann::json;

      json parseCircuit= json::parse(jsonCircuit);   // create json object
      auto cirqCheck= parseCircuit.find("cirq_type");

      json circuitToGraph;
      if (cirqCheck != parseCircuit.end()){
         // format of input circuit json: cirq
         json ionq_schema= cirq_to_ionq_schema(parseCircuit);
         circuitToGraph= ionq_schema;
      }
      // format of input circuit json: ionQ
      else {
         if (non_adjacent_gate(parseCircuit))
            qDebug() << "process aborted: non-adjacent gate in circuit";

         circuitToGraph= parseCircuit;
      }

      unsigned long cluster_state_rows= rows_m(gate_by_address, circuitToGraph);
      unsigned long cluster_state_columns= cols_n(gate_by_address
            , circuitToGraph);

      setLattice(cluster_state_rows, cluster_state_columns);
   }
   else
      qDebug() << "That file is not opening";
}

// write instructions, format .txt
void GraphView::saveGraph(const QString & wfile) {
   QFile writefile(wfile);

   // save conditions: write-only, text
   if (!writefile.open(QIODevice::WriteOnly | QIODevice::Text)){
      qDebug() << " file is not open";
      return ;
   }

   QTextStream write(&writefile);
   QList<QString> noFlips {};

   for (QGraphicsItem * item : scene->items()) {
      // save only type GraphVertex specifications
      if (item->type() == 4){
         auto v= qgraphicsitem_cast<GraphVertex *>(item);
         // vertex ID
// tuQ v0.2: read GraphVertex::IDs::vid and ::measure_prompt to .txt output
         write << *v->vertexID << " "
         // vertex pos
         << v->pos().x() << " " << v->pos().y();
         // edge coordinates (unique, i.e. no 'flips')
         for (const GraphEdge * e: *v->alledges) {
            // end positions of edge
            QString fstPos {QString::number(e->p1vertex->x())
            + " " + QString::number(e->p1vertex->y())};
            QString sndPos {QString::number(e->p2vertex->x())
            + " " + QString::number(e->p2vertex->y())};

            QStringBuilder testEdge {fstPos % " " % sndPos};
            QStringBuilder flipEdge {sndPos % " " % fstPos};

            if (!noFlips.contains(testEdge) && !noFlips.contains(flipEdge)){
               noFlips.push_back(testEdge);
               write << " " << testEdge;
            }
         }
         write << "\n";
      }
   }
   writefile.flush();
   writefile.close();
}

void GraphView::setLattice(unsigned long m, unsigned long n) {
//   define graph layout, by circuit input or menu option
//   pre-condition (from circuit input):
//      - json input is well-formed,
//      - composition and layout of gates passes input checks.
//   post-condition (from circuit input): the count of measurement patterns,
//       including edges, reconciles with the count of (input) circuit gates
   unsigned long id {1};
   qreal xinc {70};
   qreal yinc {70};
/* tuQ v0.2:
 * 1. the coordinates from a setLattice graph don't match the default
 *    setSceneRect;
 * 2. resize for big [m,n]: ~ 3 seconds to render [501,501]; ~ 12 seconds to
 *    render [1001,1001]
*/
   GraphVertex * row_to_row_edges[n];
   GraphVertex * ptr_vertex {};

   for (int i= 0; i < m; ++i) {
      for (int j= 0; j < n; ++j) {
         // create vertex[row, column]
         auto vertex= new GraphVertex(vertexmenu,id);
         vertex->setPos(j*xinc,i*yinc);
         scene->addItem(vertex);

         // set scrollbar slider positions by NW apex of lattice
         if (id == 1){
            auto hbar= new QScrollBar(Qt::Horizontal, this);
            setHorizontalScrollBar(hbar);
            auto xpos= (int) vertex->pos().x();
            hbar->setSliderPosition(xpos - 25);

            auto vbar= new QScrollBar(Qt::Vertical, this);
            setVerticalScrollBar(vbar);
            auto ypos= (int) vertex->pos().y();
            vbar->setSliderPosition(ypos - 25);
         }

         // by row, edge-per-column
         if (j > 0){
            auto * column_edge= new GraphEdge(ptr_vertex, vertex, edgemenu);
            ptr_vertex->add_edge(column_edge);
            vertex->add_edge(column_edge);
            scene->addItem(column_edge);
         }
         // by column, edge-per-row
         if (i > 0){
            auto * row_edge= new GraphEdge(row_to_row_edges[j], vertex, edgemenu);
            row_to_row_edges[j]->add_edge(row_edge);
            vertex->add_edge(row_edge);
            scene->addItem(row_edge);
         }

         // assemble (Graph-)vertices of previous row
         ptr_vertex= vertex;
         row_to_row_edges[j]= ptr_vertex;

         id += 1;
      }
   }
}


// protected:
void GraphView::keyPressEvent(QKeyEvent * event) {
   // Esc to cancel out of function
   if (cursorState && event->key() == Qt::Key_Escape){
      clabel->clear();
      cursorState= false;
   }

   if (event->key() == Qt::Key_E
   || event->key() == Qt::Key_O
   || event->key() == Qt::Key_V
   || event->key() == Qt::Key_X
   || event->key() == Qt::Key_Y
   || event->key() == Qt::Key_Z){
      cursorState= true;
      setCursorLabel(event->key());
//      clearSelection()
   }
   QGraphicsView::keyPressEvent(event);
}

void GraphView::mouseMoveEvent(QMouseEvent * event) {
   // position the label at x,y relative to cursor
   if (cursorState){
      auto qmex {(int) event->screenPos().x()};
      auto qmey{(int) event->screenPos().y()};

      clabel->move(qmex + 15, qmey + 10);
   }

   // draw the proxy (Graph)Edge instantiated through mousePressEvent
   if (tracer != nullptr && clabel->text() == "E"){
      QLineF growline(tracer->line().p1(), mapToScene(event->pos()));
      tracer->setLine(growline);
   }
   QGraphicsView::mouseMoveEvent(event);
}

void GraphView::mousePressEvent(QMouseEvent * event) {
   // LEFT click events
   if (event->button() != Qt::LeftButton)
      return ;

   // ABORT X measurement: user has not completed the X measurement in
   // contiguous steps (see 'part 2 -> measurement, x-basis', below)
   if (clabel->text() != "X" && !vertex1_X_neighbours.isEmpty()){
      for (GraphVertex * v : vertex1_X_neighbours) {
         v->resetVertexColour(Qt::black);
      }
      vertex1_X_neighbours.clear();
      vertex1_X= nullptr;
   }

   // instantiate: GraphEdge, pt. 1
   if (clabel->text() == "E"){
      // collect the p1 vertex at the cursor hotspot at 'click'
      QPointF vertex_at_scene= mapToScene(event->pos());
      QGraphicsItem * edge_origin= scene->itemAt(vertex_at_scene,QTransform());

      // prevent runtime exception from no vertex at the cursor hotspot upon
      // 'click'
      if (edge_origin == nullptr){
         cursorState= false;
         clabel->clear();
         return ;
      }
      else
         edge_origin->setFlag(QGraphicsItem::ItemIsMovable, false);

      // initialise a 'tracer' line segment.  The actual edge will be set in
      // arrears at the subsequent mouseReleaseEvent
      tracer= new QGraphicsLineItem(QLineF(vertex_at_scene, vertex_at_scene));
      tracer->setPen(QPen(Qt::black,2));
      scene->addItem(tracer);

      // reset cursor state
      cursorState= false;
      // 'E' is required as a boolean proxy in subsequent QMouseEvents
      clabel->hide();
   }
   // local complementation
   else if (clabel->text() == "O"){
      // specify in-focus vertex
      QPointF lcv_pos= mapToScene(event->pos());
      QGraphicsItem * item= scene->itemAt(lcv_pos,QTransform());
      auto lcv= qgraphicsitem_cast<GraphVertex *>(item);

      // helper function
      h_localComplementation(*lcv, *scene, edgemenu);

      // reset cursor state
      cursorState= false;
      clabel->clear();
   }
   // instantiate: GraphVertex
   else if (clabel->text() == "V"){
      // instantiate the vertex
      unsigned long v_count= h_itemCounter(4, *scene);
      auto * v= new GraphVertex(vertexmenu, v_count);

      // place vertex at 'click' position
      QPointF pos_xy= mapToScene(event->x()-15, event->y());
      v->setPos(pos_xy);

      scene->addItem(v);

      // reset cursor state
      cursorState= false;
      clabel->clear();
   }
   // part 1 -> measurement, x-basis
   else if (clabel->text() == "X" && vertex1_X_neighbours.isEmpty()){
      // select first vertex of X measurement
      QPointF x_vertex1_pos= mapToScene(event->pos());
      QGraphicsItem * item= scene->itemAt(x_vertex1_pos, QTransform());
      // 'item' is type GraphVertex or abort
      if (item == nullptr || item->type() != GraphVertex::Type){
         cursorState= false;
         clabel->clear();
         return ;
      }
      vertex1_X= qgraphicsitem_cast<GraphVertex *>(item);

      // ABORT: first vertex must have >= 1 edge for local complementation
      if (vertex1_X->alledges->isEmpty()){
         cursorState= false;
         clabel->clear();
         return ;
      }

      // populate neighbours of first vertex
      for (GraphEdge *e : *vertex1_X->alledges) {
         // obtain the address of underlying GraphVertex
         GraphVertex *address_of_p1vertex= e->p1vertex;
         // test whether pointed-at-objects are equivalent by comparing their
         // addresses
         if (address_of_p1vertex != vertex1_X)
            vertex1_X_neighbours.push_back(e->p1vertex);
         else
            vertex1_X_neighbours.push_back(e->p2vertex);
      }

      // reformat each of first vertex's neighbours as a prompt to the user to
      // select the special neighbour vertex
      for (GraphVertex *neighbour : vertex1_X_neighbours) {
         neighbour->resetVertexColour(QColor(0, 255, 0), 4
                                      , QColor(173, 255, 47));
      }
   }
   // part 2 -> measurement, x-basis
   else if (clabel->text() == "X" && !vertex1_X_neighbours.isEmpty()){
      // select special neighbour vertex of X measurement
      QPointF x_vertex2_pos= mapToScene(event->pos());
      QGraphicsItem * item= scene->itemAt(x_vertex2_pos, QTransform());
      // 'item' is type GraphVertex or abort
      if (item == nullptr || item->type() != GraphVertex::Type){
         for (GraphVertex * v : vertex1_X_neighbours) {
            v->resetVertexColour(Qt::black);
         }
         vertex1_X_neighbours.clear();
         vertex1_X= nullptr;

         cursorState= false;
         clabel->clear();
         return ;
      }
      auto vertex2_X= qgraphicsitem_cast<GraphVertex *>(item);

      // ABORT: special neighbour vertex must have >= 1 edge for local
      // complementation
      if (vertex2_X->alledges->isEmpty()){
         cursorState= false;
         clabel->clear();
         return ;
      }

      // first local complementation on special neighbour vertex
      h_localComplementation(*vertex2_X, *scene, edgemenu);

      // restore each of first vertex's neighbours to default colour scheme
      for (GraphVertex * v : vertex1_X_neighbours) {
         v->resetVertexColour(Qt::black);
      }

      // Y measurement of first vertex
      h_localComplementation(*vertex1_X, *scene, edgemenu);
      h_deleteVertex(*vertex1_X, *scene);

      // second local complementation on special neighbour vertex
      h_localComplementation(*vertex2_X, *scene, edgemenu);

      // clean up
      vertex1_X_neighbours.clear();
      vertex1_X= nullptr;

      // reset cursor state
      cursorState= false;
      clabel->clear();
   }
   // measurement, y-basis
   else if (clabel->text() == "Y"){
      // select vertex of Y measurement
      QPointF vertex_at_scene= mapToScene(event->pos());
      QGraphicsItem * y_item= scene->itemAt(vertex_at_scene, QTransform());
      auto * y_vertex= qgraphicsitem_cast<GraphVertex *>(y_item);

      // prevent the exception caused by no object at cursor hotspot upon
      // 'click'
      if (y_vertex != nullptr){
         if (y_vertex->alledges->isEmpty())
            h_deleteVertex(*y_vertex, *scene);
         else if (!y_vertex->alledges->isEmpty()){
            h_localComplementation(*y_vertex, *scene, edgemenu);
            h_deleteVertex(*y_vertex, *scene);
         }
      }

      // reset cursor state
      cursorState= false;
      clabel->clear();
   }
   // measurement, z-basis
   else if (clabel->text() == "Z"){
      // select vertex of Z measurement
      QPointF vertex_at_scene= mapToScene(event->pos());
      QGraphicsItem * z_item= scene->itemAt(vertex_at_scene, QTransform());
      auto * z_vertex= qgraphicsitem_cast<GraphVertex *>(z_item);

      if (z_vertex != nullptr)
         h_deleteVertex(*z_vertex,*scene);

      // reset cursor state
      cursorState= false;
      clabel->clear();
   }

   QGraphicsView::mousePressEvent(event);
   scene->update();
}

void GraphView::mouseReleaseEvent(QMouseEvent * event) {
   // instantiate: GraphEdge, pt. 2
   if (tracer != nullptr && clabel->text() == "E"){
      // staging: the vertex identified at p1 coordinates
      QList<QGraphicsItem *> p1items= scene->items(tracer->line().p1());
      // remove any existing instance of tracer in p1items
      if (p1items.count() && p1items.first() == tracer)
         p1items.removeFirst();

      // staging: the vertex identified at p2 coordinates
      QList<QGraphicsItem *> p2items= scene->items(tracer->line().p2());
      // as with p1items, remove any existing instance of tracer
      if (p2items.count() && p2items.first() == tracer)
         p2items.removeFirst();

      // vertices at p1 and p2 now selected so back out the tracer reference
      // from scene...
      scene->removeItem(tracer);
      // then, deallocate the memory
      delete tracer;

      if (p1items.count() > 0 && p2items.count() > 0
      && p1items.first()->type() == GraphVertex::Type
      && p2items.first()->type() == GraphVertex::Type
      && p1items.first() != p2items.first()){
         // pointer -> the GraphVertex at p1 coordinates
         auto * p1v= qgraphicsitem_cast<GraphVertex *>(p1items.first());
         // restore the 'movable' property of p1v, which was suspended at
         // mousePressEvent
         p1v->setFlag(QGraphicsItem::ItemIsMovable, true);
         // drop focus from p1v
         p1v->setSelected(false);
         // pointer -> the GraphVertex at p2 coordinates
         auto * p2v= qgraphicsitem_cast<GraphVertex *>(p2items.first());
         // use p1v and p2v as constructors to instantiate the edge
         auto * e= new GraphEdge(p1v, p2v, edgemenu);

         // add the edge to (QVector) 'edges' of vertices at both p1 and p2
         // coordinates
         p1v->add_edge(e);
         p2v->add_edge(e);

         scene->addItem(e);

         // clear the (mousePressEvent) clabel->hide()
         if (clabel->text() == "E")
            clabel->clear();
      }
   }
   tracer= nullptr;
   QGraphicsView::mouseReleaseEvent(event);
}


// private:
void GraphView::createElementMenus(QGraphicsScene * graph_scene) {
   // right-click menus for graph elements, (Graph)edge and (Graph)vertex
   edgemenu= new QMenu("edge menu");
   edgemenu->addAction(tr("&Delete"), this, [&, graph_scene](){
      // collect only the edge at the cursor hotspot, upon 'click'
      QList<QGraphicsItem *> del_edge= graph_scene->selectedItems();
      // either head of del_edge is type GraphEdge or abort
      if (del_edge.isEmpty() || del_edge.first()->type() != GraphEdge::Type)
         return ;

      // delete edge
      auto * edge= qgraphicsitem_cast<GraphEdge *>(del_edge.first());
      h_deleteEdge(edge, *graph_scene);
   });
//   edgemenu->addAction(tr("-- place 2 --"));

   vertexmenu= new QMenu("vertex menu");
   vertexmenu->addAction(tr("&Delete"), this, [&, graph_scene](){
      // collect only the vertex at the cursor hotspot, upon 'click'
      QList<QGraphicsItem *> del_vertex= graph_scene->selectedItems();
      // either head of del_vertex is type GraphVertex or abort
      if (del_vertex.isEmpty() || del_vertex.first()->type() != GraphVertex::Type)
         return ;

      // delete vertex
      auto * vertex= qgraphicsitem_cast<GraphVertex *>(del_vertex.first());
      h_deleteVertex(*vertex, *graph_scene);
   });
//   vertexmenu->addAction(tr("-- place 2 --"));
}

void GraphView::setCursorLabel(int key) {
   // create E/O/V/X/Y/Z label for cursor
   // pre-condition: cursorState == true
   // post-condition: a visible, persistent label of the cursor

   // int to upper case QString
   QString tag= (QString) toupper(key);

   if (cursorState){
      // label appears at (x,y), relative to cursor
      QPoint cursor_pos {QCursor::pos().x() + 15, QCursor::pos().y() + 10};
      clabel->move(cursor_pos);

      // set label
      clabel->setText(tag);

      // reveal a hidden cursor
      if (clabel->isHidden())
         clabel->show();
   }
}
