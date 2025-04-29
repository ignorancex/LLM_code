//
// specify compiler view/scene objects
//

#include "compilerview.hpp"


// public:
CompilerView::CompilerView(QWidget *parent)
      : QGraphicsView(parent), c_scene(new QGraphicsScene(this))
{
   c_scene->setSceneRect(-2500,-2500,11000,11000);

   setScene(c_scene);
}

// copy-and-paste from GraphView, requires <QDebug>
//void SimulatorView::readCircuit(const QString & cjson);
