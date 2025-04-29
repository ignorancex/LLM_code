//
// specify an edge object
//
#include "graphedge.hpp"
#include "graphvertex.hpp"


// public:
GraphEdge::GraphEdge(GraphVertex * p1vx, GraphVertex * p2vx, QMenu * menu
                     , QGraphicsItem * parent)
   : QGraphicsLineItem(parent), p1vertex(p1vx), p2vertex(p2vx)
   , contextmenu_e(menu)
{
   // standard edge: colour, width
   setPen(edgecolour);

   // instantiate the edge; position edge behind its p1 and p2 vertices
   QLineF edge(mapFromItem(p1vertex, 10, 10)
      , mapFromItem(p2vertex, 10, 10));
   setLine(edge);
   setZValue(-999);

   // properties of the edge
   setFlag(QGraphicsItem::ItemIsSelectable);
}

void GraphEdge::resetEdgePosition() {
   if (p1vertex != nullptr && p2vertex != nullptr){
      QLineF edge(mapFromItem(p1vertex, 10, 10)
            , mapFromItem(p2vertex, 10, 10));
      setLine(edge);
   }
}


// protected:
// edge menu options at click
void GraphEdge::contextMenuEvent(QGraphicsSceneContextMenuEvent * event) {
   // all functions @ GraphView::createElementMenus
   setSelected(true);
   contextmenu_e->exec(event->screenPos());
}