//
// specify a vertex object
//
#include "graphedge.hpp"
#include "graphvertex.hpp"


// public:
GraphVertex::GraphVertex(QMenu * menu, unsigned long vid
                         , QGraphicsItem * parent)
   : QGraphicsEllipseItem(parent), contextmenu_v(menu) {
   // standard vertex: fill, circumference pen
   setRect(vertexboundaryrect);
   setBrush(vertexfill);
   setPen(vertexcircumferencepen);

   id.vid= vid;

   // properties of the vertex
   setFlag(QGraphicsItem::ItemIsFocusable);
   setFlag(QGraphicsItem::ItemIsMovable);
   setFlag(QGraphicsItem::ItemIsSelectable);
   setFlag(QGraphicsItem::ItemSendsScenePositionChanges);
}

void GraphVertex::resetVertexColour(QColor colour, qreal pen, QColor fill) {
   vertexcircumferencepen= QPen(colour, pen);
   vertexfill= fill;
   setPen(vertexcircumferencepen);
   setBrush(fill);

   update(vertexboundaryrect);
}

void GraphVertex::resetVertexID(measure_char xy, QColor colour) {
   if (xy == measure_char::N){
      render= id_flag::idUL;
      return ;
   }

   render= id_flag::idC;

   switch (xy) {
      case measure_char::X:
         id.measure_prompt= u'X';
         break ;
      case measure_char::Y:
         id.measure_prompt= u'Y';
         break ;
      case measure_char::Z:
         id.measure_prompt= u'Z';
         break ;
      case measure_char::l_eta:
         id.measure_prompt= eta;
         break ;
      case measure_char::l_xi:
         id.measure_prompt= xi;
         break ;
      case measure_char::l_zeta:
         id.measure_prompt= zeta;
         break ;
      default:
         break ;
   }

   vertexcircumferencepen= QPen(colour, 2);
   setPen(vertexcircumferencepen);

   vertexidpen= QPen(colour, 1);

   update(vertexboundaryrect);
}


// protected:
// vertex menu options at click
void GraphVertex::contextMenuEvent(QGraphicsSceneContextMenuEvent * event) {
   // all functions @ GraphView::createElementMenus
   setSelected(true);
   contextmenu_v->exec(event->screenPos());
}

// moving a vertex moves any attached edge with it
QVariant GraphVertex::itemChange(GraphicsItemChange change
                                 , const QVariant & value) {
   if (change == QGraphicsItem::ItemPositionChange){
      for (GraphEdge * edge : qAsConst(edges))
         edge->resetEdgePosition();
   }

   return value;
}

// enable dynamic provisioning of vertices
void GraphVertex::paint(QPainter * painter
                        , const QStyleOptionGraphicsItem * option
                        , QWidget * widget) {
   Q_UNUSED(option)
   Q_UNUSED(widget)

   // call paint to render (constructor) standard vertex
   QGraphicsEllipseItem::paint(painter, option, widget);

   // font, pen and alignment of vertex id
   painter->setFont(vertexidfont);
   painter->setPen(vertexidpen);

   switch (render) {
      case id_flag::idC:
         painter->drawText(vertexboundaryrect, Qt::AlignCenter
               ,QChar(id.measure_prompt));
         break ;
      case id_flag::idUL:
         painter->drawText(vertexboundaryrect, Qt::AlignCenter
               ,QString::number(id.vid));
         break ;
   }
}

