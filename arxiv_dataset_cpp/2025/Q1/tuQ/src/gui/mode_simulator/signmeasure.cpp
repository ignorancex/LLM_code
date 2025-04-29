//
// represent measurements of x/y/z basis
//

#include "signmeasure.hpp"


// public:
SignMeasure::SignMeasure(QString & sign_measure)
   : designate(sign_measure) {}

void SignMeasure::paint(QPainter * painter
      , const QStyleOptionGraphicsItem * option
      , QWidget * widget) {
   Q_UNUSED(option)
   Q_UNUSED(widget)

   painter->setFont(QFont{"Times New Roman", 24});
   painter->setPen(QPen {Qt::black,2});
   painter->setBrush(Qt::white);

   if (designate == psi || designate == "+"){
      // row-initialising ket, framing
      QRectF backdrop {-22,-15,56,56};
      painter->drawEllipse(backdrop);

      // row-initialising ket
      painter->setPen(designatePen);
      // UTF-16: vertical line; mathematical right angle bracket
      QString initialise {QChar(0x007C) % designate % QChar(0x27E9)};
      painter->drawText(backdrop,Qt::AlignCenter,initialise);
   }
   else if (designate.mid(0,4) == "CNOT"){
      // CNOT tile
      QRectF cnotRect {-22,-15,152,130};
      painter->drawRoundedRect(cnotRect,15,15);

      // CNOT labelling and orientation
      painter->setPen(designatePen);
      QString control {"CNOT_c"};
      QString target {"CNOT_t"};

      QRectF bottomMargin {-7,47,152,56};
      QRectF topMargin {-7,-15,152,56};

      // condition = CNOT t downwards arrow
      if (designate == "CNOT t" % QChar(0x2193)){
         painter->drawText(topMargin,Qt::AlignTop,control);
         painter->drawText(bottomMargin,Qt::AlignBottom,target);
      }
      else {
         painter->drawText(topMargin,Qt::AlignTop,target);
         painter->drawText(bottomMargin,Qt::AlignBottom,control);
      }
   }
   else {
      // measurement/pattern tile
      painter->drawEllipse(patternRect);

      // operator details
      painter->setPen(designatePen);
      painter->drawText(patternRect, Qt::AlignCenter, designate);
   }
   painter->setRenderHint(QPainter::Antialiasing);
}
