/*
 * LZFactor.cpp
 *
 *  Created on: 21/set/2014
 *      Author: fernando
 */
#include <iostream>
#include "LZFactor.h"
#include "LZIndex.h"


namespace std {

LZFactor::LZFactor(Factorization *factorization):factorization(factorization) {
	this->suffixArrayPosition=-2;  //-1 is a valid value
	this->letter=-1;
	this->textPosition=-1;
}


LZFactor::LZFactor(Factorization *factorization,int suffixArrayPosition,int length,int textPosition):factorization(factorization){
	this->suffixArrayPosition=suffixArrayPosition;
	this->length=length;
	this->letter=-1;
	this->textPosition=textPosition;
}

LZFactor::LZFactor(Factorization *factorization,char letter,int textPosition):factorization(factorization){
	this->letter=letter;
	this->suffixArrayPosition=-2; //-1 is a valid value
	this->length=1;
	this->textPosition=textPosition;
}


LZFactor::~LZFactor() {
	// TODO Auto-generated destructor stub
}


string LZFactor::getString() {
	if (suffixArrayPosition==-2)
		return string(1,letter);
	else
		return factorization->index.getFactorString(suffixArrayPosition,length-1) + string(1,letter);
}

int LZFactor::getLength(){
	return length;
};



ostream& operator<<(ostream& os, const LZFactor& f) {
		os << "(";
			os << f.length << "," << f.suffixArrayPosition << "," << f.letter << "," << f.textPosition;
		os << ")";
		return os;
	}

} /* namespace std */
