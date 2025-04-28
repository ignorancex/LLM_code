/*
 * Individual.cpp
 *
 *  Created on: 21/nov/2014
 *      Author: fernando
 */

#include "Individual.h"
#include "stddef.h"
namespace std {


Individual::Individual(){
	this->id=0;
	this->code="";
}

Individual::Individual(uint64_t id,string code){
	this->id=id;
	this->code=code;
}

Individual::~Individual() {
	// TODO Auto-generated destructor stub
}


} /* namespace std */
