// contingency_iterator.cpp
// (c) 2020 David Merrell
//

#include "contingency_iterator.h"
#include <iostream>
#include <vector>

// constructors
SimplexIterator::SimplexIterator(){
    dim = 0;
    total = 0;
    cur_value = NULL;
    sub_state = NULL;
}

SimplexIterator::SimplexIterator(int new_dim, int new_total, int* new_memloc){
    dim = new_dim;
    total = new_total;
    cur_value = new_memloc;
    sub_state = NULL;
}

void SimplexIterator::set_sub(SimplexIterator* sub_iter){
    sub_state = sub_iter;
}

bool SimplexIterator::not_finished(){
    return (cur_value[dim-1] < total);
}

void SimplexIterator::reset_totals(){
    total = 0;
    if (dim > 1){
        sub_state->reset_totals();
    }
}


void SimplexIterator::restart(){
    if(dim > 1){
        cur_value[0] = total;
        cur_value[dim-1] = 0;
	sub_state->reset_totals();
    }
}


void SimplexIterator::advance(){

    if(dim == 1){
        return;
    }
    else{
        if(sub_state->not_finished()){
            sub_state->advance();
	}
	else{
            if (cur_value[0] > 0){
                sub_state->restart();
                sub_state->increment_total();
                cur_value[0] -= 1;
	    }
	}
    }
}


void SimplexIterator::pretty_print(){

    std::cout << "ADDR: " << this << "\tDIM: " << dim << "\tTOTAL: " << total << "\tCHILD: " << sub_state << std::endl;
    if (sub_state != NULL){
        sub_state->pretty_print();
    }

}


SimplexIterator& SimplexIterator::operator=(SimplexIterator it){
   dim = it.dim;
   total = it.total;
   cur_value = it.cur_value;
   sub_state = it.sub_state;
   return *this;
}

SimplexIterator::SimplexIterator(const SimplexIterator& it){
   dim = it.dim;
   total = it.total;
   cur_value = it.cur_value;
   sub_state = it.sub_state;
}


void ContingencyIterator::pretty_print(){
    state[0]->pretty_print(); 
}


std::vector<SimplexIterator*> ContingencyIterator::simplex_iter_factory(int dim, int total, int* memloc){
    std::vector<SimplexIterator*> it_stack;
    it_stack.push_back( new SimplexIterator(dim, total, memloc));
    it_stack[0]->cur_value[0] = total;

    for(int i = 1; i < dim; i++){
        it_stack.push_back( new SimplexIterator(dim - i, 0, memloc + i));
	it_stack[i-1]->set_sub(it_stack[i]);
	memloc[i] = 0;
    }

    return it_stack; 
}
