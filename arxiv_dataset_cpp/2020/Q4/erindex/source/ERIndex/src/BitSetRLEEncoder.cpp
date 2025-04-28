/*
 * BitSetRLEEncoder.cpp
 *
 *  Created on: 02/nov/2014
 *      Author: fernando
 */

#include "BitSetRLEEncoder.h"
#include <cmath>
#include <iostream>
#include <boost/dynamic_bitset.hpp>

namespace std {

enum AutomatonState{
	waitingForSymbol,
	waitingForSymbolCompletion
};

boost::dynamic_bitset<> BitSetRLEEncoder::compress(boost::dynamic_bitset<> bitSet){
		boost::dynamic_bitset<> result;
		int p=0;  //next element position in the compressed bit set
		int m=0;

		int length;
		for (length=bitSet.size()-1; length>=0;length--)
			if (bitSet[length]==1)
				break;
		length++;
		//for (int i=0;i<bitSet.size();i++){  //verify that size returns the index of the last set bit plus 1
		//		if (bitSet[i] || i==bitSet.size()-1){
		for (int i=0;i<length;i++){  //verify that size returns the index of the last set bit plus 1
			if (bitSet[i] || i==length-1){
				if (m>0){
					int nb= floor(log2(m+1));
					//compute the number N to encode in binary
					int N= m+1 - (int)pow(2, nb);
					boost::dynamic_bitset<>  binaryDigits(nb,N);
					//cout << binaryDigits << "\n";

					for (int j=binaryDigits.size()-1;j>=0;j--){
						if (binaryDigits[j]==0){	  //0a becomes 0
							result.resize(p+1);
							result.set(p,0);
							p++;
						}
						else{ //0b becomes 10
							result.resize(p+2);
							result.set(p,1);
							result.set(p+1,0);
							p+=2;
						}
					}
				}
				if (bitSet[i]){ //1 becomes 11
					result.resize(p+2);
					result.set(p,1);
					result.set(p+1,1);
					p+=2;
				}
				m=0;
			}
			else
			   m++;
		}
		return result;
	}

boost::dynamic_bitset<> BitSetRLEEncoder::decompress(boost::dynamic_bitset<> bitSet){
		boost::dynamic_bitset<> result;
		/** Decompression is done by a finite-state automaton, whose state can be:
		 * - waiting for a symbol
		 * - waiting for a symbol completion
		 */

		AutomatonState state=waitingForSymbol;
		boost::dynamic_bitset<> memory;
		int p=0;
		for (int i=0;i<bitSet.size();i++){
			if (state==waitingForSymbol){
				if (bitSet[i]==0){  //input symbol is a 0 (recognized 0=0a)
					int ms=memory.size();
					memory.resize(ms+1);
					memory.set(ms,0);
				}
				else{
					state=waitingForSymbolCompletion;
				}
			}
			else{
				if (bitSet[i]==0){  //input symbol is a 0 (recognized 10=0b)
					int ms=memory.size();
					memory.resize(ms+1);
					memory.set(ms,1);
				}
				else{
					int x=memory.size();  //input symbol is a 1 (recognized 11=1)
					if (x>0){
						boost::dynamic_bitset<> reversedMemory(x);
						for (int i=0;i<x;i++)
							reversedMemory[x-i-1]=memory[i];
						int N=reversedMemory.to_ulong();
						int m=pow(2, x) + N -1;
						result.resize(p+m);
						for (int j=0;j<m;j++){
							result.set(p,0);
							p++;
						}
					}
					result.resize(p+1);
					result.set(p,1);
					p++;
					memory.clear();
				}
				state=waitingForSymbol;
			}
		}
		if (memory.size()>0){
			int x=memory.size();
			if (x>0){
				boost::dynamic_bitset<> reversedMemory(x);
				for (int i=0;i<x;i++)
					reversedMemory[x-i-1]=memory[i];
				int N=reversedMemory.to_ulong();
				int m=pow(2, x) + N -1;
				result.resize(p+m);
				for (int j=0;j<m;j++){
					result.set(p,0);
					p++;
				}
			}
			result.resize(p+1);
			result.set(p);
			p++;
		}
		return result;
	}




} /* namespace std */
