/*
 * Utils.cpp
 *
 *  Created on: 24/nov/2014
 *      Author: fernando
 */

#include "Utils.h"
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <openssl/rsa.h>
#include <openssl/pem.h>
namespace std {

Utils::Utils() {
	// TODO Auto-generated constructor stub

}

Utils::~Utils() {
	// TODO Auto-generated destructor stub
}


// ****** Read n bits from file f
int Utils::fbit_read_simple(FILE *f,int* Bit_buffer_size, uint32* Bit_buffer,int n)
{
  uint32 u = 0;

  assert(n <= 32);
  if (n > 24){
    u =  fbit_read24_simple(f,Bit_buffer_size,Bit_buffer, n-24)<<24;
    u |= fbit_read24_simple(f,Bit_buffer_size,Bit_buffer,24);
    return((int)u);
  } else {
    return(fbit_read24_simple(f,Bit_buffer_size,Bit_buffer,n));
  }
}


int Utils::fbit_read24_simple(FILE *f, int* Bit_buffer_size, uint32* Bit_buffer, int n)
{
  int t0;
  uint32 t,u;

  assert(*Bit_buffer_size<8);
  assert(n>0 && n<=24);

  /* --- read groups of 8 bits until size>= n --- */
  while(*Bit_buffer_size<n) {
    if ((t0=getc(f)) == EOF){
      fprintf(stderr,"Unexpected end of file -bit_read-\n");
      exit(1);
    }
    t = (uint32) t0;
    *Bit_buffer |= (t << (24-*Bit_buffer_size));
    *Bit_buffer_size += 8;
  }
  /* ---- write n top bits in u ---- */
  u = *Bit_buffer >> (32-n);
  /* ---- update buffer ---- */
  *Bit_buffer <<= n;
  *Bit_buffer_size -= n;
  return((int)u);
}



/**
 * Transforms a C++ string into a C string
 */
char *Utils::toCString(string s){
		int len=s.length();
		char *cString=new char[len+1];
		len = s.copy(cString,len);
		cString[len] = '\0';
		return cString;
}


/**
 * Loads in a buffer of size sequenceLength + k-1 the sequence contained in the fastaFileName,
 * so that the buffer contains all the characters to compute the initial k-mers of the last text suffixes
 * @param fastaFileName
 * @throws Exception
 */
string Utils::loadFasta(string fastaFileName) {
	std::ifstream in(fastaFileName, std::ios::in);
	if (in) {
		std::string sequence,line;
		std::getline( in, line );
		while( std::getline( in, line ).good() ){
		        sequence += line;
		}
		return sequence;
	}
}

int Utils::generateRSAKeyPair(string &privateKeyFileName,string &publicKeyFileName){

	int             ret = 0;
	RSA             *r = NULL;
	BIGNUM          *bne = NULL;
	BIO             *bp_public = NULL, *bp_private = NULL;

	int             bits = 2048;
	unsigned long   e = RSA_F4;

	// 1. generate rsa key
	bne = BN_new();
	ret = BN_set_word(bne,e);
	if(ret != 1){
		goto free_all;
	}

	r = RSA_new();
	ret = RSA_generate_key_ex(r, bits, bne, NULL);
	if(ret != 1){
		goto free_all;
	}

	// 2. save public key
	bp_public = BIO_new_file(publicKeyFileName.c_str(), "w+");  //a PEM file
	ret = PEM_write_bio_RSAPublicKey(bp_public, r);
	if(ret != 1){
		goto free_all;
	}

	// 3. save private key
	bp_private = BIO_new_file(privateKeyFileName.c_str(), "w+");
	ret = PEM_write_bio_RSAPrivateKey(bp_private, r, NULL, NULL, 0, NULL, NULL);

	// 4. free
	free_all:

	BIO_free_all(bp_public);
	BIO_free_all(bp_private);
	RSA_free(r);
	BN_free(bne);

	return (ret == 1);

}

void Utils::generateSalsa20KeyFile(string &keyFilePath) {
		std::random_device generator;

		u8* key=new u8[64];
		std::uniform_int_distribution<u8> distribution(0,255);

		for (uint32_t i=0;i<64;i++)
			key[i]=distribution(generator);

		/*
		for (int i=0;i<64;i++){
				cout << (int)key[i] << " ";
		}
		cout << endl;
		*/


		ofstream keyFile(keyFilePath, ios::out | ios::binary);
		keyFile.write ((char*)key, 64);
		keyFile.close();
		return;
	}


} /* namespace std */
