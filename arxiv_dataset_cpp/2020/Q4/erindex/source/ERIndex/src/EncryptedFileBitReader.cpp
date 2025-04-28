/*
 * EncryptedFileBitReader.cpp
 *
 *  Created on: 27/nov/2014
 *      Author: fernando
 */

#include "EncryptedFileBitReader.h"
#include "EncryptionContext.h"
#include "../srcSalsa20/ecrypt-sync.h"

namespace std {


void EncryptedFileBitReader::open(string fileName,EncryptionContext *ec){
	this->fileName=fileName;
	this->eContext=ec;
	FileBitReader::open(fileName);
	//in.open(fileName, std::ifstream::binary);
    //BitReader::open();
}


void EncryptedFileBitReader::ECRYPT_ctrsetup(ECRYPT_ctx *x,const u8 *ctr)
{
  x->input[8] = U8TO32_BIG(ctr + 0);
  x->input[13] = U8TO32_BIG(ctr + 4);
}


EncryptedFileBitReader::EncryptedFileBitReader(string fileName, EncryptionContext *ec): FileBitReader(fileName){
	this->eContext=ec;
	ECRYPT_init();  //initialize Salsa20/20 internal tables
}

EncryptedFileBitReader::EncryptedFileBitReader(){
	ECRYPT_init();  //initialize Salsa20/20 internal tables
}



EncryptedFileBitReader::~EncryptedFileBitReader() {
	// TODO Auto-generated destructor stub
}

EncryptionContext *EncryptedFileBitReader::getEncryptionContext(){
	return eContext;
}

void EncryptedFileBitReader::setEncryptionContext(EncryptionContext *ec){
	eContext=ec;
	//the following instruction is needed because we apply the encryption on byte basis
	//(a single byte will be encrypted as a whole with a unique encryption context)
	gotoNextByteStart();
}

void EncryptedFileBitReader::intToBigEndian(uint32_t n, uint8_t *bs){
	bs[0] = (uint8_t)(n >> 24);
	bs[1] = (uint8_t)(n >> 16);
	bs[2] = (uint8_t)(n >>  8);
	bs[3] = (uint8_t)n;
	bs[4]=0;
	bs[5]=0;
	bs[6]=0;
	bs[7]=0;
}


void EncryptedFileBitReader::populateKeyStreamBuffer(){
	if (eContext->key !=NULL){
		ECRYPT_ctx x;

		ECRYPT_keysetup(&x,eContext->key,256,64);
		ECRYPT_ivsetup(&x,(u8*)&eContext->nonce);  //the nonce is in little endian format (Intel platform)

		//int64_t ctr =__builtin_bswap64 (eContext->ksBlockCounter);  //convert to big endian
		u8 *ctr=new u8[8];
		intToBigEndian(eContext->ksBlockCounter,ctr);
		ECRYPT_ctrsetup(&x,ctr);
		ECRYPT_keystream_bytes(&x, eContext->ksBlockBuffer,64);
		eContext->ksBlockCounter=eContext->ksBlockCounter+1;
	};
	eContext->ksBlockOffset=0;
}


int EncryptedFileBitReader::getByteFromBlockBuffer(){
   	if (liveBytes == 0){
   		try{
   			liveBytes=readBlockBuffer();  //
   		}
   		catch(exception ex){
   			return -1;
   		}
   	}
   	if (liveBytes < 0)
   		return -1;
   	else{
   		//populate the key stream buffer
   		if (eContext->ksBlockOffset> 63)
   				populateKeyStreamBuffer();

   		//read a byte from buffer
   		uint8_t b=(uint8_t)blockBuffer[blockBufferEffectiveLength -liveBytes]&0xFF;
   		liveBytes--;

   		//decrypt it
   		u8 ksb=eContext->ksBlockBuffer[eContext->ksBlockOffset];
   		b=b^ksb;
   		eContext->ksBlockOffset++;

   		return b;
   	}
}



} /* namespace std */
