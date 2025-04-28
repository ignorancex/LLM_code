/*
 * EncryptedFileBitWriter.cpp
 *
 *  Created on: 27/nov/2014
 *      Author: fernando
 */

#include "EncryptedFileBitWriter.h"
#include "EncryptionContext.h"
#include "../srcSalsa20/ecrypt-sync.h"
namespace std {


void EncryptedFileBitWriter::ECRYPT_ctrsetup(ECRYPT_ctx *x,const u8 *ctr)
{
  x->input[8] = U8TO32_BIG(ctr + 0);
  x->input[13] = U8TO32_BIG(ctr + 4);
}

EncryptedFileBitWriter::EncryptedFileBitWriter(string fileName, EncryptionContext *ec): FileBitWriter(fileName){
	this->eContext=ec;
	ECRYPT_init();  //initialize Salsa20/20 internal tables
}

EncryptedFileBitWriter::~EncryptedFileBitWriter() {

}


EncryptionContext *EncryptedFileBitWriter::getEncryptionContext(){
	return eContext;
}

void EncryptedFileBitWriter::setEncryptionContext(EncryptionContext *ec){
	flush();
	eContext=ec;
}

void EncryptedFileBitWriter::intToBigEndian(uint32_t n, uint8_t *bs){
	bs[0] = (uint8_t)(n >> 24);
	bs[1] = (uint8_t)(n >> 16);
	bs[2] = (uint8_t)(n >>  8);
	bs[3] = (uint8_t)n;
	bs[4]=0;
	bs[5]=0;
	bs[6]=0;
	bs[7]=0;
}


void EncryptedFileBitWriter::populateKeyStreamBuffer(){
	ECRYPT_ctx x;

	ECRYPT_keysetup(&x,eContext->key,256,64);
	ECRYPT_ivsetup(&x,(u8*)&eContext->nonce);  //the nonce is in little endian format (Intel platform)

	//int64_t ctr =__builtin_bswap64 (eContext->ksBlockCounter);  //convert to big endian
	u8 *ctr=new u8[8];
	intToBigEndian(eContext->ksBlockCounter,ctr);
	ECRYPT_ctrsetup(&x,ctr);
	ECRYPT_keystream_bytes(&x, eContext->ksBlockBuffer,64);

	eContext->ksBlockOffset=0;
	eContext->ksBlockCounter=eContext->ksBlockCounter+1;
}

void EncryptedFileBitWriter::putByteIntoBlockBuffer(int ch){
	if (blockSize==liveBytes){
			writeBlockBuffer();
			clearBlockBuffer();
			liveBytes=0;
			writtenBlocks++;
	}
	if (eContext->ksBlockOffset> 63)
		populateKeyStreamBuffer();

	//write the byte into the buffer
	blockBuffer[liveBytes]=(char)((short)ch&0xFF);

	//encrypt it
	u8 ksb=eContext->ksBlockBuffer[eContext->ksBlockOffset];
	blockBuffer[liveBytes]=blockBuffer[liveBytes]^ksb;
	eContext->ksBlockOffset++;

	//advance to the next byte
	liveBytes++;
}






} /* namespace std */

