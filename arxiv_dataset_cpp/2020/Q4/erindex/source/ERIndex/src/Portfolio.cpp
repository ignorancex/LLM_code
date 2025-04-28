/*
 * Portfolio.cpp
 *
 *  Created on: 21/nov/2014
 *      Author: fernando
 */

#include "Portfolio.h"
#include <openssl/rsa.h>
#include <iostream>


namespace std {

Portfolio::Portfolio(uint64_t userId,RSA *privateKey) {
	this->userId=userId;
	this->privateKey=privateKey;
	this->publicKey=NULL;
}

Portfolio::Portfolio(uint64_t userId) {
	this->userId=userId;
	this->privateKey=NULL;
	this->publicKey=NULL;
}


Portfolio::~Portfolio() {
	// TODO Auto-generated destructor stub
}


Key *Portfolio::getSystemKey(){
	return systemKey;
}

void Portfolio::save(string directory){
	saveSystemKey(directory);
	saveIndividualKeys(directory);
}


void Portfolio::addSystemKey(uint8_t *clearValue){
	systemKey = new Key(this);
	systemKey->clearValue=clearValue;
}


void Portfolio::saveSystemKey(string directory){
	string oFilePath=directory + "/portfolio_"+ to_string(userId)+"_s.key";
	uint8_t *buffer=NULL;
	ofstream oFile;
	oFile.open (oFilePath, ios::out | ios::binary);
	oFile.write((char *)systemKey->getEncryptedValue(),256);
	oFile.close();

}


IndividualKey *Portfolio::getIndividualKey(int64_t individualId){
	if (individualKeys.count(individualId)>0)
		return individualKeys[individualId];
	else
		return NULL;
}


void Portfolio::addIndividualKey(Individual *individual,uint8_t *clearValue){
	IndividualKey *ik=new IndividualKey(this,individual);
	individualKeys[individual->id]=ik;
	ik->clearValue=clearValue;
}


void Portfolio::saveIndividualKey(string directory,uint64_t individualId){
	IndividualKey* key=individualKeys[individualId];
	string oFilePath=directory + "/portfolio_"+ to_string(userId)+"_"+to_string(individualId)+".key";
	uint8_t *buffer=NULL;
	ofstream oFile;
	oFile.open (oFilePath, ios::out | ios::binary);
	oFile.write((char *)key->getEncryptedValue(),256);
	oFile.close();
}






void Portfolio::saveIndividualKeys(string directory){
	for (std::map<int64_t,IndividualKey*>::iterator it=individualKeys.begin(); it!=individualKeys.end(); ++it){
		int64_t individualId=it->first;
		IndividualKey* key=it->second;
		string oFilePath=directory + "/portfolio_"+ to_string(userId)+"_"+to_string(individualId)+".key";
		uint8_t *buffer=NULL;
		ofstream oFile;
		oFile.open (oFilePath, ios::out | ios::binary);
		oFile.write((char *)key->getEncryptedValue(),256);
		oFile.close();
	}

}




void Portfolio::setPublicKey(RSA* publicKey) {
	this->publicKey=publicKey;
}

void Portfolio::setPrivateKey(RSA* privateKey) {
	this->privateKey=privateKey;
}


void Portfolio::loadPublicKey(string &publicKeyFilePath) {
	RSA *pk = NULL;
	FILE *publicKeyFile = fopen(publicKeyFilePath.c_str(),"rb");
	if (PEM_read_RSAPublicKey(publicKeyFile, &pk, NULL, NULL) != NULL)
		this->publicKey=pk;
	else{
		fclose(publicKeyFile);
		throw;
	}
}





IndividualKey::IndividualKey(Portfolio *portfolio,Individual *individual,uint8_t *encryptedValue):Key(portfolio,encryptedValue){
	this->individual=individual;
}

IndividualKey::IndividualKey(Portfolio *portfolio,Individual *individual):Key(portfolio){
	this->individual=individual;
}

IndividualKey::~IndividualKey() {
	// TODO Auto-generated destructor stub
}

Key::Key(Portfolio *portfolio,uint8_t *encryptedValue){
	this->portfolio=portfolio;
	this->encryptedValue=encryptedValue;
	clearValue=NULL;
}

Key::Key(Portfolio *portfolio){
	this->portfolio=portfolio;
	this->encryptedValue=NULL;
	clearValue=NULL;
}


Key::~Key() {
	// TODO Auto-generated destructor stub
}


void Key::setEncryptedValue(uint8_t *encryptedValue){
	this->encryptedValue=encryptedValue;
}


uint8_t *Key::getEncryptedValue(){
	if (encryptedValue==NULL){
		if (clearValue==NULL || portfolio->publicKey ==NULL)
			throw new runtime_error("Cannot compute the encrypted value: unknown clear value or user's public key");

		char *encryptBuffer=new char[256];
		int encryptedkeyLength=RSA_public_encrypt(64, clearValue,
		                       (unsigned char *)encryptBuffer, portfolio->publicKey, RSA_PKCS1_PADDING);
		encryptedValue=(uint8_t*)encryptBuffer;
	}
	return encryptedValue;
}

uint8_t *Key::getClearValue(){
	if (clearValue==NULL){
		if (encryptedValue==NULL || portfolio->privateKey ==NULL)
					throw new runtime_error("Cannot compute the clear value: unknow encrypted value or user's public key not assigned");
		char *decryptBuffer = new char[64];
		int decryptedKeyLength= RSA_private_decrypt(256, encryptedValue,
								 (unsigned char*)decryptBuffer,portfolio->privateKey , RSA_PKCS1_PADDING);
		clearValue=(uint8_t*)decryptBuffer;
	}
	return clearValue;
}

void Key::computeClearValue(){
		if (encryptedValue==NULL || portfolio->privateKey ==NULL)
					throw new runtime_error("Cannot compute the clear value: unknow encrypted value or user's public key not assigned");
		char *decryptBuffer = new char[64];
		int decryptedKeyLength= RSA_private_decrypt(256, encryptedValue,
								 (unsigned char*)decryptBuffer,portfolio->privateKey , RSA_PKCS1_PADDING);
		clearValue=(uint8_t*)decryptBuffer;
}




void Key::dumpClearValue(){
	uint8_t* v=getClearValue();
	for (int i=0;i<64;i++){
		cout << (int)v[i] << " ";
	}
	cout << endl;
}


void Key::dumpEncryptedValue(){
	uint8_t* v=getEncryptedValue();
	for (int i=0;i<64;i++){
		cout << (int)v[i] << " ";
	}
	cout << endl;
}





} /* namespace std */


