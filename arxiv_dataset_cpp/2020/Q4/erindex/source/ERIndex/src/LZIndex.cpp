/*
 * LZIndex.cpp
 *
 *  Created on: 21/set/2014
 *      Author: fernando
 */
#include "LZIndex.h"
#include "LZFactor.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include "FileBitReader.h"
#include "EncryptionContext.h"
#include "EncryptedFileBitWriter.h"
#include "EncryptedFileBitReader.h"
#include "BitSetRLEEncoder.h"
#include "Utils.h"
#include <chrono>
#include <thread>

//#define INSERTION_HANDLING
#define EOF_shift(bwiOutPointer,n) (n < bwiOutPointer->bwt_eof_pos) ? n+1 :  n
extern "C" {
	//int go_forw(int row, int len, char *dest, bwi_out *s);
	char *reverse_string(char *source);
	void load_all_superbuckets(FM_INDEX *Infile,bwi_out *s);
	void load_all_buckets(FM_INDEX *Infile,bwi_out *s);
	void load_sb(FM_INDEX *Infile,bwi_out *s, int sbi);
	void load_b(FM_INDEX *Infile,bwi_out *s, int bi);
	uchar get_info_b_ferdy(FM_INDEX *Infile,bwi_out *s, uchar ch, int pos, int *occ, int flag);
}

//extern double Cache_percentage;

namespace std {

typedef pair<char, vector<int>> ExtraCharOccsPair;
typedef pair<char, unordered_map<int,int>> CharMapPair;
typedef pair<int,int> IntIntPair;





//used to sort by position the occurences of a pattern in an individual sequence
bool comparePtrToOccurrence(Occurrence* a, Occurrence* b) {
	return (a->factorIndex < b->factorIndex  ||
		        		(a->factorIndex == b->factorIndex && a->factorOffset < b->factorOffset));
}


uint64_t Header::save(EncryptedFileBitWriter &bitWriter,int *suspendedInfosStartPosition,
		EncryptionContext **suspendedInfosEncryptionContext, int *cRemappings,vector<int> &cInverseRemappings){
	bitWriter.writeInt(referenceIdentifier);
	bitWriter.writeInt(numberOfIndividuals);
	for (int i=0;i<numberOfIndividuals;i++) //individual Ids in order of factorization
		bitWriter.writeInt(individualIds[i]);
	bitWriter.writeInt(maximumFactorsNumber);
	bitWriter.writeInt(blockSize);
	bitWriter.writeInt(lmax);
	*suspendedInfosStartPosition=bitWriter.getBookmark()->getPosition();
	*suspendedInfosEncryptionContext=new EncryptionContext(*bitWriter.getEncryptionContext());
	//start of suspended informations
	bitWriter.writeInt(0); //reverseTreeDirectoryOffset
	bitWriter.writeInt(0); //forwardTreeDirectoryOffset
	bitWriter.writeInt(0); //posTreeDirectoryOffset
	for (int i=0;i<numberOfIndividuals;i++)  //individual factorizations start offsets
		bitWriter.writeInt(0); //start offset of the individual factorization

	//characters map
	bitWriter.writeUByte(compressedCharMap.size());
	for (int i=0;i<compressedCharMap.size();i++)
			bitWriter.write(1,compressedCharMap[i]);

	int firstNotAssignedCode=0;
	for (int i=0;i<256;i++){
		if (charMap[i]==1 ){
			cRemappings[i]=firstNotAssignedCode;
			cInverseRemappings.push_back(i);
			firstNotAssignedCode++;
		}
		else
			cRemappings[i]=-1;
	}
	return bitWriter.getBookmark()->getPosition();
}

void Header::load(BitReader &bitReader, int *cRemappings,vector<int> &cInverseRemappings){
	referenceIdentifier=bitReader.getInt();
	numberOfIndividuals=bitReader.getInt();
	individualIds=new int[numberOfIndividuals];
	maximumIndividualId=-1;
	for (int i=0;i<numberOfIndividuals;i++){
		individualIds[i]=bitReader.getInt(); //individual Ids in order of factorization
		factorizationIndexes[individualIds[i]]=i;  //associates each individualId with its factorization
		if (individualIds[i]>maximumIndividualId)
			maximumIndividualId=individualIds[i];
	}
	maximumFactorsNumber=bitReader.getInt();
	blockSize=bitReader.getInt();
	//blocksNumber=ceil((double)factorsNumber/blockSize); //deve essere fatto per ogni individuo
	lmax=bitReader.getInt();
	reverseTreeDirectoryOffset=bitReader.getInt();
	forwardTreeDirectoryOffset=bitReader.getInt();
	posTreeDirectoryOffset=bitReader.getInt();
	factorizationStartOffsets=new int[numberOfIndividuals];
	for (int i=0;i<numberOfIndividuals;i++)  //individual factorizations start offsets
			factorizationStartOffsets[i]= bitReader.getInt(); //start offset of the individual factorization

	//characters map
	int compressedCharMapSize=bitReader.getUByte();
	compressedCharMap.resize(compressedCharMapSize);
	for (int i=0;i<compressedCharMapSize;i++)
			compressedCharMap[i]=bitReader.read(1);
	charMap=BitSetRLEEncoder::decompress(compressedCharMap);
	charMap.resize(256);
	compressedCharMap.clear();

	int firstNotAssignedCode=0;
	for (int i=0;i<256;i++){
		//it's an extra-character only if it's in the text and not in the reference
		if (charMap[i]==1 ){
			cRemappings[i]=firstNotAssignedCode;
			cInverseRemappings.push_back(i);
			firstNotAssignedCode++;
		}
		else
			cRemappings[i]=-1;
	}
}

Factorization::Factorization(LZIndex &index,Individual *individual):index(index),individual(individual){}


uint64_t Factorization::save(EncryptedFileBitWriter &bitWriter){
	//HEADER
	//Encrypt the header with the individualKey and nonce 0
	EncryptionContext *headerEncryptionContext=new EncryptionContext(
			index.userPortfolio->getIndividualKey(individual->id)->getClearValue(),0);
	bitWriter.setEncryptionContext(headerEncryptionContext);
	bitWriter.writeInt(header.textLength);	   //length of the factorized text T (32 bit)
	bitWriter.writeInt(header.factorsNumber);
	Bookmark* blockStartsOffsetBookmark=bitWriter.getBookmark();
	EncryptionContext *blockStartsOffsetEncryptionContext=new EncryptionContext(*bitWriter.getEncryptionContext());
	bitWriter.writeInt(0);  //blockStartsOffset is a suspended information

	//ASSIGN FACTORS TO BLOCKS
	int bs=index.header.blockSize;
	int bn=ceil((double)header.factorsNumber/bs);	//number of blocks
	int fi=0;  //current factor index
	int p=0;   //actual text position
	for (int bi=0;bi<bn;bi++){
		Block *b= new Block(this,bi);
		b->firstFactorAbsolutePosition=p;
		//absolute index of the last factor in this block
		int lfi= min(fi+bs-1,header.factorsNumber-1);
		int maxRelativePosition=(factors[lfi]->textPosition + factors[lfi]->length) -b->firstFactorAbsolutePosition;
		b->saiStarts = new int[lfi+1];
		b->letters = new char[lfi+1];
		//the positions bitmap must have a bit for each possible relative position
		b->positionsBitmap.resize(maxRelativePosition+1);
		int bfi=0;
		while (fi<=lfi){
			b->saiStarts[bfi]=factors[fi]->suffixArrayPosition;
			//encode the relative position of the following factor
			//relative position of the following factor (also for the last factor, for which there isn't a following factor)
			int relPos=(factors[fi]->textPosition+ factors[fi]->length)-b->firstFactorAbsolutePosition;
			p+=factors[fi]->length;
			b->positionsBitmap.set(relPos,1);
			b->letters[bfi]=factors[fi]->letter;
			bfi++;
			fi++;
		}
		b->compressedPositionsBitmap=BitSetRLEEncoder::compress(b->positionsBitmap);
		blocks.push_back(b);
	}

	//WRITE BLOCKS TO DISK
	//Encrypt the block i with the individualKey and nonce i+1
	int nb=ceil((double)header.factorsNumber/index.header.blockSize);
	blockStarts=new int[nb];
	int lastBlockStartPosition;
	for (int i=0;i<nb;i++){
		EncryptionContext *individualEncryptionContext=new EncryptionContext(
				index.userPortfolio->getIndividualKey(individual->id)->getClearValue(),i+1);
		bitWriter.setEncryptionContext(individualEncryptionContext);
		blockStarts[i]=bitWriter.getBookmark()->getPosition();
		if (i==nb-1)
			lastBlockStartPosition=blockStarts[i];
		blocks[i]->save(bitWriter,header.textLength,index.s_rev->text_size,
				header.factorsNumber,index.header.blockSize,index.cInverseRemappings.size());
		bitWriter.flush();
		delete individualEncryptionContext;

	}
	header.blockStartsOffset=bitWriter.getBookmark()->getPosition();
	//BLOCK STARTS ARRAY
	//Encrypted with individualId and nonce 0, following the header informations
	bitWriter.setEncryptionContext(headerEncryptionContext);
	int nBits=int_log2(lastBlockStartPosition);
	bitWriter.writeInt(nBits);
	for (int i=0;i<nb;i++)
		bitWriter.write(nBits,blockStarts[i]);
	bitWriter.flush();

	//write the right header.blockStartsOffset to disk (it was a suspended information)
	Bookmark* eofBookmark=bitWriter.getBookmark();
	bitWriter.gotoBookmark(blockStartsOffsetBookmark);
	bitWriter.setEncryptionContext(blockStartsOffsetEncryptionContext);
	bitWriter.writeInt(header.blockStartsOffset);
	bitWriter.gotoBookmark(eofBookmark);
	bitWriter.setEncryptionContext(headerEncryptionContext);

	delete blockStartsOffsetEncryptionContext;
}

void Factorization::load(EncryptedFileBitReader &bitReader){
	EncryptionContext *headerEncryptionContext=new EncryptionContext(
				index.userPortfolio->getIndividualKey(individual->id)->getClearValue(),0);
	bitReader.setEncryptionContext(headerEncryptionContext);
	header.textLength=bitReader.getInt();
	header.factorsNumber=bitReader.getInt();
	header.blockStartsOffset=bitReader.getInt();

	int nb=ceil((double)header.factorsNumber/index.header.blockSize);
	//BLOCK STARTS
	//the block starts array is encrypted with the header's encryption context, without re-initializing it
	bitReader.gotoBookmark(new Bookmark(header.blockStartsOffset,0));
	int nBits=bitReader.getInt();
	blockStarts=new int[nb];
	for (int i=0;i<nb;i++)
		blockStarts[i]=bitReader.read(nBits);

	//initialize the block array
	blocks.resize(nb);
	for (int i=0;i<nb;i++)
		blocks[i]=NULL;

	header.blocksNumber=nb;
}


void Factorization::loadAllBlocks(EncryptedFileBitReader &bitReader){
	int nb=ceil((double)header.factorsNumber/index.header.blockSize);
	for (int i=0;i<nb;i++)
		getBlock(i,bitReader);
}


Block::Block(Factorization *factorization,int blockIndex){
	this->factorization=factorization;
	this->blockIndex=blockIndex;
}

int Block::getEstimatedSize(int blockSize,int textLength,int referenceLength){
	return int_log2(textLength)+  //first factor position
	32+ //here we'll store the size in bits of the compressed positions bitmap (32 bit)
	compressedPositionsBitmap.size()+ //here we'll store the compressed positions bitmap;
	blockSize* int_log2(referenceLength); //here we'll store the saiStarts array
}


LZFactor* Block::getFactor(int factorOffset){
	if (factors[factorOffset]==NULL){
		LZFactor *factor=new LZFactor(factorization);
		//find the start positions of the factor and of the following one
		int factorStartPos;
		int followingFactorStartPos=0;
		for (int i=0;i<=factorOffset;i++){
			factorStartPos=followingFactorStartPos;
			followingFactorStartPos=positionsBitmap.find_next(followingFactorStartPos);
		}
		factor->length=followingFactorStartPos-factorStartPos;
		factor->textPosition=firstFactorAbsolutePosition+factorStartPos;
		factor->suffixArrayPosition=saiStarts[factorOffset];
		factor->letter=letters[factorOffset];
		factors[factorOffset]=factor;
	};
	return factors[factorOffset];
}




uint64_t Block::save(BitWriter &bitWriter, int textLength, int referenceLength,/*int distinctExtraCharacters,*/
		int factorsNumber,int blockSize,int compactAlphabetSize){
	//here we'll store the absolute text position of the first factor (over the number of bit required to encode the textLength)
	int nBits=int_log2(textLength);
	bitWriter.write(nBits,firstFactorAbsolutePosition);

	//here we'll store the size in bits of the compressed positions bitmap
	int s=compressedPositionsBitmap.size();
	bitWriter.writeInt(s);
	//here we'll store the compressed positions bitmap;
	for (int i=0;i<s;i++)
		bitWriter.write(1,compressedPositionsBitmap[i]);
	//here we'll store the saiStarts array
	//nBits=int_log2(referenceLength+distinctExtraCharacters +1);
	nBits=int_log2(referenceLength +2);

	int bn=ceil((double)factorsNumber/blockSize);	//total number of blocks
	int fn; //number of factors in this block
	if (blockIndex<bn-1)
		fn=blockSize;
	else
		fn=factorsNumber-blockIndex*blockSize;
	int nBitsLetter=int_log2(compactAlphabetSize);
	for (int i=0;i<fn;i++){
		bitWriter.write(nBits,saiStarts[i]+2); //the term 2 must be added because the suffixArrayPosition is -2 if the referential part is missing
		int remappedLetter=factorization->index.cRemappings[letters[i]];
		bitWriter.write(nBitsLetter,remappedLetter);
	}
}

void Block::load(BitReader &bitReader, int textLength,  int referenceLength,int totalFactorsNumber,int blockSize,
		int compactAlphabetSize){
	//absolute text position of the first factor (over the number of bit required to encode the textLength)
	int nBits=int_log2(textLength);
	firstFactorAbsolutePosition=bitReader.read(nBits);
	//size in bits of the compressed positions bitmap
	int compressedPositionsBitmapSize=bitReader.getInt();
	//compressed positions bitmap;
	compressedPositionsBitmap.resize(compressedPositionsBitmapSize);
	for (int i=0;i<compressedPositionsBitmapSize;i++)
		compressedPositionsBitmap[i]=bitReader.read(1);
	positionsBitmap=BitSetRLEEncoder::decompress(compressedPositionsBitmap);

	compressedPositionsBitmap.clear();

	//saiStarts array
	int bn=ceil((double)totalFactorsNumber/blockSize);	//total number of blocks
	if (blockIndex<bn-1)
		factorsNumber=blockSize;
	else
		factorsNumber=totalFactorsNumber-blockIndex*blockSize;
	//allocates the factors array
	factors.resize(factorsNumber);
	for (int i=0;i<factorsNumber;i++)
		factors[i]=NULL;
	//allocates the saiStarts and letters array
	saiStarts=new int[factorsNumber];
	letters=new char[factorsNumber];
	nBits=int_log2(referenceLength +2);
	int nBitsLetter=int_log2(compactAlphabetSize);
	for (int i=0;i<factorsNumber;i++){
		saiStarts[i]=bitReader.read(nBits)-2;
		int remappedLetter=bitReader.read(nBitsLetter);
		letters[i]=factorization->index.cInverseRemappings[remappedLetter];
	}
}

class LZFactor;

void LZIndex::checkReverseReferenceIndex(){  //ensures that the reference index is opened
	if (s_rev==NULL)
		openReverseReferenceIndex();
}

void LZIndex::checkForwardReferenceIndex(){  //ensures that the reference index is opened
	if (s_for==NULL)
		openForwardReferenceIndex();
}

void LZIndex::checkForwardSuffixArray(){  //ensures that the reference index is opened
	if (forwardSA==NULL)
		openForwardSuffixArray();
}


void LZIndex::checkCorrespondenceFiles(){  //ensures that the reference index is opened
	if (r2f==NULL)
		openCorrespondenceFiles();
}

void LZIndex::loadCorrespondenceArrayFromDisk(string fileName,int *a,int n){
	FileBitReader br(fileName);
	br.open();
	int psize = int_log2(n);
	for(int i=0;i<n;i++)
		a[i]=br.read(psize);
	br.close();
}

void LZIndex::openCorrespondenceFiles(){  //ensures that the reference index is opened
	checkForwardReferenceIndex();
	checkReverseReferenceIndex();
	r2f=new int[s_for->text_size];
	f2r=new int[s_for->text_size];
	loadCorrespondenceArrayFromDisk(correspondenceFilesName + ".r2f",r2f,s_for->text_size);
	loadCorrespondenceArrayFromDisk(correspondenceFilesName + ".f2r",f2r,s_for->text_size);
}





/*
// ****** Read n bits from file f
int fbit_read_simple(FILE *f,int* Bit_buffer_size, uint32* Bit_buffer,int n)
{
  int fbit_read24_simple(FILE *f,int *,uint32*,int n);
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


int fbit_read24_simple(FILE *f, int* Bit_buffer_size, uint32* Bit_buffer, int n)
{
  int t0;
  uint32 t,u;

  assert(*Bit_buffer_size<8);
  assert(n>0 && n<=24);

  // --- read groups of 8 bits until size>= n ---
  while(*Bit_buffer_size<n) {
    if ((t0=getc(f)) == EOF){
      fprintf(stderr,"Unexpected end of file -bit_read-\n");
      exit(1);
    }
    t = (uint32) t0;
    *Bit_buffer |= (t << (24-*Bit_buffer_size));
    *Bit_buffer_size += 8;
  }
  //* ---- write n top bits in u ----
  u = *Bit_buffer >> (32-n);
  // ---- update buffer ----
  *Bit_buffer <<= n;
  *Bit_buffer_size -= n;
  return((int)u);
}
*/

inline int get_suffix(FILE *Safile, int pointerSize,int i)
{
  int pos,rem;
  int fbit_read_simple(FILE *,int*, uint32*,int);
  //int Pointer_size=int_log2(n);
  uint32* Bit_buffer=(uint32*)malloc(sizeof(uint32));
  int*  Bit_buffer_size=(int*)malloc(sizeof(int));  /* number of unread/unwritten bits in Bit_buffer */

  pos = i*pointerSize;
  rem = pos& 0x7;         // last 3 bits
  fseek(Safile, pos>>3, SEEK_SET);

  *Bit_buffer= (uint32) 0;
  *Bit_buffer_size=0;

  if(rem)
    Utils::fbit_read_simple(Safile,Bit_buffer_size,Bit_buffer,rem);
  return Utils::fbit_read_simple(Safile, Bit_buffer_size,Bit_buffer,pointerSize);
}



void LZIndex::openForwardSuffixArray(){
	checkForwardReferenceIndex();
	int n=s_for->text_size;
	ilog2n_reference=int_log2(n);
	forwardSA = new sdsl::int_vector<>(n,0,ilog2n_reference);
	load_from_file(*forwardSA,forwardSuffixArrayFileName);
	//FILE *forwardSAFile = fopen(forwardSuffixArrayFileName.c_str(), "a+b");
	//for (int i=0;i<n;i++)
	//	forwardSA[i]=get_suffix(forwardSAFile,ilog2n_reference,i);
}



LZIndex::LZIndex(Database &database,int referenceIdentifier,int blockSize):database(database){

	this->referenceIdentifier=referenceIdentifier;
	ReferenceSequence *refSeq=database.getReferenceSequence(referenceIdentifier);
	this->forwardReferenceIndexFileName=refSeq->forwardIndexFileName;
	this->reverseReferenceIndexFileName=refSeq->reverseIndexFileName;
	this->correspondenceFilesName=refSeq->correspondenceFilesName;
	this->forwardSuffixArrayFileName=refSeq->forwardSuffixArrayFileName;
	this->bs=blockSize;

	s_for=NULL;
	s_rev=NULL;
	forwardReferenceIndex=NULL;
	reverseReferenceIndex=NULL;
	forwardSA=NULL;
	reverseTree=NULL;
	forwardTree=NULL;
	posTree=NULL;
	f2r=NULL;
	r2f=NULL;
	Type_mem_ops = IN_MEM;
	Cache_percentage=1.0;


	//open reference indexes
	checkReverseReferenceIndex();
	checkForwardReferenceIndex();
	checkCorrespondenceFiles();
	checkForwardSuffixArray();

	//clear the factors indexes
	reverseTree=new BPlusTree(this,BTREE_HALF_PAGE_SIZE,s_rev->text_size);
	forwardTree=new BPlusTree(this,BTREE_HALF_PAGE_SIZE,s_rev->text_size);
	posTree=new BPlusTree(this,BTREE_HALF_PAGE_SIZE,s_rev->text_size);

	//clears the char map
	charMap.clear();
	charMap.resize(256);
	//the index is empty: set to 0 the maximum factor length
	lmax=0;
	maximumFactorsNumber=0;

	loadReferenceInMemory();

}


LZIndex::LZIndex(Database &database):database(database){
	s_for=NULL;
	s_rev=NULL;
	forwardReferenceIndex=NULL;
	reverseReferenceIndex=NULL;
	forwardSA=NULL;
	reverseTree=NULL;
	forwardTree=NULL;
	posTree=NULL;
	f2r=NULL;
	r2f=NULL;
	Type_mem_ops = IN_MEM;
	Cache_percentage=1.0;
}


LZIndex::~LZIndex() {

}



void LZIndex::open(string indexFileName){

	//Create a system key encryption context
	userPortfolio=database.getPortfolio();
	EncryptionContext *systemEncryptionContext=new EncryptionContext(userPortfolio->getSystemKey()->getClearValue(),0);

	this->indexFilename=indexFileName;
	bitReader.open(indexFileName,systemEncryptionContext);

	//HEADER
	header.load(bitReader,cRemappings,cInverseRemappings);

	factorizations.resize(header.numberOfIndividuals);

	this->referenceIdentifier=header.referenceIdentifier;
	ReferenceSequence *refSeq=database.getReferenceSequence(referenceIdentifier);
	this->forwardReferenceIndexFileName=refSeq->forwardIndexFileName;
	this->reverseReferenceIndexFileName=refSeq->reverseIndexFileName;
	this->correspondenceFilesName=refSeq->correspondenceFilesName;
	this->forwardSuffixArrayFileName=refSeq->forwardSuffixArrayFileName;

	checkReverseReferenceIndex();
	checkForwardReferenceIndex();
	checkForwardSuffixArray();


	//B-TREES
	reverseTree=new BPlusTree(this,BTREE_HALF_PAGE_SIZE,s_rev->text_size);
	reverseTree->setMaximumNumberOfFactorsPerIndividual(header.maximumFactorsNumber);
	reverseTree->setNumberOfIndividuals(header.numberOfIndividuals);
	reverseTree->setMaximumIndividualId(header.maximumIndividualId);
	reverseTree->setBaseNonce(10000000);
	reverseTree->load(&bitReader, header.reverseTreeDirectoryOffset);

	forwardTree=new BPlusTree(this,BTREE_HALF_PAGE_SIZE,s_rev->text_size);
	forwardTree->setMaximumNumberOfFactorsPerIndividual(header.maximumFactorsNumber);
	forwardTree->setNumberOfIndividuals(header.numberOfIndividuals);
	forwardTree->setMaximumIndividualId(header.maximumIndividualId);
	forwardTree->setBaseNonce(20000000);
	forwardTree->load(&bitReader,header.forwardTreeDirectoryOffset);


	posTree=new BPlusTree(this,BTREE_HALF_PAGE_SIZE,s_rev->text_size);
	posTree->setMaximumNumberOfFactorsPerIndividual(header.maximumFactorsNumber);
	posTree->setNumberOfIndividuals(header.numberOfIndividuals);
	posTree->setMaximumIndividualId(header.maximumIndividualId);
	posTree->setBaseNonce(30000000);
	posTree->load(&bitReader,header.posTreeDirectoryOffset);

	//loadAllInMemory();


	/*
	for (int i=0;i<header.numberOfIndividuals;i++){
		Factorization *f=getFactorization(header.individualIds[i]);
		for (int fi=0;fi<f->header.factorsNumber;fi++){
			LZFactor *fa=f->getFactor(fi,bitReader);
			if (fa->length == 3100741)
				cout << endl;
		}
	}*/

	//for (int i=1;i<10;i++)
	//	cout << getFactorization(i)->header.factorsNumber <<endl;
	//string s=f->getString();
	//cout << *f << " " << f->getString() <<  endl;
}


void LZIndex::close(){
	closeForwardReferenceIndex();
	closeReverseReferenceIndex();
	//fclose(forwardSAFile);

	bitReader.close();
}

/*
void LZIndex::save(string indexFileName){
	ofstream os;
	os.open (indexFileName, ios::out | ios::binary);
	int outputValue;
	for (unsigned int i=0;i< factors.size();i++){
		outputValue=(*factors[i]).getLength();
		char * c = (char *)&outputValue;
		os.write(c,4);

		outputValue=(*factors[i]).letter;
		c = (char *)&outputValue;
		os.write(c,1);

		outputValue=(*factors[i]).suffixArrayPosition;
		c = (char *)&outputValue;
		os.write(c,4);

	}
	os.close();
}*/

int LZIndex::save(string indexFileName){
	userPortfolio=database.getPortfolio();
	//Create a system key encryption context
	EncryptionContext *systemEncryptionContext=new EncryptionContext(userPortfolio->getSystemKey()->getClearValue(),0);
	EncryptedFileBitWriter bitWriter(indexFileName,systemEncryptionContext);
	bitWriter.open();

	//HEADER
	header.referenceIdentifier=referenceIdentifier;
	header.numberOfIndividuals=factorizations.size();
	header.individualIds=new int[header.numberOfIndividuals];
	for (int i=0;i<header.numberOfIndividuals;i++)
		header.individualIds[i]=factorizations[i]->individual->id;
	header.maximumFactorsNumber=maximumFactorsNumber;
	header.factorizationStartOffsets=new int[header.numberOfIndividuals];
	header.blockSize=bs;
	header.lmax=lmax;
	header.charMap=charMap;
	header.compressedCharMap=BitSetRLEEncoder::compress(charMap);

	int suspendedInfosStartPosition;
	EncryptionContext *suspendedInfosEncryptionContext;

	header.save(bitWriter,&suspendedInfosStartPosition,&suspendedInfosEncryptionContext, cRemappings,cInverseRemappings);
	bitWriter.flush();

	int indexSize=bitWriter.getBookmark()->getPosition()*8;

	compactAlphabetSize=cInverseRemappings.size();
	/*
	for (int i=0;i<256;i++){
		if (cRemappings[i]!=-1){
			char c=i;
			cout <<  c << "," << cRemappings[i] << endl;
		}
	}

	cout << "Inverse" << endl;
	for (int i=0;i<compactAlphabetSize;i++){
		cout <<  i << "," << cInverseRemappings[i] << endl;
	}
	*/

	//FACTORIZATIONS
	for (int i=0;i<header.numberOfIndividuals;i++){
		header.factorizationStartOffsets[i]=bitWriter.getBookmark()->getPosition();
		factorizations[i]->save(bitWriter);
		bitWriter.flush();
	}
	indexSize=bitWriter.getBookmark()->getPosition()*8;

	bitWriter.setEncryptionContext(systemEncryptionContext);

	//B-TREES
	//Encrypt the reverse tree header using the system key and nonce 1
	uint64_t treeDirectoryOffset;
	reverseTree->setMaximumNumberOfFactorsPerIndividual(maximumFactorsNumber);
	reverseTree->setNumberOfIndividuals(header.numberOfIndividuals);
	reverseTree->setBaseNonce(10000000);
	reverseTree->save(bitWriter,&treeDirectoryOffset);
	header.reverseTreeDirectoryOffset=treeDirectoryOffset;
	indexSize=bitWriter.getBookmark()->getPosition()*8;

	//Encrypt the forward tree header using the system key and nonce 2
	forwardTree->setMaximumNumberOfFactorsPerIndividual(maximumFactorsNumber);
	forwardTree->setNumberOfIndividuals(header.numberOfIndividuals);
	forwardTree->setBaseNonce(20000000);
	forwardTree->save(bitWriter,&treeDirectoryOffset);
	header.forwardTreeDirectoryOffset=treeDirectoryOffset;

	//Encrypt the positions tree header using the system key and nonce 3
	posTree->setMaximumNumberOfFactorsPerIndividual(maximumFactorsNumber);
	posTree->setNumberOfIndividuals(header.numberOfIndividuals);
	posTree->setBaseNonce(30000000);
	posTree->save(bitWriter,&treeDirectoryOffset);
	header.posTreeDirectoryOffset=treeDirectoryOffset;
	indexSize=bitWriter.getBookmark()->getPosition()*8;

	//WRITE SUSPENDED INFORMATIONS
	bitWriter.gotoBookmark(new Bookmark(suspendedInfosStartPosition,0));
	bitWriter.setEncryptionContext(suspendedInfosEncryptionContext);
	bitWriter.writeInt(header.reverseTreeDirectoryOffset);
	bitWriter.writeInt(header.forwardTreeDirectoryOffset);
	bitWriter.writeInt(header.posTreeDirectoryOffset);
	for (int i=0;i<header.numberOfIndividuals;i++)
		bitWriter.writeInt(header.factorizationStartOffsets[i]);

	bitWriter.flush();
	bitWriter.close();

	delete systemEncryptionContext;
	delete suspendedInfosEncryptionContext;

	return indexSize;
}

void LZIndex::dumpIndex(string dumpFileName){
	ofstream out(dumpFileName, std::ios::out | std::ios::binary);
	for (int i=0;i<header.numberOfIndividuals;i++){
		int64_t iid=header.individualIds[i];
		IndividualKey *ik=userPortfolio->getIndividualKey(iid);
		if (ik!=NULL){
			Factorization *f= new Factorization(*this,database.getIndividual(iid));  //i+1 is the individualId
			bitReader.gotoBookmark(new Bookmark(header.factorizationStartOffsets[i],0));

			f->load(bitReader);
			f->loadAllBlocks(bitReader);
			factorizations[i]=f;
			for (int fi=0;fi<f->header.factorsNumber;fi++){
				LZFactor *factor=f->getFactor(fi,bitReader);
				string s=factor->getString();
				out << iid << "," << fi << "," << s << endl;
			}
		}
	}
	out.close();
}






char *toCString(string s){
	int len=s.length();
	char *cString=new char[len+1];
	len = s.copy(cString,len);
	cString[len] = '\0';
	return cString;
}

void LZIndex::loadReferenceInMemory(){
	load_all_superbuckets(forwardReferenceIndex,s_for);
	load_all_buckets(forwardReferenceIndex,s_for);
	load_all_superbuckets(reverseReferenceIndex,s_rev);
	load_all_buckets(reverseReferenceIndex,s_rev);
}

void LZIndex::openForwardReferenceIndex(){
	s_for=new bwi_out();
	char *cString=toCString(forwardReferenceIndexFileName);
	forwardReferenceIndex=my_open_file(cString);
	forwardReferenceIndex->Use_bwi_cache=1;
	forwardReferenceIndex->Cache_percentage=1;
	init_bwi_cache(forwardReferenceIndex);
	read_basic_prologue(forwardReferenceIndex,s_for);

}

void LZIndex::openReverseReferenceIndex(){
	s_rev=new bwi_out();
	char *cString=toCString(reverseReferenceIndexFileName);
	reverseReferenceIndex=my_open_file(cString);
	reverseReferenceIndex->Use_bwi_cache=1;
    reverseReferenceIndex->Cache_percentage=1;
	init_bwi_cache(reverseReferenceIndex);
	read_basic_prologue(reverseReferenceIndex,s_rev);

}


void LZIndex::closeForwardReferenceIndex(){
	my_fclose(forwardReferenceIndex);
	delete s_for;
	s_for=NULL;
	delete forwardReferenceIndex;
	forwardReferenceIndex=NULL;
}





void LZIndex::closeReverseReferenceIndex(){
	my_fclose(reverseReferenceIndex);
	delete s_rev;
	s_rev=NULL;
	delete reverseReferenceIndex;
	reverseReferenceIndex=NULL;
}

/**
 * Loads in a buffer of size sequenceLength + k-1 the sequence contained in the fastaFileName,
 * so that the buffer contains all the characters to compute the initial k-mers of the last text suffixes
 * @param fastaFileName
 * @throws Exception
 */
 string LZIndex::loadFasta(string fastaFileName) {
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


 /**
  * Loads in a buffer of size sequenceLength + k-1 the sequence contained in the fastaFileName,
  * so that the buffer contains all the characters to compute the initial k-mers of the last text suffixes
  * @param fastaFileName
  */
  string LZIndex::loadFasta(string fastaFileName,int size) {
 	std::ifstream in(fastaFileName, std::ios::in);
 	if (in) {
 		std::string sequence,line;
 		std::getline( in, line );
 		while( std::getline( in, line ).good()  && sequence.size() < size){
 		        sequence += line;
 		}
 		in.close();
 		return sequence.substr(0,size);
 	} else return "";
}



void LZIndex::buildIndexFromMultiFASTA(int numberOfIndividuals,
		      Individual **individuals,
			  seqan::StringSet<seqan::IupacString> *seqs){
	this->numberOfIndividuals=numberOfIndividuals;
	this->individuals=individuals;
	this->fastaFileNames=fastaFileNames;
	this->multiFASTASequences=seqs;
	nextToFactorize=0;

	int nt = std::thread::hardware_concurrency();
	//nt=1;

	cout << "Number of threads: " << nt << endl;
	std::vector<std::thread*> threads;
	for (int t = 0; t < nt; t++)
		threads.push_back(new std::thread(&LZIndex::factorizationThread,this,t,SequencesSourceType::MULTIFASTA));

	for (int i = 0; i < nt; i++)
		threads[i]->join();
	for (int i = 0; i < nt; i++)
		delete threads[i];


	//Free the memory used for the reverse to forward and the forward to reverse map, used only during the factorization
	delete[] r2f;
	delete[] f2r;

}




void LZIndex::buildIndexFromDirectory(int numberOfIndividuals,Individual **individuals,string *fastaFileNames){
	this->numberOfIndividuals=numberOfIndividuals;
	this->individuals=individuals;
	this->fastaFileNames=fastaFileNames;

	nextToFactorize=0;

	int nt = std::thread::hardware_concurrency();
	//int nt=1;

	cout << "Number of threads: " << nt << endl;
	std::vector<std::thread*> threads;
	for (int t = 0; t < nt; t++)
		threads.push_back(new std::thread(&LZIndex::factorizationThread,this,t,SequencesSourceType::DIRECTORY));

	for (int i = 0; i < nt; i++)
		threads[i]->join();
	for (int i = 0; i < nt; i++)
		delete threads[i];


	//Free the memory used for the reverse to forward and the forward to reverse map, used only during the factorization
	delete[] r2f;
	delete[] f2r;


}



void LZIndex::factorizationThread(int threadNumber, SequencesSourceType sourceType){
	int toFactorize;
	boost::dynamic_bitset<> threadCharMap;
	int threadLMax=-1;
	threadCharMap.resize(256);
	bool finished=false;
	while (!finished){
		nextToFactorizeMutex.lock();
		if (nextToFactorize < numberOfIndividuals)
			toFactorize=nextToFactorize;
		else
			finished=true;
		nextToFactorize++;
		nextToFactorizeMutex.unlock();
		if (!finished){
			string sequence;
			if (sourceType == SequencesSourceType::MULTIFASTA ){
				stringstream ss;
				seqan::StringSet<seqan::IupacString> mfs=*multiFASTASequences;
				ss << mfs[toFactorize];
				sequence=ss.str();
			}
			else
				sequence=loadFasta(fastaFileNames[toFactorize]);

			Factorization *f=addToIndex(individuals[toFactorize],sequence,1,threadCharMap,threadLMax);
			coutMutex.lock();
			cout << "Thread " << threadNumber << ": factorized " << individuals[toFactorize]->code << endl;
			cout << "\t" << f->factors.size() << " factors" << endl;
			coutMutex.unlock();
		}
	}

	charMapLmaxMutex.lock();
	if (lmax < threadLMax)
		lmax=threadLMax;
	charMap|=threadCharMap;  //adds to charMap the characters in this thread's charMap
	charMapLmaxMutex.unlock();
}


Factorization *LZIndex::addToIndex(Individual *individual,string &sequence,int lookAheadWindowSize,boost::dynamic_bitset<> &threadCharMap,int &threadLMax){
	Factorization *f=new Factorization(*this,individual);
	doLZFactorization(f, sequence.c_str(),sequence.length(),threadCharMap,threadLMax);

	factorizationsMutex.lock();
	factorizations.push_back(f);
	if (f->header.factorsNumber > maximumFactorsNumber)
			maximumFactorsNumber=f->header.factorsNumber;
	factorizationsMutex.unlock();

	return f;
}

vector<int64_t> LZIndex::getIndividualIds(){
	vector<int64_t> result;
	for(auto it = header.factorizationIndexes.begin(); it != header.factorizationIndexes.end(); it++)
		result.push_back(it->first);
	return result;
}


inline bool LZIndex::backwardStepSuccessful(FM_INDEX *referenceIndex, bwi_out *s,
		unordered_map<char, unordered_map<int,int>> *occsCache,
		char c,int sp,int ep,int *trySp,int *tryEp){
	/*
	int o;
	bool cacheEnabled=true;
	//c is a compact alphabet code
	if (cacheEnabled){
		if ((*occsCache).count(c)>0 && (*occsCache)[c].count(sp-1)>0)
			o=(*occsCache)[c][sp-1];
		else{
			o=occ(referenceIndex,s, EOF_shift(s,sp - 1),c);
			if ((*occsCache).count(c)==0)
				(*occsCache).insert(CharMapPair(c,unordered_map<int,int>()));
			(*occsCache)[c].insert(IntIntPair(sp-1,o));
		}
		*trySp = s->bwt_occ[c]+o;

		if ((*occsCache).count(c)>0 && (*occsCache)[c].count(ep)>0)
			o=(*occsCache)[c][ep];
		else{
			o=occ(referenceIndex,s, EOF_shift(s,ep),c);
			if ((*occsCache).count(c)==0)
				(*occsCache).insert(CharMapPair(c,unordered_map<int,int>()));
			(*occsCache)[c].insert(IntIntPair(ep,o));
		}
		*tryEp = s->bwt_occ[c]+o-1;
	}
	else{*/
		*trySp = s->bwt_occ[c]+occ(referenceIndex,s, EOF_shift(s,sp - 1),c);
		*tryEp = s->bwt_occ[c]+occ(referenceIndex,s, EOF_shift(s,ep),c)-1;
	//}
	return *tryEp>=*trySp;
}

/**
 * This function must be called with the following constraint:
 * 	numOfSteps > 0;
 */
inline void LZIndex::forwardBackwardSteps(int sai,int numOfSteps, int *sai_start,int *sai_pref){

	  int steps,n;
	  uchar c,c_sb;
	  //int occ_b[256];
	  int occ_b;

	  int t_sai_pref;
	  for(steps=0;steps<numOfSteps; ) {  //at least one step
		 t_sai_pref=sai;
		 if(sai==s_for->bwt_eof_pos){
			sai=-1;
		  	break;
		}

		sai=EOF_shift(s_for,sai);

	    // fetches info from the header of the superbucket
	    //get_info_sb(forwardReferenceIndex ,s_for,sai,occ_sb);
		int sbi =  sai/forwardReferenceIndex->Bucket_size_lev1;
		if (s_for->buclist_lev1[sbi]==NULL)
			 load_sb(forwardReferenceIndex, s_for,sbi);
		bucket_lev1 *sb=s_for->buclist_lev1[sbi];


	    // fetches occ into occ_b properly remapped and returns
	    // the remapped code for occ_b of the char in the  specified position
	    //c = get_info_b_ferdy(forwardReferenceIndex,s_for,NULL_CHAR,sai,occ_b,WHAT_CHAR_IS);
		c = get_info_b_ferdy(forwardReferenceIndex,s_for,NULL_CHAR,sai,&occ_b,WHAT_CHAR_IS);

	    c_sb = sb->inv_map_sb[c];
	    assert(c_sb < s_for->alpha_size);

	    steps++;

	    //n = sb->occ[c_sb] + occ_b[c] - 1;         // # of occ before curr_row
	    n = sb->occ[c_sb] + occ_b - 1;              // # of occ before curr_row
	    sai = s_for->bwt_occ[c_sb] + n;             // get next row

	  }
	  *sai_pref=t_sai_pref;
	  *sai_start=sai;
}


//Perform  a backward step, returning:
//1) the new suffix array index as return value;
//2) the character in position sai as output parameter
inline int LZIndex::reverseBackwardStep(int sai, char *character){
	uchar c,c_sb;
	//int occ_b[256];
	int occ_b;
	if (sai < s_rev->bwt_eof_pos)
		sai++;

	// fetches info from the header of the superbucket
	//get_info_sb(reverseReferenceIndex ,s_rev,sai,occ_sb);
	int sbi =  sai/reverseReferenceIndex->Bucket_size_lev1;
	if (s_rev->buclist_lev1[sbi]==NULL)
		load_sb(reverseReferenceIndex, s_rev,sbi);
	bucket_lev1 *sb=s_rev->buclist_lev1[sbi];

	// fetches occ into occ_b properly remapped and returns
	// the remapped code for occ_b of the char in the  specified position
	//c = get_info_b_ferdy(reverseReferenceIndex,s_rev,NULL_CHAR,sai,occ_b,WHAT_CHAR_IS);
	c = get_info_b_ferdy(reverseReferenceIndex,s_rev,NULL_CHAR,sai,&occ_b,WHAT_CHAR_IS);

	c_sb = sb->inv_map_sb[c];
	assert(c_sb < s_rev->alpha_size);
	*character=s_rev->inv_char_map[c_sb];  //correzione per errore di ricerca

	//int n = sb->occ[c_sb] + occ_b[c] - 1;  // # of occ before curr_row
	int n = sb->occ[c_sb] + occ_b - 1;  // # of occ before curr_row
	sai = s_rev->bwt_occ[c_sb] + n;       // get next row
	if(sai==s_rev->bwt_eof_pos)
		sai=-1;
	//else
	//	sai = EOF_shift(s_rev,sai);
    return sai;
}








/**
 * Find, within the suffix array of the reverse reference string, the starting suffix for the length l factor
 * whose reverse is a prefix of the sai_rev suffix; to increase the performance of this operation we
 * have used the straight reference index instead of doing forward steps on the reverse reference index.
 */
inline void LZIndex::findStartingAndPrefixSuffix(int sai_rev, int l,int *sai_rev_start,int *sai_pref,int *straightTextPosition){
	//Find, within the suffix array of the straight reference string, the suffix array index corresponding
	//to the same text position of sai_rev and
	//do l backward steps to find the suffix sai_start whose correspondent in the reverse string
	//is the wanted starting suffix
	int sai=r2f[sai_rev];
	int sai_start;
	forwardBackwardSteps(sai,l,&sai_start,sai_pref);
	if (sai_start> -1)
		*sai_rev_start=f2r[sai_start];
	else
		*sai_rev_start=-1;

	//*straightTextPosition=get_suffix(forwardSAFile,ilog2n_reference,*sai_pref);
	*straightTextPosition=(*forwardSA)[*sai_pref];


	//if (textPositionsCache.count(*sai_pref)>0)
	//		*straightTextPosition=textPositionsCache[*sai_pref];
	//else{
			//*straightTextPosition=get_occ_pos(forwardReferenceIndex,s_for,*sai_pref);
			//*straightTextPosition=forwardSA[*sai_pref];
	//		textPositionsCache.insert(IntIntPair(*sai_pref,*straightTextPosition));
	//}

	/*int straightTextPosition2=get_occ_pos(forwardReferenceIndex,s_for,*sai_pref);
	if (straightTextPosition2 != *straightTextPosition)
		exit(1);
	*/
}







//lookahead by h
void LZIndex::findLongestFactor(const char *buffer,int len,int i,int h,int &longest_i,int &longest_l,int &longest_sai_rev,int &longest_numberOfN,
		boost::dynamic_bitset<> &threadCharMap){
	int l;
	int sp,ep;
	int trySp,tryEp;
	char c;
	char lastNrc; //last not remapped character
	longest_l=0;
	longest_i=i;
	longest_sai_rev=-1;
	longest_numberOfN = 0;

	for (int j=0;j<h && i<len;j++){
		int numberOfN=0;
		char nrc=buffer[i];  //not yet remapped character
		if (nrc=='N')
			numberOfN++;

		threadCharMap[(int)nrc]=1;
		l=1;
		if (i<len-1 && s_rev->bool_char_map[nrc]){
			lastNrc=nrc;
			c=s_rev->char_map[nrc];
			sp=s_rev->bwt_occ[c];
			if(c==s_rev->alpha_size-1)
				ep=s_rev->text_size-1;
			else
				ep=s_rev->bwt_occ[c+1]-1;
			bool stepSuccessful=true;
			while (i+l < len-1 && s_rev->bool_char_map[nrc=buffer[i+l]] && stepSuccessful &&
			    (lastNrc!='N' && nrc!='N' || lastNrc=='N' && nrc=='N')){
					if (nrc=='N')
						numberOfN++;
					threadCharMap[(int)nrc]=1;
					c=s_rev->char_map[nrc];
					if (stepSuccessful=backwardStepSuccessful(reverseReferenceIndex,s_rev,&reverseOccs,c,sp,ep,&trySp,&tryEp)){
					  //if [trySp,tryEp] is not a valid suffix array range (i.e. ep<sp) then
					  //the factor ended at the previous step and we can bring as factor any suffix in
					  //the range [sp,ep] computed at the previous step (we can take the first, i.e. the suffix
					  //corresponding to the index sp
						sp=trySp;
						ep=tryEp;
						l++;
					}
					lastNrc=nrc;

			}
			if (l>longest_l){
				longest_i=i;
				longest_l=l;
				longest_sai_rev=sp;
				longest_numberOfN=numberOfN;
			}
	   }
	   i++;
	}
}

void LZIndex::noLookAheadFactorization(Factorization *f,const char *buffer,int len,int &currentTextPosition,int &currentFactorIndex,
		boost::dynamic_bitset<> &threadCharMap){
	int l=0;
	int c;
	int sp,ep,trySp,tryEp;
	int i=currentTextPosition;
	int saiStart;
	int saiPrefix;
	int textPosition;
	int64_t individualId=f->individual->id;
	while (i<currentTextPosition+len){
		char nrc=buffer[i];  //not yet remapped character
		charMap[(int)nrc]=1;
		l=1;
		if (i<currentTextPosition+len-1 && s_rev->bool_char_map[nrc]){
			c=s_rev->char_map[nrc];
			sp=s_rev->bwt_occ[c];
			if(c==s_rev->alpha_size-1)
				ep=s_rev->text_size-1;
			else
				ep=s_rev->bwt_occ[c+1]-1;
			bool stepSuccessful=true;
			while (i+l < currentTextPosition+len-1 && s_rev->bool_char_map[nrc=buffer[i+l]] && stepSuccessful){
				charMap[(int)nrc]=1;
				c=s_rev->char_map[nrc];
				if (stepSuccessful=backwardStepSuccessful(reverseReferenceIndex,s_rev,&reverseOccs,c,sp,ep,&trySp,&tryEp)){
				   //if [trySp,tryEp] is not a valid suffix array range (i.e. ep<sp) then
				   //the factor ended at the previous step and we can bring as factor any suffix in
				   //the range [sp,ep] computed at the previous step (we can take the first, i.e. the suffix
				   //corresponding to the index sp
					sp=trySp;
					ep=tryEp;
					l++;
				}
			}
			findStartingAndPrefixSuffix(sp,l,&saiStart,&saiPrefix,&textPosition);
			f->factors.push_back(new LZFactor(f,saiStart,l+1,i));
			char c=buffer[i+l];
			f->factors[currentFactorIndex]->letter=c;
			charMap[(int)c]=1;
			reverseTree->insert(sp,Value(individualId,currentFactorIndex));
			forwardTree->insert(saiPrefix,Value(individualId,currentFactorIndex));
			posTree->insert(textPosition,Value(individualId,currentFactorIndex));
			currentFactorIndex++;
			i+=l+1;
	   } else{
		   f->factors.push_back(new LZFactor(f,nrc,i));
		   i++;
		   currentFactorIndex++;
	   }
	}
}

void LZIndex::doLZFactorization(Factorization *f,const char *buffer,int len,boost::dynamic_bitset<> &threadCharMap,
		int &threadLMax){
	int saiStart;
	int saiPrefix;
	int textPosition;

	checkReverseReferenceIndex();
	checkForwardReferenceIndex();
	checkCorrespondenceFiles();
	checkForwardSuffixArray();

	f->header.textLength=len;

	int64_t individualId=f->individual->id;

	int currentFactorIndex=0;
	int h=1;

	//initializes the maximum length of a factor
	int i=0;
	while (i<len){
		//find the longest factor
		int longest_i,longest_l,longest_sai_rev,longest_numberOfN;
		findLongestFactor(buffer,len,i,h,longest_i,longest_l,longest_sai_rev,longest_numberOfN,threadCharMap);
		if (longest_i>i)
			noLookAheadFactorization(f,buffer,longest_i-i,i,currentFactorIndex,threadCharMap);
		//longest_l è la lunghezza della sola parte referenziale
		findStartingAndPrefixSuffix(longest_sai_rev,longest_l,&saiStart,&saiPrefix,&textPosition);
		f->factors.push_back(new LZFactor(f,saiStart,longest_l+1,longest_i));
		char c=buffer[longest_i+longest_l];
		f->factors[currentFactorIndex]->letter=c;
		threadCharMap[(int)c]=1;

		//updates the auxiliary data structures
		reverseTreeMutex.lock();
		reverseTree->insert(longest_sai_rev,Value(individualId,currentFactorIndex));
		reverseTreeMutex.unlock();

		forwardTreeMutex.lock();
		forwardTree->insert(saiPrefix,Value(individualId,currentFactorIndex));
		forwardTreeMutex.unlock();

		posTreeMutex.lock();
		posTree->insert(textPosition,Value(individualId,currentFactorIndex));
		posTreeMutex.unlock();

		if (longest_numberOfN<longest_l && longest_l+1>threadLMax)  //the "only  'N'" factors don't contribute to lmax
			threadLMax=longest_l+1;

		currentFactorIndex++;
		i=longest_i+longest_l+1;
	}

	f->header.factorsNumber=f->factors.size();

}




inline bool LZIndex::searchPatternReverseInReferenceIndex(FM_INDEX *referenceIndex, bwi_out *s,
		unordered_map<char, unordered_map<int,int>> *occsCache,
		string pattern,int *sp,int *ep){
	char c;
	char nrc=pattern[0];  //not yet remapped character
	if (s->bool_char_map[nrc])
		c=s->char_map[nrc];
	else
		return false;
	*sp=s->bwt_occ[c];
	if(c==s->alpha_size-1)
		*ep=s->text_size-1;
	else
		*ep=s->bwt_occ[c+1]-1;

	int i=1;
	int pl=pattern.size();
	while (*sp<=*ep && i<pl){
		if (s->bool_char_map[pattern[i]])
			c=s->char_map[pattern[i]];
		else
			return false;
		if (backwardStepSuccessful(referenceIndex,s,occsCache,c,*sp,*ep,sp,ep))
			i++;
	}
	return *ep>=*sp;
}

inline bool LZIndex::searchPatternInReferenceIndex(FM_INDEX *referenceIndex, bwi_out *s,
		unordered_map<char, unordered_map<int,int>> *occsCache,
		string pattern,int *sp,int *ep){
	char c;
	int pl=pattern.size();
	char nrc=pattern[pl-1];  //not yet remapped character
	if (s->bool_char_map[nrc]) //there is no need
		c=s->char_map[nrc];
	else
		return false;
	*sp=s->bwt_occ[c];
	if(c==s->alpha_size-1)
		*ep=s->text_size-1;
	else
		*ep=s->bwt_occ[c+1]-1;

	int i=pl-2;
	while (*sp<=*ep && i>=0){
		if (s->bool_char_map[pattern[i]])
			c=s->char_map[pattern[i]];
		else
			return false;
		//*sp = s->bwt_occ[c]+occ(referenceIndex,s, EOF_shift(s,*sp - 1), c);
		//*ep = s->bwt_occ[c]+occ(referenceIndex,s, EOF_shift(s,*ep), c)-1;
		if (backwardStepSuccessful(referenceIndex,s,occsCache,c,*sp,*ep,sp,ep))
			i--;
	}
	return *ep>=*sp;
}


Factorization *LZIndex::getFactorization(int64_t individualId){
	if (header.factorizationIndexes.count(individualId)==0){
		string msg="The individual "+ to_string(individualId) + "is not in this index";
		throw msg;
	}
	int fi=header.factorizationIndexes[individualId];
	Factorization *f;
	if (factorizations[fi]==NULL){
		f=new Factorization(*this,database.getIndividual(individualId));
		bitReader.gotoBookmark(new Bookmark(header.factorizationStartOffsets[fi],0));
		f->load(bitReader);
		factorizations[fi]=f;
	}
	return factorizations[fi];
}


void LZIndex::loadAllInMemory(){
	for (int i=0;i<header.numberOfIndividuals;i++){
		int64_t iid=header.individualIds[i];
		IndividualKey *ik=userPortfolio->getIndividualKey(iid);
		if (ik!=NULL){
			Factorization *f= new Factorization(*this,database.getIndividual(iid));  //i+1 is the individualId
			bitReader.gotoBookmark(new Bookmark(header.factorizationStartOffsets[i],0));
			f->load(bitReader);
			f->loadAllBlocks(bitReader);
			factorizations[i]=f;
		}
	}
	reverseTree->loadAllNodesInMemory();
	forwardTree->loadAllNodesInMemory();
	posTree->loadAllNodesInMemory();

}

inline Block* Factorization::getBlock(int blockIndex,EncryptedFileBitReader &bitReader){
	if (blocks[blockIndex]==NULL){
		EncryptionContext *individualEncryptionContext=new EncryptionContext(
					index.userPortfolio->getIndividualKey(individual->id)->getClearValue(),blockIndex+1);
		bitReader.setEncryptionContext(individualEncryptionContext);

		bitReader.gotoBookmark(new Bookmark(blockStarts[blockIndex],0));
		Block *block=new Block(this,blockIndex);

		block->load(bitReader,header.textLength,index.s_rev->text_size,
				header.factorsNumber,index.header.blockSize,index.cInverseRemappings.size());
		blocks[blockIndex]=block;
		delete individualEncryptionContext;
	}
	return blocks[blockIndex];
}



inline LZFactor* Factorization::getFactor(int factorIndex,EncryptedFileBitReader &bitReader){
	int blockIndex=factorIndex/index.header.blockSize;
	int factorOffset=factorIndex%index.header.blockSize;
	return getBlock(blockIndex,bitReader)->getFactor(factorOffset);
}



string LZIndex::getSequence(int64_t individualId){
	string s="";
	Factorization *f=getFactorization(individualId);

	for (unsigned int i=0;i<f->header.factorsNumber ;i++){
		LZFactor *factor=f->getFactor(i,bitReader);
		//cout << i << "-" << *factor << "\t" << (*factor).getString() << endl;
		s+=factor->getString();
	}
	return s;
}

map<int,vector<int>> LZIndex::getFactorsInRange(BPlusTree *tree,int sp,int ep){
	map<int,vector<int>> result;
	map<int,vector<pair<int,int>>> queryResult=tree->rangeQuery(sp,ep);
	for( auto it = queryResult.begin(), end = queryResult.end(); it != end; ++it){  //for each individual
		for (unsigned int j=0;j< it->second.size();j++)  //for each factor
				  result[it->first].push_back(it->second[j].second);
		sort(result[it->first].begin(),result[it->first].end());
	}
	return result;
}


bool LZIndex::tryMatch(LZFactor *factor,int factorOffset, string pattern, int actualPosition,unsigned int &steps){
	char fc; //current factor character

	int pl=pattern.length();
	int fl=factor->length-1;  //solo la parte referenziale
	bool match=true;
	int sai=factor->suffixArrayPosition;
	for (int i=0; i<factorOffset;i++)  //salta i primi fo caratteri del fattore
		sai=reverseBackwardStep(sai,&fc);

	int i=actualPosition; //current pattern position to be matched
	int fo=factorOffset;  //current factor position to be matched
	while (match && i<pl && fo<fl){
		char pc=pattern[i];  //not yet remapped character
		sai=reverseBackwardStep(sai,&fc);
		match=fc==pc;
		if (match){
			i++;
			fo++;
		}
	}
	steps=i-actualPosition;
	return match;
}

//"backward" refers to the pattern, i.e. the match operation "logically" starts from the actualPosition and continues "backward" on the previous characters of the pattern,
//instead the algorithm must proceed "forward", because we are able to retrieve the factors characters from the backward reference index starting from the first character
//of the factor, the second, the third, etc...
//1) find the initial

bool LZIndex::tryBackwardMatch(LZFactor *factor,int factorOffset, string pattern, int actualPosition,unsigned int &steps){
	char fc; //current factor character
	unsigned int p;
	int fl=factor->length-1;  //la sola parte referenziale
	int fo=factorOffset;
	if (actualPosition > factorOffset){
		fo=0;
		p=actualPosition-factorOffset;
	} else{
		fo=factorOffset-actualPosition;
		p=0;
	}

	bool match=true;
	int sai=factor->suffixArrayPosition;
	//skip the characters preceding the factor offset
	for (int i=0;i<fo;i++)
		sai=reverseBackwardStep(sai,&fc);

	int i=p; //current pattern position to be matched
	while (match && i<=actualPosition){
		char pc=pattern[i];
		sai=reverseBackwardStep(sai,&fc);
		match=fc==pc;
		if (match){
			i++;
			fo++;
		}
	}
	steps=actualPosition-p+1;
	return match;
}



inline void LZIndex::locateInternalOccs(const string &pattern,map<int64_t,vector<Occurrence*>> &results){
	int sp;
	int ep;
	int m=pattern.length();
	//Cerca il pattern nella sequenzza di riferimento, determinando il range [sp,ep] dei suffissi che lo hanno come prefisso
	if (searchPatternInReferenceIndex(forwardReferenceIndex,s_for,&forwardOccs,pattern,&sp, &ep)){
		//cout << "\nRange length: " << ep-sp+1 << endl;
		for (int i=sp;i<=ep;i++){
			//determina  la posizione della specifica occorrenza del pattern all'interno della sequenza di riferimento
			//int tp=get_occ_pos(forwardReferenceIndex,s_for, i);
			//int tp=get_suffix(forwardSAFile,ilog2n_reference,i);
			int tp=(*forwardSA)[i];
			//Determina i fattori che iniziano in una posizione del testo compresa in [tp+m-header.lmax,tp]: solo i fattori
			//con tp appartenente a tale intervallo possono contenere il pattern.
			map<int,vector<pair<int,int>>> queryResult=posTree->rangeQuery(tp+m-header.lmax,tp);
			for( auto it = queryResult.begin(), end = queryResult.end(); it != end; ++it){  //for each individual
			    Factorization *fn=getFactorization(it->first);
				for (unsigned int j=0;j<it->second.size();j++){ //for each pair
						LZFactor *f=fn->getFactor(it->second[j].second,bitReader);
						//The first element of each pair is the position of the factor's referential part into the reference sequence
						int tpf=it->second[j].first;
						//The second element of each pair is a factorIndex
						int factorIndex=it->second[j].second;
						LZFactor *factor=fn->getFactor(factorIndex,bitReader);
						int l=factor->length-1; //considero solo la parte "referenziale" del fattore
						if (tpf >= tp+m-l){
							Occurrence *occurrence=new Occurrence();
							occurrence->factorIndex=factorIndex;
							occurrence->factorOffset=tp -tpf;
							occurrence->endingFactorIndex=factorIndex;
							occurrence->endingFactorOffset=occurrence->factorOffset+m-1;
							occurrence->textPosition=factor->textPosition + occurrence->factorOffset;
							results[it->first].push_back(occurrence);
						}
				}
			}
		}
	}

}




//returns the length of the left side longest suffix, i.e. the longest leftSide suffix that occurs in the reference string
//A consideration: we are exploring a factor whose referential part ends with a suffix of leftSide, from the factor's last
//character to its first character; the factorization happens instead in the opposite direction and
//so the left side longest suffix can go beyond the current factor referential part.
//The left side longest suffix can be used to find left side factors for which the left side longest suffix doesn't go beyond
//the current factor referential part: otherwise the pattern will be found in correspondence with a previous split point
inline int LZIndex::findLeftSideLongestSuffix(string leftSide){
		int sp,ep;
		int trySp,tryEp;
		char c;
		char nrc;
		int len=leftSide.length();
		int i=len-1; //current position in the string
		nrc=leftSide[i];
		if (s_for->bool_char_map[nrc]){
			if (len>1){
				c=s_for->char_map[nrc];
				sp=s_for->bwt_occ[c];
				if(c==s_for->alpha_size-1)
					ep=s_for->text_size-1;
				else
					ep=s_for->bwt_occ[c+1]-1;
				i=len-2;
				while (i>=0 &&
					   s_for->bool_char_map[nrc=leftSide[i]] &&
					   backwardStepSuccessful(forwardReferenceIndex,s_for,&forwardOccs,s_for->char_map[nrc],
								              sp,ep,&trySp,&tryEp))
				{
					sp=trySp;
					ep=tryEp;
					i--;
				}
				return len - i -1;
			}
			else
				return 1;
		}
		else
			return 0;

}




//returns the length of the right side longest prefix, i.e. the longest rightSide prefix that occurs in the reference string
//Some considerations: we are exploring a factor referential part from its first to its last character,
//and so the first mismatch found should match with the mismatch letter of the current factor:
//therefore the left side longest suffix cannot go beyond the current factor referential part.
inline int LZIndex::findRightSideLongestPrefix(string rightSide){
		int i=0; //current position in the string
		int sp,ep;
		int trySp,tryEp;
		char c;
		char nrc;
		int len=rightSide.length();
		nrc=rightSide[0];
		if (s_rev->bool_char_map[nrc]){
			if (len>1){
				c=s_rev->char_map[nrc];
				sp=s_rev->bwt_occ[c];
				if(c==s_rev->alpha_size-1)
					ep=s_rev->text_size-1;
				else
					ep=s_rev->bwt_occ[c+1]-1;
				i=1;
				while (i<len &&
					   s_rev->bool_char_map[nrc=rightSide[i]] &&
					   backwardStepSuccessful(reverseReferenceIndex,s_rev,&reverseOccs,s_rev->char_map[nrc],
								              sp,ep,&trySp,&tryEp))
				{
					sp=trySp;
					ep=tryEp;
					i++;
				}
				return i;
			}
			else
				return 1;
		}
		else
			return 0;

}


//Trova i fattori le cui parti "referenziali" terminano con la stringa ls
inline map<int,vector<int>> LZIndex::findLeftSideFactors(string ls,int *lslsLength){
	//find the left side longest suffix (the longest left side suffix that occurs in the reference string)
	int l=findLeftSideLongestSuffix(ls);
	int sp,ep;

	string lsls=ls.substr(ls.length()-l,l);
	if (searchPatternReverseInReferenceIndex(reverseReferenceIndex,s_rev,&reverseOccs,lsls,&sp,&ep)){
		*lslsLength=l;
		//find the factors whose suffixArrayPosition is included into the [sp,ep] interval
		return getFactorsInRange(reverseTree,sp,ep);
	}
	else{
		map<int,vector<int>> emptyMap;
		return emptyMap;
	}
}



//Trova i fattori le cui parti "referenziali" terminano con la stringa ls
inline map<int,map<int,vector<int>>> LZIndex::findLeftSideFactorsEC(string ls){
	//find the left side longest suffix (the longest left side suffix that occurs in the reference string)
	int l=findLeftSideLongestSuffix(ls);
	int sp,ep;

	string lsls=ls.substr(ls.length()-l,l);
	map<int,map<int,vector<int>>> allMap;
	for (int i=0;i<lsls.length();i++){

		string slsls=lsls.substr(i);
		//cout <<slsls << endl;
		if (searchPatternReverseInReferenceIndex(reverseReferenceIndex,s_rev,&reverseOccs,slsls,&sp,&ep)){

			//find the factors whose suffixArrayPosition is included into the [sp,ep] interval
			map<int,vector<int>> m=getFactorsInRange(reverseTree,sp,ep);
			if (m.size()>0){
				map<int,vector<int>> thisLengthMap;
				//cout << "\t" << m.size() << endl;
				for (auto it=m.begin(); it!=m.end(); ++it){
					auto thisLengthMapIt=thisLengthMap.find(it->first);
					if (thisLengthMapIt!=thisLengthMap.end())
						thisLengthMapIt->second.insert(thisLengthMapIt->second.end(), it->second.begin(),it->second.end());
					else
						thisLengthMap[it->first]=it->second;
					/*
					std::cout << "\t\t" << it->first << endl;
					for (int fid:it->second){
						std::cout << "\t\t\t" << fid << endl;
					}*/
				}
				allMap[l]=thisLengthMap;
			}
		}


		//else{
		//	map<int,vector<int>> emptyMap;
		//	return emptyMap;
		//}
	}
	/*
	cout << "\t" << allMap.size() << endl;
	for (auto it=allMap.begin(); it!=allMap.end(); ++it){
			std::cout << "\t\t" << it->first << endl;
			for (int fid:it->second){
				std::cout << "\t\t\t" << fid << endl;
			}
	}
	*/
	return allMap;
}




void strrev(char *p)
{
  char *q = p;
  while(q && *q) ++q;
  for(--q; p < q; ++p, --q)
    *p = *p ^ *q,
    *q = *p ^ *q,
    *p = *p ^ *q;
}




inline map<int,vector<int>> LZIndex::findRightSideFactors(string rs,int *rslpLength){
	//find the right side longest prefix (the longest right side prefix that occurs in the reference string)
	int l=findRightSideLongestPrefix(rs);
	int sp,ep;

	string rslp=rs.substr(0,l);
	std::reverse(std::begin(rslp), std::end(rslp));

	if (searchPatternReverseInReferenceIndex(forwardReferenceIndex,s_for,&forwardOccs,rslp,&sp,&ep)){
	//if (searchPatternInReferenceIndex(forwardReferenceIndex,s_for,&forwardOccs,rslp,&sp,&ep)){
				*rslpLength=l;
				//find the factors whose suffixArrayPosition is included into the [sp,ep] interval
				return getFactorsInRange(forwardTree,sp,ep);
	} else{
		map<int,vector<int>> emptyMap;
		return emptyMap;
	}
	//delete[] crslp;
}





//TODO: riabilitare il metodo
bool LZIndex::verifyPatternRemainingPart(Factorization *fn,string pattern,int splitPoint,Occurrence *occurrence,int lsVerifiedLength,int rsVerifiedLength){

	unsigned int pl=pattern.length();
	unsigned int vl=1 + //the split point character has been verified
			        lsVerifiedLength + rsVerifiedLength;
	bool match=true;
	if (vl<pl){
		int fi; //factor from which we'll start the check
		int fo; //initial factor offset from wich we'll start the check
		//find the left and right side of the pattern that have not been verified
		int lsEnd=splitPoint-1-lsVerifiedLength;
		unsigned int rsStart=splitPoint+rsVerifiedLength+1;
		//verify the remaining right side part
		if (rsStart < pl){
			if (occurrence->endingFactorOffset< fn->getFactor(occurrence->endingFactorIndex,bitReader)->length-1){
				fi=occurrence->endingFactorIndex;
				fo=occurrence->endingFactorOffset+1;
			}else{
				fi=occurrence->endingFactorIndex+1;
				fo=0;
			}
			unsigned int p=rsStart;
			while (match && p<pl && fi<fn->header.factorsNumber){
				//if (factors[fi]->suffixArrayPosition==-1){
				//		if (factors[fi]->letter==pattern[p]){
				LZFactor *factor=fn->getFactor(fi,bitReader);
				if (fo==factor->length-1){
					if (factor->letter==pattern[p]){
							occurrence->endingFactorIndex=fi;
							occurrence->endingFactorOffset=fo;
							p++;
							if (p<pl){  //this check has been added to ensure that fi is the last factor index and fo the offset of the last character of the pattern
								fi++;
								fo=0;
							}
						}
						else
							match=false;
				}
				else{
					if (factor->suffixArrayPosition>-2){
						//LZFactor *factor=factors[fi];
						unsigned int steps;
						match=tryMatch(factor,fo,pattern,p,steps);
						if (match){
								occurrence->endingFactorIndex=fi;
								occurrence->endingFactorOffset=fo+steps-1;
								p+=steps;
								fo+=steps;
							}
					}
				}
			}
		}
		if (match && lsEnd>=0){
			if (occurrence->factorOffset==0){
				if (occurrence->factorIndex>0){ //if a previous factor exists, then  go to its last character
					fi=occurrence->factorIndex-1;
					fo=fn->getFactor(fi,bitReader)->length-1;
				} else
					match=false;
			} else{
				fi=occurrence->factorIndex; //go to the previous character of the same factor
				fo=occurrence->factorOffset-1;
			}

			int p=lsEnd;
			while (match && p>=0 && fi>=0){
				LZFactor *factor=fn->getFactor(fi,bitReader);
				if (fo==factor->length-1){
					if (factor->letter==pattern[p]){
						occurrence->factorIndex=fi;
						occurrence->factorOffset=factor->length-1;
						fo--;
						p--;
						if (p>=0 && fo<0){
							if (fi>0){
								fi--;
								factor=fn->getFactor(fi,bitReader);
								fo=factor->length-1;
							} else
								match=false;
						}
					}
					else
						match=false;
				}
				else{
					if (factor->suffixArrayPosition>-2){
						unsigned int steps;
						match=tryBackwardMatch(factor,fo,pattern,p,steps);
						if (match){
							occurrence->factorIndex=fi;
							occurrence->factorOffset=fo-steps+1;
							p-=steps;
							if (p >= 0){  //this check has been added to ensure that fi is the last factor index and fo the offset of the last character of the pattern
								fi--;
								factor=fn->getFactor(fi,bitReader);
								fo=factor->length-1;
							}
						}
					}
				}
			}
		}
	}
	return match;
}

//TODO: riabilitare il metodo

void LZIndex::locateOverlappingOccs(const string &pattern,map<int64_t,vector<Occurrence*>> &results){
	unsigned int pl=pattern.length();
	char splitPointCharacter;
	for (unsigned int splitPoint=0;splitPoint<pl;splitPoint++){
		if (splitPoint > pl/2 ){
			//ls è più lunga di rs e quindi si presuppone che il numero di fattori che termina con ls sia inferiore rispetto
			//al numero di quelli che inizia con rs
			splitPointCharacter=pattern[splitPoint];
			string ls=pattern.substr(0,splitPoint);
			int lslsLength;
			map<int,vector<int>> lsFactorsPerIndividual=findLeftSideFactors(ls,&lslsLength);  //the factors are grouped by individualId
			for( auto it = lsFactorsPerIndividual.begin(), end = lsFactorsPerIndividual.end(); it != end; ++it){  //for each individual
				Factorization *fn=getFactorization(it->first);  //it->first is the individualId
				if (it->second.size()>0){ //it->second contains the individual occurrences
					string rs=pattern.substr(splitPoint+1,pl-splitPoint-1);
					for (unsigned int i=0;i<it->second.size();i++){
						int lsFactorIndex=it->second[i];
						LZFactor *lsFactor=fn->getFactor(lsFactorIndex,bitReader);
						if (lsFactor->length-1 >= ls.size()){   //esclude il caso in cui esplorando il left side si sconfini nel fattore precedente:
																//in tal caso la soluzione verrà trovata in corrispondenza di uno split point precedente
							if (lsFactor->letter==splitPointCharacter){
								Occurrence *occurrence=new Occurrence();
								occurrence->factorIndex=lsFactorIndex;
								occurrence->factorOffset=lsFactor->length-1-ls.length();
								occurrence->endingFactorIndex=lsFactorIndex;
								occurrence->endingFactorOffset=lsFactor->length-1;
								int rsVerifiedLength=0;
								//int lsVerifiedLength=ls.size();
								int lsVerifiedLength=lslsLength;
								if (verifyPatternRemainingPart(fn,pattern,splitPoint,occurrence,lsVerifiedLength,rsVerifiedLength))
									results[it->first].push_back(occurrence);
							}
						}
					}
				}
			}
		}
		else{
			splitPointCharacter=pattern[splitPoint];
			string rs=pattern.substr(splitPoint+1,pl-splitPoint-1);
			int rslpLength;
			map<int,vector<int>> rsFactorsPerIndividual=findRightSideFactors(rs,&rslpLength);
			for( auto it = rsFactorsPerIndividual.begin(), end = rsFactorsPerIndividual.end(); it != end; ++it){  //for each individual
				Factorization *fn=getFactorization(it->first);
				if (it->second.size()>0){
					string ls=pattern.substr(0,splitPoint);
					for (unsigned int i=0;i<it->second.size();i++){
						int rsFactorIndex=it->second[i];
						int rsFactorLength=fn->getFactor(rsFactorIndex,bitReader)->length;
						int rsVerifiedLength;
						//verificare se realmente occorre la condizione seguente: in quali casi si verifica?
						//un caso potenziale è la fine della stringa  S da fattorizzare (l'ultimo fattore)
						//analogo caso si potrebbe verificare per il lsls rispetto all'inizio della stringa S da fattorizzare (il primo fattore)
						if (rslpLength<rsFactorLength-1)
							rsVerifiedLength=rslpLength;
						else
							rsVerifiedLength=rsFactorLength-1;
						if (rsFactorIndex>0){
							LZFactor *lsFactor=fn->getFactor(rsFactorIndex-1,bitReader);
							if (lsFactor->letter==splitPointCharacter){
								Occurrence *occurrence=new Occurrence();
								occurrence->factorIndex=rsFactorIndex-1;
								occurrence->factorOffset=lsFactor->length-1;
								occurrence->endingFactorIndex=rsFactorIndex;
								occurrence->endingFactorOffset=rsVerifiedLength-1;
								int lsVerifiedLength=0;
								if (verifyPatternRemainingPart(fn,pattern,splitPoint,occurrence,lsVerifiedLength,rsVerifiedLength))
									results[it->first].push_back(occurrence);
							}
						}

					}

				}
			}
		}
	}
}


bool LZIndex::checkPatternCharacters(const string &pattern){
	vector<ExtraSequence> extraSequences;
	int currentExtraSequenceIndex=-1;
	for (unsigned int i=0;i<pattern.length() ;i++){
		int c=pattern[i];
		if (header.charMap[c]!=1)
			return false;
	}
	return true;
}




bool LZIndex::checkPatternCharacters(const string &pattern,bool *withExtraCharacters,ExtraSequence *extraSequence){
	vector<ExtraSequence> extraSequences;
	int currentExtraSequenceIndex=-1;
	*withExtraCharacters=false;
	for (unsigned int i=0;i<pattern.length() ;i++){
		int c=pattern[i];
		if (header.charMap[c]!=1)
			return false;
		else{
			if (cRemappings[c]>-1 && !s_for->bool_char_map[c]){  //it's an extra-character
				if (currentExtraSequenceIndex==-1 ||
					(extraSequences[currentExtraSequenceIndex].position + extraSequences[currentExtraSequenceIndex].length <i)){
					ExtraSequence extraSeq;
					extraSeq.position=i;
					extraSeq.length=1;
					if (currentExtraSequenceIndex==-1)
						extraSeq.lsLength=i;
					else{
						extraSeq.lsLength=i - (extraSequences[currentExtraSequenceIndex].position+
											   extraSequences[currentExtraSequenceIndex].length);
						extraSequences[currentExtraSequenceIndex].rsLength=extraSeq.lsLength;
					}
					extraSequences.push_back(extraSeq);
					currentExtraSequenceIndex++;
				}
				else
					extraSequences[currentExtraSequenceIndex].length++;
			}
		}
	}
	unsigned int ess=extraSequences.size();
	if (ess>0){
		//choose the extra character sequence that will lead the pattern search
		*withExtraCharacters=true;
		extraSequences[ess-1].rsLength=pattern.length()-(extraSequences[ess-1].position+
				   	   	   	   	   	   	   	   	   	   	 extraSequences[ess-1].length);
		int maxLength=-1;
		int maxLengthIndex=-1;
		for (unsigned int i=0;i<ess;i++){
			/*if (extraSequences[i].lsLength > maxLength ){
				maxLength=extraSequences[i].lsLength;
				maxLengthIndex=i;
			} else */if ((int)extraSequences[i].rsLength > maxLength ) {
				maxLength=extraSequences[i].rsLength;
				maxLengthIndex=i;
			}
		}
		*extraSequence=extraSequences[maxLengthIndex];
		extraSequence->sequence=extraSequence->sequence=pattern.substr(extraSequence->position,extraSequence->length);
		if (extraSequence->lsLength>0)
			extraSequence->ls=pattern.substr(extraSequence->position-extraSequence->lsLength,extraSequence->lsLength);
		else
			extraSequence->ls="";
		if (extraSequence->rsLength>0)
				extraSequence->rs=pattern.substr(extraSequence->position+extraSequence->length,extraSequence->rsLength);
		else
				extraSequence->rs="";

	}
	else
		*withExtraCharacters=false;
	return true;
}


 void LZIndex::locate(const string &pattern,SearchStatistics &stats,map<int64_t,vector<Occurrence*>> &results){
	//checkReverseReferenceIndex();
	//checkForwardReferenceIndex();
	//checkForwardSuffixArray();

	//bool withExtraCharacters;
	//ExtraSequence extraSequence;

	//bool patternOK=checkPatternCharacters(pattern,&withExtraCharacters, &extraSequence);
	bool patternOK=checkPatternCharacters(pattern);
	if (patternOK){
		/*
		if (withExtraCharacters){
			stats.internalLocateTime=0;
			stats.overlappingLocateTime=0;
			locateExtraCharacters(pattern,extraSequence,results,stats);
		}
		else*/
			locateNoExtraCharacters(pattern,results,stats);
	};

	//Sort the results by position and remove duplicates
	for( auto it = results.begin(), end = results.end(); it != end; ++it){  //for each individual
		if (it->second.size()>0){
			//sort the results by position
			sort(it->second.begin(),it->second.end(),comparePtrToOccurrence);
			//remove duplicates
			int resultsSize=it->second.size();
			for (int i=resultsSize-1;i>0;i--)
				if (it->second[i]->factorIndex==it->second[i-1]->factorIndex &&
					it->second[i]->factorOffset==it->second[i-1]->factorOffset)
							it->second.erase(it->second.begin()+i);
			//find the text positions
			for (unsigned int i=0;i<it->second.size();i++){
				int factorStartPosition= getFactorization(it->first)->getFactor(it->second[i]->factorIndex,bitReader)->textPosition;
				it->second[i]->textPosition=factorStartPosition + it->second[i]->factorOffset;
			}

		}
	}

	//stats.withExtraCharacters=withExtraCharacters;
	stats.withExtraCharacters=false;

}




inline void LZIndex::locateNoExtraCharacters(const string &pattern,map<int64_t,vector<Occurrence*>> &results,SearchStatistics &stats){
	std::chrono::time_point<std::chrono::system_clock> startTime,endTime1,endTime2,endTime3;
	startTime=std::chrono::system_clock::now();
	//map<int64_t,vector<Occurrence*>> results=
	locateInternalOccs(pattern,results);
	endTime1=std::chrono::system_clock::now();
	//map<int64_t,vector<Occurrence*>> overlappingResults=locateOverlappingOccs(pattern);
	locateOverlappingOccs(pattern,results);
	endTime2=std::chrono::system_clock::now();

	//for( auto it = overlappingResults.begin(), end = overlappingResults.end(); it != end; ++it)
	//	results[it->first].insert(results[it->first].end(), overlappingResults[it->first].begin(), overlappingResults[it->first].end());
	//
	//endTime3=std::chrono::system_clock::now();
	//gather statistics
	stats.internalLocateTime=std::chrono::duration_cast<std::chrono::microseconds>(endTime1 - startTime).count();
	stats.overlappingLocateTime=std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - endTime1).count();
	stats.assemblingTime=std::chrono::duration_cast<std::chrono::microseconds>(endTime3 - endTime2).count();
	//return results;
}


inline void LZIndex::locateExtraCharacters(const string &pattern,ExtraSequence &extraSequence,map<int64_t,vector<Occurrence*>> &results,
		SearchStatistics &stats){
	std::chrono::time_point<std::chrono::system_clock> startTime,endTime;
	startTime=std::chrono::system_clock::now();

	//if (extraSequence.rsLength==0 || extraSequence.lsLength >= extraSequence.rsLength){
	if (extraSequence.rsLength==0){
		//search for factors whose referential parts end with the left side
		int lslsLength;  //left side longest suffix length
		map<int,map<int,vector<int>>> lengthsMap=findLeftSideFactorsEC(extraSequence.ls);  //the factors are grouped by left side longest prefix length and then by individualId

		for (auto itLength=lengthsMap.begin(), end = lengthsMap.end(); itLength != end; ++itLength){

		lslsLength=itLength->first;
		auto lsFactorsPerIndividual=itLength->second;

		for( auto it = lsFactorsPerIndividual.begin(), end = lsFactorsPerIndividual.end(); it != end; ++it){  //for each individual
			Factorization *fn=getFactorization(it->first);  //it->first is the individualId
			if (it->second.size()>0){ //it->second contains the individual occurrences
				string rs=extraSequence.sequence.substr(1,extraSequence.length-1) + extraSequence.rs;
				for (unsigned int i=0;i<it->second.size();i++){
					int lsFactorIndex=it->second[i];
					LZFactor *lsFactor=fn->getFactor(lsFactorIndex,bitReader);
					if (lsFactor->letter==extraSequence.sequence[0]){
						int lsFactorLength=lsFactor->length;
						//int lsVerifiedLength=extraSequence.ls.size();
						int lsVerifiedLength;
						if (lslsLength < lsFactorLength-1)
							//lsVerifiedLength=lslsLength+1;
							lsVerifiedLength=lslsLength;
						else
							//lsVerifiedLength=lsFactorLength;
							lsVerifiedLength=lsFactorLength-1;
						Occurrence *occurrence=new Occurrence();
						occurrence->factorIndex=lsFactorIndex;
						occurrence->factorOffset=lsFactor->length-1-lsVerifiedLength;
						occurrence->endingFactorIndex=lsFactorIndex;
						occurrence->endingFactorOffset=lsFactor->length-1;
						int rsVerifiedLength=0;

						//the split point is the position of the first extra-character of the extra-sequence
						if (verifyPatternRemainingPart(fn,pattern,extraSequence.position,occurrence,lsVerifiedLength,rsVerifiedLength))
							results[it->first].push_back(occurrence);
					}
				}
			}
		}
		}

	} else{
		int rslpLength;
		map<int,vector<int>> rsFactorsPerIndividual=findRightSideFactors(extraSequence.rs,&rslpLength);
		for( auto it = rsFactorsPerIndividual.begin(), end = rsFactorsPerIndividual.end(); it != end; ++it){  //for each individual
			Factorization *fn=getFactorization(it->first);
			//if (it->first==34)
			//	cout << endl;
			if (it->second.size()>0){
				string ls=extraSequence.ls + extraSequence.sequence;
				for (unsigned int i=0;i<it->second.size();i++){
					int rsFactorIndex=it->second[i];
					int rsFactorLength=fn->getFactor(rsFactorIndex,bitReader)->length;
					int rsVerifiedLength;
					//verificare se realmente occorre la condizione seguente: in quali casi si verifica?
					//un caso potenziale è la fine della stringa  S da fattorizzare (l'ultimo fattore)
					//analogo caso si potrebbe verificare per il lsls rispetto all'inizio della stringa S da fattorizzare (il primo fattore)
					if (rslpLength<rsFactorLength-1)
						rsVerifiedLength=rslpLength;
					else
						rsVerifiedLength=rsFactorLength-1;
					if (rsFactorIndex>0){
						LZFactor *lsFactor=fn->getFactor(rsFactorIndex-1,bitReader);
						if (lsFactor->letter==extraSequence.sequence[extraSequence.length-1]){
							Occurrence *occurrence=new Occurrence();
							occurrence->factorIndex=rsFactorIndex-1;
							occurrence->factorOffset=lsFactor->length-1;
							occurrence->endingFactorIndex=rsFactorIndex;
							occurrence->endingFactorOffset=rsVerifiedLength-1;
							int lsVerifiedLength=0;
							if (verifyPatternRemainingPart(fn,pattern,extraSequence.position+extraSequence.length-1,occurrence,lsVerifiedLength,rsVerifiedLength))
								results[it->first].push_back(occurrence);
						}
					}

				}

			}
		}
	}
	endTime=std::chrono::system_clock::now();
	stats.locateTime=std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}







void LZIndex::setBlockSize(int bs){
	this->bs=bs;
}


int LZIndex::getFactorsNumber(int64_t factorizationIndex){
	return factorizations[factorizationIndex]->factors.size();
}


vector<LZFactor*> LZIndex::getFactors(int64_t factorizationIndex){
	return factorizations[factorizationIndex]->factors;
}


/* *********************************************************************
   read len chars before the position given by row using the LF mapping
   stop if the beginning of the file is encountered
   return the number of chars actually read
   La riga row indica la posizione nell'array dei suffissi e non nella matrice delle rotazioni
   e quindi deve essere sottoposta ad EOF_shift prima di iniziare l'elaborazione
   ********************************************************************* */
int  LZIndex::go_back(FM_INDEX *Infile,int row, int len, char *dest, bwi_out *s)
{
  int written,curr_row,n;
  uchar c,c_sb;
  //int occ_b[256];
  int occ_b;


  curr_row=EOF_shift(s,row);
  for(written=0;written<len; ) {
    // fetches info from the header of the superbucket
	//get_info_sb(Infile,s,curr_row,occ_sb);
	int sbi =  curr_row/Infile->Bucket_size_lev1;
	if (s->buclist_lev1[sbi]==NULL)
		load_sb(Infile, s,sbi);
	bucket_lev1 *sb=s->buclist_lev1[sbi];


    // fetches occ into occ_b properly remapped and returns
    // the remapped code for occ_b of the char in the  specified position
    //c = get_info_b_ferdy(Infile,s,NULL_CHAR,curr_row,occ_b,WHAT_CHAR_IS);
	c = get_info_b_ferdy(Infile,s,NULL_CHAR,curr_row,&occ_b,WHAT_CHAR_IS);

    c_sb = sb->inv_map_sb[c];  //TODO: rivedere la logica

    assert(c_sb < s->alpha_size);
    //printf("%c\n",s->inv_char_map[c_sb]);

    dest[written++] = s->inv_char_map[c_sb]; // store char
    //n = sb->occ[c_sb] + occ_b[c] - 1;         // # of occ before curr_row
    n = sb->occ[c_sb] + occ_b - 1;         // # of occ before curr_row
    curr_row = s->bwt_occ[c_sb] + n;         // get next row
    if(curr_row==s->bwt_eof_pos) break;
    curr_row = EOF_shift(s,curr_row);
  }
  return written;
}


string LZIndex::getFactorString(int suffixArrayPosition,int length){
	char *factorString=new char[length+1];
	go_back(reverseReferenceIndex,suffixArrayPosition,length,factorString,s_rev);
	factorString[length]=0;
	return string(factorString);
}



}/* namespace std */
