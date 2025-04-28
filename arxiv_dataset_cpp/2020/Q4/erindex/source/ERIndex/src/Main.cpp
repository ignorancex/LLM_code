/*
 * Main.cpp
 *
 *  Created on: 21/set/2014
 *      Author: fernando
 */
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <map>
#include <cmath>
#include <thread>
#include <mutex>
#include <fstream>
#include <cerrno>
#include <unordered_map>
#include <vector>
#include <random>
#include <limits>
#include <libgen.h>
#include <cstring>
#include "LZFactor.h"
#include "LZIndex.h"
#include "BPlusTree.h"
#include "FileBitWriter.h"
#include "FileBitReader.h"
#include "EncryptedFileBitWriter.h"
#include "EncryptedFileBitReader.h"
#include "Portfolio.h"
#include "Test.h"
#include <boost/foreach.hpp>
#include <sdsl/suffix_arrays.hpp>
#include <stdio.h>
#include "Database.h"
#include "BitSetRLEEncoder.h"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <iostream>
#include <cmath>






using namespace std;
using namespace sdsl;
using namespace boost::accumulators;

#define EOF_shift(bwiOutPointer,n) (n < bwiOutPointer->bwt_eof_pos) ? n+1 :  n
extern "C" {
	#include "../srcFMIndex/common.h"
	//int go_forw(int row, int len, char *dest, bwi_out *s);
	char *reverse_string(char *source);
	void load_all_superbuckets(FM_INDEX *Infile,bwi_out *s);
	void load_all_buckets(FM_INDEX *Infile,bwi_out *s);
	void load_sb(FM_INDEX *Infile,bwi_out *s, int sbi);
	void load_b(FM_INDEX *Infile,bwi_out *s, int bi);
	uchar get_info_b_ferdy(FM_INDEX *Infile,bwi_out *s, uchar ch, int pos, int *occ, int flag);
}

typedef pair<char, unordered_map<int,int>> CharMapPair;
/*
void testBTree(){
	BPlusTree t(3,1000);

    int conta=0;
	for (int i=0;i<100;i++){
		int r=rand()%100;
		int s=100-r;
		//int s=i;
		//int r=10000-i;
		if (r>=50 && r<=60)
			conta++;
		t.insert(r,s);
		//t.dumpTree();
	}
	cout << "Numero di chiavi comprese nel range: "<<conta << endl;


	t.dumpTree();
    cout << "Traversal of the constructed tree is: \n";
    //t.traverse();
    //cout<<endl;
    t.treeTraverse();
    cout<<endl;


    //for (int i=0;i<10;i++){
	//	int k = i*100;
	//	cout<<k<<" is ";
	//	int value;
	//	(t.search(k,&value))? cout << "Present: value=" << value << " \n" : cout << "Not Present\n";
    //}

    vector<pair<int,vector<int>>> result=t.rangeQuery(50,60);
    for (unsigned int i=0;i<result.size();i++){
    	cout << result[i].first << ",<";
    	for (unsigned int j=0;j< result[i].second.size();j++){
    					  cout << result[i].second[j];
    					  if (j<result[i].second.size()-1)
    						  cout << ",";
    	}
    	cout << ">," << endl;
    }
    cout << "Numero di chiavi comprese nel range: "<<result.size() << endl;
}
*/


string *loadFasta(string fastaFileName) {
	std::ifstream in(fastaFileName, std::ios::in);
	if (in) {
		string line;
		string *sequence=new string();
		std::getline( in, line );
		while( std::getline( in, line ).good() ){
		        *sequence += line;
		}
		return sequence;
	}
}

map<int64_t,vector<int>> naiveSearch(vector<int64_t> ids,vector<string*> sequences, string pattern){
	map<int64_t,vector<int>> results;
	for (int i=0;i<ids.size();i++){
		int id=ids[i];
		string *s=sequences[i];
		string::size_type pos = s->find(pattern);
		while(pos != std::string::npos){
			results[id].push_back(pos);
			pos = s->find(pattern,pos+1);
		}
	}
	return results;
}


double computeMedian(vector<double>::const_iterator begin,
              vector<double>::const_iterator end) {
    int len = end - begin;
    auto it = begin + len / 2;
    double m = *it;
    if ((len % 2) == 0) m = (m + *(--it)) / 2;
    return m;
}

tuple<double, double, double> computeQuartiles(const vector<double>& v) {
    auto it_second_half = v.cbegin() + v.size() / 2;
    auto it_first_half = it_second_half;
    if ((v.size() % 2) == 0) --it_first_half;

    double q1 = computeMedian(v.begin(), it_first_half);
    double q2 = computeMedian(v.begin(), v.end());
    double q3 = computeMedian(it_second_half, v.end());
    return make_tuple(q1, q2, q3);
}


tuple<vector<double>,vector<double>> removeOutliers(const vector<double>& v,double range_lb,double range_ub){
	vector<double> no_outliers_v;
	vector<double> outliers_v;
	for (double d:v){
		if (d < range_lb || d> range_ub)
			outliers_v.push_back(d);
		else
			no_outliers_v.push_back(d);
	}

	return make_tuple(no_outliers_v,outliers_v);

}


double computeMean(vector<double> v){
	double s=0;
	for (double d:v)
		s+=d;
	s=s/(double)v.size();
	return s;
}

double computeStdDev(vector<double> v){
	double m=computeMean(v);
	double s=0;
	for (double d:v)
		s+=pow(d-m,2);
	s=s/((double)v.size()-1);
	return sqrt(s);
}




void dumpIndex(string xmlFilePath,string dumpFileName){
	Test test(xmlFilePath);
	for (auto const& testGroup : test.testGroups){
		Database database(testGroup.databaseRoot);
		//login
		cout << "userId " << testGroup.userId  << endl;
		database.login(testGroup.userId,testGroup.userPrivateKey);  //TODO: add the password
		std::chrono::time_point<std::chrono::system_clock> startTime,endTime;
		startTime=std::chrono::system_clock::now();
		cout << "indexFileName: " << testGroup.indexFileName  << endl;
		LZIndex *index=database.openIndex(testGroup.indexFileName);
		if (testGroup.loadReferenceInMemory){
			cout << "Loading reference in memory" << endl;
			index->loadReferenceInMemory();
		}


		index->dumpIndex(dumpFileName);


	}
}

void testLocate(string xmlFilePath){
	Test test(xmlFilePath);
	for (auto const& testGroup : test.testGroups){
		Database database(testGroup.databaseRoot);
		//login
		cout << "userId " << testGroup.userId  << endl;
		database.login(testGroup.userId,testGroup.userPrivateKey);  //TODO: add the password
		std::chrono::time_point<std::chrono::system_clock> startTime,endTime;
		startTime=std::chrono::system_clock::now();
		cout << "indexFileName: " << testGroup.indexFileName  << endl;
		LZIndex *index=database.openIndex(testGroup.indexFileName);
		if (testGroup.loadReferenceInMemory){
			cout << "Loading reference in memory" << endl;
			index->loadReferenceInMemory();
		}
		if (testGroup.loadIndexInMemory){
			cout << "Loading index in memory" << endl;
			index->loadAllInMemory();
		}
		endTime=std::chrono::system_clock::now();
		double openingTime=std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		//il codice è cambiato per supportare più individui
		//int factorsNumber=index.getFactorsNumber();
		cout << "Index opening time: " << openingTime << "ms" << endl;
		//cout << "Number of factors: " << factorsNumber << endl;  //il codice è cambiato per supportare più individui

		vector<string*> sequences;
		vector<int64_t> individualIds=index->getIndividualIds();
		if (testGroup.naiveSearch){
			sequences.resize(individualIds.size());
			for (unsigned int i=0;i<individualIds.size();i++){
				string code=database.getIndividual(individualIds[i])->code;
				string sequenceFilePath=testGroup.sequencesDirectory + "/" + code + "_" + testGroup.referenceIdentifier +".fa";
				sequences[i]=loadFasta(sequenceFilePath);
				/*//check factorization correctness
				string t=index->getSequence(individualIds[i]);
				if (t!=*sequences[i])
					cout << "FACTORIZATION "<<  i<< " ERROR" << endl;
				else
					cout << "FACTORIZATION "<<  i<< " OK" << endl;
				*/
			}
		}

		for (auto const& patternSet : testGroup.patternSets){
			cout << "Pattern set length: " << patternSet.length << endl ;
			vector<double> times;
			vector<double> timesPerOccurrence;
			for(auto const& pattern:patternSet.patterns){
				startTime=std::chrono::system_clock::now();
				cout << "\tPattern " << pattern ;
				SearchStatistics stats;
				map<int64_t,vector<Occurrence*>> overallResults;
				index->locate(pattern,stats,overallResults);



				endTime=std::chrono::system_clock::now();
				double locateTime=(double)std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()/(double)1000;
				times.push_back(locateTime);

				//count the number of results
				int numberOfResults=0;
				for (auto it=overallResults.begin(); it!=overallResults.end(); ++it){
					numberOfResults+=it->second.size();
				}

				double locateTimePerOccurrence=locateTime/(double)numberOfResults;
				timesPerOccurrence.push_back(locateTimePerOccurrence);

				cout << "\tLocate time: " << locateTime << "ms";
				cout << "\t(" ;
				cout << stats.internalLocateTime;
				cout << "," << stats.overlappingLocateTime;
				cout << ")";
				cout << "\toccurrences: " << numberOfResults << " ";
				if (testGroup.naiveSearch){
						startTime=std::chrono::system_clock::now();
						map<int64_t,vector<int>> naiveOverallResults=naiveSearch(individualIds,sequences,pattern);
						int numberOfNaiveResults=0;
						for (auto it=naiveOverallResults.begin(); it!=naiveOverallResults.end(); ++it){
							numberOfNaiveResults+=it->second.size();
						}
						endTime=std::chrono::system_clock::now();
						locateTime=std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
						cout << "(Naive " << numberOfNaiveResults << ", in " << locateTime << " ms)";
						bool searchError=false;
						for (unsigned int i=0;i<individualIds.size();i++){
							numberOfResults=overallResults[individualIds[i]].size();
							numberOfNaiveResults=naiveOverallResults[individualIds[i]].size();
							if (numberOfResults!=numberOfNaiveResults)
								searchError=true;
							else
								for (int j=0;j<numberOfResults;j++){
									//cout << "\t" << result[i].textPosition << " <" << result[i].factorIndex  << "," << result[i].factorOffset << ">";
									int naiveResult=naiveOverallResults[individualIds[i]][j];
									if (overallResults[individualIds[i]][j]->textPosition != naiveResult){
										cout << endl << overallResults[individualIds[i]][j]->textPosition << endl;
										cout << naiveResult << endl;
										searchError=true;
										break;
									}

								}
							if (searchError)
								break;
						}
						cout << (searchError?"\tERROR":"");
				}
				cout << endl;
			}

			double m=computeMean(times);
			double d=computeStdDev(times);
			double m2=computeMean(timesPerOccurrence);
			double d2=computeStdDev(timesPerOccurrence);

			sort(times.begin(),times.end());
			auto q=computeQuartiles(times);
			double q1=get<0>(q);
			double q3=get<2>(q);
			double iqr=q3-q1;
			double range_lb=q1-3*iqr;
			double range_ub=q3+3*iqr;

			auto t=removeOutliers(times,range_lb,range_ub);
			cout << "Number of outliers in times: " << get<1>(t).size() << "/" << times.size() << endl;
			double mno=computeMean(get<0>(t));
			double dno=computeStdDev(get<0>(t));

			double median=get<1>(q);
			sort(timesPerOccurrence.begin(),timesPerOccurrence.end());
			q=computeQuartiles(timesPerOccurrence);
			q1=get<0>(q);
			q3=get<2>(q);
			iqr=q3-q1;
			range_lb=q1-3*iqr;
			range_ub=q3+3*iqr;

			t=removeOutliers(timesPerOccurrence,range_lb,range_ub);
			cout << "Number of outliers in timesPerOccurrence: " << get<1>(t).size() << "/" << times.size() << endl;
			double mno2=computeMean(get<0>(t));
			double dno2=computeStdDev(get<0>(t));
			double median2=get<1>(q);



			std::cout << "------------------------------------------" << endl;
			std::cout << "indexFileName" << "," <<  "patternSetLength" << ","<< "median" << "," << "mean" << "," << "stdDev" << "," << "mean_no" << "," << "stddev_no" << ","<< "medianPerOccurrence" << "," << "meanPerOccurrence" << "," << "stdDevPerOccurrence" << ","<< "meanPerOccurrence_no" << "," << "stdDevperOccurrrence_no" << endl;
			std::cout << testGroup.indexFileName << "," <<  patternSet.length << ","<< median << "," << m << "," << d << "," << mno << "," << dno << ","<< median2 << "," << m2 << "," << d2 << ","<< mno2 << "," << dno2 << endl;
			std::cout << "------------------------------------------" << endl;
			std::cout << endl;
			std::cout << endl;

		}
		index->close();
	}
}






/*
void testSearch(){
	string referenceIndexFileName="/work/phd/progettoRicerca/dati/1000genomes/indexes/hs37d5_chr20_rev.bwi";
	string indexFileName="/work/phd/progettoRicerca/dati/1000genomes/indexes/NA06989_20.elz";

	LZIndex indx(referenceIndexFileName);
	std::chrono::time_point<std::chrono::system_clock> startTime,endTime;
	startTime=std::chrono::system_clock::now();
	indx.open(indexFileName);
	vector<LZFactor*> factors= indx.getFactors();

	endTime=std::chrono::system_clock::now();
	double openingTime=std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	cout << "Opening time: " << openingTime << " ms" << endl;
	cout << "Number of factors: " << factors.size() << endl;

	indx.locate("CAACT");cout << "Verifying factor " << lsFactorIndex << endl;

	indx.close();

}
*/

bool testBinarySearch(){
	int n=7;
	int keys[7]={1,3,5,10,15,18,20};
    int k=0;

	int p=0; int u=n-1;
	int pos=-1;
	while (pos ==-1 && p<=u){
			int m=(p+u)/2;
			if (keys[m] > k )
					u=m-1;
			else if (keys[m] < k)
					p=m+1;
			else{
			   pos=m;
			   return true;
			}
	}
	pos=p;
	return false;
}




// ****** Write in file f the n bits taken from vv (possibly n > 24)
void fbit_write(FILE *f,int* Bit_buffer_size, uint32* Bit_buffer, int n, int vv)
{
  void fbit_write24(FILE *f,int* Bit_buffer_size, uint32* Bit_buffer,int n, int vv);
  uint32 v = (uint32) vv;

  assert(n <= 32);
  if (n > 24){
    fbit_write24(f,Bit_buffer_size,Bit_buffer,n-24, (v>>24) & 0xffL);
    fbit_write24(f,Bit_buffer_size,Bit_buffer,24, v & 0xffffffL);
  } else {
    fbit_write24(f,Bit_buffer_size,Bit_buffer,n,v);
  }
}


void fbit_write24(FILE *f,int* Bit_buffer_size, uint32* Bit_buffer, int n, int vv)
{                              // v contains bits to read starting from
                               // the least significant bits
  uint32 v = (uint32) vv;

  assert(*Bit_buffer_size<8);
  assert(n>0 && n<=24);
  assert( v < 1u <<(n+1) );

  /* ------- add n bits to Bit_buffer -------- */
  *Bit_buffer_size += n;       // add first, to compute the correct shift
  *Bit_buffer |= (v << (32 - *Bit_buffer_size));  // compact to end of the buffer

  /* ------- flush Bit_buffer as much as possible ----- */
  while (*Bit_buffer_size>=8) {
    if( putc((*Bit_buffer>>24),f) == EOF) {
      fprintf(stderr,"Error writing to output file -fbit_write-\n");
      exit(1);
    }
    *Bit_buffer <<= 8;
    *Bit_buffer_size -= 8;
  }
}

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


__inline__ int get_suffix(FILE *Safile, int n,int i)
{
  //int pos,rem;
  long pos,rem;

  //int fbit_read_simple(FILE *,int*, uint32*,int);
  int Pointer_size=int_log2(n);
  uint32* Bit_buffer=(uint32*)malloc(sizeof(uint32));
  int*  Bit_buffer_size=(int*)malloc(sizeof(int));  /* number of unread/unwritten bits in Bit_buffer */

  pos = (long)i*(long)Pointer_size;
  rem = int(pos& 0x7);         // last 3 bits

  fseek(Safile, pos>>3, SEEK_SET);

  *Bit_buffer= (uint32) 0;
  *Bit_buffer_size=0;

  if(rem)
    fbit_read_simple(Safile,Bit_buffer_size,Bit_buffer,rem);

  int val=fbit_read_simple(Safile, Bit_buffer_size,Bit_buffer,Pointer_size);
  free(Bit_buffer);
  free(Bit_buffer_size);
  return val;
}

void fbit_flush(FILE *f,int* Bit_buffer_size, uint32* Bit_buffer)
{
  //void fbit_write24(FILE  *f,int* Bit_buffer_size, uint32* Bit_buffer, int n, int vv);

  if(*Bit_buffer_size!=0)
    fbit_write24(f, Bit_buffer_size, Bit_buffer,  8 - (*Bit_buffer_size%8) , 0);  // pad with zero !
}



string getSuffix(string s,int i){
	return s.substr(i,s.length()-i);
}

string getRotation(string s,int i){
	return s.substr(i,s.length()-i) + s.substr(0,i);
}


void saveCorrespondenceArrayToDisk(string fileName,int *a,int n){
	FileBitWriter bw(fileName);
	bw.open();
	int psize = int_log2(n);
	for(int i=0;i<n;i++)
	    bw.write(psize,a[i]);
	bw.flush();
	bw.close();
}

void loadCorrespondenceArrayFromDisk(string fileName,int *a,int n){
	FileBitReader br(fileName);
	br.open();
	int psize = int_log2(n);
	for(int i=0;i<n;i++)
		a[i]=br.read(psize);
	br.close();
}


void saveCorrespondenceArrayToDisk(FILE *f,int *a,int n){
	uint32* Bit_buffer=(uint32*)malloc(sizeof(uint32));
	int*  Bit_buffer_size=(int*)malloc(sizeof(int));  /* number of unread/unwritten bits in Bit_buffer */
	*Bit_buffer=0;
	*Bit_buffer_size = 0;

	int psize = int_log2(n);
	for(int i=0;i<n;i++)
	    fbit_write(f,Bit_buffer_size,Bit_buffer,psize,a[i]);
	  fbit_flush(f,Bit_buffer_size,Bit_buffer);
}





void loadCorrespondenceArrayFromDisk(FILE *f,int *a,int n){
	//int fbit_read_simple(FILE *,int*, uint32*,int);

	uint32* Bit_buffer=(uint32*)malloc(sizeof(uint32));
	int*  Bit_buffer_size=(int*)malloc(sizeof(int));  /* number of unread/unwritten bits in Bit_buffer */
	*Bit_buffer=0;
	*Bit_buffer_size = 0;

	int psize = int_log2(n);
	for(int i=0;i<n;i++)
		a[i]=fbit_read_simple(f, Bit_buffer_size,Bit_buffer,psize);
}



typedef pair<int, int> IntIntPair;

/**
 *
 */
void buildSACorrespondence(int n,string reverseSaFileName,string saFileName,string r2fFileName,string f2rFileName){
	//int sa[n];
	int revSa[n];

	cout << "Reverse suffix array" << endl;
	FILE *revSaFile = fopen(reverseSaFileName.c_str(), "a+b");
	fseek(revSaFile, 0L, SEEK_SET);

	//revSa[60000000]=get_suffix(revSaFile,135006516,60000000);
	//revSa[127717681]=get_suffix(revSaFile,135006516,127717681);
	//revSa[135006515]=get_suffix(revSaFile,135006516,135006515);


	for (int i=0;i<n;i++){
		revSa[i]=get_suffix(revSaFile,n,i);
	}
	fclose(revSaFile);

	//for (int i=0;i<n;i++)
	//	cout << revSa[i] << endl;

	unordered_map<int, int> saSuffixes;
	cout << "Suffix array" << endl;

	FILE *saFile = fopen(saFileName.c_str(), "a+b");
	fseek(saFile, 0L, SEEK_SET);
	for (int i=0;i<n;i++){
		if (i%100000==0)
			cout << i << endl;
		int s=get_suffix(saFile,n,i);
		//cout << s << endl;
		saSuffixes.insert(IntIntPair(s,i));
	}
	fclose(saFile);

	//Calcolo i due vettori r2f ed f2r, che consentono rispettivamente di passare da un suffisso dellla stringa reverse al corrispondente della straight e viceversa
	int r2f[n];
	int f2r[n];
	cout << "Calcolo R2F e F2R" << endl;
	for (int i=0;i<n;i++){
		if (i%100000==0)
			cout << i << endl;
		if (saSuffixes.count(n-revSa[i]-1)==0){
			cout << "NNCUC" << endl;
			exit(1);
		}
		int j=saSuffixes[n-revSa[i]-1]; //suffisso che corrisponde alla stessa posizione del testo
		r2f[i]=j;
		f2r[j]=i;
	}

	/*
	cout << "R2F" <<endl;
	for (int i=0;i<n;i++)
			cout << r2f[i] << endl;

	cout << "F2R" <<endl;
		for (int i=0;i<n;i++)
				cout << f2r[i] << endl;
	*/
	//Salvo i due array su disco
	saveCorrespondenceArrayToDisk(r2fFileName,r2f,n);
	saveCorrespondenceArrayToDisk(f2rFileName,f2r,n);

}


void buildSACorrespondence_2(int n,string reverseSaFileName,string saFileName,string r2fFileName,string f2rFileName){
	//int sa[n];
	//int revSa[n];

	cout << "Reverse suffix array" << endl;
	FILE *revSaFile = fopen(reverseSaFileName.c_str(), "a+b");
	fseek(revSaFile, 0L, SEEK_SET);
	//for (int i=0;i<n;i++){
	//	revSa[i]=get_suffix(revSaFile,n,i);
	//}


	//for (int i=0;i<n;i++)
	//	cout << revSa[i] << endl;

	//map<int, int> saSuffixes;
	int saSuffixes[n];
	std:fill_n(saSuffixes,n,-1);
	cout << "Suffix array" << endl;

	FILE *saFile = fopen(saFileName.c_str(), "a+b");
	fseek(saFile, 0L, SEEK_SET);
	for (int i=0;i<n;i++){
		if (i%100000==0)
			cout << i << endl;

		int s=get_suffix(saFile,n,i);
		//cout << s << endl;
		//saSuffixes.insert(IntIntPair(s,i));
		saSuffixes[s]=i;
	}
	fclose(saFile);

	//Calcolo i due vettori r2f ed f2r, che consentono rispettivamente di passare da un suffisso dellla stringa reverse al corrispondente della straight e viceversa
	int r2f[n];
	int f2r[n];
	int revSa_di_i;
	int j;
	cout << "Calcolo R2F e F2R" << endl;
	for (int i=0;i<n;i++){
		if (i%100000==0)
			cout << i << endl;
		revSa_di_i=get_suffix(revSaFile,n,i);
		if (saSuffixes[n-revSa_di_i-1]==-1){
			cout << "NNCUC" << endl;
			exit(1);
		}
		j=saSuffixes[n-revSa_di_i-1]; //suffisso che corrisponde alla stessa posizione del testo
		r2f[i]=j;
		f2r[j]=i;
	}

	fclose(revSaFile);
	/*
	cout << "R2F" <<endl;
	for (int i=0;i<n;i++)
			cout << r2f[i] << endl;

	cout << "F2R" <<endl;
		for (int i=0;i<n;i++)
				cout << f2r[i] << endl;
	*/
	//Salvo i due array su disco
	saveCorrespondenceArrayToDisk(r2fFileName,r2f,n);
	saveCorrespondenceArrayToDisk(f2rFileName,f2r,n);

}





/*
void testBuildSACorrespondence(void){
	int nReverseSa;
	int nSa;
	int n;

	FILE *revSaFile;
	FILE *saFile;

	string s="ACTGAACCGTA$";
	string sRev="ATGCCAAGTCA$";

	string reverseSaFileName="/work/phd/progettoRicerca/dati/artificial/revShortString.seq.sa";
	string saFileName="/work/phd/progettoRicerca/dati/artificial/shortString.seq.sa";
	n=11;
	int sa[n];
	int revSa[n];

	revSaFile = fopen(reverseSaFileName.c_str(), "a+b");
	fseek(revSaFile, 0L, SEEK_END);
	nReverseSa=ftell(revSaFile);
	fseek(revSaFile, 0L, SEEK_SET);

	cout << "Dump dell'array dei suffissi ("<< sRev << ")" << endl;
	for (int i=0;i<n;i++){
		revSa[i]=get_suffix(revSaFile,11,i);
		cout <<  getRotation(sRev.substr(0,sRev.length()-1),revSa[i]) << "\t" <<  getRotation(sRev,revSa[i]) << "\t(" <<revSa[i] << ")"<< "\t" << getSuffix(sRev,revSa[i]) << endl;
	}

	saFile = fopen(saFileName.c_str(), "a+b");
	fseek(saFile, 0L, SEEK_END);
	nSa=ftell(saFile);
	fseek(saFile, 0L, SEEK_SET);
	cout << "Dump dell'array dei suffissi ("<< s << ")" << endl;
	for (int i=0;i<n;i++){
		sa[i]=get_suffix(saFile,11,i);
		cout   << getRotation(s.substr(0,s.length()-1),sa[i]) << "\t" << getRotation(s,sa[i]) << "\t(" <<sa[i] << ")" << "\t" << getSuffix(s,sa[i]) << endl;
	}

	//Determino la corrispondenza tra saRev ed sa
	int r2f[n];
	int f2r[n];
    for (int i=0;i<n;i++){
    	int pos=-1;
    	int j=0;
    	while (pos==-1 && j<n){
    		if (sa[j]==n-revSa[i]-1)
    			pos=j;
    		else
    			j++;
    	}
    	r2f[i]=j;
    	f2r[j]=i;
    }
    for (int i=0;i<11;i++)
    		cout   << i << "\t" << r2f[i] << "\t" << getSuffix(sRev.substr(0,sRev.length()-1),revSa[i]) << "\t\t\t" << getSuffix(s.substr(0,s.length()-1),sa[r2f[i]]) << endl;


    for (int i=0;i<11;i++)
        		cout   << i << "\t" << f2r[i] << "\t" << getSuffix(s.substr(0,s.length()-1),sa[i]) << "\t\t\t" << getSuffix(sRev.substr(0,sRev.length()-1),revSa[f2r[i]]) << endl;

	fclose(revSaFile);
	fclose(saFile);

	saveCorrespondenceArrayToDisk("/work/phd/progettoRicerca/dati/artificial/shortString.r2f",r2f,n);
	saveCorrespondenceArrayToDisk("/work/phd/progettoRicerca/dati/artificial/shortString.f2r",f2r,n);


	loadCorrespondenceArrayFromDisk("/work/phd/progettoRicerca/dati/artificial/shortString.r2f",r2f,n);
	loadCorrespondenceArrayFromDisk("/work/phd/progettoRicerca/dati/artificial/shortString.f2r",f2r,n);

	for (int i=0;i<11;i++)
	    		cout   << i << "\t" << r2f[i] << "\t" << getSuffix(sRev.substr(0,sRev.length()-1),revSa[i]) << "\t\t\t" << getSuffix(s.substr(0,s.length()-1),sa[r2f[i]]) << endl;


	for (int i=0;i<11;i++)
	        		cout   << i << "\t" << f2r[i] << "\t" << getSuffix(s.substr(0,s.length()-1),sa[i]) << "\t\t\t" << getSuffix(sRev.substr(0,sRev.length()-1),revSa[f2r[i]]) << endl;

}


*/


void testBuildSACorrespondence(void){
	/*
	buildSACorrespondence(63025520,
			"/work/phd/progettoRicerca/dati/1000genomes/indexes/hs37d5_chr20_rev.seq.sa",
			"/work/phd/progettoRicerca/dati/1000genomes/indexes/hs37d5_chr20.seq.sa",
			"/work/phd/progettoRicerca/dati/1000genomes/indexes/hs37d5_chr20.r2f",
			"/work/phd/progettoRicerca/dati/1000genomes/indexes/hs37d5_chr20.f2r");*/

	buildSACorrespondence_2(135006516,
				"/work/phd/progettoRicerca/dati/1000genomes/indexes/hs37d5_chr11_rev.seq.sa",
				"/work/phd/progettoRicerca/dati/1000genomes/indexes/hs37d5_chr11.seq.sa",
				"/work/phd/progettoRicerca/dati/1000genomes/indexes/hs37d5_chr11.r2f",
				"/work/phd/progettoRicerca/dati/1000genomes/indexes/hs37d5_chr11.f2r");

	/*
	buildSACorrespondence(11,
				"/work/phd/progettoRicerca/dati/artificial/revShortString.seq.sa",
				"/work/phd/progettoRicerca/dati/artificial/shortString.seq.sa",
				"/work/phd/progettoRicerca/dati/artificial/shortString.r2f",
				"/work/phd/progettoRicerca/dati/artificial/shortString.f2r");
	*/
}


/*
void testBitset(){

	/*
	boost::dynamic_bitset<>  binaryDigits(6,10);
	for (boost::dynamic_bitset<>::size_type i = 0; i < binaryDigits.size(); ++i)
			std::cout << binaryDigits[i];
		cout << "\n";
	cout << binaryDigits << "\n";
	cout << binaryDigits.size() << "\n";


	boost::dynamic_bitset<> x(30); // all 0's by default
	x[0] = 1;
	x[1] = 1;
	x[20] = 1;
	x[25] = 1;

	cout << x << "\n";

	boost::dynamic_bitset<> c=BitSetRLEEncoder::compress(x);
	cout << c << "\n";

	boost::dynamic_bitset<> d=BitSetRLEEncoder::decompress(c);
	cout << d << "\n";


}
*/

void testStream(){
	 std::ifstream is ("/work/phd/progettoRicerca/dati/1000genomes/indexes/NA06989_20_1000000.elz", std::ifstream::binary);
	 if (is) {
	    // get length of file:
		 is.seekg (0, is.end);
		int length = is.tellg();
		is.seekg (0, is.beg);
		length=is.tellg();
		is.seekg (10000, is.beg);
		is.seekg (10000);
	    length = is.tellg();


	    is.close();

	  }


}


void testBitWriter(){
	FileBitWriter w("/tmp/prova.bw");
	w.open();

	w.write(26,2);
	w.write(26,2);
	w.write(4,5);
	w.write(4,5);


	w.close();

	FileBitReader r;
	r.open(("/tmp/prova.bw"));

	int n=r.read(26);
	n=r.read(26);
	n=r.read(4);
	n=r.read(4);

	r.close();

}


//Con i numeri unsigned lo shift right è di tipo zero-fill,
//quindi il codice seguente funziona correttamente
void intToLittleEndian(uint32_t n, uint8_t *bs){
    bs[0] = (uint8_t)n;
    bs[1] = (uint8_t)(n >>  8);
    bs[2] = (uint8_t)(n >> 16);
    bs[3] = (uint8_t)(n >> 24);
    bs[4] = 0;
    bs[5] = 0;
    bs[6] = 0;
    bs[7] = 0;
}


void intToBigEndian(uint32_t n, uint8_t *bs){
	bs[0] = (uint8_t)(n >> 24);
	bs[1] = (uint8_t)(n >> 16);
	bs[2] = (uint8_t)(n >>  8);
	bs[3] = (uint8_t)n;
	bs[4]=0;
	bs[5]=0;
	bs[6]=0;
	bs[7]=0;
}


void testEncryptedBitWriter(){
	Database database("/work/phd/progettoRicerca/dati/1000genomes/egcdb");
	database.login(1,"/work/phd/progettoRicerca/dati/1000genomes/egcdb/security/user_1.pvt");
	Portfolio *portfolio=database.getPortfolio();



	EncryptionContext *eContext1=new EncryptionContext(portfolio->getIndividualKey(1)->getClearValue(),65536);
	EncryptionContext *eContext2=new EncryptionContext(portfolio->getIndividualKey(1)->getClearValue(),65536);
	EncryptedFileBitWriter w("/tmp/prova.bw",eContext1);

	w.open();
	for (int i=0;i<100;i++)
		w.write(32,i+1);
	w.write(3,4);

	w.setEncryptionContext(eContext2);
	w.write(3,5);
	for (int i=100;i<200;i++)
		w.write(32,i+1);

	w.setEncryptionContext(eContext1);
	for (int i=200;i<300;i++)
		w.write(32,i+1);

	w.close();


	EncryptionContext *rContext1=new EncryptionContext(portfolio->getIndividualKey(1)->getClearValue(),65536);
	EncryptionContext *rContext2=new EncryptionContext(portfolio->getIndividualKey(1)->getClearValue(),65536);

	int num;
	EncryptedFileBitReader r;
	r.open("/tmp/prova.bw",rContext1);
	for (int i=0;i<100;i++){
		num=r.read(32);
		cout << num << endl;
	}

	num=r.read(3);
	cout << num << endl;
	r.setEncryptionContext(rContext2);
	num=r.read(3);
	cout << num << endl;
	for (int i=100;i<200;i++){
		num=r.read(32);
		cout << num << endl;
	}

	r.setEncryptionContext(rContext1);
	for (int i=200;i<300;i++){
		num=r.read(32);
		cout << num << endl;
	}

	r.close();

}


void build(string databaseRoot,string multiFastaFilePath,string indexFileName,int referenceId){
	/*
	Database database("/work/phd/progettoRicerca/dati/1000genomes/egcdb");
	//Build
	database.buildIndex("20_1MB.egc","/work/phd/progettoRicerca/dati/1000genomes/sequences1MB",20,128);
	//Open an existing index
	bool result=database.verifyIndex("20_1MB.egc","/work/phd/progettoRicerca/dati/1000genomes/sequences1MB",20);
	cout <<  (result?"Index OK":"ERROR: the index doesn't correspond to the original sequences");
 	//LZIndex *index=database.openIndex("20_1MB.egc");
 	 *
 	 * */
	Database database(databaseRoot);

	//login
	database.login(3,databaseRoot+"/security/user_3.pvt");

	std::chrono::time_point<std::chrono::system_clock> startTime,endTime;
	startTime=std::chrono::system_clock::now();

	//Build
		//database.buildIndex("20_5MB.egc",databaseRoot+"/work/phd/progettoRicerca/dati/1000genomes/sequences5MB",20,128);
	database.buildIndexFromMultiFASTA(indexFileName, multiFastaFilePath, referenceId, 128);
	endTime=std::chrono::system_clock::now();
	double buildTime=std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()/double(1000);
	cout <<  "Build time: " << buildTime << " ms" << endl;
	//Open an existing index
	bool result=database.verifyIndexFromMultiFASTA(indexFileName,multiFastaFilePath,referenceId);
	cout <<  (result?"Index OK":"ERROR: the index doesn't correspond to the original sequences");
	//LZIndex *index=database.openIndex("20_1MB.egc");

}






void testBuild(string databaseRoot,string sequencesDirectory,string indexFileName,int referenceId){
	/*
	Database database("/work/phd/progettoRicerca/dati/1000genomes/egcdb");
	//Build
	database.buildIndex("20_1MB.egc","/work/phd/progettoRicerca/dati/1000genomes/sequences1MB",20,128);
	//Open an existing index
	bool result=database.verifyIndex("20_1MB.egc","/work/phd/progettoRicerca/dati/1000genomes/sequences1MB",20);
	cout <<  (result?"Index OK":"ERROR: the index doesn't correspond to the original sequences");
 	//LZIndex *index=database.openIndex("20_1MB.egc");
 	 *
 	 * */
	Database database(databaseRoot);

	//login
	database.login(3,databaseRoot+"/security/user_3.pvt");

	std::chrono::time_point<std::chrono::system_clock> startTime,endTime;
	startTime=std::chrono::system_clock::now();

	//Build
		//database.buildIndex("20_5MB.egc",databaseRoot+"/work/phd/progettoRicerca/dati/1000genomes/sequences5MB",20,128);
	database.buildIndexFromDirectory(indexFileName,sequencesDirectory,referenceId,128);
	endTime=std::chrono::system_clock::now();
	double buildTime=std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()/double(1000);
	cout <<  "Build time: " << buildTime << " ms" << endl;
	//Open an existing index
	bool result=database.verifyIndexFromDirectory(indexFileName, sequencesDirectory, referenceId);
	cout <<  (result?"Index OK":"ERROR: the index doesn't correspond to the original sequences");
	//LZIndex *index=database.openIndex("20_1MB.egc");

}
/*
void testMultipleBTree(){

   BPlusTree t(3,1000);
   t.setNumberOfIndividuals(4);
   t.setMaximumNumberOfFactorsPerIndividual(11);


   t.insert(10,Value(1,0));
   t.insert(10,Value(2,0));
   t.insert(20,Value(1,1));
   t.insert(5,Value(1,2));
   t.insert(6,Value(1,3));
   t.insert(12,Value(8,4));
   t.insert(30,Value(3,5));
   t.insert(30,Value(3,4));
   t.insert(30,Value(2,5));
   t.insert(3,Value(1,6));
   t.insert(1,Value(1,10));

   cout << "Traversal of the constucted tree is ";
   t.traverse();
   cout<<endl;

   t.dumpTree();

   FileBitWriter bw("/tmp/prova.tree");
   bw.open();
   uint64_t dirOffset;
   t.save(bw,&dirOffset);
   bw.close();

   BPlusTree t2(3,1000);
   t2.setNumberOfIndividuals(4);
   t2.setMaximumNumberOfFactorsPerIndividual(11);
   t2.setMaximumIndividualId(8);
   FileBitReader br("/tmp/prova.tree");
   br.open();
   t2.load(&br,dirOffset);
   t2.loadAllNodesInMemory();
   br.close();

   t2.dumpTree();

   map<int,vector<int>> results=t2.rangeQuery(3,25);
   for (auto it=results.begin();it!=results.end();++it){
	   cout << "Individual " << it->first << endl;
	   for (int i=0;i<it->second.size();i++)
		   cout << it->second[i] << " ";
	   cout << endl;
   }




  }
  */

char *toCString(string s){
	int len=s.length();
	char *cString=new char[len+1];
	len = s.copy(cString,len);
	cString[len] = '\0';
	return cString;
}


inline bool backwardStepSuccessful(FM_INDEX *referenceIndex, bwi_out *s,
		unordered_map<char, unordered_map<int,int>> *occsCache,
		char c,int sp,int ep,int *trySp,int *tryEp){
	int o;
	//c is a compact alphabet code
	if ((*occsCache).count(c)>0 && (*occsCache)[c].count(sp-1)>0)
		o=(*occsCache)[c][sp-1];
	else{
		o=occ(referenceIndex,s, EOF_shift(s,sp - 1),c);
		if ((*occsCache).count(c)==0)
			(*occsCache).insert(CharMapPair(c,unordered_map<int,int>()));
		(*occsCache)[c].insert(IntIntPair(sp-1,o));
	}
	*trySp = s->bwt_occ[c]+o;

	//tryEp=s->bwt_occ[c]+occ(s, EOF_shift(ep), c)-1;
	if ((*occsCache).count(c)>0 && (*occsCache)[c].count(ep)>0)
		o=(*occsCache)[c][ep];
	else{
		o=occ(referenceIndex,s, EOF_shift(s,ep),c);
		if ((*occsCache).count(c)==0)
			(*occsCache).insert(CharMapPair(c,unordered_map<int,int>()));
		(*occsCache)[c].insert(IntIntPair(ep,o));
	}
	*tryEp = s->bwt_occ[c]+o-1;
	return *tryEp>=*trySp;
}


inline bool searchPatternInReferenceIndex(FM_INDEX *referenceIndex, bwi_out *s,
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



void createSdslSuffixArray(){
	//int n=63025520;
	int n=135006516;
	//Create

	string s="/work/phd/progettoRicerca/dati/1000genomes/egcdb/references/hs37d5_chr11.seq.sa";
	string t="/work/phd/progettoRicerca/dati/1000genomes/egcdb/references/hs37d5_chr11.sa";
	//forwardSA = new int[n];
	FILE *forwardSAFile = fopen(s.c_str(), "a+b");
	int ilog2n_reference=int_log2(n);
	int_vector<> iv(n,0,ilog2n_reference);
	for (int i=0;i<n;i++){
		int suf=get_suffix(forwardSAFile,n,i);
		iv[i]=suf;
	}
	store_to_file(iv,t);
	fclose(forwardSAFile);

/*
	//Verify
	string s="/work/phd/progettoRicerca/dati/1000genomes/egcdb/references/hs37d5_chr11.seq.sa";
	string t="/work/phd/progettoRicerca/dati/1000genomes/egcdb/references/hs37d5_chr11.sa";
	//forwardSA = new int[n];
	FILE *forwardSAFile = fopen(s.c_str(), "a+b");
	int ilog2n_reference=int_log2(n);
	int_vector<> iv(n,0,ilog2n_reference);
	load_from_file(iv,t);
	for (int i=0;i<n;i++){
		int suf=get_suffix(forwardSAFile,n,i);
		if (iv[i]!=suf)
			break;
	}
	fclose(forwardSAFile);
*/
}


void testSdsl(){
	//string pattern="GGGTCAAATGGTATTTCTGGTTCTACATCCTTGAGGAATCGCCACACTGTCTTCCACAATGGTTGAACTAATTTACATTC";
	string pattern="GGGTCAAATG";
	string index_file = "/tmp/chr20.fm";
	double queryMeanTime=0;
	int iter=1;

	csa_wt<wt_huff<rrr_vector<127> >,1 , 1024> fm_index;


	construct(fm_index, "/work/phd/progettoRicerca/dati/1000genomes/reference/hs37d5_chr20.seq", 1); // generate index
	store_to_file(fm_index, index_file); // save it

	load_from_file(fm_index, index_file);


	for (int i=0;i<iter;i++){
		std::chrono::time_point<std::chrono::system_clock> startTime,endTime;
		startTime=std::chrono::system_clock::now();
		size_t occs = sdsl::count(fm_index, pattern.begin(), pattern.end());
		if (occs > 0) {
			auto locations = locate(fm_index, pattern.begin(), pattern.end());
			sort(locations.begin(), locations.end());
		}
		endTime=std::chrono::system_clock::now();
		double queryTime=std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()/double(1000);
		queryMeanTime+=queryTime;

	}
	cout << "Query mean time: " << queryMeanTime/iter << "ms" << endl;


	unordered_map<char, unordered_map<int,int>> forwardOccs;
	bwi_out *s_for=new bwi_out();
	string forwardReferenceIndexFileName="/work/phd/progettoRicerca/dati/1000genomes/egcdb/references/hs37d5_chr20.bwi";
	char *cString=toCString(forwardReferenceIndexFileName);
	Type_mem_ops = IN_MEM;
	FM_INDEX *forwardReferenceIndex=my_open_file(cString);
	forwardReferenceIndex->Use_bwi_cache=1;
	forwardReferenceIndex->Cache_percentage=1;

	init_bwi_cache(forwardReferenceIndex);
	read_basic_prologue(forwardReferenceIndex,s_for);
	load_all_superbuckets(forwardReferenceIndex,s_for);
	load_all_buckets(forwardReferenceIndex,s_for);
	queryMeanTime=0;

	int sp,ep;
	for (int i=0;i<iter;i++){
			std::chrono::time_point<std::chrono::system_clock> startTime,endTime;
			startTime=std::chrono::system_clock::now();
			searchPatternInReferenceIndex(forwardReferenceIndex, s_for,
					&forwardOccs,pattern,&sp,&ep);
			endTime=std::chrono::system_clock::now();
					double queryTime=std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()/double(1000);
					queryMeanTime+=queryTime;
	}
	cout << "Query mean time: " << queryMeanTime/iter << "ms" << endl;
}



int test_accumulator()
{
  accumulator_set<int, features<tag::count,tag::mean,tag::variance>> acc;
  acc(4);
  acc(-6);
  acc(9);

  double n=boost::accumulators::count(acc);
  std::cout << n << '\n';
  std::cout << boost::accumulators::mean(acc) << '\n';
  double v=(boost::accumulators::variance(acc)*n)/(n-1);
  std::cout << sqrt(v) << '\n';
}



int main(int argc, char* argv[]) {
	//testSdsl();

	cout << "argc = " << argc << endl;
	for(int i = 0; i < argc; i++)
	      cout << "argv[" << i << "] = " << argv[i] << endl;

	//test_accumulator();
	//testBinarySearch();
	//testBTree();
	//testStream();
	//testFactorization(std::string(argv[1]));
	//testBitset();
	//testBuildSACorrespondence();
	//testShortStringFactorization();
	//testSearch();
	//testBitWriter();
	//testEncryptedBitWriter();

	//testMultipleBTree();
	//testSdsl();
	//createSdslSuffixArray();

	string command=argv[1];
	if (command=="init"){
		if (argc < 3){
			cout << "The following parameters are required by the init command:" << endl;
			cout << "1) Database root directory full path" << endl;
			exit(1);
		}
		string databaseRoot=argv[2];
		string systemKeyFile;
		if (argc==4)
			systemKeyFile=argv[3];
		Database::initialize(databaseRoot, systemKeyFile);
	} else if (command=="addref"){ //add a reference sequence
		if (argc < 5){
			cout << "The following parameters are required by the addref command:" << endl;
			cout << "\t1) Database root directory full path" << endl;
			cout << "\t2) Reference sequence numeric identifier (it could be the chromosome number, for example)"<< endl;
			cout << "\t3) Reference sequence FASTA file full path" << endl;
			exit(1);
		}
		string databaseRoot=argv[2];
		int id=stoi(argv[3]);
		string refSeqPath=argv[4];


		std::chrono::time_point<std::chrono::system_clock> startTime,endTime;
		startTime=std::chrono::system_clock::now();
		Database database(databaseRoot);
		database.addReferenceSequence(id, refSeqPath);
		endTime=std::chrono::system_clock::now();
		double openingTime=std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		cout << "Elapsed: " << openingTime/1000 << "s" << endl;

	} else if (command=="addind"){ //add an individual
		if (argc < 4){
			cout << "The following parameters are required by the addind command:" << endl;
			cout << "\t1) Database root directory full path" << endl;
			cout << "\t2) Individual code" << endl;
			cout << "Optional parameters:" << endl;
			cout << "\t3) Individual Salsa20 key file path (if not given,a new key will be generated)" << endl;
			exit(1);
		}
		string databaseRoot=argv[2];
		string code=argv[3];
		string individualKeyFilePath;
		Database database(databaseRoot);
		if (argc==5){
			individualKeyFilePath=argv[4];
			database.addIndividual(code, &individualKeyFilePath);
		} else
			database.addIndividual(code, NULL);

	} else if (command=="adduser"){
		if (argc < 4){
			cout << "The following parameters are required by the adduser command:" << endl;
			cout << "\t1) Database root directory full path" << endl;
			cout << "\t2) Username" << endl;
			cout << "Optional parameters:" << endl;

			cout << "\t3) userId (user internal identifier)" << endl;
			cout << "\t4) user's private key, in PEM format (mandatory if userId was given)" << endl;
			cout << "\t5) user's public key, in PEM format (mandatory if userId was given)" << endl;
			exit(1);
		}
		string databaseRoot=argv[2];
		string username=argv[3];
		string privateKeyFilePath;
		string publicKeyFilePath;
		int userId;
		Database database(databaseRoot);
		if (argc==7){
			userId=stoi(argv[4]);
			privateKeyFilePath=argv[5];
			publicKeyFilePath=argv[6];
			database.addUser(username,&userId, &privateKeyFilePath,&publicKeyFilePath);
		} else
			database.addUser(username,NULL,NULL,NULL);
	}
	else if (command=="buildindex"){
		if (argc < 6){
			cout << "The following parameters are needed for the build command:" << endl;
			cout << "1) database root directory" << endl;
			//cout << "2) sequences directory" << endl;
			cout << "2) MultiFASTA file containing the sequences to be indexed" << endl;
			cout << "3) file name of the index to build" << endl;
			cout << "4) reference identifier" << endl;
			exit(1);
		}
		string databaseRoot=argv[2];
		//string sequencesDirectory=argv[3];
		string multiFastaFileName=argv[3];
		string indexFileName=argv[4];
		int referenceId=stoi(argv[5]);
		//testBuild(databaseRoot,sequencesDirectory,indexFileName,referenceId);
		build(databaseRoot,multiFastaFileName,indexFileName,referenceId);
	} else if (command=="locate"){
		if (argc < 3){
			cout << "The following parameters are needed for the locate command:" << endl;
			cout << "1) XML file path" << endl;
			exit(1);
		}
		string xmlFilePath=argv[2];
		testLocate(xmlFilePath);
	}  else if (command=="dump"){
		if (argc < 4){
			cout << "The following parameters are needed for the dump command:" << endl;
			cout << "1) XML file path" << endl;
			cout << "2) Dump File name" << endl;
			exit(1);
		}
		string xmlFilePath=argv[2];
		string dumpFileName=argv[3];
		dumpIndex(xmlFilePath,dumpFileName);
	}
	//testLocate(std::string(argv[1]));




}
