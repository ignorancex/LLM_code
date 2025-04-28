/*
 * Database.cpp
 *
 *  Created on: 21/nov/2014
 */

#include "Database.h"
#include "Utils.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <seqan/sequence.h>
#include <seqan/seq_io.h>
#include <seqan/modifier.h>

#include "LZIndex.h"
#include <sdsl/int_vector.hpp>


extern "C" {
	int createFMIndex(char *infile_name, char *outfile_name,int bs_lev1, int bs_lev2, float mc_freq,
			bwi_input *bwi,int save_sa_to_disk);
}

namespace std {


Database::Database(string rootDirectory) {
	this->rootDirectory=rootDirectory;
	this->loggedUserId=NULL;
	this->loggedUser=NULL;

	//Check if the database root directory exists
	if (!boost::filesystem::is_directory(rootDirectory))
		throw runtime_error("The root directory " + rootDirectory + " does not exist");

	//Check if the following sub-directories exist:
	// - references
	// - security
	// - indexes
	// and, if required, create them.
	string subdir=rootDirectory + "/references";
	if (!boost::filesystem::is_directory(subdir))
		if (!boost::filesystem::create_directory(subdir))
			throw runtime_error("Error while creating directory "+subdir);
	subdir=rootDirectory + "/security";
	if (!boost::filesystem::is_directory(subdir))
		if (!boost::filesystem::create_directory(subdir))
			throw runtime_error("Error while creating directory "+subdir);

	string systemKeyPath=rootDirectory+"/security/system.key";
	//Checks if the system key exists and creates it if necessary
	if (!boost::filesystem::is_regular_file(systemKeyPath))
		Utils::generateSalsa20KeyFile(systemKeyPath);

	subdir=rootDirectory + "/indexes";
	if (!boost::filesystem::is_directory(subdir))
		if (!boost::filesystem::create_directory(subdir))
			throw runtime_error("Error while creating directory "+subdir);

	this->catalog= new Catalog(rootDirectory);

	//Test
	//string s="MANU";
	//Individual *ind=addIndividual(s, NULL);
}


void Database::initialize(string rootDirectory,string systemKeyFile){
	//Check if database root directory exists and is empty
	//Ensures that the database root directory exists
	if (!boost::filesystem::is_directory(rootDirectory)){
		cout << "The root directory " + rootDirectory + " does not exist";
		boost::filesystem::create_directories(rootDirectory);
		cout << "-->root directory has been successfully created";
	}
	//Ensures that root directory is empty
	if(!boost::filesystem::is_empty(rootDirectory))
		throw runtime_error("Can't initialize a new database: the root directory " + rootDirectory + " is not empty");

	//Creates the references subdirectory
	string subdir=rootDirectory + "/references";
	if (!boost::filesystem::create_directory(subdir))
		throw runtime_error("Error while creating directory "+subdir);

	//Creates the security subdirectory
	subdir=rootDirectory + "/security";
	if (!boost::filesystem::create_directory(subdir))
		throw runtime_error("Error while creating directory "+subdir);
	//Creates the Salsa20 system key
	//TODO: store the system key in a more secure way
	string systemKeyDatabasePath=rootDirectory+"/security/system.key";
	if (boost::filesystem::is_regular_file(systemKeyFile))
		boost::filesystem::copy_file(systemKeyFile,systemKeyDatabasePath,boost::filesystem::copy_option::overwrite_if_exists);
	else
		Utils::generateSalsa20KeyFile(systemKeyDatabasePath);

	//Creates the indexes subdirectory
	subdir=rootDirectory + "/indexes";
	if (!boost::filesystem::create_directory(subdir))
		throw runtime_error("Error while creating directory "+subdir);

	//Creates an empty XML database catalog
	Catalog catalog(rootDirectory);

}



Database::~Database() {
	// TODO Auto-generated destructor stub
}


//TODO: replace userId with username
void Database::login(int64_t userId,string privateKeyFilePath){

	//set the current user
	loggedUserId=new uint64_t;
	*loggedUserId=userId;

	loggedUser=getUser(userId);
	if (loggedUser==NULL)
		throw new runtime_error("The user "+ to_string(userId)+" does not exist!");

	loggedUserPrivateKeyFilePath=privateKeyFilePath;


	//TODO: implement login: decrypt the user object login value field and verify that the value
	// decrypted value is exactly ERINDEX_LOGIN_VALUE

	portfolio=getPortfolio(*loggedUserId,loggedUserPrivateKeyFilePath);
}

/*
void Database::buildIndex(string indexFileName,string sequencesDirectory,int referenceId,int blockSize){
	LZIndex index(*this,referenceId,blockSize);
	for( auto it = individuals.begin(), end = individuals.end(); it != end; ++it){
	    Individual *individual=it->second;
	    cout << "Adding individual "<< individual->id << "(" << individual->code << "): ";
	    string sequenceFilePath=sequencesDirectory + "/" + individual->code + "_" + to_string(referenceId) +".fa";
	    Factorization *f=index.addToIndex(individual,sequenceFilePath,1);
	    cout << "\t" << f->factors.size() << " factors" << endl;
	}
	index.save(rootDirectory + "/indexes/"+ indexFileName);
	index.close();
}
*/

void Database::buildIndexFromDirectory(string indexFileName,string sequencesDirectory,int referenceId,int blockSize){
	LZIndex index(*this,referenceId,blockSize);
	Individual **individs=new Individual*[catalog->individuals.size()];
	string *fastaFileNames=new string[catalog->individuals.size()];
	int i=0;
	for( auto it = catalog->individuals.begin(), end = catalog->individuals.end(); it != end; ++it){
	    Individual *individual=it->second;
	    individs[i]=individual;
	    string sequenceFilePath=sequencesDirectory + "/" + individual->code + "_" + to_string(referenceId) +".fa";
	    fastaFileNames[i]=sequenceFilePath;
	    i++;
	}
	index.buildIndexFromDirectory(catalog->individuals.size(),individs,fastaFileNames);
	index.save(rootDirectory + "/indexes/"+ indexFileName);
	index.close();
}

void Database::buildIndexFromMultiFASTA(string indexFileName, string multiFastaFilePath,
		int referenceId, int blockSize) {
	if (!boost::filesystem::is_regular_file(multiFastaFilePath))
			throw new runtime_error("The given reference sequence file is not valid or not accessible!");

		cout << "Reading sequences from the given multiFASTA file...";
		seqan::SeqFileIn seqFileIn(multiFastaFilePath.c_str());
		seqan::StringSet<seqan::CharString> codes;
		seqan::StringSet<seqan::IupacString> seqs;
		seqan::readRecords(codes, seqs, seqFileIn);

		int numSeqs=length(seqs);
		int totalLength=0;
		if (numSeqs >0){
			LZIndex index(*this,referenceId,blockSize);
			Individual **individs=new Individual*[numSeqs];

			int i=0;
			for (i=0;i<numSeqs;i++){
				string sequenceCode(seqan::toCString(codes[i]));
				totalLength+=length(seqs[i]);
				//if the individual does not exist in the database catalog, adds it
				Individual *individ=catalog->getIndividual(sequenceCode);
				if (individ==NULL)
					individ=addIndividual(sequenceCode,NULL);
				individs[i]=individ;
			}

			cout << "Building the index in memory..." <<endl;
			index.buildIndexFromMultiFASTA(numSeqs, individs, &seqs);
			cout << "Saving the index to disk..." <<endl;
			cout << "Collection size: "<< to_string(totalLength) << "bytes" << endl;
			int indexSize=index.save(rootDirectory + "/indexes/"+ indexFileName);
			cout << "Index size: "<< indexSize << " bits (" <<  ceil((double)indexSize/8)  << " bytes)" << endl;
			//cout << "Compression ratio: "<< ceil((double)indexSize/8)/textLength << endl;
			double cr=((double)ceil((double)indexSize/8)/(double)totalLength);
			cout << "Compession ratio: " << cr << endl;
			index.close();



		} else
			throw new runtime_error("The given MultiFASTA file is empty!");
}




LZIndex *Database::openIndex(string indexFileName){
	LZIndex *index=new LZIndex(*this);
	index->open(rootDirectory + "/indexes/"+ indexFileName);

	return index;
}



bool Database::verifyIndexFromMultiFASTA(string indexFileName,string multiFastaFilePath,int referenceId){
	if (!boost::filesystem::is_regular_file(multiFastaFilePath))
					throw new runtime_error("The given index file is not valid or not accessible!");

	if (!boost::filesystem::is_regular_file(multiFastaFilePath))
				throw new runtime_error("The given reference sequence file is not valid or not accessible!");

	LZIndex *index=new LZIndex(*this);
	index->open(rootDirectory + "/indexes/"+ indexFileName);


	seqan::SeqFileIn seqFileIn(multiFastaFilePath.c_str());
	seqan::StringSet<seqan::CharString> codes;
	seqan::StringSet<seqan::IupacString> seqs;
	seqan::readRecords(codes, seqs, seqFileIn);

	bool ok=true;
	vector<int64_t> ids=index->getIndividualIds();
	for (unsigned int i=0;i<ids.size();i++){
		string s=index->getSequence(ids[i]);
		Individual *individual=catalog->individuals[ids[i]];
		cout << "Verifying the individual " << i << "(" << individual->code << ")" << endl;
		int ssize=s.size();

		seqan::IupacString t=seqs[i];
		int tsize=length(t);


		//if (ssize!=tsize){
		ok=true;
		int minSize;
		if (ssize<tsize)
			minSize=ssize;
		else
			minSize=tsize;
		for (int i=0;i<minSize;i++){
			if (s[i]!=t[i]){
				cout << s[i] << "!=" << t[i] << endl;
				cout << "ERROR at " << i << endl;
				ok=false;
				break;
			}
		}
		//}
	}
	return ok;

}



bool Database::verifyIndexFromDirectory(string indexFileName,string sequencesDirectory,int referenceId){
	LZIndex *index=new LZIndex(*this);
	index->open(rootDirectory + "/indexes/"+ indexFileName);
	bool ok=true;
	vector<int64_t> ids=index->getIndividualIds();
	for (int i=0;i<ids.size();i++){

		string s=index->getSequence(ids[i]);
		Individual *individual=catalog->individuals[ids[i]];
		cout << "Verifying the individual " << i << "(" << individual->code << ")" << endl;
		string sequenceFilePath=sequencesDirectory + "/" + individual->code + "_" + to_string(referenceId) +".fa";
		string t=Utils::loadFasta(sequenceFilePath);
		if (s!=t){
			ok=false;
			int minSize;
			if (s<t)
				minSize=s.size();
			else
				minSize=t.size();
			for (int i=0;i<minSize;i++){
				if (s[i]!=t[i]){
					cout << "ERROR at " << i << endl;
					break;
				}
			}
		}
	}
	return ok;
}


/*
void Database::loadCatalog(){
	//Check if the catalog XML file exists in the root directory.
	//If it doesn't exist, create an empty catalog XML file, otherwise load it
	string catalogFilePath=rootDirectory+"/catalog.xml";

	boost::property_tree::ptree pTree;
	if (boost::filesystem::is_regular_file(catalogFilePath)){

		boost::property_tree::read_xml(catalogFilePath,pTree);

		//read parameters from the XML file
		//Using boost::property_tree
		for( auto const& individualChildNode: pTree.get_child("catalog.individuals") ) {
				boost::property_tree::ptree individualSubTree = individualChildNode.second;
				int id=individualSubTree.get<int>("id");
				string code=individualSubTree.get<string>("code");
				Individual *individual=new Individual(id,code);
				individuals[id]=individual;
		}
		for( auto const& referenceChildNode: pTree.get_child("catalog.references") ) {
				boost::property_tree::ptree referenceSubTree = referenceChildNode.second;
				ReferenceSequence seq;
				string referencesPath=rootDirectory+ "/references/";
				seq.id=referenceSubTree.get<int>("id");
				seq.forwardIndexFileName=referencesPath+referenceSubTree.get<string>("forwardIndexFileName");
				seq.reverseIndexFileName=referencesPath+referenceSubTree.get<string>("reverseIndexFileName");
				seq.correspondenceFilesName=referencesPath+referenceSubTree.get<string>("correspondenceFilesName");
				seq.forwardSuffixArrayFileName=referencesPath+referenceSubTree.get<string>("forwardSuffixArrayFileName");
				references[seq.id]=seq;
		}
	} else {

		boost::property_tree::ptree catalogElement;
		boost::property_tree::ptree referencesElement;
		boost::property_tree::ptree individualsElement;
		catalogElement.push_back(std::make_pair("references", referencesElement));
		catalogElement.push_back(std::make_pair("individuals", individualsElement));

		pTree.push_back(std::make_pair("catalog",catalogElement));

		boost::property_tree::write_xml(catalogFilePath, pTree);
	}
}
*/

Portfolio *Database::getPortfolio (){
	return portfolio;
}


Portfolio *Database::getPortfolio (int64_t userId,string privateKeyFilePath){
	RSA *privateKey = NULL;
	//string privateKeyFilePath=rootDirectory+"/security/user_"+ boost::lexical_cast<std::string>(userId)+ ".pvt";
	FILE *privateKeyFile = fopen(privateKeyFilePath.c_str(),"rb");

	//load user's private key
	if (PEM_read_RSAPrivateKey(privateKeyFile, &privateKey, NULL, NULL) != NULL  ){
		Portfolio *portfolio=new Portfolio(userId,privateKey);

		//load system key
		string filePath=rootDirectory+"/security/portfolio_"+ to_string(userId)+"_s.key";
		uint8_t *buffer=NULL;
		if (fileExists(filePath,&buffer,true)){
			Key *sk=new Key(portfolio,buffer);
			portfolio->systemKey=sk;
		}
		//load individual keys
		for (auto it=catalog->individuals.begin(); it!=catalog->individuals.end(); ++it){
			//filePath=rootDirectory+"/security/portfolio_"+ boost::lexical_cast<std::string>(userId)+"_"+boost::lexical_cast<std::string>(individuals[i]->id)+ ".key";
			filePath=rootDirectory+"/security/portfolio_"+ to_string(userId)+"_"+to_string(it->first)+ ".key";
			uint8_t *buffer=NULL;
			if (fileExists(filePath,&buffer,true)){
				IndividualKey *ik=new IndividualKey(portfolio,it->second,buffer);
				portfolio->individualKeys[it->second->id]=ik;
			}
		}
		return portfolio;
	} else{
	  fclose(privateKeyFile);
	  throw;
	}

}

void Database::savePortfolio(Portfolio *portfolio){
	RSA *publicKey = NULL;
	string publicKeyFilePath=rootDirectory+"/security/user_"+ boost::lexical_cast<std::string>(portfolio->userId)+ ".pub";
	FILE *publicKeyFile = fopen(publicKeyFilePath.c_str(),"rb");
	//load user's public key
	if (PEM_read_RSAPublicKey(publicKeyFile, &publicKey, NULL, NULL) != NULL){
		portfolio->setPublicKey(publicKey);
		portfolio->save(rootDirectory+"/security");
	} else{
		fclose(publicKeyFile);
		throw;
	}

}



inline bool Database::fileExists (const string& filePath,uint8_t **buffer,bool load) {
	ifstream f(filePath.c_str());
	if (f.good()) {
		if (load){
			f.seekg (0, f.end);
			int fileSize = f.tellg();
			*buffer=new uint8_t[fileSize];
			f.seekg (0, f.beg);
			f.read((char *)*buffer,fileSize);
		}
        f.close();
        return true;
    } else {
        return false;
    }
}



void Database::saveCorrespondenceArrayToDisk(string fileName,int *a,int n){
	FileBitWriter bw(fileName);
	bw.open();
	int psize = int_log2(n);
	for(int i=0;i<n;i++){
	    bw.write(psize,a[i]);
	    //cout  << i << " -> " << a[i] << endl;
	}
	bw.flush();
	bw.close();
}

void Database::loadCorrespondenceArrayFromDisk(string fileName,int *a,int n){
	FileBitReader br(fileName);
	br.open();
	int psize = int_log2(n);
	for(int i=0;i<n;i++)
		a[i]=br.read(psize);
	br.close();
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
    Utils::fbit_read_simple(Safile,Bit_buffer_size,Bit_buffer,rem);

  int val=Utils::fbit_read_simple(Safile, Bit_buffer_size,Bit_buffer,Pointer_size);
  free(Bit_buffer);
  free(Bit_buffer_size);
  return val;
}


void Database::buildSuffixArrayCorrespondenceFiles(int n,int *saSuffixes,int *reverse_saSuffixes,string r2fFileName,string f2rFileName){

	int *saSuffixesPerPosition=new int[n];
	for (int i=0;i<n;i++){
			saSuffixesPerPosition[i]=-1;
	}

	int s;
	for (int i=0;i<n;i++){
		s=saSuffixes[i];
		saSuffixesPerPosition[s]=i;
	}


	//Calcolo i due vettori r2f ed f2r, che consentono rispettivamente di passare da un suffisso dellla stringa reverse al corrispondente della straight e viceversa
	int *r2f=new int[n];
	int *f2r=new int[n];
	int revSa_di_i;
	int j;
	cout << "Computing R2F and F2R" << endl;
	for (int i=0;i<n;i++){
		if (i%10000000==0)
			cout << "Computing the "<< (i+1) << "-th R2F and F2R element"<< endl;
		revSa_di_i=reverse_saSuffixes[i];
		if (saSuffixesPerPosition[n-revSa_di_i-1]==-1){
			cout << "NNCUC" << endl;
			exit(1);
		}
		j=saSuffixesPerPosition[n-revSa_di_i-1]; //suffix related to the same text position
		r2f[i]=j;
		f2r[j]=i;
	}


	//Salvo i due array su disco
	saveCorrespondenceArrayToDisk(r2fFileName,r2f,n);
	saveCorrespondenceArrayToDisk(f2rFileName,f2r,n);

	free(r2f);
	free(f2r);
}








void Database::buildSuffixArrayCorrespondenceFiles(int n,string reverseSaFileName,string saFileName,string r2fFileName,string f2rFileName){

	FILE *revSaFile = fopen(reverseSaFileName.c_str(), "a+b");
	fseek(revSaFile, 0L, SEEK_SET);

	int *saSuffixes=new int[n];
	//fill_n(saSuffixes,n,-1);
	for (int i=0;i<n;i++)
		saSuffixes[i]=-1;



	cout << "Reading suffix array from disk" << endl;
	cout << "It may take a few minutes" << endl;

	FILE *saFile = fopen(saFileName.c_str(), "a+b");
	fseek(saFile, 0L, SEEK_SET);
	for (int i=0;i<n;i++){
		if (i%10000000==0)
			cout << "Reading the "<< (i+1) << "-th suffix array element"<< endl;

		int s=get_suffix(saFile,n,i);
		saSuffixes[s]=i;
	}
	fclose(saFile);

	//Calcolo i due vettori r2f ed f2r, che consentono rispettivamente di passare da un suffisso dellla stringa reverse al corrispondente della straight e viceversa
	int *r2f=new int[n];
	int *f2r=new int[n];
	int revSa_di_i;
	int j;
	cout << "Computing R2F and F2R" << endl;
	for (int i=0;i<n;i++){
		if (i%10000000==0)
			cout << "Computing the "<< (i+1) << "-th R2F and F2R element"<< endl;
		revSa_di_i=get_suffix(revSaFile,n,i);
		if (saSuffixes[n-revSa_di_i-1]==-1){
			cout << "NNCUC" << endl;
			exit(1);
		}
		j=saSuffixes[n-revSa_di_i-1]; //suffix related to the same text position
		r2f[i]=j;
		f2r[j]=i;
	}

	fclose(revSaFile);

	cout << "Saving R2F and F2R to disk" << endl;
	//Salvo i due array su disco
	saveCorrespondenceArrayToDisk(r2fFileName,r2f,n);
	saveCorrespondenceArrayToDisk(f2rFileName,f2r,n);

	free(saSuffixes);
	free(r2f);
	free(f2r);
}




ReferenceSequence *Database::addReferenceSequence(int id,string referenceSequenceFastaFilePath){
	if (catalog->references.count(id)>0)
		throw new runtime_error("Reference sequence with id " +  to_string(id) + "already exists in the database!");

	if (!boost::filesystem::is_regular_file(referenceSequenceFastaFilePath))
		throw new runtime_error("The given reference sequence file is not valid or not accessible!");

	cout << "Reading the reference sequence from FASTA file... "<< endl;
	seqan::SeqFileIn seqFileIn(referenceSequenceFastaFilePath.c_str());
	seqan::StringSet<seqan::CharString> ids;
	seqan::StringSet<seqan::IupacString> seqs;
	seqan::readRecords(ids, seqs, seqFileIn);
	vector<string> sequenceFileNames;
	vector<string> sequenceDescriptions;
	int size=length(seqs);
	if (size>0){

		//Gets the sequence length
		int n=seqan::length(seqs[0]);

		boost::filesystem::path p(referenceSequenceFastaFilePath);
		string fileName=p.stem().string();
		string indexFilePath=this->rootDirectory + "/references/" + fileName + ".bwi";

		//Creates the FM-index and the suffix array of the sequence and saves them to disk
		//Creates a temporary file containing only the reference sequence characters
		string sequenceDescription(seqan::toCString(ids[0]));
		sequenceDescriptions.push_back(sequenceDescription);
		char *tfn=std::tmpnam(nullptr);


		std::ofstream temporaryFileStream(tfn);
		temporaryFileStream << seqs[0];
		temporaryFileStream.flush();
		temporaryFileStream.close();

		string temporaryFileName=tfn;
		cout << "Creating the FM-index of the forward sequence... " << endl;
		bwi_input *bwi = new bwi_input();
		createFMIndex(Utils::toCString(temporaryFileName), Utils::toCString(indexFilePath),
				8192, 1024, 0.02, bwi,  1);

		cout << "Writing forward suffix array to disk... ";
		n=bwi->text_size;
		int ilog2n_reference=int_log2(n);
		sdsl::int_vector<> *forwardSA = new sdsl::int_vector<>(n,0,ilog2n_reference);
		for (int i=0;i<n;i++){
			(*forwardSA)[i]=bwi->sa[i];
			//cout << "sa[" << i << "]=" << bwi->sa[i] << endl;
		}

		string forwardSuffixArrayFilePath=this->rootDirectory + "/references/" + fileName + ".sa";
		store_to_file(*forwardSA,forwardSuffixArrayFilePath);

		//Creates the FM-index and the suffix array of the reversed sequence and saves them to disk
		tfn=std::tmpnam(nullptr);
		std::ofstream temporaryFileStream2(tfn);
		seqan::IupacString revSeq=seqs[0];
		//reverseComplement(revSeq);
		reverse(revSeq);
		temporaryFileStream2 << revSeq;
		temporaryFileStream2.flush();
		temporaryFileStream2.close();

		string revIndexFilePath=this->rootDirectory + "/references/" + fileName + "_rev.bwi";
		temporaryFileName=tfn;
		cout << "Creating the FM-index of the reversed sequence... "  << endl;
		bwi_input *bwi_rev = new bwi_input();
		createFMIndex(Utils::toCString(temporaryFileName), Utils::toCString(revIndexFilePath),
						8192, 1024, 0.02,bwi_rev,1);


		sdsl::int_vector<> *reverseSA = new sdsl::int_vector<>(n,0,ilog2n_reference);
		for (int i=0;i<n;i++){
			(*reverseSA)[i]=bwi_rev->sa[i];
			//cout << "rev_sa[" << i << "]=" << bwi_rev->sa[i] << endl;
		}

		/*
		string reverseSuffixArrayFilePath=this->rootDirectory + "/references/" + fileName + "_rev.sa";
		store_to_file(*reverseSA,reverseSuffixArrayFilePath);
		*/

		cout << "Creating R2F and F2R mapping files..." << endl;
		//builds the R2F and F2R suffix array correspondence files
		//string revSaFileName=this->rootDirectory + "/references/" + fileName + "_rev.sa";
		//string saFileName=this->rootDirectory + "/references/" + fileName + ".sa";
		string r2fFileName=this->rootDirectory + "/references/" + fileName + ".r2f";
		string f2rFileName=this->rootDirectory + "/references/" + fileName + ".f2r";
		buildSuffixArrayCorrespondenceFiles(n, bwi->sa,bwi_rev->sa, r2fFileName, f2rFileName);
		//buildSuffixArrayCorrespondenceFiles(n, revSaFileName, saFileName, r2fFileName, f2rFileName);

        /*
		int *nuovo=new int[n];
		int *vecchio = new int[n];
		loadCorrespondenceArrayFromDisk(this->rootDirectory + "/references/" + fileName + ".r2f", nuovo, n);

		string path="/home/fernando/camilla/egcdb/references/" + fileName + ".r2f";
		loadCorrespondenceArrayFromDisk(path, vecchio, n);
		for (int i=0;i<n;i++){
			if (vecchio[i]!=nuovo[i]){
				cout << i << " " << vecchio[i] << " " << nuovo[i] <<endl;
			}
		}
		*/
		delete bwi;
		delete bwi_rev;

		//deletes from disk the reverse suffix array file
		//boost::filesystem::remove(revSaFileName);






		cout << "Adding the reference sequence to the catalog... ";
		//adds the reference sequence to the catalog
		ReferenceSequence *ref=new ReferenceSequence();
		ref->id=id;
		ref->forwardIndexFileName=indexFilePath;
		ref->reverseIndexFileName=revIndexFilePath;
		ref->correspondenceFilesName=this->rootDirectory + "/references/" + fileName;
		ref->forwardSuffixArrayFileName=forwardSuffixArrayFilePath;
		catalog->addReferenceSequence(ref);
		catalog->save();

	}

}



ReferenceSequence *Database::getReferenceSequence(int id){
	if (catalog->references.count(id)>0)
		return catalog->references[id];
	else
		return NULL;
}


Individual *Database::getIndividual(int id){
	if (catalog->individuals.count(id)>0)
		return catalog->individuals[id];
	else
		return NULL;
}

Individual *Database::addIndividual(string &code, string *individualKeyFilePath){
	//verify if the individual already exists and retrieve the greatest id in the catalog
	int greatestId=0;
	for (std::map<int64_t,Individual*>::iterator it=catalog->individuals.begin(); it!=catalog->individuals.end(); ++it){

		if (it->second->code == code)
			throw new runtime_error("An individual with code " +  code + "already exists in the database!");

		if (it->second->id>greatestId)
			greatestId=it->second->id;
	}
	int newId=greatestId+1;

	//creates, if needed, the Salsa 20 key for the individual
	string *ifp = new string(rootDirectory + "/security/" + code + ".key");
	if (individualKeyFilePath ==NULL){
		Utils::generateSalsa20KeyFile(*ifp);
	}
	else{
		if (!boost::filesystem::is_regular_file(*individualKeyFilePath))
				throw new runtime_error("The given key file path for the individual "+ code +  " is not valid or not accessible!");
		boost::filesystem::copy_file(*individualKeyFilePath,*ifp,boost::filesystem::copy_option::overwrite_if_exists);
	}

	//creates the new individual
	Individual *ind=new Individual(newId,code);

	//reads the value of the individual key and adds it to the user's portfolio
	std::ifstream ifile;
    ifile.open (*ifp);
    uint8_t *clearValue=new uint8_t[64];
    ifile.read((char*)clearValue, 64);
	//ifile >> clearValue;
	ifile.close();
	portfolio->addIndividualKey(ind, clearValue);

	//loads, if not yet loaded, the user's public key from the security sub-directory
	//because saving an individual key into the user's portfolio requires encoding it with the user's public key
	if (portfolio->publicKey == NULL){
		string dbPublicKeyFilePath=rootDirectory+"/security/user_"+ boost::lexical_cast<std::string>(portfolio->userId)+ ".pub";
		portfolio->loadPublicKey(dbPublicKeyFilePath);
	}
	portfolio->saveIndividualKey(rootDirectory + "/security",newId);

	//adds the individual to the catalog
	catalog->addIndividual(ind);
	catalog->save();

	//TODO: remove the clear text individual key (I don't remove it at the moment for testing purposes)
	return ind;
}






User *Database::addUser(string username,int *userId,string *privateKeyFilePath,string *publicKeyFilePath){

	User *u=new User();
	u->username=username;
	//if the parameter userid was given, check if it exists in the database
	if (userId!=NULL){
		if (catalog->users.count(*userId)>0)
			throw new runtime_error("The user with id " +  to_string(*userId) + "already exists in the database!");
		u->id=*userId;
	}
	else{
		int greatestId=0;
		for (std::map<int64_t,User*>::iterator it=catalog->users.begin(); it!=catalog->users.end(); ++it){

			if (it->second->username == username)
				throw new runtime_error("The user " +  username + "already exists in the database!");

			if (it->second->id>greatestId)
				greatestId=it->second->id;
		}
		u->id=greatestId+1;
	}




	string dbPrivateKeyFilePath=this->rootDirectory + "/security/user_"+   to_string(u->id) + ".pvt";
	string dbPublicKeyFilePath=this->rootDirectory + "/security/user_"+ to_string(u->id) + ".pub";
	if (privateKeyFilePath == NULL || publicKeyFilePath == NULL)
		Utils::generateRSAKeyPair(dbPrivateKeyFilePath, dbPublicKeyFilePath);
	else{
		boost::filesystem::copy_file(*privateKeyFilePath,dbPrivateKeyFilePath,boost::filesystem::copy_option::overwrite_if_exists);
		boost::filesystem::copy_file(*publicKeyFilePath,dbPublicKeyFilePath,boost::filesystem::copy_option::overwrite_if_exists);
	}



	//initializes the user's portfolio, storing in it the system key
	Portfolio *portfolio=new Portfolio(u->id);

	RSA *publicKey = NULL;
	FILE *publicKeyFile = fopen(dbPublicKeyFilePath.c_str(),"rb");
	if (PEM_read_RSAPublicKey(publicKeyFile, &publicKey, NULL, NULL) != NULL)
			portfolio->setPublicKey(publicKey);
	else{
			fclose(publicKeyFile);
			throw;
	}
	portfolio->setPublicKey(publicKey);


	RSA *privateKey = NULL;
	FILE *privateKeyFile = fopen(dbPrivateKeyFilePath.c_str(),"rb");
	if (PEM_read_RSAPrivateKey(privateKeyFile, &privateKey, NULL, NULL) != NULL)
			portfolio->setPrivateKey(privateKey);
	else{
			fclose(privateKeyFile);
			throw;
	}
	portfolio->setPublicKey(publicKey);



	//reads the clear value of the system key from the corresponding file and adds it to the user's portfolio
	std::ifstream ifile;
	ifile.open (rootDirectory+"/security/system.key");
	uint8_t *clearValue=new uint8_t[64];
	ifile.read((char*)clearValue, 64);
	//ifile >> clearValue;
	ifile.close();



	portfolio->addSystemKey(clearValue);
	//Key *sk=portfolio->getSystemKey();
	//sk->dumpClearValue();
	//sk->dumpEncryptedValue();
	//sk->computeClearValue();
	//sk->dumpClearValue();
	portfolio->saveSystemKey(rootDirectory + "/security");

	//TODO:encrypts the value ERINDEX_LOGIN_VALUE with the user's public key
	//and stores it into the loginValue field

	catalog->addUser(u);
	catalog->save();

}


User *Database::getUser(int userId){
	if (catalog->users.count(userId)>0)
		return catalog->users[userId];
	else
		return NULL;
}





} /* namespace std */

