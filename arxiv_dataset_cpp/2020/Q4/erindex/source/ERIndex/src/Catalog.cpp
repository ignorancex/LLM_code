/*
 * Catalog.cpp
 *
 *  Created on: Aug 19, 2019
 *      Author: fernando
 */

#include "Catalog.h"


#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>



namespace std {

Catalog::Catalog(string rootDirectory) {
	this->rootDirectory=rootDirectory;
	catalogFilePath=rootDirectory+"/catalog.xml";
	//Check if the catalog XML file exists in the root directory.
	//If it exists load it, otherwise create an empty catalog file and save it to disk
	if (boost::filesystem::is_regular_file(catalogFilePath))
		load();
	else
		initialize();


	//test
	/*
	Individual *i = new Individual(3,"CHIARA");
	addIndividual(i);

	ReferenceSequence *ref= new ReferenceSequence();
	ref->id=22;
	ref->forwardIndexFileName="hs37d5_chr22.bwi";
    ref->reverseIndexFileName="hs37d5_chr22_rev.bwi";
    ref->correspondenceFilesName="hs37d5_chr22";
    ref->forwardSuffixArrayFileName="hs37d5_chr22.sa";
    addReferenceSequence(ref);

	save();
	*/
}

Catalog::~Catalog() {

}

/**
 * Creates an empty catalog
 */
void Catalog::initialize(){
	boost::property_tree::ptree catalogElement;
	boost::property_tree::ptree referencesElement;
	boost::property_tree::ptree individualsElement;
	boost::property_tree::ptree usersElement;
	catalogElement.push_back(std::make_pair("references", referencesElement));
	catalogElement.push_back(std::make_pair("individuals", individualsElement));
	catalogElement.push_back(std::make_pair("users", usersElement));
	catalogTree.push_back(std::make_pair("catalog",catalogElement));

	save();
}


/**
 * Loads catalog from disk
 */
void Catalog::load(){
		boost::property_tree::read_xml(catalogFilePath,catalogTree);

		//read parameters from the XML file
		//Using boost::property_tree
		for( auto const& individualChildNode: catalogTree.get_child("catalog.individuals") ) {
				boost::property_tree::ptree individualSubTree = individualChildNode.second;
				int id=individualSubTree.get<int>("id");
				string code=individualSubTree.get<string>("code");
				Individual *individual=new Individual(id,code);
				individuals[id]=individual;
		}
		for( auto const& referenceChildNode: catalogTree.get_child("catalog.references") ) {
				boost::property_tree::ptree referenceSubTree = referenceChildNode.second;
				ReferenceSequence *seq=new ReferenceSequence();
				string referencesPath=rootDirectory+ "/references/";
				seq->id=referenceSubTree.get<int>("id");
				seq->forwardIndexFileName=referenceSubTree.get<string>("forwardIndexFileName");
				seq->reverseIndexFileName=referenceSubTree.get<string>("reverseIndexFileName");
				seq->correspondenceFilesName=referenceSubTree.get<string>("correspondenceFilesName");
				seq->forwardSuffixArrayFileName=referenceSubTree.get<string>("forwardSuffixArrayFileName");
				references[seq->id]=seq;
		}
		for( auto const& referenceChildNode: catalogTree.get_child("catalog.users") ) {
						boost::property_tree::ptree referenceSubTree = referenceChildNode.second;
						User *u=new User();
						u->id=referenceSubTree.get<int>("id");
						u->username=referenceSubTree.get<string>("username");
						u->loginValue=referenceSubTree.get<string>("loginValue");
						users[u->id]=u;
		}

}


/**
 * Save catalog to disk
 */
void Catalog::save(){
	if (boost::filesystem::is_regular_file(catalogFilePath))
		boost::filesystem::rename(catalogFilePath,catalogFilePath + ".old");

	boost::property_tree::xml_writer_settings<std::string> w('\t', 1);
	boost::property_tree::write_xml(catalogFilePath, catalogTree, std::locale(),w);
}

void Catalog::addReferenceSequence(ReferenceSequence *referenceSequence){
	references[referenceSequence->id]=referenceSequence;

	boost::property_tree::ptree &referencesElement=catalogTree.get_child("catalog.references");
	boost::property_tree::ptree referenceSubTree;
	referenceSubTree.put("id",referenceSequence->id);
	referenceSubTree.put("forwardIndexFileName",referenceSequence->forwardIndexFileName);
	referenceSubTree.put("reverseIndexFileName",referenceSequence->reverseIndexFileName);
	referenceSubTree.put("correspondenceFilesName",referenceSequence->correspondenceFilesName);
	referenceSubTree.put("forwardSuffixArrayFileName",referenceSequence->forwardSuffixArrayFileName);
	referencesElement.add_child("reference", referenceSubTree);


}

void Catalog::addIndividual(Individual *individual){
	individuals[individual->id]=individual;

	boost::property_tree::ptree &individualsElement=catalogTree.get_child("catalog.individuals");
	boost::property_tree::ptree individualSubTree;
	individualSubTree.put("id",individual->id);
	individualSubTree.put("code",individual->code);
	individualsElement.add_child("individual", individualSubTree);

}

Individual* Catalog::getIndividual(string code){
	for( auto it = individuals.begin(), end = individuals.end(); it != end; ++it){
		    Individual *individual=it->second;
		    if (individual->code==code)
		    	return individual;
	}
	return NULL;
}



void Catalog::addUser(User *user){
	users[user->id]=user;

	boost::property_tree::ptree &usersElement=catalogTree.get_child("catalog.users");
	boost::property_tree::ptree userSubTree;
	userSubTree.put("id",user->id);
	userSubTree.put("username",user->username);
	userSubTree.put("loginValue",user->loginValue);
	usersElement.add_child("user", userSubTree);

}



}


