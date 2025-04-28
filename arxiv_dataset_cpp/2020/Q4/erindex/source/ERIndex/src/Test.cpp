/*
 * Test.cpp
 *
 *  Created on: 05/nov/2014
 *      Author: fernando
 */

#include "Test.h"
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>


namespace std {

Test::Test(string xmlFilePath) {
	loadParameters(xmlFilePath);
}

Test::~Test() {
	// TODO Auto-generated destructor stub
}

void Test::loadParameters(string xmlFilePath)  {
	boost::property_tree::ptree pTree;
	boost::property_tree::read_xml(xmlFilePath,pTree);
	//read the test parameters from the XML file
	//Using boost::property_tree
	for( auto const& elementNode : pTree.get_child("tns:parameters") ) {
		//string s=elementNode.first;
		boost::property_tree::ptree groupTree = elementNode.second;
		TestGroup group;
		group.inputFileType = groupTree.get<string>("tns:inputFileType");
		group.databaseRoot = groupTree.get<string>("tns:databaseRoot");
		group.sequencesDirectory = groupTree.get<string>("tns:sequencesDirectory");
		group.referenceIdentifier = groupTree.get<string>("tns:referenceIdentifier");
		group.indexFileName = groupTree.get<string>("tns:indexFileName");
		group.buildIndex = groupTree.get<bool>("tns:buildIndex");
		group.naiveSearch = groupTree.get<bool>("tns:naiveSearch");
		group.loadReferenceInMemory = groupTree.get<bool>("tns:loadReferenceInMemory");
		group.loadIndexInMemory = groupTree.get<bool>("tns:loadIndexInMemory");
		group.blockSize = groupTree.get<int>("tns:blockSize");
		group.userId = groupTree.get<int>("tns:userId");
		group.userPrivateKey = groupTree.get<string>("tns:userPrivateKey");
		group.userPassword = groupTree.get<string>("tns:userPassword");
		for( auto const& groupChildNode : groupTree.get_child("") ) {
			if (groupChildNode.first=="tns:patternSets"){
				PatternSet patternSet;
				boost::property_tree::ptree patternSetSubTree = groupChildNode.second;
				patternSet.length= patternSetSubTree.get<int>("tns:length");
				for( auto const& patternSetChildNode : patternSetSubTree.get_child("") ) {
					if (patternSetChildNode.first=="tns:patterns"){
						//string s=patternSetChildNode.first;
						string p=patternSetChildNode.second.data();
						patternSet.patterns.push_back(p);
					}
				}
				group.patternSets.push_back(patternSet);
			}
		}
		testGroups.push_back(group);
	}

}


} /* namespace std */
