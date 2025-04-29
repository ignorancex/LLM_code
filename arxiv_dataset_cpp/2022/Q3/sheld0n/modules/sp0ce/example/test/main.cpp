// std includes
#include <memory> // shared_ptr
#include <cmath>
#include <iostream>
// thirdparties includes
#include <Eigen/Dense>
// lib includes
#include "sp0ce/bin.h"

using TypeScalar = double;
// linear Algebra
template<int Size>
using tVector = Eigen::Matrix<TypeScalar, Size, 1>;
// space
constexpr unsigned int Dim = 2;
using tSpaceVector = Eigen::Matrix<TypeScalar, Dim, 1>;
// view
template<typename ...Args>
using tView = Eigen::Map<Args...>;

const TypeScalar Step = 1.0;

int main () {
	std::vector<tSpaceVector> positions(128);
	for(unsigned int index = 0; index < positions.size(); ++index) {
		positions[index] = tSpaceVector::Random() * 16.0 * Step;
	}
	// bin
	sp0ce::Bin<Dim> bin(Step);
	for(unsigned int index = 0; index < positions.size(); ++index) {
		bin.addIndex(index, positions[index].data());
	}
	std::array<int, Dim> ijk = {0, 0};
	std::vector<std::array<int, Dim>> siblings = bin.ijkToSiblingIjk(ijk.data());
	// binTree
	sp0ce::BinTree<Dim> binTree(Step);
	for(unsigned int index = 0; index < positions.size(); ++index) {
		binTree.addIndex(index, positions[index].data());
	}
	// create equation and solution
	std::cout << "\n";
	std::cout << "binTree.data.size() : " << binTree.data.size() << "\n";
	std::cout << "binTree.data.front().step : " << binTree.data.front().step << "\n";
	std::cout << "binTree.data.back().step : " << binTree.data.back().step << "\n";
	std::cout << "\n";
	std::cout << std::endl;
}
