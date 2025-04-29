#include <iostream>
#include <stack>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include "bifrost/CompactedDBG.hpp"
#include "bifrost/ColoredCDBG.hpp"
#include "graph.h"
#include "bridging.h"

using namespace std;

string output, tmp;

void print_help()
{
	printf("\n");
	printf("Usage: <output-bridge-sequence> <tmpfile>\n");
	printf("\n");
}

bridging findbridge;

int main(int argc, char **argv)
{
	if(argc < 3)
	{
		printf("Parameters Error\n");
		print_help();
		return 0;
	}

	output = argv[1];
	tmp = argv[2];

	findbridge.read(tmp + "/graph", tmp + "/query");
	findbridge.findPaths();
	findbridge.write(output);

	rmdir(tmp.c_str());
	return 0;
}