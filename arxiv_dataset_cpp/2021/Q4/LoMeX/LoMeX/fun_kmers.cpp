/*

	k-mer handling functionality

*/

#include "fun_kmers.hpp"
#include <iostream>
#include <fstream>
//#include <bitset>
#include <regex>
#include <string>
#include <tuple>
//#include <math.h>
//#include <boost/log/trivial.hpp>

//namespace po = boost::program_options;

using namespace std;


// Idea from Squeakr
char map_int2nuc(uint8_t nuc)
{
	switch(nuc) {
		case NUC_MAP::A: {return 'A';}
		case NUC_MAP::T: {return 'T';}
		case NUC_MAP::C: {return 'C';}
		case NUC_MAP::G: {return 'G';}
		default: {return NUC_MAP::G+1;}
		//default: {std::cout << "WEIRD THING 1" << std::endl; return NUC_MAP::G+1;}
	}
}


// Idea from Squeakr
uint8_t map_nuc2int(char nuc)
{
	switch(nuc) {
		case 'A': {return NUC_MAP::A;}
		case 'T': {return NUC_MAP::T;}
		case 'C': {return NUC_MAP::C;}
		case 'G': {return NUC_MAP::G;}
		default: {return NUC_MAP::G+1;}
		//default: {std::cout << "WEIRD THING 2" << std::endl; return NUC_MAP::G+1;}
	}
}


// Idea from Squeakr
string map_int2str(__uint128_t seq, uint64_t len)
{
	uint8_t nuc;
	string str;
	for (uint32_t i = len; i > 0; i--) {
		nuc = (seq >> (i*2-2)) & 3ULL;
		char nucc = map_int2nuc(nuc);
		str.push_back(nucc);
	}
	return str;
}


// Idea from Squeakr
__uint128_t map_str2int(std::string str)
{
	__uint128_t strint = 0;
	for (auto it = str.begin(); it != str.end(); it++) {
		uint8_t nuc = 0;
		switch (*it) {
			case 'A': {nuc = NUC_MAP::A; break;}
			case 'T': {nuc = NUC_MAP::T; break;}
			case 'C': {nuc = NUC_MAP::C; break;}
			case 'G': {nuc = NUC_MAP::G; break;}
		}
		strint = strint | nuc;
		strint = strint << 2;
	}
	return strint >> 2;
}


void map_str2intR(std::string & seq, __uint128_t & bin_seq)
{
	bin_seq = 0;
	for (auto it = seq.begin(); it != seq.end(); it++) {
		uint8_t nuc = 0;
		switch (*it) {
			case 'A': {nuc = NUC_MAP::A; break;}
			case 'T': {nuc = NUC_MAP::T; break;}
			case 'C': {nuc = NUC_MAP::C; break;}
			case 'G': {nuc = NUC_MAP::G; break;}
		}
		bin_seq <<= 2;
		bin_seq |= nuc;
	}
}



// Idea from Squeakr
int reverse_complement_nucint(int nuc)
{ 
	return 3 - nuc; 
}


char reverse_complement_nucchar(char nuc)
{
	switch(nuc) {
		case 'A': {return 'T';}
		case 'T': {return 'A';}
		case 'C': {return 'G';}
		case 'G': {return 'C';}
		default: {return 'N';}
	}
}


__uint128_t reverse_complement_seqint(__uint128_t seq, uint64_t len)
{
	__uint128_t rev_seq = 0;
	uint8_t nuc = 0;
	for (uint64_t i = 0; i < len; i++) {
		nuc = seq & 3ULL; // 3 unsigned long long
		nuc = reverse_complement_nucint(nuc);
		seq >>= 2;
		rev_seq <<= 2; // MODIFIED RIGHT HERE, SWAP WITH BELOW IF PROBLEMS OCCUR
		rev_seq |= nuc;
		
	}
	//rev_seq >>=2;
	return rev_seq;
}


// Slow, where is this used?
string reverse_complement_seqstr(std::string seq)
{
	string rev_seq = "";
	char cur_char, rev_char;
	int seqlen = (int)seq.length();

	for(int i = 0; i < seqlen; i++)
	{
		cur_char = seq.at(i);
		rev_char = reverse_complement_nucchar(cur_char);
		rev_seq = rev_char + rev_seq;
	}
	return rev_seq;
}


bool compare_seqs(__uint128_t seq1, __uint128_t seq2)
{
	return seq1 <= seq2;
}

bool compare_seqs_string(std::string seq1, std::string seq2)
{
	int seqlen = (int)seq1.length();
	for (int i = 0; i < seqlen; i+= 1)
	{
		if (seq1[i] < seq2[i]){return true;}
		if (seq2[i] < seq1[i]){return false;}
	}
	return true;
}


bool is_zero(uint64_t x, int y)
{
	x = x << (62-y);
	x = x >> (62-y);
	return x == 0;
}


bool no_Ns_present(uint64_t nbs[], vector<int> &lengths, int blocks)
{
	for (int i = 0; i < blocks; i++)
	{
		if (is_zero(nbs[i], lengths[i]) == false){return false;}
	}
	return true;
}


// Requires the spaced seed pattern to be in the form of "a-b-c-d-e-f-g"
std::tuple< std::vector<bool>, int, int, std::vector<int>, std::vector<int>, std::vector<int> > interpret_spaced_seed_pattern(string pattern)
{
	regex rgx("-+");
    sregex_token_iterator iter(pattern.begin(), pattern.end(), rgx, -1);
    sregex_token_iterator last{};
    vector<int> segments, block_starts, block_lengths, fixed_nucs;
    int number_of_segments = 0;
    int total_length = 0;
    int fixed_length = 0;

    for ( ; iter != last; ++iter)
    {
    	number_of_segments += 1;
    	int segment_length = stoi(*iter);
    	segments.insert(segments.end(), 1, segment_length);
    }

	vector<bool> character_status;
	bool current_status = true;
	for (int i = 0; i < number_of_segments; i++)
	{
		int segment_size = segments[i];
		character_status.insert(character_status.end(), segment_size, current_status);
		total_length += segment_size;
		if(current_status)
		{
			fixed_length += segment_size;
			block_lengths.insert(block_lengths.end(), 1, segment_size);
			block_starts.insert(block_starts.end(), 1, total_length - segment_size);
		}
		current_status = !current_status;
	}
	for (int j = 0; j < total_length; j+=1)
	{
		if (character_status[j]){fixed_nucs.push_back(j);}

	}
	return make_tuple(character_status, total_length, fixed_length, block_starts, block_lengths, fixed_nucs);
}



std::string extract_spaced_kmer(std::string long_kmer, std::vector<bool> & is_fixed_character)
{
	std::string spaced_kmer = "";
	
	//std::cout << "SPACED K-MER EXTRACTION FROM ITS OCCURRENCE" << std::endl;
	//std::cout << "LONG K-MER IS BELOW" << std::endl;
	//std::cout << long_kmer << std::endl;

	int veclen = (int)is_fixed_character.size();
	//std::cout << "OCCURRENCE LENGTH IS: " << veclen << std::endl;

	//std::cout << "PRINTING SPACED SEED BELOW" << std::endl;

	for (int i = 0; i < veclen; i++)
	{
		if (is_fixed_character[i])
		{
			//std::cout << "1";
			spaced_kmer = spaced_kmer +	long_kmer.at(i);
		}
		//else
		//{
		//	std::cout << "0";
		//}
	}
	//std::cout << std::endl;
	//std::cout << "AND THE EXTRACTED SPACED K-MER IS BELOW" << std::endl;
	//std::cout << spaced_kmer << std::endl;

	return spaced_kmer;
}
