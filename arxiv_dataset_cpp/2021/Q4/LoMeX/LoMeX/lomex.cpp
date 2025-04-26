#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <regex>
#include <string>
#include <tuple>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <math.h>
#include "fun_kmers.hpp"
#include "fun_consensus.hpp"
#include "file_writer.hpp"
#include "file_reader.hpp"
#include "search_step.hpp"
#include "consensus_step.hpp"


//namespace chrono = std::chrono; 
namespace po = boost::program_options;
namespace bfs = boost::filesystem;
//using namespace std;
//using namespace boost::filesystem;


int main(int argc, char *argv[])
{

	/*
		Initialize random
	*/

	int random_seed = 999;
	std::srand(random_seed);


	/*
	 * Step 1: Parse input arguments
	 *
	*/

	// Initialize arguments
	std::string kmers_path, reads_path, output_path, output_file_name, work_dir, spaced_seed_pattern;
	int kmer_min, buffer_size, delete_files, skip_occurrences, skip_consensus, abs_min_nuc, iterations, search_threads, consensus_threads;
	float relative_nuc_threshold;

	// Parse arguments
	po::options_description desc("k-mer index options");

	desc.add_options()
		("help,h", "Give help")
		("k-mers,k", po::value<std::string>(& kmers_path)->default_value("na"), "Path to the k-mer file")
		("reads,r", po::value<std::string>(& reads_path)->default_value("na"), "Path to the read file")
		("output-path,o", po::value<std::string>(& output_path)->default_value(""), "Path to the output directory")
		("output-file,q", po::value<std::string>(& output_file_name)->default_value("lomex-kmers.txt"), "Output file name")
		("work-dir,w", po::value<std::string>(& work_dir)->default_value("work_tmp"), "Path to a working directory")
		("spaced-seed,s", po::value<std::string>(& spaced_seed_pattern)->default_value("na"), "Spaced seed pattern")
		("min-occ,m", po::value<int>(& kmer_min)->default_value(2), "Minimum number of k-mer occurrences required")
		("min-nuc,n", po::value<int>(& abs_min_nuc)->default_value(2), "Minimum nucleotide threshold")
		("rel-nuc,l", po::value<float>(& relative_nuc_threshold)->default_value(0.1), "Relative nucleotide threshold")
		("buffer-size,b", po::value<int>(& buffer_size)->default_value(5000000), "Buffer size for k-mer occurrences")
		("delete-files,d", po::value<int>(& delete_files)->default_value(0), "Delete temporary location files")
		("skip-occurrences,f", po::value<int>(& skip_occurrences)->default_value(0), "Skip occurrence finding step")
		("skip-consensus,c", po::value<int>(& skip_consensus)->default_value(0), "Skip consensus finding step")
		("search-threads,t", po::value<int>(& search_threads)->default_value(1), "Search threads")
		("consensus-threads,e", po::value<int>(& consensus_threads)->default_value(1), "Consensus threads")
		("iterations,i", po::value<int>(& iterations)->default_value(1), "Iterations");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
 		std::cout << "Print help message here" << std::endl;
    	return 0;
	}

	int initialization_seconds, search_seconds, consensus_seconds;
	initialization_seconds = 0;
	search_seconds = 0;
	consensus_seconds = 0;
	std::chrono::high_resolution_clock::time_point start_time, end_time;
	std::chrono::duration<double> elapsed_time;

	std::cout << "Initialization started" << std::endl;

	
	//auto init_start = chrono::high_resolution_clock::now();
	start_time = std::chrono::high_resolution_clock::now();

	/*
	 * Step 2: Check spaced seed pattern characteristics
	 *
	*/

	// Initialize k-mer stuff
	vector<bool> character_status; // bit vector for spaced seed pattern
	int total_length, fixed_length; // pattern total length, pattern fixed characters
	vector<int> fixed_block_positions, fixed_block_lengths, fixed_nuc_positions; // fixed character block start positions and lengths

	// Get gapped k-mer characteristics, std::tie
	tie(character_status, total_length, fixed_length, fixed_block_positions, fixed_block_lengths, fixed_nuc_positions) = interpret_spaced_seed_pattern(spaced_seed_pattern);


	// Pattern length cannot be even, throw an exception and kill the program
	if (total_length % 2 == 0)
	{
		std::cout << "The spaced seed pattern length cannot be even. Choose a pattern with an odd length." << std::endl;
		return 0;
	}

	std::cout << "Calculating how to split the job between threads" << std::endl;
	int number_of_reads = (int)std::floor((double)count_fastq_lines(reads_path) / 4.0);
	//std::cout << number_of_reads << " reads are split between " << threads << " threads" << std::endl;

	end_time = std::chrono::high_resolution_clock::now();
	elapsed_time = end_time - start_time;
	initialization_seconds += (elapsed_time.count());

	std::cout << "Initialization done" << std::endl;

	std::cout << "Long k-mer extraction started" << std::endl;

	// Keep track of some stats to print at the end of the program
	int ua = 0;
	int sa = 0;
	int ca = 0;

	int nua = 0;
	int nsa = 0;
	int nca = 0;

	int narkc = 0;
	int nwrkc = 0;

	uint64_t current_iteration_bytes;
	uint64_t tmp_memory_bytes = 0;

	int written_regular_kmers_count = 0;
	int all_regular_kmers_count = 0;

	for (int i = 0; i < iterations; i+=1)
	{

		std::cout << "== Iteration " << i+1 << "/" << iterations << "==" <<std::endl;

		/*
		 * Step 3: Run search step if it is not skipped
		 *
		*/
		if(skip_occurrences != 1)
		{
			std::cout << "- Search step -" << std::endl;
			start_time = std::chrono::high_resolution_clock::now();
			tie(current_iteration_bytes, nwrkc) = run_search_step(work_dir, kmers_path, reads_path, buffer_size, total_length, fixed_length, character_status, kmer_min, i, iterations, search_threads, number_of_reads);
			tmp_memory_bytes += current_iteration_bytes;
			end_time = std::chrono::high_resolution_clock::now();
			elapsed_time = end_time - start_time;
			search_seconds += (elapsed_time.count());
			//std::cout << "FILE SIZES: " << tmp_memory_bytes << std::endl;
			written_regular_kmers_count += nwrkc;
			
		}

		/*
		 * Step 4: Run consensus step if it is not skipped
		 *
		*/
		if (skip_consensus != 1)
		{
			std::cout << "- Consensus step -" << std::endl;
			start_time = std::chrono::high_resolution_clock::now();
			//tie(nua, nsa, nca) = run_consensus_step_multithread(work_dir, output_path, total_length, fixed_length, relative_nuc_threshold, abs_min_nuc, delete_files, character_status, iterations, i, consensus_threads);
			tie(nua, nsa, nca, narkc) = run_consensus_step(work_dir, output_path, output_file_name, total_length, fixed_length, relative_nuc_threshold, abs_min_nuc, delete_files, character_status);
			end_time = std::chrono::high_resolution_clock::now();
			elapsed_time = end_time - start_time;
			consensus_seconds += (elapsed_time.count());
			ua += nua;
			sa += nsa;
			ca += nca;
			all_regular_kmers_count += narkc;
		}
	}
	
	/*
	 *
	 * Step 5: End of the program
	 *
	*/

	std::cout << "################# k-mer extraction stats #################" << std::endl;
	std::cout << "Total number of consensus k-mers: " << ua + sa + ca << std::endl;
	std::cout << "Unambiguous consensus k-mers: " << ua << std::endl;
	std::cout << "Simple ambiguous consensus k-mers: " << sa << std::endl;
	std::cout << "Complex ambiguous consensus k-mers: " << ca << std::endl;
	std::cout << "##########################################################" << std::endl << std::endl;


	std::cout << "################# Run time break down #################" << std::endl;
	std::cout << "Total time (seconds): " << initialization_seconds + search_seconds + consensus_seconds << std::endl;
	std::cout << "Initialization time (seconds): " << initialization_seconds << std::endl;
	std::cout << "Search time (seconds): " << search_seconds << std::endl;
	std::cout << "Consensus time (seconds): " << consensus_seconds << std::endl;
	std::cout << "#######################################################" << std::endl << std::endl;


	//std::cout << "THIS MANY TMP BYTES USED: " << tmp_memory_bytes << std::endl;
	//std::cout << "THIS MANY REGULAR K-MERS WRITTEN: " << written_regular_kmers_count << std::endl;
	//std::cout << "THIS MANY REGULAR K-MERS READ: " << all_regular_kmers_count << std::endl;


	std::cout << "Program run finished successfully. Thank you for using LoMeX." << std::endl;
	return 0;

}
