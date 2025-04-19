/*

	Consensus step

*/

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include "fun_consensus.hpp"
#include "fun_kmers.hpp"
#include "file_writer.hpp"
#include "file_reader.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>


//namespace po = boost::program_options;
namespace bfs = boost::filesystem;
using namespace std;

tuple<int, int, int> run_consensus_step_multithread(std::string work_dir, std::string output_path, int total_length, int fixed_length, double relative_nuc_threshold, int abs_min_nuc, int delete_files, vector<bool> & character_status, int iterations, int iteration, int threads)
{

	//std::cout << "Building long consensus k-mers" << std::endl;
	// Temporary files
	std::vector<std::string> tmp_file_paths;

	// Get all temporary file name
	std::cout << "Finding spaced k-mer occurrence files" << std::endl;
	bfs::path wd{work_dir};
    if(is_directory(wd)) {
    	std::cout << "Temp file directory found" << std::endl;
        //std::cout << wd << " is a directory containing following files:\n";
        for(auto& entry : boost::make_iterator_range(bfs::directory_iterator(wd), {}))
        {
            //std::cout << entry << std::endl;
            tmp_file_paths.push_back(entry.path().string());
        }
    }
    else
    {
    	std::cout << "No directory for temporary files found" << std::endl;
    	return make_tuple(0, 0, 0);
    }

	// Make a 2d matrix for storing consensus k-mer stats
    int consensus_kmer_stats[threads][4];
    for (int thri = 0; thri < threads; thri+=1)
    {
    	for (int stai = 0; stai < 4; stai += 1){consensus_kmer_stats[thri][stai]=0;}
    }

	// Make an array for thread specific file names
	std::string thread_output_files[threads];

	//int solved_kmer_counter = 0;
	
	// DO NOT DELETE FILES IN THIS IMPLEMENTATION
	delete_files = 0; 

	// PARALLEL JOBS START HERE
    #pragma omp parallel for
    for (int thread = 0; thread < threads; thread+=1)
    {

    	int unambiguous_consensus = 0;
		int simple_ambiguous_consensus = 0;
		int complex_ambiguous_consensus = 0;
		int undecided_consensus = 0;

    	std::string thread_output_path = output_path + "/lomex_kmers_" + format_number(thread) + ".tmp";

    	thread_output_files[thread] = thread_output_path;
    	// Output file
		std::ofstream thread_output_file;
		// Open a file to write the k-mers
		thread_output_file.open(thread_output_path, ios::out);

    	// k-mer strings
		std::string current_spaced_kmer, consensus_spaced_kmer;
		char curr_kmer_char;
		std::vector<std::string> consensus_spaced_kmer_occurrences;
		int kmer_pattern_occurrences;
		bool undecided_kmer;

    	// Create file merger
    	TmpFileMerger tmp_merger;
		tmp_merger.initialize_me(tmp_file_paths.size(), tmp_file_paths, fixed_length, total_length, character_status, delete_files, iterations, iteration, threads, thread);
	    
		std::cout << "File merger for successfully created for thread: " << thread << std::endl;
		
		// Nucleotide matrices
		vector<int> consensus_nuc_init(total_length, -1);
		vector<vector<int> > consensus_nucleotides(4, consensus_nuc_init);
		
		vector<int> nuc_counts_init(total_length, -1);
		vector<vector<int> > kmer_nucleotide_occurrences(4, nuc_counts_init);
		//int kmer_nucleotide_occurrences[4][total_length];

		// Reset matrices
		for (int row = 0; row < 4; row++)
		{			
			for (int column = 0; column < total_length; column++)
			{
				kmer_nucleotide_occurrences[row][column] = 0;
				consensus_nucleotides[row][column] = 0;
			}
		}

		//while(tmp_merger.get_kmer_remains())
		while(true)
		{
			bool correct_kmer = false;
			while(!correct_kmer){correct_kmer = tmp_merger.solve_next_kmer();}
			
			if (!tmp_merger.get_kmer_remains()){break;}

			// Debug printing
			//if (solved_kmer_counter % 10000 == 0)
			//{
			//	std::cout << "Solved k-mers: " << solved_kmer_counter << std::endl;
			//}
			//solved_kmer_counter += 1;

			//tie(consensus_spaced_kmer, consensus_spaced_kmer_occurrences) = tmp_merger.get_next_kmer(); // FIX THIS SO THAT IT RETURN THE DATA IN CORRECT FORM PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

			consensus_spaced_kmer = tmp_merger.get_next_spaced_kmer();
			consensus_spaced_kmer_occurrences = tmp_merger.get_next_spaced_kmer_regular_kmers();

			undecided_kmer = false;

			current_spaced_kmer = consensus_spaced_kmer;
			kmer_pattern_occurrences = consensus_spaced_kmer_occurrences.size(); // reset the new spaced k-mer counter to zero

			// Start going through the current spaced k-mer occurrences in the data and fill the nucleotide count matrix

			for (int kmerlen = 0; kmerlen < total_length; kmerlen++)
			{
				for (int occs = 0; occs < (int)consensus_spaced_kmer_occurrences.size(); occs++)
				{
					curr_kmer_char = consensus_spaced_kmer_occurrences[occs].at(kmerlen);
					if (curr_kmer_char == 'C'){kmer_nucleotide_occurrences[0][kmerlen] += 1;}
					else if (curr_kmer_char == 'A'){kmer_nucleotide_occurrences[1][kmerlen] += 1;}
					else if (curr_kmer_char == 'T'){kmer_nucleotide_occurrences[2][kmerlen] += 1;}
					else if (curr_kmer_char == 'G'){kmer_nucleotide_occurrences[3][kmerlen] += 1;}
				}
			}

			// Now find which characters are trusted in each gap position

			int ambiguous_count = 0; // Count the number of ambiguous consensus positions
			int ambiguous_positions[total_length]; // Array to specify which positions are ambiguous
			int relative_threshold; // Required character count for it to be trusted
			int absolute_min_char_count = abs_min_nuc; // Absolute minimum required character count 	

			relative_threshold = static_cast<int>(ceil(kmer_pattern_occurrences * relative_nuc_threshold)); 

			// Set the minimum count threshold
			if (relative_threshold < absolute_min_char_count)
			{
				relative_threshold = absolute_min_char_count;
			}

			//relative_threshold = absolute_min_char_count; //                  RELATIVE THRESHOLD REMOVED

			// Reset ambiguous positions to zero
			for (int i = 0; i < total_length; i++)
			{
				ambiguous_positions[i] = 0;
			}

			int trusted_nucs; // How many trusted characters a position has

			// Find out the trusted nuclotides for each gap position

			// Loop through all spaced k-mer positions
			for (int consensus_i = 0; consensus_i < total_length; consensus_i++)
			{
				trusted_nucs = 0; // Reset trusted nucleotides
				// Loop through all 4 nucleotides
				for (int nuc_i = 0; nuc_i < 4; nuc_i++)
				{
					if (kmer_nucleotide_occurrences[nuc_i][consensus_i] >= relative_threshold)
					{
						consensus_nucleotides[nuc_i][consensus_i] = 1;
						trusted_nucs += 1;
					}
					if (trusted_nucs == 2)
					{
						//ambiguous_count += 1;
						ambiguous_positions[consensus_i] = 1;
					}
				
				// If none of the nucleotides appears enough (more than once)	
				}
				if (trusted_nucs == 0)
				{
					undecided_kmer = true;
				
				}
			}

			for (const int &ambistatus : ambiguous_positions){ambiguous_count+=ambistatus;}

			// Finally find the consensus sequenses

			if (undecided_kmer)
			{
				undecided_consensus += 1;
				goto consensus_cleanup;
			}


			/*
				-- First case: unambiguous consensus --

				Print the consensus k-mer if it is unambiguous
			*/
			else if (ambiguous_count == 0)
			{
				unambiguous_consensus = determine_unambiguous_consensus(thread_output_file, total_length, consensus_nucleotides, kmer_nucleotide_occurrences, unambiguous_consensus);
				//unambiguous_consensus = determine_unambiguous_consensus(thread_output_file, total_length, consensus_nucleotides, unambiguous_consensus);
			}

			/*
				-- Second case: simple ambiguous consensus --

				i.e. Print the consensus k-mers if they differ in only one position
			*/
			else if (ambiguous_count == 1)
			{

				simple_ambiguous_consensus = determine_simple_ambiguous_consensus(thread_output_file, total_length, consensus_nucleotides, kmer_nucleotide_occurrences, ambiguous_positions, simple_ambiguous_consensus);
			
				//bool unfinished_consensus = true;	
				//while(unfinished_consensus)
				//{
				//	tie(simple_ambiguous_consensus, unfinished_consensus) = determine_simple_ambiguous_consensus(thread_output_file, total_length, consensus_nucleotides, simple_ambiguous_consensus);
				//}
			}

			/*
				-- Third case: complex ambiguous consensus --

				Solve the more ambiguous consensus k-mers with different means
			*/
			else if (ambiguous_count > 1)
			{
				vector<vector<char> > ambiguous_patterns;
				ambiguous_patterns = extract_ambiguous_position_matrix(consensus_spaced_kmer_occurrences.size(), ambiguous_count, consensus_spaced_kmer_occurrences, ambiguous_positions, total_length);
				complex_ambiguous_consensus = determine_complex_ambiguous_consensus(thread_output_file, total_length, consensus_nucleotides, kmer_nucleotide_occurrences, kmer_pattern_occurrences, ambiguous_positions,
										  										ambiguous_patterns, ambiguous_count, complex_ambiguous_consensus);
				//complex_ambiguous_consensus = determine_complex_ambiguous_consensus(thread_output_file, total_length, consensus_nucleotides, kmer_pattern_occurrences, ambiguous_positions,
				//							  										ambiguous_patterns, ambiguous_count, complex_ambiguous_consensus);
			}

consensus_cleanup:

			// Reset matrices for next iteration
			for (int row = 0; row < 4; row++)
			{			
				for (int column = 0; column < total_length; column++)
				{
					kmer_nucleotide_occurrences[row][column] = 0;
					consensus_nucleotides[row][column] = 0;
				}
			}
		} 

		thread_output_file.close(); // close file
		thread_output_file.clear(); // clear flags

		consensus_kmer_stats[thread][0] = unambiguous_consensus;
		consensus_kmer_stats[thread][1] = simple_ambiguous_consensus;
		consensus_kmer_stats[thread][2] = complex_ambiguous_consensus;
		consensus_kmer_stats[thread][3] = undecided_consensus;
    }


    std::cout << "MADE IT HERE" << std::endl;
    // MERGE THREAD OUTPUT FILES
    // Path to the output file
    std::string consensus_kmers_path = output_path + "/lomex_kmers.txt";
	// Output file
	std::ofstream lomex_output_file;
	// Open a file to write the k-mers
	lomex_output_file.open(consensus_kmers_path, ios::out | ios::app);
	// Put all thread consensus k-mers into the file
    for(auto & tmpfilepath : thread_output_files)
    {
    	std::string tmpkmer;
    	std::ifstream tmp_thread_file;
    	tmp_thread_file.open(tmpfilepath, std::ifstream::in);
    	while (std::getline(tmp_thread_file, tmpkmer, '\n')){lomex_output_file << tmpkmer << "\n";}
    	tmp_thread_file.close();
		tmp_thread_file.clear();
    }

    // DELETE TEMPORARY FILES HERE 
    for(auto & tmpfilepath : tmp_file_paths)
    {
    	const char * removed_file = tmpfilepath.c_str();
		int removed = remove(removed_file);
		if (removed == 0) {std::cout << "Regular k-mer file " << tmpfilepath << " permanently deleted" << std::endl;}
		else {std::cout << "Regular k-mer file deletion was unsuccessful for some reason..." << std::endl;}

    }
    // DELETE THREAD OUTPUT FILES
    for(auto & threadfilepath : thread_output_files)
    {
    	const char * removed_file = threadfilepath.c_str();
		int removed = remove(removed_file);
		if (removed == 0) {std::cout << "Thread spaced k-mer file " << threadfilepath << " permanently deleted" << std::endl;}
		else {std::cout << "Thread spaced k-mer file deletion was unsuccessful for some reason..." << std::endl;}
    }

    // COMBINE THREAD STATS
	int total_unambiguous_consensus = 0;
	int total_simple_ambiguous_consensus = 0;
	int total_complex_ambiguous_consensus = 0;
	int total_undecided_consensus = 0;

	for (int thri = 0; thri < threads; thri+=1)
	{
		total_unambiguous_consensus += consensus_kmer_stats[thri][0];
		total_simple_ambiguous_consensus += consensus_kmer_stats[thri][1];
		total_complex_ambiguous_consensus += consensus_kmer_stats[thri][2];
		total_undecided_consensus += consensus_kmer_stats[thri][3];
	}
    

	std::cout << "-- Printing consensus k-mer stats --" << std::endl;
	std::cout << "Total number of consensus k-mers: " << total_unambiguous_consensus + total_simple_ambiguous_consensus + total_complex_ambiguous_consensus << std::endl;
	std::cout << "Unambiguous consensus k-mers: " << total_unambiguous_consensus << std::endl;
	std::cout << "Simple ambiguous consensus k-mers: " << total_simple_ambiguous_consensus << std::endl;
	std::cout << "Complex ambiguous consensus k-mers: " << total_complex_ambiguous_consensus << std::endl;

	return make_tuple(total_unambiguous_consensus, total_simple_ambiguous_consensus, total_complex_ambiguous_consensus);

	//return 0;
}


tuple<int, int, int, int> run_consensus_step(std::string work_dir, std::string output_path, std::string output_file_name, int total_length, int fixed_length, double relative_nuc_threshold, int abs_min_nuc, int delete_files, vector<bool> & character_status)
{

	int all_regular_kmers_count = 0;

	//std::cout << "Building long consensus k-mers" << std::endl;

	// Temporary files
	std::vector<std::string> tmp_file_paths;
	TmpFileMerger tmp_merger;
	
	// Nucleotide matrices
	vector<int> consensus_nuc_init(total_length, -1);
	vector<vector<int> > consensus_nucleotides(4, consensus_nuc_init);

	vector<int> nuc_counts_init(total_length, -1);
	vector<vector<int> > kmer_nucleotide_occurrences(4, nuc_counts_init);
	//int kmer_nucleotide_occurrences[4][total_length];
	
	// k-mer strings
	std::string current_spaced_kmer, consensus_spaced_kmer;
	char curr_kmer_char;
	
	std::vector<std::string> consensus_spaced_kmer_occurrences;
	
	int kmer_pattern_occurrences;

	bool undecided_kmer;
	int solved_kmer_counter = 0;
	int unambiguous_consensus = 0;
	int simple_ambiguous_consensus = 0;
	int complex_ambiguous_consensus = 0;
	int undecided_consensus = 0;

	
	// Output file
	std::ofstream output_file;
	// Open a file to write the k-mers
	//std::string consensus_kmers_path = output_path + "/lomex_kmers.txt";
	std::string consensus_kmers_path = output_path + "/" + output_file_name;
	// Output file
	//std::ofstream lomex_output_file;
	// Open a file to write the k-mers
	output_file.open(consensus_kmers_path, ios::out | ios::app);
	//output_file.open(output_path, ios::out | ios::app);
	
	// Get all temporary file name
	std::cout << "Finding k-mer location files" << std::endl;

	bfs::path wd{work_dir};
    if(is_directory(wd)) {
        std::cout << "Temp file directory found" << std::endl;
        for(auto& entry : boost::make_iterator_range(bfs::directory_iterator(wd), {}))
        {
            //std::cout << entry << std::endl;
            tmp_file_paths.push_back(entry.path().string());
        }
    }
    else
    {
    	std::cout << "No directory for temporary files found" << std::endl;
    	return make_tuple(0, 0, 0, 0);
    }

    // Always delete files
	delete_files = 1;

    // Create file merger
    //tmp_merger.initialize_me(tmp_file_paths.size(), tmp_file_paths, fixed_length, total_length, character_status, delete_files);

    tmp_merger.initialize_me(tmp_file_paths.size(), tmp_file_paths, fixed_length, total_length, character_status, delete_files, 1, 0, 1, 0);
	    
	std::cout << "File merger successfully created" << std::endl;
	
	// Reset matrices
	for (int row = 0; row < 4; row++)
	{			
		for (int column = 0; column < total_length; column++)
		{
			kmer_nucleotide_occurrences[row][column] = 0;
			consensus_nucleotides[row][column] = 0;
		}
	}

	//while(tmp_merger.get_kmer_remains())
	while(true)
	{
		tmp_merger.solve_next_kmer();

		if (!tmp_merger.get_kmer_remains()){break;}

		// Debug printing
		/*
		if (solved_kmer_counter % 100000 == 0)
		{
			std::cout << "Solved k-mers: " << solved_kmer_counter << std::endl;
		}
		*/
		
		solved_kmer_counter += 1;
		//tie(consensus_spaced_kmer, consensus_spaced_kmer_occurrences) = tmp_merger.get_next_kmer(); // FIX THIS SO THAT IT RETURN THE DATA IN CORRECT FORM PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

		consensus_spaced_kmer = tmp_merger.get_next_spaced_kmer();
		consensus_spaced_kmer_occurrences = tmp_merger.get_next_spaced_kmer_regular_kmers();

		//####################################################################
		//####################################################################
		//####################################################################
		//####################################################################
		//  DEBUG BLOCK
		//for (auto& sunshineamigo : consensus_spaced_kmer_occurrences)
		//{
		//	output_file << sunshineamigo << "\n";
		//}
		//all_regular_kmers_count += (int)consensus_spaced_kmer_occurrences.size();
		//continue;
		//####################################################################
		//####################################################################
		//####################################################################
		//####################################################################

		// Write something random and don't solve the consensus FOR TESTING
		//output_file << consensus_spaced_kmer;
		//output_file << " T";
		//output_file << '\n';
		//continue;
		// Block above is for test purposes

		undecided_kmer = false;

		current_spaced_kmer = consensus_spaced_kmer;
		kmer_pattern_occurrences = consensus_spaced_kmer_occurrences.size(); // reset the new spaced k-mer counter to zero

		// Start going through the current spaced k-mer occurrences in the data and fill the nucleotide count matrix

		for (int kmerlen = 0; kmerlen < total_length; kmerlen++)
		{
			for (int occs = 0; occs < (int)consensus_spaced_kmer_occurrences.size(); occs++)
			{
				curr_kmer_char = consensus_spaced_kmer_occurrences[occs].at(kmerlen);
				if (curr_kmer_char == 'C'){kmer_nucleotide_occurrences[0][kmerlen] += 1;}
				else if (curr_kmer_char == 'A'){kmer_nucleotide_occurrences[1][kmerlen] += 1;}
				else if (curr_kmer_char == 'T'){kmer_nucleotide_occurrences[2][kmerlen] += 1;}
				else if (curr_kmer_char == 'G'){kmer_nucleotide_occurrences[3][kmerlen] += 1;}
			}
		}

		// Now find which characters are trusted in each gap position

		int ambiguous_count = 0; // Count the number of ambiguous consensus positions
		int ambiguous_positions[total_length]; // Array to specify which positions are ambiguous
		int relative_threshold; // Required character count for it to be trusted
		int absolute_min_char_count = abs_min_nuc; // Absolute minimum required character count 	

		relative_threshold = static_cast<int>(ceil(kmer_pattern_occurrences * relative_nuc_threshold)); 

		// Set the minimum count threshold
		if (relative_threshold < absolute_min_char_count)
		{
			relative_threshold = absolute_min_char_count;
		}

		//RELATIVE THRESHOLD REMOVER
		//relative_threshold = absolute_min_char_count; 

		// Reset ambiguous positions to zero
		for (int i = 0; i < total_length; i++)
		{
			ambiguous_positions[i] = 0;
		}

		int trusted_nucs; // How many trusted characters a position has

		// Find out the trusted nuclotides for each gap position

		// Loop through all spaced k-mer positions
		for (int consensus_i = 0; consensus_i < total_length; consensus_i++)
		{
			trusted_nucs = 0; // Reset the number of trusted nucleotides
			// Loop through all 4 nucleotides
			for (int nuc_i = 0; nuc_i < 4; nuc_i++)
			{
				if (kmer_nucleotide_occurrences[nuc_i][consensus_i] >= relative_threshold)
				{
					consensus_nucleotides[nuc_i][consensus_i] = kmer_nucleotide_occurrences[nuc_i][consensus_i];
					//consensus_nucleotides[nuc_i][consensus_i] = 1;
					trusted_nucs += 1;
				}
				if (trusted_nucs == 2)
				{
					//ambiguous_count += 1;
					ambiguous_positions[consensus_i] = 1;
				}
			
			// If none of the nucleotides appears enough (more than once)	
			}
			if (trusted_nucs == 0)
			{
				undecided_kmer = true;
			}
		}

		for (const int &ambistatus : ambiguous_positions){ambiguous_count+=ambistatus;}

		// Finally find the consensus sequenses

		if (undecided_kmer)
		{
			undecided_consensus += 1;
			goto consensus_cleanup;
		}

		/*
			-- First case: unambiguous consensus --

			Print the consensus k-mer if it is unambiguous
		*/
		else if (ambiguous_count == 0)
		{
			unambiguous_consensus = determine_unambiguous_consensus(output_file, total_length, consensus_nucleotides, kmer_nucleotide_occurrences, unambiguous_consensus);
		}

		/*
			-- Second case: simple ambiguous consensus --

			i.e. Print the consensus k-mers if they differ in only one position
		*/
		else if (ambiguous_count == 1)
		{
			simple_ambiguous_consensus = determine_simple_ambiguous_consensus(output_file, total_length, consensus_nucleotides, kmer_nucleotide_occurrences, ambiguous_positions, simple_ambiguous_consensus);
			
			//bool unfinished_consensus = true;	
			//while(unfinished_consensus)
			//{
			//	tie(simple_ambiguous_consensus, unfinished_consensus) = determine_simple_ambiguous_consensus(output_file, total_length, consensus_nucleotides, kmer_nucleotide_occurrences, simple_ambiguous_consensus);
			//}
		}

		/*
			-- Third case: complex ambiguous consensus --

			Solve the more ambiguous consensus k-mers with different means
		*/
		else if (ambiguous_count > 1)
		{
			vector<vector<char> > ambiguous_patterns;
			ambiguous_patterns = extract_ambiguous_position_matrix(consensus_spaced_kmer_occurrences.size(), ambiguous_count, consensus_spaced_kmer_occurrences, ambiguous_positions, total_length);
			complex_ambiguous_consensus = determine_complex_ambiguous_consensus(output_file, total_length, consensus_nucleotides, kmer_nucleotide_occurrences, kmer_pattern_occurrences, ambiguous_positions,
										  										ambiguous_patterns, ambiguous_count, complex_ambiguous_consensus);
		}

consensus_cleanup:

		// Reset matrices for next iteration
		for (int row = 0; row < 4; row++)
		{			
			for (int column = 0; column < total_length; column++)
			{
				kmer_nucleotide_occurrences[row][column] = 0;
				consensus_nucleotides[row][column] = 0;
			}
		}
	} 

	output_file.close(); // close file
	output_file.clear(); // clear flags	

	std::cout << "** Consensus k-mer stats for this iteration**" << std::endl;
	std::cout << "1) Total number of consensus k-mers: " << unambiguous_consensus + simple_ambiguous_consensus + complex_ambiguous_consensus << std::endl;
	std::cout << "2) Unambiguous consensus k-mers: " << unambiguous_consensus << std::endl;
	std::cout << "3) Simple ambiguous consensus k-mers: " << simple_ambiguous_consensus << std::endl;
	std::cout << "4) Complex ambiguous consensus k-mers: " << complex_ambiguous_consensus << std::endl;

	return make_tuple(unambiguous_consensus, simple_ambiguous_consensus, complex_ambiguous_consensus, all_regular_kmers_count);

	//return 0;
}
