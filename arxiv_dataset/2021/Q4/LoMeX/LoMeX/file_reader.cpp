/*

	File reading functionality

*/

#include "file_reader.hpp"
#include "fun_kmers.hpp"
#include "file_writer.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <deque>
#include <map>
#include <random>
#include <math.h>

using namespace std;


/*
	Function to count fastq file lines
*/
int count_fastq_lines(std::string reads_path)
{
	std::ifstream fastq_file;
	fastq_file.open(reads_path, std::ifstream::in);
	std::string fastq_line;
	int number_of_lines = 0;
	while (std::getline(fastq_file, fastq_line, '\n')){number_of_lines+=1;}
	return number_of_lines;
}


/*
	Class for reading fastq files
*/

//FastqFileReader::FastqFileReader();

void FastqFileReader::initialize_me(std::string file_path, int start_position, int end_position)
{
	first = start_position; // included
	last = end_position; // excluded
	//std::cout << "START:"<< first << " END:" << last << std::endl; 
	reads_left = true;
	read_line = 0;
	current_read_number = -1;
	fastq_line = "";
	fastq_file.open(file_path, std::ifstream::in);

}

void FastqFileReader::kill_me()
{
	fastq_file.close();
	fastq_file.clear();
}

void FastqFileReader::roll_to_next_read()
{
	// Read fastq file line by line
	bool new_read_gotten = false;
	char current_nucleotide;
	char push_nucleotide;
	current_read.clear();
	while (std::getline(fastq_file, fastq_line, '\n'))
	{
		// Line with the actual read encountered, it is the second line
		if (read_line == 1)
		{
			current_read_number += 1;
			if (current_read_number >= last)
			{
				reads_left = false;
				kill_me();
				break;
			}

			if (current_read_number >= first)
			{
				int linelen = fastq_line.length();
				for (int i = 0; i < linelen; i++)
				{
					current_nucleotide = fastq_line.at(i);
					if (current_nucleotide == 'C' || current_nucleotide == 'c'){push_nucleotide = 'C';}
					else if (current_nucleotide == 'A' || current_nucleotide == 'a'){push_nucleotide = 'A';}
					else if (current_nucleotide == 'T' || current_nucleotide == 't'){push_nucleotide = 'T';}
					else if (current_nucleotide == 'G' || current_nucleotide == 'g'){push_nucleotide = 'G';}
					else {push_nucleotide = 'N';}
					current_read.push_back(push_nucleotide);
				}
				new_read_gotten = true;
				reads_left = true;
			}
		}
		// Update line counters
		read_line += 1;
		// Reached the end of a read
		if (read_line == 4){read_line = 0;}
		// Break after a new read has been found
		if (new_read_gotten){break;}
	}
	if (!new_read_gotten)
	{
		reads_left = false;
		kill_me();
	}
}



/*
	Class for handling a single temporary k-mer file
*/
TmpFileManager::TmpFileManager(int n, std::string path, int total_length, std::vector<bool> & character_status, int delfile, uint64_t fixed_length_to_save)
{
	fixed_length = fixed_length_to_save;
	is_fixed_character = character_status;
	current_spaced_kmer_int = 0;
	current_spaced_kmer_string = "EMPTY SPACED K-MER1";
	current_spaced_kmer_string_from_file = "EMPTY SPACED K-MER2";
	kmer_remains = true;
	kmer_length = total_length;
	my_file_path = path;
	sudoku = delfile;
	location_file.open(path, ios::in | ios::binary);
	read_file_info();
	read_next_line_dupesnt();
	//read_next_line();
}

void TmpFileManager::read_file_info()
{	
	location_file.read((char*)(&occurrence_bytes), sizeof(occurrence_bytes));
	//std::cout << "BYTES PER K-MER IS: " << occurrence_bytes << std::endl;
}


void TmpFileManager::read_next_line()
{
	clear_current_kmer();
	if (!kmer_remains){return;}

	int32_t kmer_occurrences = 0;
	location_file.read((char*) (&kmer_occurrences), sizeof(kmer_occurrences));

	if(kmer_occurrences == 0)
	{
		close_file();
		if (sudoku == 1)
		{
			delete_file();
		}
		return;
	}

	uint8_t occbyte;
	std::string current_kmer_occurrence_string;
	for (int occi = 0; occi < kmer_occurrences; occi+=1)
	{
		current_kmer_occurrence_string = "";
		for (int occb = 0; occb < occurrence_bytes; occb+=1)
		{
			location_file.read((char *) & bytebuffer, 1);
			occbyte = bytebuffer[0];
			current_kmer_occurrence_string = current_kmer_occurrence_string + map_byte2fournucs(occbyte);
		}
		current_kmer_occurrence_string = current_kmer_occurrence_string.substr(0, kmer_length);
		current_kmer_occurrences_string.push_back(current_kmer_occurrence_string);
	}
	current_spaced_kmer_string = extract_spaced_kmer(current_kmer_occurrences_string[0], is_fixed_character);
	current_spaced_kmer_int = map_str2int(current_spaced_kmer_string);
}


void TmpFileManager::read_next_line_dupesnt()
{
	__uint128_t previous_spaced_kmer_int = current_spaced_kmer_int;
	
	clear_current_kmer();
	if (!kmer_remains){return;}
	
	int32_t kmer_occurrences = 0;
	location_file.read((char*) (&kmer_occurrences), sizeof(kmer_occurrences));

	if(kmer_occurrences == 0)
	{
		close_file();
		if (sudoku == 1)
		{
			delete_file();
		}
		//std::cout << "FILE ENDS HERE" << std::endl;
		return;
	}

	// NEW STUFF
	//location_file.read((char*) (&spaced_kmer_from_file), sizeof(spaced_kmer_from_file));

	uint8_t occbyte = 0;
	int32_t regular_kmer_count = 0;

	int regular_kmer_count_readable;
	std::string current_kmer_occurrence_string;

	for (int occi = 0; occi < kmer_occurrences; occi+=1)
	{
		current_kmer_occurrence_string = "";
		for (int occb = 0; occb < occurrence_bytes; occb+=1)
		{
			location_file.read((char*) (&occbyte), sizeof(occbyte));
			current_kmer_occurrence_string = current_kmer_occurrence_string + map_byte2fournucs(occbyte);
		}
		// How many times does it occur
		location_file.read((char*) (&regular_kmer_count), sizeof(regular_kmer_count));
		regular_kmer_count_readable = static_cast<int>(regular_kmer_count);
		
		current_kmer_occurrence_string = current_kmer_occurrence_string.substr(0, kmer_length);

		for (int addi = 0; addi < regular_kmer_count_readable; addi+=1)
		{
			current_kmer_occurrences_string.push_back(current_kmer_occurrence_string);
		}
	}
	
	current_spaced_kmer_string = extract_spaced_kmer(current_kmer_occurrences_string[0], is_fixed_character);
	current_spaced_kmer_int = map_str2int(current_spaced_kmer_string);

	if (current_spaced_kmer_int < previous_spaced_kmer_int)
	{
		std::cout << "READING SPACED K-MERS IN WRONG ORDER!!! AYAYA !!" << std::endl;
	}
}

void TmpFileManager::read_next_line_dupesnt_DEBUG()
{
	__uint128_t previous_spaced_kmer_int = current_spaced_kmer_int;
	std::string previous_spaced_kmer_str = current_spaced_kmer_string;
	std::string previous_spaced_kmer_str_from_file = current_spaced_kmer_string_from_file;

	std::string loop_test_string = "SUNSHINEAMIGO";

	__uint128_t spaced_kmer_from_file;
	
	clear_current_kmer();
	if (!kmer_remains){return;}
	
	int32_t kmer_occurrences = 0;
	location_file.read((char*) (&kmer_occurrences), sizeof(kmer_occurrences));

	//std::cout << "SPACED K-MER HAS THIS MANY OCCURRENCES: " << kmer_occurrences << std::endl;
	
	if(kmer_occurrences == 0)
	{
		close_file();
		if (sudoku == 1)
		{
			delete_file();
		}
		std::cout << "FILE ENDS HERE" << std::endl;
		return;
	}

	// NEW STUFF
	location_file.read((char*) (&spaced_kmer_from_file), sizeof(spaced_kmer_from_file));

	uint8_t occbyte = 0;
	int32_t regular_kmer_count = 0;

	int regular_kmer_count_readable;
	std::string current_kmer_occurrence_string;

	for (int occi = 0; occi < kmer_occurrences; occi+=1)
	{
		current_kmer_occurrence_string = "";
		for (int occb = 0; occb < occurrence_bytes; occb+=1)
		{
			//location_file.read((char*) & bytebuffer, 1);
			//occbyte = bytebuffer[0];

			location_file.read((char*) (&occbyte), sizeof(occbyte));
			
			current_kmer_occurrence_string = current_kmer_occurrence_string + map_byte2fournucs(occbyte);
		}
		// How many times does it occur
		location_file.read((char*) (&regular_kmer_count), sizeof(regular_kmer_count));
		regular_kmer_count_readable = static_cast<int>(regular_kmer_count);
		if (regular_kmer_count_readable <= 0)
		{
			std::cout << "OMG ERROR ERROR I READ ZERO COUNT FOR A K-MER!!!!!!! XPOTATO " << regular_kmer_count_readable << std::endl;
		}
	
		current_kmer_occurrence_string = current_kmer_occurrence_string.substr(0, kmer_length);

		if (loop_test_string == "SUNSHINEAMIGO")
		{
			loop_test_string = extract_spaced_kmer(current_kmer_occurrence_string, is_fixed_character);
		}
		else
		{
			if (loop_test_string != extract_spaced_kmer(current_kmer_occurrence_string, is_fixed_character))
			{
				std::cout << "SPACED K-MER HAS WRONG OCCURRENCES!!!!! SONNA !!!!" << std::endl;
			}
			loop_test_string = extract_spaced_kmer(current_kmer_occurrence_string, is_fixed_character);
		}

		for (int addi = 0; addi < regular_kmer_count_readable; addi+=1)
		{
			current_kmer_occurrences_string.push_back(current_kmer_occurrence_string);
		}
	}
	current_spaced_kmer_string = extract_spaced_kmer(current_kmer_occurrences_string[0], is_fixed_character);
	current_spaced_kmer_int = map_str2int(current_spaced_kmer_string);

	std::string current_spaced_kmer_string_from_file = map_int2str(spaced_kmer_from_file, fixed_length);


	if (current_spaced_kmer_int != spaced_kmer_from_file)
	{
		std::cout << "TMP FILE AND EXTRACTED ARE DIFFERENT !!!!!!!! DESUDESU" << std::endl;
		std::cout << "EXTRACTED IS BELOW" << std::endl;
		std::cout << current_spaced_kmer_string << std::endl;
		std::cout << "FROM FILE IS BELOW" << std::endl;
		std::cout << current_spaced_kmer_string_from_file << std::endl;
		std::cout << "PREVIOUS DIRECTLY FROM FILE IS BELOW" << std::endl;
		std::cout << previous_spaced_kmer_str_from_file << std::endl;
		std::cout << "OCCURRENCES FOR CURRENT SPACED K-MER ARE: " << current_kmer_occurrences_string.size() << std::endl;
	}
	else
	{
		current_spaced_kmer_int = spaced_kmer_from_file;
	}

	if (current_spaced_kmer_int < previous_spaced_kmer_int)
	{
		std::cout << "READING SPACED K-MERS IN WRONG ORDER!!! AYAYA !!" << std::endl;
		std::cout << "CURRENT IS BELOW" << std::endl;
		std::cout << current_spaced_kmer_string << std::endl;
		std::cout << "PREVIOUS IS BELOW" << std::endl;
		std::cout << previous_spaced_kmer_str << std::endl;
	}
}

void TmpFileManager::close_file()
{
	kmer_remains = false;
	location_file.close();
	location_file.clear();
}

void TmpFileManager::delete_file()
{
	const char * removed_file = my_file_path.c_str();
	int removed = remove(removed_file);
	//if (removed == 0) {std::cout << "Location file " << my_file_path << " permanently deleted" << std::endl;}
	//else {std::cout << "Location file deletion was unsuccessful for some reason..." << std::endl;}
}

void TmpFileManager::clear_current_kmer()
{
	current_spaced_kmer_string = "";
	current_spaced_kmer_int = 0;
	current_kmer_occurrences_string.clear();
}


/*
	Class for merging multiple temporary k-mer files
*/

//TmpFileMerger::TmpFileMerger();


void TmpFileMerger::initialize_me(int n_files, vector<std::string> file_paths, uint64_t spaced_length, int total_length, std::vector<bool> & character_status, int delfiles, int iterations, int iteration, int threads, int thread)
{
	all_iterations = iterations;
	my_iteration = iteration;
	all_threads = threads;
	my_thread = thread;
	tmp_kmer_files = n_files;
	kmer_remains = true;
	fixed_length = spaced_length;
	kmer_length = total_length;
	is_fixed_character = character_status;
	cleanup = delfiles;
	previous_spaced_kmer = 0;
	my_spaced_kmers = 0;

	for (int i = 0; i < n_files; i+=1)
	{
		if (boost::algorithm::ends_with(file_paths[i], ".bin"))
		{	
			files.push_back(TmpFileManager(i, file_paths[i], kmer_length, is_fixed_character, cleanup, static_cast<uint64_t>(fixed_length)));
		}
	}
	//solve_next_kmer();
}

void TmpFileMerger::delete_location_files()
{
	std::cout << "Deleting location files" << std::endl;
	for(int i = 0; i < tmp_kmer_files; i+=1){files[i].delete_file();}
}


bool TmpFileMerger::solve_next_kmer()
{
	// First find the smallest k-mer in the input files
	//std::cout << "SOLVING NEXT K-MER" << std::endl;
	__uint128_t smallest_kmer = 0;
	bool first_in = false;

	for (int i = 0; i<tmp_kmer_files; i+=1)
	{
		if (files[i].get_kmer_remains())
		{
			__uint128_t file_smallest_kmer = files[i].get_current_kmer_int();
			if (!first_in)
			{
				smallest_kmer = file_smallest_kmer;
				first_in = true;
			}
			else if (file_smallest_kmer < smallest_kmer)
			{
				smallest_kmer = file_smallest_kmer;
			}
		}
	}

	if (smallest_kmer < previous_spaced_kmer)
	{
		std::cout << "FATAL ERROR IN READING SPACED K-MERS FROM TEMP FILES" << std::endl;
	}
	previous_spaced_kmer = smallest_kmer;

	// If no file has more k-mers, take note
	if (!first_in)
	{
		kmer_remains = false;
		current_spaced_kmer = 0;
		current_spaced_kmer_occurrences.clear();
		std::cout << "NO MORE SPACED K-MERS LEFT IN TEMP FILES" << std::endl;
		std::cout << "I HAD THIS MANY SPACED K-MERS: " << my_spaced_kmers << std::endl;
		return true;
	}
	// Otherwise update the stored smallest k-mer and its locations
	else
	{
		my_spaced_kmers += 1;
		current_spaced_kmer = smallest_kmer;
		current_spaced_kmer_string = map_int2str(current_spaced_kmer, fixed_length);
		current_spaced_kmer_occurrences.clear();

		//#pragma omp parallel for
		for (int i = 0; i<tmp_kmer_files; i+=1)
		{
			//std::cout << "DOING THIS LOOP" << std::endl;
			bool file_kmer_remains = files[i].get_kmer_remains();
			if (file_kmer_remains)
			{
				__uint128_t file_current_kmer = files[i].get_current_kmer_int();
				if (file_current_kmer == current_spaced_kmer)
				{
					std::vector<std::string> append_occurrences = files[i].get_current_kmer_occurrences();
					// Tell the file manager to read next line discarding the currently asked information
					int info_amount = append_occurrences.size();

					for (int info_counter = 0; info_counter < info_amount; info_counter+=1)
					{
						current_spaced_kmer_occurrences.push_back(append_occurrences[info_counter]);
					}
					files[i].read_next_line_dupesnt();
					//files[i].read_next_line();
				}
			}
		}
	}
	//(spaced_kmer - (spaced_kmer % iteration)) % threads
	//int a;
	//a = 
	//int a = (int)((current_spaced_kmer - (current_spaced_kmer % my_iteration)) / (float)all_iterations);

	return true;

	// ABSOLUTE MESS, FIX THIS LATER

	std::cout << "OK??" << std::endl;

	int tester = (int)current_spaced_kmer;
	if (tester < 0){tester = 0 - tester;}
	std::cout << "OK??2" << std::endl;

	if (tester % all_iterations == my_iteration){
		int tester2 = (int)((tester - (tester % my_iteration)) / all_iterations);

		std::cout << "OK??3" << std::endl;
		if(tester2 < 0){tester2 = 0 - tester2;}
		if (tester2 % all_threads == my_thread){
			std::cout << "OK??4" << std::endl;
			return true;
		} else {
			return false;
		}


	} else {
		std::cout << "FATAL ERROR IN FILE READER" << std::endl;
		return false;
	}

	//if ((((current_spaced_kmer-my_iteration)/all_iterations) % all_threads) != my_thread){return false;}
	//std::cout << "NEXT K-MER SOLVED" << std::endl;
	//return true;
}

tuple<std::string, std::vector<std::string> > TmpFileMerger::get_next_kmer()
{
	return make_tuple(current_spaced_kmer_string, current_spaced_kmer_occurrences);
}

