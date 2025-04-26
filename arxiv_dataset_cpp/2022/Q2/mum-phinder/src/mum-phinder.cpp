/* matching_statistics - Computes the matching statistics from BWT and Thresholds
    Copyright (C) 2020 Massimiliano Rossi
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/ .
*/
/*!
   \file matching_statistics.cpp
   \brief matching_statistics.cpp Computes the matching statistics from BWT and Thresholds.
   \author Massimiliano Rossi
   \date 13/07/2020
*/

#include <iostream>

#define VERBOSE

#include <common.hpp>

#include <sdsl/io.hpp>

#include <mum-phinder.hpp>

#include <malloc_count.h>

#include <kseq.h>
#include <zlib.h>



template<class matchingstats>
void run(const Args& args) {
  verbose("Deserializing the MUM-PHINDER index");
  std::chrono::high_resolution_clock::time_point t_insert_start = std::chrono::high_resolution_clock::now();

  matchingstats ms;
  {
    ifstream in(args.filename + ".mum-phinder");
    ms.load(in, args.filename);
  }

  std::chrono::high_resolution_clock::time_point t_insert_end = std::chrono::high_resolution_clock::now();

  verbose("MUM-PHINDER index construction complete");
  verbose("Memory peak: ", malloc_count_peak());
  verbose("Elapsed time (s): ", std::chrono::duration<double, std::ratio<1>>(t_insert_end - t_insert_start).count());

  verbose("Processing patterns");
  t_insert_start = std::chrono::high_resolution_clock::now();

  std::ofstream f_mums(args.patterns + ".mums");
  if (!f_mums.is_open())
    error("open() file " + std::string(args.patterns) + ".mums failed");


  gzFile fp = gzopen(args.patterns.c_str(), "r");
  kseq_t *seq = kseq_init(fp);
  int l;
  while ((l = kseq_read(seq)) >= 0)
  {
    auto res = ms.query_mum(seq->seq.s, seq->seq.l);
    // MUMmer-like print
    f_mums<<"> "<<std::string(seq->name.s) << std::endl;
    ms.print_MUMs_to_file(f_mums, res);
  }

  kseq_destroy(seq);
  gzclose(fp);

  t_insert_end = std::chrono::high_resolution_clock::now();
  verbose("Memory peak: ", malloc_count_peak());
  verbose("Elapsed time (s): ", std::chrono::duration<double, std::ratio<1>>(t_insert_end - t_insert_start).count());

  verbose("Printing plain output");
  t_insert_start = std::chrono::high_resolution_clock::now();

  f_mums.close();

  t_insert_end = std::chrono::high_resolution_clock::now();

  verbose("Memory peak: ", malloc_count_peak());
  verbose("Elapsed time (s): ", std::chrono::duration<double, std::ratio<1>>(t_insert_end - t_insert_start).count());
}

int main(int argc, char *const argv[]) {
  Args args;
  parseArgs(argc, argv, args);

#ifdef NDEBUG
  verbose("RELEASE build");
#else
  verbose("DEBUG build");
#endif

  verbose("Memory peak: ", malloc_count_peak());

  if(args.grammar == "naive") {
    verbose("using naive grammar");
    run<ms_pointers<PlainSlp<var_t, Fblc, Fblc>> >(args);
  } else {
    verbose("using default grammar");
    run<ms_pointers<> >(args);
  }


  auto mem_peak = malloc_count_peak();
  verbose("Memory peak: ", malloc_count_peak());
  return 0;
}
