/* sample_lcp - Samples the LCP array to compute the mums
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
   \file sample_lcp.cpp
   \brief sample_lcp.cpp Samples the LCP array to compute the mums.
   \author Massimiliano Rossi
   \date 12/02/2022
*/

#include <iostream>

#define VERBOSE

#include <common.hpp>

#include <sdsl/io.hpp>

#include <phoni.hpp>

#include <malloc_count.h>

typedef std::pair<std::string, std::vector<uint8_t>> pattern_t;


int main(int argc, char *const argv[]) {
  Args args;
  parseArgs(argc, argv, args);

#ifdef NDEBUG
  verbose("RELEASE build");
#else
  verbose("DEBUG build");
#endif

  verbose("Compressing the LCP");
  std::chrono::high_resolution_clock::time_point t_insert_start = std::chrono::high_resolution_clock::now();


  ms_pointers<> ms;
  ms.build(args.filename);

  FILE *lcp_file;

  std::string outfile = args.filename + std::string(".mums.lcp");
  if ((lcp_file = fopen(outfile.c_str(), "w")) == nullptr)
      error("open() file " + outfile + " failed");
  for (size_t i = 0; i < ms.lcp_start.size(); ++i)
  {
    size_t s_lcp = ms.lcp_start[i];
    if (fwrite(&s_lcp, THRBYTES, 1, lcp_file) != 1)
        error("LCP write error 1");
    size_t e_lcp = ms.lcp_last[i];
    if (fwrite(&e_lcp, THRBYTES, 1, lcp_file) != 1)
        error("LCP write error 1");
        
  }
  fclose(lcp_file);

  std::chrono::high_resolution_clock::time_point t_insert_end = std::chrono::high_resolution_clock::now();

  verbose("PHONI index construction complete");
  verbose("Memory peak: ", malloc_count_peak());
  verbose("Elapsed time (s): ", std::chrono::duration<double, std::ratio<1>>(t_insert_end - t_insert_start).count());

  {
  ofstream outfile(args.filename + ".phoni", std::ios::binary);
  ms.serialize(outfile);
  }

  return 0;
}
