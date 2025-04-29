#include <sstream>
#include <fstream>
#include <iostream>
#include "util.h"

// Thisf function will update the values inputs, errors and trues(spectro info).
bool annz_util::parse_input_data(const std::string data, const int nIn, const int nOut, bool& truesPresent, std::vector<double>& inputs, std::vector<double>& errors, std::vector<double>& trues) {
	
	// Recap:
	// data = one line of data (string)
	// nIn and nOut = number of in and outputs
	// truesPresent = whether spectro info available
	// inputs, errors & trues are vectors.
		
  // Clear data vectors.
  inputs.clear();
  errors.clear();
  trues.clear();

  // Attach a stringstream to the data string.
  std::stringstream ss(data, std::stringstream::in);

  // Temporary data storage.
  double temp;
  
  // Read inputs (expect n_inputs values).
  for (int ii = 0; ii < nIn; ++ii) {
    if (ss >> temp) {
      inputs.push_back(temp);
    } else {
      std::cerr << "ERROR: Bad input data (insufficient fields)" << std::endl;
      return false;
    }
  }
  
  // Read errors (again, expect n_inputs values).
  for (int ii = 0; ii < nIn; ++ii) {
    if (ss >> temp) {
      errors.push_back(temp);
    } else {
      std::cerr << "ERROR: Bad input data (insufficient fields)" << std::endl;
      return false;
    }
  }
  
  // Is there any more?
  if (ss >> temp) {
    // Assume it's evaluation data.
    trues.push_back(temp);
    
    // Look for the rest of the outputs.
    for (int ii = 1; ii < nOut; ++ii) {
      if (ss >> temp) {
        trues.push_back(temp);
	  } else {
		std::cerr << "ERROR: Bad input data (insufficient fields)" << std::endl;
		return false;
	  }
	}
    

    // Ensure we are now at the end of the buffer.
    ss >> std::ws;
    if (!ss.eof()) {
      std::cerr << "ERROR: Bad input data (too many fields)" << std::endl;
      return false;
    }

    truesPresent = true;

  } else {
    truesPresent = false;
  }
  
  return true;
  
}

//-------------------------------------------------
// Skip past comments/ blank lines in a file.
void annz_util::skip_comments(std::ifstream& fstr)
{
  // Read the next non-whitespace character on the stream.
  char a;
  fstr >> a;
  
  // Check whether it's a commenting mark.
  if (a == '#') {
    // Skip to the end of the line.
    //fstr.ignore(numeric_limits<streamsize>::max(), '\n'); // Seems to be incompatible with gcc v2.x
    fstr.ignore(2147483647, '\n');
    // Recurse.
    skip_comments(fstr);

  } else {
    // Step back to the start of the line.
    fstr.unget();
  }
}

// Locate the next non-comment and non-blank line from the given file stream.
bool annz_util::get_next_dataline(std::ifstream& file, std::string& line) {
  
  //int c;
  //c = file.get();
  //std::cerr << c << " ";
  skip_comments(file);
  
  // Return fail if end of file.
  if (file.fail()) return false;
  // if (c == -1) return false;

  // Put the next line from the file to the string 'line'
  getline(file, line);
  //std::cerr << line << " ";

  return true;
}

// Extract the flag and data from the given line. Updates the flag (N_INPUT etc) and data as well.
int annz_util::parse_param_line(const std::string& buffer, std::string& flag, std::vector<std::string>& data) {
  
  std::stringstream ss(buffer, std::stringstream::in);
  
  ss >> flag;    // First item is flag.
  ss >> std::ws; // extract white space
  std::string temp;
  data.clear();
  while (!ss.eof()) {
    ss >> temp;
    data.push_back(temp);
    ss >> std::ws;
  }
}

// Read int from string.
int annz_util::string_to_int(const std::string s) {
  int i = 0;
  std::stringstream ss(s, std::stringstream::in);
  ss >> i;
  return i;
}

// Read double from string.
double annz_util::string_to_double(const std::string s) {
  double d = 0.0;
  std::stringstream ss(s, std::stringstream::in);
  ss >> d;
  return d;
}
