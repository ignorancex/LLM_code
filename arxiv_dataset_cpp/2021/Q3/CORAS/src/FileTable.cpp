#include "FileTable.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <limits>

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

FileTable::FileTable(const std::string &fileName, const std::size_t columnNumber, const char delimiterCharacter, const char commentCharacter, const std::size_t skippedColumnNumber)
	: ColumnNumber(columnNumber),
	  TableValues(columnNumber)
{
	read_values_from_file(fileName, delimiterCharacter, commentCharacter, skippedColumnNumber);
}

std::vector<std::string> &FileTable::column(const std::size_t columnIndex)
{
	return TableValues[columnIndex];
}

const std::vector<std::string> &FileTable::column(const std::size_t columnIndex) const
{
	return TableValues[columnIndex];
}

template std::vector<double> FileTable::column(const std::size_t columnIndex) const; // explicitly instantiate the method template defined in FileTable.tpp for the types used in the code
template std::vector<std::size_t> FileTable::column(const std::size_t columnIndex) const;
template std::vector<std::string> FileTable::column(const std::size_t columnIndex) const;

std::size_t FileTable::number_of_columns() const
{
	return ColumnNumber;
}

std::size_t FileTable::number_of_rows() const
{
	return TableValues[0].size();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

void FileTable::read_values_from_file(const std::string &fileName, const char delimiterCharacter, const char commentCharacter, const std::size_t skippedColumnNumber)
{
	std::ifstream file(fileName);

	if (file.fail())
	{
		std::cout << std::endl
				  << " FileTable::read_values_from_file: File \"" << fileName << "\" can not be opened" << std::endl
				  << std::endl;

		exit(EXIT_FAILURE);
	}

	auto ignoreColumn = [&](std::istringstream &lineStream)
	{
		lineStream.ignore(std::numeric_limits<std::streamsize>::max(), delimiterCharacter);
	};

	auto parseColumn = (delimiterCharacter == ' ') ? std::function<std::string(std::istringstream &)>([](std::istringstream &lineStream)
																									  {
																										  std::string columnEntry;

																										  lineStream >> columnEntry;

																										  return columnEntry;
																									  })
												   : std::function<std::string(std::istringstream &)>([&](std::istringstream &lineStream)
																									  {
																										  std::string columnEntry;

																										  std::getline(lineStream, columnEntry, delimiterCharacter);

																										  return columnEntry;
																									  });

	std::string line;

	while (std::getline(file, line)) // read in the file line by line
	{
		if (line.front() != commentCharacter) // ignore line if it starts with the comment character
		{
			std::istringstream lineStream(line);

			for (std::size_t i_c = 0; i_c < skippedColumnNumber; ++i_c) // ignore values of first skippedColumnNumber columns
			{
				ignoreColumn(lineStream);
			}

			for (std::size_t i_c = 0; i_c < ColumnNumber; ++i_c) // extract values column by column and write them to the array TableValues
			{
				TableValues[i_c].push_back(parseColumn(lineStream));
			}
		}
	}

	file.close();
}