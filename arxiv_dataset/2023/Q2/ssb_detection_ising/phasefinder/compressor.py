# https://stackoverflow.com/questions/5940416/compress-a-series-of-1s-and-0s-into-the-shortest-possible-ascii-string

# A set of 64 characters, which allows a maximum chunk length of 6 .. because
# int('111111', 2) == 63 (plus zero)
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'

def encode(bin_string):
	# Split the string of 1s and 0s into lengths of 6.
	chunks = [bin_string[i:i+6] for i in range(0, len(bin_string), 6)]
	# Store the length of the last chunk so that we can add that as the last bit
	# of data so that we know how much to pad the last chunk when decoding.
	last_chunk_length = len(chunks[-1])
	# Convert each chunk from binary into a decimal
	decimals = [int(chunk, 2) for chunk in chunks]
	# Add the length of our last chunk to our list of decimals.
	decimals.append(last_chunk_length)
	# Produce an ascii string by using each decimal as an index of our charset.
	ascii_string = ''.join([charset[i] for i in decimals])

	return ascii_string

def decode(ascii_string):
	# Convert each character to a decimal using its index in the charset.
	decimals = [charset.index(char) for char in ascii_string]
	# Take last decimal which is the final chunk length, and the second to last
	# decimal which is the final chunk, and keep them for later to be padded
	# appropriately and appended.
	last_chunk_length, last_decimal = decimals.pop(-1), decimals.pop(-1)
	# Take each decimal, convert it to a binary string (removing the 0b from the
	# beginning, and pad it to 6 digits long.
	bin_string = ''.join([bin(decimal)[2:].zfill(6) for decimal in decimals])
	# Add the last decimal converted to binary padded to the appropriate length
	bin_string += bin(last_decimal)[2:].zfill(last_chunk_length)

	return bin_string
