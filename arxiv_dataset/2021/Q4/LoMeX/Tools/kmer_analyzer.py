import sys

nuc_prio = {'C' : 0, 'A' : 1, 'T' : 2, 'G' : 3}
nuc_comp = {'C' : 'G', 'A' : 'T', 'T' : 'A', 'G' : 'C'}


def clean_line(line):
	if line:
		return line.split(" ")[0].strip()
	else:
		return ""

def orient(kmer):
	if len(kmer) == 0:
		return ""
	complement = ""
	for nuc in kmer:
		if nuc not in nuc_prio:
			return ""
		complement = nuc_comp[nuc] + complement
	for i in range(len(kmer)):
		if nuc_prio[kmer[i]] == nuc_prio[complement[i]]:
			continue
		if nuc_prio[kmer[i]] > nuc_prio[complement[i]]:
			return kmer
		if nuc_prio[kmer[i]] < nuc_prio[complement[i]]:
			return complement
	return kmer


# THIS IS THE MOST RECENT COMPARATOR. GIVE K-MERS IN ALPHABETICAL ORDER!
def compare(real_path, extracted_path, k):
	real = 0
	extracted = 0
	rex = 0
	lastreal = ""
	lastex = ""
	exdup = 0
	realdup = 0
	with open(real_path, 'r') as rfile, open(extracted_path, 'r') as xfile:
		curreal = clean_line(rfile.readline())
		curex = clean_line(xfile.readline())
		if len(curreal) == k:
			real += 1
		if len(curex) == k:
			extracted += 1
		while True:
			if len(curreal) == 0 and len(curex) == 0:
				break
			# CASE 1
			if curreal == curex:
				rex += 1 # same k-mer
				lastreal = curreal # save last real
				lastex = curex # save last extracted
				curreal = clean_line(rfile.readline()) # read new real
				curex = clean_line(xfile.readline()) # read new extracted
				# examine real
				if  lastreal != curreal:
					if len(curreal) == k:
						real += 1
				else:
					realdup += 1
				# examine extracted
				if  lastex != curex:
					if len(curex) == k:
						extracted += 1
				else:
					exdup += 1
			# CASE 2
			elif (len(curreal) == k and curreal < curex) or len(curex) == 0:
				lastreal = curreal # save last real
				curreal = clean_line(rfile.readline())
				# examine real
				if  lastreal != curreal:
					if len(curreal) == k:
						real += 1
				else:
					realdup += 1
			# CASE 3
			elif (len(curex) == k and curex < curreal) or len(curreal) == 0:
				lastex = curex # save last extracted
				curex = clean_line(xfile.readline())
				if  lastex != curex:
					if len(curex) == k:
						extracted += 1
				else:
					exdup += 1

	precision = rex/extracted
	recall = rex/real

	if precision+recall != 0:
		f1 = (2*precision*recall)/(precision+recall)
	else:
		f1 = 0

	print("Real k-mers:", real)
	print("Real k-mer duplicates:", realdup)
	print("Extracted k-mers:", extracted)
	print("Extracted k-mer duplicates:", exdup)
	print("Shared k-mers:", rex)
	print("Precision:", precision)
	print("Recall:",recall)
	print("F1 score:",f1)


def main(real_path, extracted_path, k):
	print("Analyzing k-mers")
	compare(real_path, extracted_path, int(k))

if __name__ == "__main__":
   main(sys.argv[1], sys.argv[2], sys.argv[3])

	
