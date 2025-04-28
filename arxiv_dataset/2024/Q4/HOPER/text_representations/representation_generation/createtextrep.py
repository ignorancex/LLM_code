'''
This script provides a convenient way to create different types of text representations by specifying the desired representation techniques and the paths to the input files via command-line arguments.
Creating TFIDF representations: If the tfidf or all flags are set, the script sets the file paths for the create_tfidf module and calls its main() function to create TFIDF representations.
Creating biosentvec representations: If the biosentvec or all flags are set, the script sets the file paths for the create_biosentvec module and calls its main() function to create biosentvec representations.
Creating biowordvec representations: If the biowordvec or all flags are set, the script sets the file paths for the create_biowordvec module and calls its main() function to create biowordvec representations.
'''

import argparse
import os
import create_tfidf as tf
import create_biosentvec as bs
import create_biowordvec as bw

parser = argparse.ArgumentParser(description='Create text representations')
parser.add_argument("-tfidf","--tfidf", action='store_true', help="Create TFIDF representations")
parser.add_argument("-bsv", "--biosentvec", action='store_true',  help="Create biosentvec representations")
parser.add_argument("-bwv", "--biowordvec", action='store_true',  help="Create biowordvec representations")
parser.add_argument("-upfp", "--uniprotfilespath", required=True,  help="Path for the uniprot files")
parser.add_argument("-pmfp", "--pubmedfilespath", required=True,  help="Path for the pubmed files")
parser.add_argument("-mdw", "--model_download", help="Download biosentvec and biowordvec pre-trained models automatically")
parser.add_argument("-a", "--all", action='store_true',  help="Create TFIDF, biosentvec and biowordvec representations")


try:
    args = parser.parse_args()
    if not (args.uniprotfilespath or args.pubmedfilespath):
            parser.error('At least one path should be specified!')
except:
    parser.print_help()

print(args)

if args.tfidf or args.biosentvec or args.biowordvec or args.all:
    print("Loading files... \n\n")
    
 
if args.tfidf or args.all:
    print("\n\n Creating tfidf representations...\n")
    tf.ufiles_path = args.uniprotfilespath
    tf.pfiles_path = args.pubmedfilespath
    tf.main()
      
if args.biosentvec or args.all:
    if args.model_download == "y":
        print("\n\nDownloading biosentvec model...\n")
        os.system("wget -P " + os.path.join(os.getcwd(),'text_representations/representation_generation/models') + " https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin")
    
    print("\n\nCreating biosentvec representations...\n")
    bs.ufiles_path = args.uniprotfilespath
    bs.pfiles_path = args.pubmedfilespath
    bs.main()

if args.biowordvec or args.all:
    if args.model_download == "y":
        print("\n\nDownloading biowordvec model...\n")
        os.system("wget -P " + os.path.join(os.getcwd(),'text_representations/representation_generation/models') + " https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin")
    
    print("\n\nCreating biowordvec representations...\n")   
    bw.ufiles_path = args.uniprotfilespath
    bw.pfiles_path = args.pubmedfilespath
    bw.main()
