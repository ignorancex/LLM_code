#!/usr/bin/env python

import yaml
from Bio import SeqIO
from Bio import SwissProt
import os
import os.path
from os import path
import gzip
from tqdm.notebook import tqdm
from Bio import Entrez
from Bio.Entrez import efetch
import pdb

import CC_subsection_extractor
import subsection
import parsing_pubmed_ids
import extracting_abstracts



CC_subsection_extractor.main()
subsection.main()
#subsection.removing_parentheses()
#subsection.removing_dots()
#subsection.removing_spaces()
parsing_pubmed_ids.main()
extracting_abstracts.main()
