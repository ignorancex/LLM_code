import cdspyreadme

tablemaker = cdspyreadme.CDSTablesMaker()

# add a table
table = tablemaker.addTable("GCS_MRT.dat", description="Classification of good CSCv2 sample")
table = tablemaker.addTable("TD_MRT.dat", description="Classification of training dataset")
table = tablemaker.addTable("HESS_MRT.dat", description="Classification of HESS field sources")
# write table in CDS-ASCII aligned format (required)
tablemaker.writeCDSTables()

# Customize ReadMe output
tablemaker.catalogue = "J/ApJ/941/104       Classification of Chandra sources"
tablemaker.title = "Classifying Unidentified X-Ray Sources in the Chandra Source Catalog Using a Multiwavelength Machine-learning Approach."
tablemaker.author = 'Yang+'
tablemaker.authors= 'Yang H., Hare J., Kargaltsev O., Volkov I., Chen S., Rangelov B.'
tablemaker.date = 2022
tablemaker.bibcode = '<Astrophys. J. 941, 104 (2022)> =2022ApJ...941..104Y        (SIMBAD/NED BibCode)'
# ADC_Keywords: Active gal. nuclei ; Binaries, X-ray ; Binaries, cataclysmic ; X-ray sources
tablemaker.keywords = 'Catalogs - X-ray sources - Classification - Random Forests - X-ray binary stars - Active galactic nuclei - X-ray stars - Young stellar objects - Cataclysmic variable stars - Astrostatistics tools - X-ray surveys - Compact objects'
tablemaker.abstract = "The rapid increase in serendipitous X-ray source detections requires the development of novel approaches to efficiently explore the nature of X-ray sources. If even a fraction of these sources could be reliably classified, it would enable population studies for various astrophysical source types on a much larger scale than currently possible. Classification of large numbers of sources from multiple classes characterized by multiple properties (features) must be done automatically and supervised machine learning (ML) seems to provide the only feasible approach. We perform classification of Chandra Source Catalog version 2.0 (CSCv2) sources to explore the potential of the ML approach and identify various biases, limitations, and bottlenecks that present themselves in these kinds of studies. We establish the framework and present a flexible and expandable Python pipeline, which can be used and improved by others. We also release the training data set of 2941 X-ray sources with confidently established classes. In addition to providing probabilistic classifications of 66,369 CSCv2 sources (21% of the entire CSCv2 catalog), we perform several narrower-focused case studies (high-mass X-ray binary candidates and X-ray sources within the extent of the H.E.S.S. TeV sources) to demonstrate some possible applications of our ML approach. We also discuss future possible modifications of the presented pipeline, which are expected to lead to substantial improvements in classification confidences."
tablemaker.more_description = "The following tables present the properties and classification results of the good CSCv2 sample (GCS), the training dataset (TD), and the X-ray sources within the unidentified HESS sources using the multiwavelength machine-learning method (MUWCLASS)."
# tablemaker.putRef("II/246", "2mass catalogue")
# tablemaker.putRef("http://...", "external link")

# Print ReadMe
tablemaker.makeReadMe()

# Save ReadMe into a file
with open("ReadMe", "w") as fd:
    tablemaker.makeReadMe(out=fd)
