VIEW_URL = "https://portal.gdc.cancer.gov/files/"

SITES_DICT = {
    "Brain": "brain",
    "Breast": "breast",
    "Bronchus and lung": "lung",
    "Colon": "colon",
    "Liver and intrahepatic bile ducts": "liver",
}

DIAGNOSES_DICT= {
    "Brain Lower Grade Glioma": "LGG",
    "Glioblastoma Multiforme": "GBM",
    "Breast Invasive Carcinoma": "BRCA",
    "Lung Adenocarcinoma": "LUAD",
    "Lung Squamous Cell Carcinoma": "LUSC",
    "Colon Adenocarcinoma": "COAD",
    "Liver Hepatocellular Carcinoma": "LIHC",
    "Cholangiocarcinoma": "CHOL",
}

SITES_DIAGNOSES_DICT = {
    "brain": ["LGG", "GBM"],
    "breast": ["BRCA"],
    "lung": ["LUAD", "LUSC"],
    "colon": ["COAD"],
    "liver": ["LIHC", "CHOL"],
}