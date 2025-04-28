all_names= { 
#     possible inputs for variable `delta`
    "kol": ["0.3333333333333333", "1/3", "kol", "KOL", "Kol", "0.3", "0.33", "kolmogorov", "Kolmogorov", 
            "KOLMOGOROV", "third", "onethird"],
    "kra": ["1/2", "kra", "KRA", "kraichnan", "Kraichnan", "KRAICHNAN", "0.5", "half", "one half", "onehalf"],
    "sx": ["sx", "0.16666666666666666", "1/6", "sixth", "one sixth", "onesixth", "0.17"],
    "twth": ["twth", "2/3", "0.6666666666666666", "twothirds", "two thirds", "0.67", "0.6", "2by3"],
#     possible inputs for variable `hyp`
    "wimp": ['dm_ann', 'dmann', 'dm ann', 'ann', 'annihilation', 'dark matter annihilation', 'wimp', 'wimp ann', 'wimp_ann', 'wimps'],
    "decay": ['decaying', 'dm_dec', 'dmdec', 'dm dec', 'dec', 'decay', 'dark matter decay', 'decaying dark matter', 'decaying dm'],
    "generic": ['gen', 'generic', 'astrophysics', 'astro', 'non-dm', 'pw', 'pow', 'pwlaw', 'pw_law', 'pow law', 'powlaw', 'power law', 'generic'],
#     possible inputs for variable `galaxy`
   'Bootes I' : ['Bootes I' , 'Boo I', 'Bootes 1' , 'Boo 1', 'BootesI' , 'BooI', 'Bootes1' , 'Boo1'],
 'Canes Venatici I' : ['Canes Venatici I','Canes Venatici 1', 'CVn I', 'CVnI', 'CVn1', 'CVn 1'],
 'Canes Venatici II' : ['Canes Venatici II','Canes Venatici 2', 'CVn II', 'CVnII', 'CVn2', 'CVn 2'],
 'Carina' : ['Carina', 'Car'],
 'Coma Berenices' : ['Coma Berenices', 'Coma', 'Berenices', 'Com', 'ComB' ], 
 'Draco' : ['Draco', 'Dra', 'Drc'],
 'Fornax': ['Fornax', 'For', 'Frnx'],
 'Hercules': ['Hercules', 'Her', 'Hrcls', 'Hrc'],
 'Leo I' : ['Leo I', 'Leo 1', 'LeoI', 'Leo1'],
 'Leo II' : ['Leo II', 'Leo 2', 'LeoII', 'Leo2'],
 'Leo IV' : ['Leo IV', 'Leo 4', 'LeoIV', 'Leo4'],
 'Leo T' : ['Leo T', 'LeoT'],
 'Leo V' : ['Leo V', 'Leo 5', 'LeoV', 'Leo5'] ,
 'Sculptor' : ['Sculptor', 'Scl', 'Sclptr'],
 'Segue 1' : ['Segue 1', 'Segue I', 'Segue1', 'SegueI', 'Seg I', 'Seg 1', 'SegI', 'Seg1'],
 'Segue 2' : ['Segue 2', 'Segue II', 'Segue2', 'SegueII', 'Seg II', 'Seg 2', 'SegII', 'Seg2'],
 'Sextans' : ['Sextans', 'Sex', 'Sxtns'],
 'Ursa Major I' : ['Ursa Major I', 'Ursa Major 1', 'UMa I', 'UMa 1', 'UMaI', 'UMa1'],
 'Ursa Major II' : ['Ursa Major II', 'Ursa Major 2', 'UMa II', 'UMa 2', 'UMaII', 'UMa2'],
 'Willman I' : ['Willman I', 'Will I', 'Wil I', 'Willman 1', 'Will 1', 'Wil 1', 'WillmanI', 'WillI', 'WilI', 'Willman1', 'Will1', 'Wil1',],
 'Ursa Minor': ['Ursa Minor', 'UMi'],
#     possible inputs for variable `ref`
    '1309.2641' : ['1309.2641', 'Martinez', 'Martinez 2013', 'Mar2013', 'Mar13',],
    '1408.0002' : ['1408.0002', 'Geringer-Sameth et al', 'Geringer-Sameth 2014', 'Geringer-Sameth 2015', 'GS 2014', 'GS14', 'GS15',],
    '1706.05481' : ['1706.05481', 'Ichikawa et al 2017', 'Ichikawa 2017', 'Kashiwa 2017', 'Ich 2017', 'Ich2017'],
    '1608.01749 I' : ['1608.01749 I', '1608.01749 Draco I', 'Ichikawa-I 2016', 'Kashiwa 2016 I', 'Ich 2016 I', 'Ich 16 I'],
    '1608.01749 II' : ['1608.01749 II', '1608.01749 Draco II', 'Ichikawa-II 2016', 'Kashiwa 2016 II', 'Ich 2016 II', 'Ich 16 II'],
#     possible inputs for variable `rad_temp`
    'HDZ' : ['HDZ', 'hdz', 'Hernquist', 'Diemand', 'Zhao', 'gen NFW', 'gNFW', 'genNFW'],
    'Enst' : ['Enst', 'Einasto'],
    'NFW' : ['NFW', 'Navarro', 'Frenk', 'White', 'Navarro-Frenk-White', 'nfw'],
    'cNFW' : ['cored NFW', 'cNFW', 'CNFW'],
    'Bkrt' : ['Bkrt', 'Burkert'],
    'sis' : ['sis', 'SIS', 'singular isothermal sphere', 'isothermal sphere'],
    'ps' : ['ps', 'point', 'point source', 'point-source'],
    'const' : ['c', 'const', 'constant density', 'constant'],
    'plmm' : ['plmm', 'Plummer'],
    'ps_iso' : ['ps_iso', 'pseudo isothermal sphere']
}

delta_to_float = {
    "sx": 1/6, 
    "kol": 1/3, 
    "kra": 1/2, 
    "twth": 2/3
}

#  #######################################################################

ch_to_grid = {
    "ee_MC": "lmgen1",
    "mumu": "lmgen1",
    "tautau": "lmgen1",
    "qq": "lmgen",
    "cc": "lmgen",
    "bb": "lmgen1",
    "tt": "lmtt",
    "WW": "lmWW",
    "ZZ": "lmZZ",
    "hh": "lmhh",
    "nunu": "lmhh"
}