import numpy as np

Simbad_dict = {'AGN':['QSO','AGN','Galaxy','Seyfert_1','Seyfert_2','EmG','RadioG','BLLac','LINER',\
                      'StarburstG','Seyfert','Blazar','PartofG','HII_G',\
                     'LensedG','LSB_G','LensedQ','IG','GinPair','GinGroup','GinCl','BClG','GroupG'], 
           'YSO':['YSO','TTau*','Orion_V*','Ae*','HH','Y*O','Or*','TT*'], #'YSO_Candidate','Ae*_Candidate'

          'STAR':['Star','sg*','BlueStraggler',\
                  'V*','Pec*','RotV*','Em*','Irregular_V*','PulsV*','PM*','EB*','LPV*','Eruptive*',\
                  'HotSubdwarf','s*r','s*y','RedSG*','RGB*','HB*','Cepheid','Mira','Symbiotic*',\
                  'RRLyr','WD*','RotV*alf2CVn','deltaCep','PulsV*delSct','Erupt*RCrB',\
                  'EllipVar','SB*','BYDra','RSCVn','RotV*','**'\
                  ],
        'HM-STAR':['Ma*','bC*','s*b','WR*','BlueSG*','Be*','PulsV*bCep'],
        'LM-STAR':['low-mass*','brownD*'],
        'CV':['CataclyV*','CV*','XB','XRB','Nova'],#'CV*_Candidate'
        'NS':['Neutron*','Pulsar','N*','Psr'], 
        'NS_BIN':['Neutron*','Pulsar','N*','Psr'],
        'HMXB':['HMXB','HXB'], #'HMXB_Candidate'
        'LMXB':['LMXB','LXB'],
        'XRB':['XB','XRB'],#'XB*_Candidate',
        'Common':['X','IR','NIR','Radio','MIR','UV','gamma','Radio(cm)','EmObj'],
        'Uncla': ['Planet','EB*_Candidate','Cl*','Unknown_Candidate','GlCl','ULX?_Candidate','HII','PN',\
                'LensSystem_Candidate','Maser','SNR','SN','BH_Candidate','WD*_Candidate',\
                'MolCld','LP*_Candidate','denseCore','GravLensSystem',\
                'RGB*_Candidate','LensedImage','MolCld','Cloud','ClG','GlCl?_Candidate','PartofCloud','Transient','EllipVar',\
                'V*?_Candidate','Compact_Gr_G', 'ClG_Candidate','AGN_Candidate','QSO_Candidate','YSO_Candidate','Ae*_Candidate',\
                'CV*_Candidate','HMXB_Candidate'],
        'Unclear':['EB*_Candidate','Cl*','Unknown_Candidate','GlCl','ULX?_Candidate','HII',\
                'LensSystem_Candidate','Maser','BH_Candidate','WD*_Candidate',\
                'MolCld','LP*_Candidate','denseCore','GravLensSystem',\
                'RGB*_Candidate','LensedImage','MolCld','Cloud','ClG','GlCl?_Candidate','PartofCloud','Transient',\
                'V*?_Candidate','Compact_Gr_G', 'ClG_Candidate','AGN_Candidate','QSO_Candidate','YSO_Candidate','Ae*_Candidate',\
                'CV*_Candidate','HMXB_Candidate'],
       'others':['Planet','PN','SNR','SN'],
       'rare-type': ['CV','CataclyV*','CV*','Neutron*','Pulsar','N*','Psr','XB','XRB','HMXB','LMXB','HXB','LXB','NS','NS_BIN']}




crowd_fields_dict = {'NGC_55':      {'ra':3.723333,                 'dec':-39.196667,             'r':12.05*180/(2*2.1e3*np.pi)},
                'IC_10':       {'ra':(20+17.3/60)/4,           'dec':(59+(18+14/60)/60), 'r':4./60}, 
                'Haro_11':     {'ra':(36+52.7/60)/4,           'dec':-(33+(33+17.2/60)/60), 'r':6./3600}, 
                'M_31':        {'ra':10.68458,                 'dec':41.26916,              'r':1.},
                'SMC':         {'ra':13.1867,                  'dec':-72.8286,              'r':5.33},
                'NGC_300':     {'ra':13.7229,                  'dec':-37.6844,              'r':0.17}, 
                'IC_1623':     {'ra':(1+(7+47.2/60)/60)*15,    'dec':-(17+(30+25/60)/60),   'r':20./3600}, 
                'NGC_404':     {'ra':17.362500,                'dec':35.718056,             'r':3.25*180/(2*3.05e3*np.pi)},
                'M_33':        {'ra':(1+(33+50.9/60)/60)*15,    'dec':(30+(39+36.6/60)/60), 'r':71./(2*60)},
                'Maffei_1':    {'ra':(2+(36+35.47/60)/60)*15,  'dec':(59+(39+17.51/60)/60), 'r':1.7/60}, 
                'NGC_1333':    {'ra':52.2971,                  'dec':31.31,                 'r':6./60},
                'IC_348':      {'ra':56.13833,                 'dec':32.16333,              'r':42./60},
                'LMC':         {'ra':80.89375,                 'dec':-69.75611,             'r':10.75},
                'NGC_2264':    {'ra':100.25,                   'dec':9.8833,                'r':45./60},
                'NGC_2403':    {'ra':114.214167,               'dec':65.602500,             'r':19.43*180/(2*3.3e3*np.pi)},
                'M_81':        {'ra':(9+(55+33.173/60)/60)*15, 'dec':69+(3+55.06/60)/60,    'r':10./60},
                'NGC_3115':    {'ra':(10+(5+14./60)/60)*15,    'dec':-(7+(43+7./60)/60),    'r':3.6/60},
                'NGC_3379':    {'ra':(10+(47+49.6/60)/60)*15,  'dec':12+(34+53.87/60)/60,   'r':2.7/60},
                'NGC_4214':    {'ra':183.913333,               'dec':36.326944 ,            'r':7.05*180/(2*2.9e3*np.pi)},   
                'NGC_4278':    {'ra':(12+(20+6.8/60)/60)*15,   'dec':29+(16+51/60)/60,      'r':2.1/60},
                'NGC_4697':    {'ra':(12+(48+35.9/60)/60)*15,  'dec':-(5+(48+2.5/60)/60),   'r':2.2/60},
                'Cen_A':       {'ra':(13+(25+27.604/60)/60)*15,'dec':-(41+(1+9.49/60)/60),  'r':10./60},
                'NGC_5457':    {'ra':(14+(3+12.583/60)/60)*15, 'dec':(54+(20+55.5/60)/60),  'r':10./60},
                'Circinus':    {'ra':213.29125,                'dec':-65.339167,            'r':6.9/60},
                'M_13':        {'ra':(16+(41+41.24/60)/60)*15, 'dec':36+(27+35.5/60)/60,    'r':10./60},
                'Westerlund_1':{'ra':251.76667,                'dec':-45.85136,             'r':3./60},     
                'Liller_1':    {'ra':263.3520,                 'dec':-33.3889,              'r':0.003}, 
                'NGC_6388':    {'ra':(17+(36+17.461/60)/60)*15,'dec':-(44+(44+8.34/60)/60), 'r':3.1/60},
                'NGC_6397':    {'ra':(17+(40+42.09/60)/60)*15, 'dec':-(53+(40+27.6/60)/60), 'r':16./60},
                'NGC_6440':    {'ra':(17+(48+52.7/60)/60)*15,  'dec':-(20+(21+36.9/60)/60), 'r':1.6/60},
                'M_28':        {'ra':(18+(24+32.89/60)/60)*15, 'dec':-(24+(52+11.4/60)/60), 'r':5.6/60},
                'NGC_6652':    {'ra':(18+(35+45.6/60)/60)*15,  'dec':-(32+(59+26.6/60)/60), 'r':2./60}, 
                'NGC_6791':    {'ra':290.22083,                'dec':37.771667,             'r':16./60}, 
                'M_27':        {'ra':299.901417,               'dec':22.721136,             'r':8./60}
               }

# why are they removed from the old TD?
# 2CXO J073751.2-303940 
# 2CXO J161736.2-510224
# 2CXO J122758.7-485343
# 2CXO J141730.5-440257
# 2CXO J170634.5+235818
# 2CXO J182943.9-095123

# sources removed from old TD

rare_sources_removed_dict = {'2CXO J083520.6-451034': 'Vela pulsar, complicated environment', 
                        '2CXO J090835.4-491305': 'PSR B0906-49, complicated environment',
                        '2CXO J163905.4-464212': 'AX J1639.0-4642, complicated environment',
                        '2CXO J184343.3-040805': 'PSR J1843-0408, complicated environment',
                        '2CXO J195258.2+325240': 'PSR B1951+32, complicated environment',
                        '2CXO J102347.6+003840': 'PSR J1023+0038, a LMXB/NS_BIN in complicated environment, 2005JAD....11....2D misclassified as a CV',
                        '2CXO J151355.6-590809': 'PSR B1509-58, complicated environment',
                        '2CXO J183333.5-103407': 'PSR J1833-1034, SNR complicated environment',
                        '2CXO J052728.2-124150': 'HD 35914, a CV from a PN (IC 418) environment',
                        '2CXO J182943.9-095123': 'HMXB classified by 2006A&A...455.1165L but SIMBAD classified as a pulsar, in crowded environment',#{'Class':'HMXB', 'Comment':'HMXB classified by 2006A&A...455.1165L but SIMBAD classified as a pulsar'},
                        '2CXO J174451.1-292116': 'A LMXB in a messy region, SIMBAD match to a different source 0.35" away while LXB is 34.21" away (center of a nebula), in Galactic Bulge, coordinates confirmed by 2007A&A...462.1065M',
                        '2CXO J174813.1-360758': '1A 1744-361: a Galactic LMXB in a crowded region',
                        '2CXO J174819.2-360716': '1A 1744-361: a Galactic LMXB in a crowded region, the inaccurate BAT coordinate',
                        '2CXO J170019.2-422019': 'AX J1700.2-4220, confused with 2CXO J170025.2-421900, ambiguious source',
                        '2CXO J170025.2-421900': 'AX J1700.2-4220, ambiguious source',
                        '2CXO J061801.4+222228': 'CV candidate, see https://iopscience.iop.org/article/10.3847/1538-3881/ac0efb/pdf',
                        '2CXO J110505.3-611044': 'Classified as CV based on variability from https://arxiv.org/pdf/1307.1238.pdf',
                        '2CXO J133833.0+292908': 'CV candidate, Cannot find where the class is coming from.',
                        '2CXO J141106.0+122252': 'CV candidate, do not see any strong H alpha emission features typical of CVs, but maybe it is identified through some other spectroscopic/timing features. See https://www.wis-tns.org/object/2018cjc.',
                        '2CXO J194129.2+401124': 'CV candidate, 2012ApJ...745...57G.',
                        '2CXO J155246.9-502953':{'CV candidate, from INTEGRAL'},
                        '2CXO J033108.1+435750':{'CV candidate, from 2005JAD....11....2D'},
                        '2CXO J183545.7-325923':{'CV candidate, from 2005JAD....11....2D'},
                        '2CXO J123907.9-453344':{'SIMBAD classified as EB*. 2003AA...404..301R classified as a DN with an Uncertainty flag raised. Unusual variable star?'},
                        '2CXO J132430.3-631349':{'A new LMXB classified by 2018ApJS..235....4O, SIMBAD identifies as an X-ray source, not too much info.'},
                        '2CXO J140846.0-610754':{'A new CV from 2018ApJS..235....4O, SIMBAD match to a star 0.28" away while a CV candidate is 2.6" away. IP? from 2016ApJ...816...38T, 2022MNRAS.511.4937S.'},
                        '2CXO J155748.3-542453':{'a new HMXB classified by 2018ApJS..235....4O (we use the counterpart coordinate while the X-ray coordinate is 2.8 arcmin. away), SIMBAD HXB, 2S 1553-542, is 6.2" away.2022ApJ...927..194M claims that no Gaia nor Chandra sources are found within 5 arcsec from this transient Be/X-ray binary.'},
                        '2CXO J161736.2-510224':{'LMXB from 2007A&A...469..807L, X from SIMBAD with 0.42" away, with 2E 3623 1.95" away, ambiguous classification.'},
                        '2CXO J162826.8-502239':{'Comment':'A new LMXB classified by 2018ApJS..235....4O, confirmed by SIMBAD, no individual study.'},
                        '2CXO J183228.3-075641':{'A new HMXB classified by 2018ApJS..235....4O, SIMBAD identifies as a Gamma-ray source, not too much info'},
                        '2CXO J182608.6-125631':{'Confused with 2CXO J182608.5-125634'},
                        '2CXO J181506.2-120545':{'Readout streakm, confused with 2CXO J181506.1-120548'},
                        '2CXO J181506.1-120548':{'Readout streakm, confused with 2CXO J181506.2-120545'},
                        '2CXO J170004.3-415805':{'Classified as HMXB from simbad, could be an IP CV see 2010MNRAS.402.2388K'},
                        '2CXO J145252.7-594908':{'Classified as HMXB from simbad, could be an IP CV see 2009MNRAS.394.1597K'},
                        '2CXO J140421.8+541921':{'This is controversial. Could be either in another galaxy (ULX?) or a flare in galactic Be star.'},
                        '2CXO J043715.9-471509X':{'Confused with 2CXO J043715.8-471508.'},
                        '2CXO J053854.5+261855X':{'Confused with 2CXO J053854.5+261856.'},
                        '2CXO J063354.3+174614':{'Confused with 2CXO J063354.2+174613.'},
                        '2CXO J063354.2+174616':{'Confused with 2CXO J063354.2+174613.'},
                        '2CXO J105007.5-595321':{'Confused with 2CXO J105007.1-595321.'},
                        '2CXO J112115.1-603725X':{'Confused with 2CXO J112115.1-603725.'},
                        '2CXO J113106.9-625648X':{'Confused with 2CXO J113106.9-625648.'},
                        '2CXO J130848.1+212707':{'Confused with 2CXO J130848.2+212706.'},
                        '2CXO J140045.7-632542X':{'Confused with 2CXO J140045.7-632542.'},
                        '2CXO J141842.9-605804':{'Confused with 2CXO J141842.6-605802.'},
                        '2CXO J142008.4-604815':{'Confused with 2CXO J142008.1-604817.'},
                        '2CXO J155058.3-562835X':{'Confused with 2CXO J155058.6-562835.'},
                        '2CXO J161243.1-522523X':{'Confused with 2CXO J161243.0-522523.'},
                        '2CXO J162046.2-513004X':{'Confused with 2CXO J162046.2-513006.'},
                        '2CXO J170249.4-484723X':{'Confused with 2CXO J170249.3-484723.'},
                        '2CXO J171810.0-371853':{'Confused with 2CXO J171809.8-371851.'},
                        '2CXO J171935.9-410054X':{'Confused with 2CXO J171935.8-410053.'},
                        '2CXO J174354.8-294443':{'Unclear HMXB or LMXB, Confused with 2CXO J174354.9-294444X.'},
                        '2CXO J174354.9-294444X':{'Unclear HMXB or LMXB, Confused with 2CXO J174354.8-294443.'},
                        '2CXO J184624.9-025828X':{'Confused with 2CXO J184624.9-025830.'},
                        '2CXO J191404.2+095258X':{'Confused with 2CXO J191404.2+095258.'},
                        '2CXO J193029.9+185213X':{'Confused with 2CXO J193030.1+185214.'},
                        '2CXO J202105.4+365104X':{'Confused with 2CXO J202105.4+365104.'},
                        '2CXO J203225.7+405728':{'Extremely bright, excluded. Confused with 2CXO J203225.8+405727X.'},
                        '2CXO J203225.8+405727X':{'Extremely bright, excluded. Confused with 2CXO J203225.7+405728.'},
                        '2CXO J214441.1+381917':{'Extremely bright, excluded. Confused with 2CXO J214441.2+381916X.'},
                        '2CXO J214441.2+381916X':{'Extremely bright, excluded. Confused with 2CXO J214441.1+381917.'},
                        '2CXO J003719.6-721413':{'A CV candidate, SIMBAD classified as XB, in SMC'},
                        '2CXO J141329.9-620534':{'Confused with 2CXO J141330.2-620535'},
                         
                        
}



# rare-type sources saving

# from  TD_rare_multi_comment_good.csv

#print(TD_simbad.loc[TD_simbad.name_cat.isin(['B1259-63','J1124-3653','J2032+4127','XTE J1858+034','J0737-3039B','B1534+12']), ['name_cat','Class']])


rare_sources_saving_dict = {# Reclassification
                       '2CXO J073751.2-303940':{'Class':'NS', 'Comment':'change from NS_BIN to NS since it is a double pulsar system'},
                       '2CXO J153709.9+115555':{'Class':'NS', 'Comment':'change from NS_BIN to NS since it is a double pulsar system'},
                       '2CXO J112401.1-365319':{'Class':'NS_BIN', 'Comment':'change from NS to NS_BIN as a black widow pulsar'},
                       '2CXO J185843.6+032606':{'Class':'LMXB', 'Comment':'2021ApJ...909..154T'},
                       '2CXO J130247.6-635008':{'Class':'HMXB', 'Comment':'recorded in the ATNF catalog'},
                       '2CXO J203213.1+412724':{'Class':'HMXB', 'Comment':'recorded in the ATNF catalog, SIMBAD classified as a Be*'},
                       '2CXO J124850.7-412654':{'Class':'YSO', 'Comment': 'A CV candidate from Open CV catalog, but Spectra taken by two independent groups classify this source as a Classic T Tauri Type YSO.'},
    
                        # sources with Ambiguous classifications 
                       '2CXO J122637.5-624613':{'Class':'HMXB', 'Comment':'https://www.aanda.org/articles/aa/full_html/2019/09/aa36045-19/aa36045-19.html'},
                       '2CXO J163553.8-472540':{'Class':'HMXB', 'Comment':'https://www.aanda.org/articles/aa/pdf/2008/24/aa8774-07.pdf'},
                       '2CXO J180438.9-145647':{'Class':'CV', 'Comment':'https://arxiv.org/pdf/1108.1105.pdf'},
                       '2CXO J230108.2+585244':{'Class':'NS', 'Comment':'Magnetar'},
                       '2CXO J182608.5-125634':{'Class':'NS', 'Comment':' It is an isolated pulsar with compact faint PWN. see 2005AJ....129.1993M, 2019MNRAS.487.1964K, 2015ApJ...814..128K'},
    
    
                       # good sources from Galactic Bulge
                       '2CXO J174445.7-271344':{'Class':'HMXB', 'Comment':'Gamma Cas analog, in Galactic Bulge'},
                       '2CXO J180632.1-221417':{'Class':'LMXB', 'Comment':'in Galactic Bulge, Updated Coordinates here: https://www.astronomerstelegram.org/?read=3218'},
                       '2CXO J173527.5-325554':{'Class':'HMXB', 'Comment':'2019MNRAS.485..286G, SIMBAD classified as HMXB_Candidate, in Galactic Bulge'},
                       '2CXO J174621.1-284343':{'Class':'LMXB', 'Comment':'SIMBAD match to a different source 0.75" away while LXB is 1.64" away, in Galactic Bulge'}, 
                       '2CXO J174931.7-280805':{'Class':'LMXB', 'Comment':'SIMBAD match to a different source 0.1" away while LXB is 2.9" away, in Galactic Bulge'},
                       '2CXO J173413.4-260518':{'Class':'LMXB', 'Comment':'in Galactic Bulge'},
                       '2CXO J174502.3-285449':{'Class':'LMXB', 'Comment':'in Galactic Bulge'},
                       '2CXO J174702.5-285259':{'Class':'LMXB', 'Comment':'in Galactic Bulge'},
                       '2CXO J181044.4-260901':{'Class':'LMXB', 'Comment':'in Galactic Bulge'},
                       '2CXO J173953.9-282946':{'Class':'LMXB', 'Comment':'in Galactic Bulge'},
                       '2CXO J181921.6-252425':{'Class':'LMXB', 'Comment':'Well known BH XRB, companion mass puts it somewhere between HMXB and LMXB, but given the counterpart is a BH, its probably best to place it in the LMXB class'},
                       
                        
                       # sources from the Galactic Bulge that are confused, but kept in previous version
                       ## '2CXO J174433.0-284426':{'Class':'LMXB', 'Comment':'SIMBAD match to a star 0.1" away with LMXB 0.4" away, in Galactic Bulge'},
                       ## '2CXO J175834.5-212321':{'Class':'HMXB', 'Comment':'in Galactic Bulge'},
                       ## '2CXO J174819.2-360716':{'Class':'LMXB', 'Comment':'in Galactic Bulge'},
                       ## '2CXO J174354.8-294443':{'Class':'LMXB', 'Comment':'SIMBAD match to a different source 0.05" away while LXB is 0.47" away, in Galactic Bulge'},
    
                       #  SIMBAD wrong info. 
                       '2CXO J020537.9+644941':{'Class':'NS', 'Comment':'SIMBAD match to another source while NS PSR J0205+6449 is a little far away with 1.36" sep'},
                       '2CXO J110526.2-610749':{'Class':'NS', 'Comment':'SIMBAD match to another source while NS PSR J1105-6107 is a little far away with 0.74" sep'},
                       '2CXO J113603.1+155111':{'Class':'NS', 'Comment':'SIMBAD match to a different source due to proper motion (2013MNRAS.435.2227Z) while PSR B1133+16 is 3.93" away'},
                       '2CXO J140045.7-632542':{'Class':'NS', 'Comment':'SIMBAD classified as SNR, confused with 2CXO J140045.7-632542X.'},
                       '2CXO J170846.8-400852':{'Class':'NS', 'Comment':'SIMBAD offset by 29.7" since it uses Gamma coordinate'},
                       '2CXO J182218.0-160425':{'Class':'NS', 'Comment':'SIMBAD offset by 4" since it uses XRT coordinate'},
                       '2CXO J185610.6+011322':{'Class':'NS', 'Comment':'SIMBAD classified as SNR'},
    
                       '2CXO J131145.7-343030':{'Class':'NS_BIN', 'Comment':'SIMBAD offset by 24" since it uses Gamma coordinate'},
                       '2CXO J195936.7+204814':{'Class':'NS_BIN', 'Comment':'SIMBAD match to another source while Psr is exact the same distance away(0.25")'},
                       '2CXO J141730.5-440257':{'Class':'NS_BIN', 'Comment':'SIMBAD misclassifed as NS, Redback? 2019MNRAS.483.4495D'}, 
    
                       '2CXO J030346.9+645435':{'Class':'CV', 'Comment':'SIMBAD classified as a PN'}, 
                       '2CXO J054320.3-410154':{'Class':'CV', 'Comment':'SIMBAD match to another source while CV is a little far away with 2.18" sep'},
                       '2CXO J141249.0-402136':{'Class':'CV', 'Comment':'SIMBAD classified as Erupt*RCrB'},
                       '2CXO J175013.1-064228':{'Class':'CV', 'Comment':'SIMBAD classified as Symbiotic*'}, 
                       '2CXO J190109.3-220005':{'Class':'CV', 'Comment':'SIMBAD offset by 2.2" since it uses XMM coordinate'},
                       '2CXO J223715.5+821027':{'Class':'CV', 'Comment':'SIMBAD classified as WD*, spectroscopically classified WD'},
    
                       '2CXO J042142.7+325426':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, while spectral type is M4.5V D'},
                       '2CXO J062244.5-002044':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, while spectral type is K3V-K7V C'}, 
                       '2CXO J111810.7+480212':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, while spectral type is K5V-M1V C'}, 
                       '2CXO J135809.6-644405':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB'}, 
                       '2CXO J155058.6-562835':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, while spectral type is K3III D, confused with 2CXO J155058.3-562835X.'}, 
                       '2CXO J165000.9-495744':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, while spectral type is K4V C'}, 
                       '2CXO J165400.1-395044':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, while spectral type is F5IV D'}, 
                       '2CXO J170249.3-484723':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, confused with 2CXO J170249.4-484723X.'},
                       '2CXO J185841.4+223929':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, while spectral type is G5V-K0V C'}, 
                       '2CXO J190853.0+092305':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB'},
                       '2CXO J195542.3+320549':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, while spectral type is M4/5III'}, 
                       '2CXO J202403.8+335201':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB, while spectral type is G9/K0III/V E'}, 
                       '2CXO J152040.8-571000':{'Class':'LMXB', 'Comment':'SIMBAD misclassified as HMXB'},
    
                       '2CXO J170206.5-295644':{'Class':'LMXB', 'Comment':'SIMBAD match to a wrong planet while LXB is exact the same distance away(0.26")'},
                       '2CXO J191116.0+003505':{'Class':'LMXB', 'Comment':'SIMBAD match to another source (0.42") while LMXB is a little far away with 0.51" sep'},
                       
                       '2CXO J114800.0-621224':{'Class':'HMXB', 'Comment':'SIMBAD classified as a Be*'},
    
                       #  literature wrong info. 
                       '2CXO J163201.8-475228':{'Class':'HMXB', 'Comment':'2018ApJS..235....4O misclassified as NS'},
                       
    
                      ##### new sources
                      
                      '2CXO J062153.2-271002':{'Class':'CV', 'Comment':'A new CV from the open CV catalog, confirmed by SIMBAD, Appears to have a spectroscopic classification, See https://www.wis-tns.org/object/2016hvt'},
                      '2CXO J065036.5+413053':{'Class':'CV', 'Comment':'A new CV from the open CV catalog, confirmed by SIMBAD, spectroscopically confirmed from https://arxiv.org/pdf/1407.3315.pdf'},
                      '2CXO J152351.2+083604':{'Class':'CV', 'Comment':'Classified as a Disk system by https://iopscience.iop.org/article/10.3847/1538-3881/ab6ded, which it seems to mean the source is a non-magnetic accreting CV.'},
                      '2CXO J215544.4+380116':{'Class':'CV', 'Comment':'Confirmed magnetic CV, see https://arxiv.org/pdf/1610.05008v1.pdf'},
                      '2CXO J225848.2+251543':{'Class':'CV', 'Comment':'A new CV from the open CV catalog, confirmed by SIMBAD, spectroscopically confirmed from https://simbad.cds.unistra.fr/simbad/sim-ref?bibcode=2016yCat....1.2035M'},
                      '2CXO J033131.4+435648':{'Class':'CV', 'Comment':'A new CV from the open CV catalog, CV candidate from 2005JAD....11....2D, confirmed by LAMOST 2020AJ....159...43H'},
    
                      
        
    
                      '2CXO J141842.6-605802':{'Class':'NS', 'Comment':'Rabbit, NS from 2018ApJS..235....4O, comfirmed by SIMBAD, a NS that has two potential counterpart 2CXO J141842.6-605802 and 2CXO J141842.9-605804 that are 2.4" away'},
                      '2CXO J122758.7-485343':{'Class':'NS_BIN', 'Comment':'2018ApJS..235....4O misclassified as a CV, transitional pulsar, see 2020A&A...635A..30M, 2022ApJ...924...91A'},
                    
                      '2CXO J161933.3-280740':{'Class':'LMXB', 'Comment':'A new LMXB classified by 2018ApJS..235....4O, SIMBAD identifies as an X-ray source. SyXB, fairly rare, accreting from cool M giant.'},
                      '2CXO J170634.5+235818':{'Class':'LMXB', 'Comment':'LMXB classified by 2007A&A...469..807L and 2018ApJS..235....4O, SIMBAD identifies as a LPV* and LXB in othertype, LAMOST classified as LM-STAR. Observed by HRC-I.'},
                      '2CXO J170907.5-362425':{'Class':'LMXB', 'Comment':'a new LMXB classified by BAT-105, confirmed by SIMBAD, with 200+ refs'},
                      
                      '2CXO J182119.7-131838':{'Class':'HMXB', 'Comment':'A new HMXB classified by 2018ApJS..235....4O, confirmed by SIMBAD, see 2020MNRAS.498.2750C'},
                      
                      ### new sources candidates: 
                      
                     ### New sources solely from SIMBAD ordered by nbref
    
                     '2CXO J043715.8-471508':{'Class':'NS', 'Comment':'PSR J0437-47, 2008ApJ...685L..67D, however there is a nearby source 2CXO J043715.9-471509X 1.3" away'},
                     ## '2CXO J182608.6-125631':{'Class':'NS', 'Comment':'PSR J1826-1256, 2019MNRAS.487.1964K, 2015ApJ...814..128K'},
                     '2CXO J162636.5-515630':{'Class':'HMXB', 'Comment':'SWIFT J1626.5-5156, classified as a HMXB by 2013ApJ...762...61D and BAT-105, the BAT counterpart is 4.3" away from the X-ray source while the SIMBAD coordinate is more accurate with 0.12" sep'},
                     '2CXO J063344.1+063230':{'Class':'NS', 'Comment':'PSR J0633+0632, a gamma-ray NS. identified by the 1FGL catalog (2010ApJS..188..405A), with a refined coordinate from 2015ApJ...814..128K and 2020MNRAS.493.1874D'},
                     ## '2CXO J081228.3-311452':{'Class':'HMXB', 'Comment':'V* V572 Pup,  first identified by ROSAT as an X-ray source coincides with a Be star (LS 992, B0.2IVe), 1999MNRAS.306...95R, 2001A&A...367..266R,2019MNRAS.488.4427Z'}, 
                     '2CXO J141330.2-620535':{'Class':'NS', 'Comment':'PSR J1413-6205, a gamma-ray NS. identified by the 1FGL catalog (2010ApJS..188..405A), with a refined coordinate from 2015ApJ...814..128K, confused with 2CXO J141329.9-620534 (2.8" sep) are nearby'},
                     '2CXO J140514.4-611827':{'Class':'HMXB', 'Comment':'2MASS J14051441-6118282, identified by 2019ApJ...884...93C 4FGL J1405.1-6119 (=3FGL J1405.4-6119) as a high-mass gamma-ray binary'}, 
                     '2CXO J184625.8+091949':{'Class':'NS', 'Comment':'PSR J1846+0919, a CXO source with low S/N ~3. a gamma-ray NS. identified by the 1FGL catalog (2010ApJS..188..405A), with a refined coordinate from 2015ApJ...814..128K'},
                     '2CXO J223827.9+590343':{'Class':'NS', 'Comment':'PSR J2238+5903, a gamma-ray NS. identified by the 1FGL catalog (2010ApJS..188..405A), with a refined coordinate from 2015ApJ...814..128K'},
                     '2CXO J183327.7-103524':{'Class':'HMXB', 'Comment':'ATO J278.3657-10.5901, identified from 2013A&A...553A..12N, an Gamma Cas analog. May be related to Kes 75.'}, 
                     
                     '2CXO J101011.8-565532':{'Class':'HMXB', 'Comment':'https://arxiv.org/pdf/2210.10363.pdf'},
                     '2CXO J114400.3-610736':{'Class':'HMXB', 'Comment':'https://arxiv.org/pdf/2210.10363.pdf'},
                     '2CXO J124846.4-623743':{'Class':'CV', 'Comment':'https://arxiv.org/pdf/2210.10363.pdf'},
                     '2CXO J132426.6-620119':{'Class':'HMXB', 'Comment':''},
                     '2CXO J182154.8-134726':{'Class':'HMXB', 'Comment':'This object is identified by Brendan as BeXR, see 2022ApJ...927..139O'},
                     '2CXO J181642.7-161322':{'Class':'HMXB', 'Comment':'According to 2019A%26A...622A.198N they identified B0-2e counterpart hence it is an likely accreting X--ray pulsar'},
                         
                     #### confused X-ray sources
                     '2CXO J053854.5+261856':{'Class':'HMXB', 'Comment':'confused with 2CXO J053854.5+261855X'},
                     '2CXO J063354.2+174613':{'Class':'NS', 'Comment':'confused with 2CXO J063354.3+174614 and 2CXO J063354.2+174616'},
                     '2CXO J105007.1-595321':{'Class':'NS', 'Comment':'confused with 2CXO J105007.5-595321.'},
                     '2CXO J112115.1-603725':{'Class':'HMXB', 'Comment':'confused with 2CXO J112115.1-603725X.'},  
                     '2CXO J113106.9-625648':{'Class':'HMXB', 'Comment':'confused with 2CXO J113106.9-625648X.'}, 
                     '2CXO J130848.2+212706':{'Class':'NS', 'Comment':'confused with 2CXO J130848.1+212707.'},
                     '2CXO J141842.6-605802':{'Class':'NS', 'Comment':'confused with 2CXO J141842.9-605804.'},
                     '2CXO J142008.1-604817':{'Class':'NS', 'Comment':'confused with 2CXO J142008.4-604815.'},
                     '2CXO J161243.0-522523':{'Class':'LMXB', 'Comment':'confused with 2CXO J161243.1-522523X.'},
                     '2CXO J162046.2-513006':{'Class':'HMXB', 'Comment':'confused with 2CXO J162046.2-513004X.'},
                     '2CXO J171809.8-371851':{'Class':'NS', 'Comment':'confused with 2CXO J171810.0-371853.'},
                     '2CXO J171935.8-410053':{'Class':'CV', 'Comment':'confused with 2CXO J171935.9-410054X.'},
                     '2CXO J184624.9-025830':{'Class':'NS', 'Comment':'confused with 2CXO J184624.9-025828X.'},
                     '2CXO J191404.2+095258':{'Class':'HMXB', 'Comment':'confused with 2CXO J191404.2+095258X.'},
                     '2CXO J193030.1+185214':{'Class':'NS', 'Comment':'confused with 2CXO J193029.9+185213X.'},
                     '2CXO J202105.4+365104':{'Class':'NS', 'Comment':'confused with 2CXO J202105.4+365104X.'},
                     '2CXO J222552.8+653536':{'Class':'NS', 'Comment':'same NS as 2CXO J222552.6+653535 in different epoch due to large proper motion.'},
                     '2CXO J222552.6+653535':{'Class':'NS', 'Comment':'same NS as 2CXO J222552.8+653536 in different epoch due to large proper motion.'},
    
    
                      }