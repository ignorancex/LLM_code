# Author: Yicheng Rui
# Email:  ruiyicheng@sjtu.edu.cn
# This code is used for downloading the GOST data given a list of ra dec and target name. You can revise line 98 and 99 for your own purpose.
# usage:
# python Obtain_GOST.py --file /absolute/path/to/the.csv
import argparse
parser = argparse.ArgumentParser(description='target folder path')
parser.add_argument("--file", type=str, default='demo.csv')
args = parser.parse_args()
import glob

import requests
import time
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from datetime import timedelta
import os
# t0=time.time()

def find_target(ra,dec,tgname):
    # proxies = {                                           #The proxy of SJTU Siyuan cluster
    # 'http': 'http://proxy2.pi.sjtu.edu.cn:3128',
    # 'https': 'http://proxy2.pi.sjtu.edu.cn:3128',
    # }
    # Similar to the code of HTOF (Brandt, G. Mirek 2021)
    url = f"https://gaia.esac.esa.int/gost/GostServlet?ra="+str(ra)+"&dec="+str(dec)

    with requests.Session() as s:
        s.get(url)
        headers = {"Cookie": f"JSESSIONID={s.cookies.get_dict()['JSESSIONID']}"}
        response = s.get(url, headers=headers, timeout=1000)#,proxies=proxies)
    root = ET.fromstring(response.text)
    columns = ["Target", "ra[rad]", "dec[rad]", "ra[h:m:s]", "dec[d:m:s]", "ObservationTimeAtGaia[UTC]",
                    "CcdRow[1-7]", "zetaFieldAngle[rad]", "scanAngle[rad]", "Fov[FovP=preceding/FovF=following]",
                    "parallaxFactorAlongScan", "parallaxFactorAcrossScan", "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"]
    rows = []
    name = root.find('./targets/target/name').text
    raR = root.find('./targets/target/coords/ra').text
    decR = root.find('./targets/target/coords/dec').text
    raH = root.find('./targets/target/coords/raHms').text
    decH = root.find('./targets/target/coords/decDms').text
    for event in root.findall('./targets/target/events/event'):
        details = event.find('details')
        observationTimeAtGaia = event.find('eventUtcDate').text
        ccdRow = details.find('ccdRow').text
        zetaFieldAngle = details.find('zetaFieldAngle').text
        scanAngle = details.find('scanAngle').text
        fov = details.find('fov').text
        parallaxFactorAl = details.find('parallaxFactorAl').text
        parallaxFactorAc = details.find('parallaxFactorAc').text
        observationTimeAtBarycentre = event.find('eventTcbBarycentricJulianDateAtBarycentre').text
        rows.append([name, raR, decR, raH, decH, observationTimeAtGaia, ccdRow,
                        zetaFieldAngle, scanAngle, fov, parallaxFactorAl, parallaxFactorAc, observationTimeAtBarycentre])
    data = pd.DataFrame(rows, columns=columns)
    data = data.astype({"Target": str,"ra[rad]": float, "dec[rad]": float,"ra[h:m:s]": str,"dec[d:m:s]": str,"ObservationTimeAtGaia[UTC]": str,"CcdRow[1-7]": int,"zetaFieldAngle[rad]": float,"scanAngle[rad]": float,"Fov[FovP=preceding/FovF=following]": str,"parallaxFactorAlongScan": float,"parallaxFactorAcrossScan": float,"ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": float })
    data['Target']=[tgname]*len(data)
     #print(data)
    #data.to_csv('this_data_all.csv',index=False)
    lookingfornext=0
    ttemp=0
    tp_list=[]
    t_end=datetime.fromisoformat('2017-05-28T08:46:29')
    for i,r in data.iterrows():

        t_this=r['ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]']
        #The raw GOST data downloaded from the pipeline is a bit different from the GOST data downloaded from the website.
        #These filtering criteria can convert it into the same form with the GOST downloaded from https://gaia.esac.esa.int/gost/
        if t_this-ttemp>1/24/20:
            
            ttemp=t_this
            lookingfornext=1
            continue
        if lookingfornext:
            timedt = (t_end - datetime.fromisoformat(r['ObservationTimeAtGaia[UTC]']))/ timedelta(seconds=1)

            #print(timedt)
            if timedt<0:
                break
            tp_list.append(pd.DataFrame(r).T)
            lookingfornext=0

    res=pd.concat(tp_list)
    #print(res)
    #res.to_csv('this_data_reduced.csv',index=False)
    return res
if __name__=='__main__':
    targets=pd.read_csv(args.file,dtype={'dr3_source_id': 'Int64'})
    # try: #create the folder. uncomment it when running the script for the first time.
    #     os.system('mkdir '+args.file[:-4])
    # except:
    #     pass
    datapaths=glob.glob(args.file[:-4]+'/*.csv')
    for i,r in targets.iterrows():

        prepare_name=args.file[:-4]+'/'+str(r['dr3_source_id'])+'.csv'
        if prepare_name in datapaths:#if the target have already been downloaded, skip it.
            continue
        print('processing:',r['dr3_source_id'],'  ra, dec:',r['dr3_ra'],r['dr3_dec'])
        try:
            gost_res=find_target(r['dr3_ra'],r['dr3_dec'],r['dr3_source_id'])
            gost_res.to_csv(args.file[:-4]+'/'+str(r['dr3_source_id'])+'.csv',index=False)
        except:
            print('failed:',r['dr3_source_id'],'  ra, dec:',r['dr3_ra'],r['dr3_dec'])
        #print(gost_res)
