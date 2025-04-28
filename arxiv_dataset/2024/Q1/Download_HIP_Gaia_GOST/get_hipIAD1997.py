# Author: Yifan Xuan; E-mail: yifan.xuan@sjtu.edu.cn 
'''
This code is dedicated to download Intermediate Astrometric Data (IAD) of the Hipparcos 1997 reductions. There are a total of 118,204 entries which have non-empty Hipparcos IAD information in the 1997 version. Using the \verb|get_hipIAD1997| function, Hipparcos IAD of the corresponded source can be downloaded as a csv file. If the input HIP entry is non-empty, this csv file will contain columns of orbit number, source of abscissa (FAST or NDAC), partial derivatives of the abscissa with respect to five astrometric parameters ($\alpha* = \alpha \cos\delta, \delta, \pi, \mu_{\alpha*}, \mu_{\delta}$), abscissa residual in mas, standard error of the abscissa in mas, correlation coefficient between abscissae, reference great circle mid-epoch in years, reference great circle mid-epoch in days, RA of the great circle pole in degrees, Dec of the great circle pole in degrees. Otherwise, \verb|The Hipparcos IAD of this HIP entry cannot be found| will be printed.
'''
import numpy as np
import pandas as pd
import urllib
import bs4

# Get access to Hipparcos IAD 1997 version.
def get_hipIAD1997(HIP,relative_path = 'results/HIP_epoch/test.csv'):
    url = f'https://hipparcos-tools.cosmos.esa.int/cgi-bin/HIPcatalogueSearch.pl?noLinks=1&tabular=1&hipiId={HIP}'
    webpage = str(urllib.request.urlopen(url).read())
    soup = bs4.BeautifulSoup(webpage,'html.parser')
    text = soup.find(name='pre').get_text().lstrip("\\n").rstrip("\\r\\n\\r\\n\\r\\n'")
    text = text.split('\\r\\n\\r\\n')
    text_list = []
    try:
        text_list.append(text[0].split('\\n',3)[3])
        for line in text[1:]:
            text_list.append(line)
        data_list = []
        for row in text_list:
            row = row.replace('\\n','|').split('|')
            if ('F' in row) or ('f' in row):
                data = row[0:9]
                data.append(np.nan) if row[9] == '     ' else data.append(row[9])
                for x in row[11:14]:
                    data.append(x)
            elif ('N' in row) or ('n' in row):
                data = row[0:9]
                data.append(np.nan) if row[9] == '     ' else data.append(row[9])
                for x in row[14:]:
                    data.append(x)
            data_list.append(data)
        data_list_t = list(map(list, zip(*data_list)))
        index_float = [2,3,7,8,9,10]
        for i in index_float:
            data_list_t[i] = [float(x) for x in list(data_list_t[i])]
        epoch_time = [x+1991.25 for x in list(data_list_t[10])]
        data_list_t = np.array(data_list_t).transpose()
        data_list_t = np.insert(data_list_t,11,epoch_time,axis=1)
        data_list_tt = list(zip(*data_list_t))

        for i in np.arange(2,14):
            data_list_tt[i] = [float(x) for x in list(data_list_tt[i])]
        colnames = ['orbit_number','source_absc','absc/ra','absc/dec','absc/parallax','absc/pmra','absc/pmdec','absc_residual [mas]','absc_error [mas]','absc_corr',
                    'ref_great-circle_mid-epoch [yr]','ref_great-circle_epoch_time [yr]',
                    'great-circle_pole_ra [deg]','great-circle_pole_dec [deg]']
        df = pd.DataFrame(zip(*data_list_tt),columns=colnames)
        df.to_csv(relative_path,index=False)  # Adjust the path to save the IAD csv files by yourself

    except:  # A minority of HIP intermediate data cannot be found.
        print(f'The Hipparcos intermediate astrometric data of HIP{HIP} cannot be found.')