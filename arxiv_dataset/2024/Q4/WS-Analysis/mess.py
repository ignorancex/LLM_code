if __name__ == "__main__":

        mess  = 'Analysis code for weather station data\n\n'
        mess += 'This code has been used to produce the paper \"Detailed Analysis of Local Climate at the CTAO-North Site from 20 Years of MAGIC Weather Station Data\", https://dx.doi.org/10.1093/mnras/stae2214\n\n'
        mess += 'Download the data from: https://dx.doi.org/10.5281/zenodo.11279074\n\n'
        mess += 'For correlations with the NAOI download also https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.cdas.z500.19500101_current.csv\n\n'        
        mess += 'Edit and run the python script create_files.py\n\n'
        mess += 'Either use a jupyter notebook: \n'
        mess += 'make jupyter \n\n'
        mess += 'Or work directly with the python script WS.py:\n'
        mess += 'Edit the main part of the script WS.py to uncomment different parts of the analysis\n'
        mess += "   #plot_datacount()\n"       
        mess += "   #plot_downtime() \n"
        mess += "   #plot_temperature() \n"
        mess += "   #plot_DTR()\n"
        mess += "   #plot_temperature_not()\n"
        mess += "   #plot_snow() \n"
        mess += "   #plot_rainy_periods()\n"
        mess += "   #plot_humidity() \n"
        mess += "   #plot_pressure()\n"
        mess += "   #plot_wind()\n"
        mess += "   #plot_huracans()\n\n"
        mess += "As each individual part takes several minutes to finish and produces a long series of plots, I recommend to run WS.py by parts.\n\n"
        mess += "The script has been written for analysis of the MAGIC weather data only, a more user-friendly version easily adaptable for other weather station data is still being developed. Contributions are welcome.\n"

        print (mess)
