# Required libraries are either numpy or pandas
# Matplotlib required for standalone execution

# Suport functions ############################################################
#########################################################
def read_FFT_file(filename, use_pandas=True):
    """
    Reads an FFT file and extracts simulation header, FFT header, units header, and data sections.

    Parameters:
    filename (str): The path to the file to be read.
    use_pandas (bool): If True, returns data as a pandas DataFrame. If False, returns data as a numpy array. Default is True.

    Returns:
    tuple: A tuple containing four elements:
        - simulation_header (dict): A dictionary containing simulation header key-value pairs.
        - FFT_header (dict): A dictionary containing FFT header key-value pairs.
        - units_header (dict): A dictionary containing units header key-value pairs.
        - df_data (pandas.DataFrame or numpy.ndarray): Data extracted from the file. Type depends on use_pandas parameter.

    Example:
    simulation_header, FFT_header, units_header, df_data = read_FFT_file('path/to/file.dat')
    """
    with open(filename, 'r') as file:
        lines = file.readlines()  # Read all lines from the file

    sectionID = 0  # Initialize section identifier
    # Initialize dictionaries for headers and units
    simulation_header = {}
    FFT_header = {}
    units_header = {}
    data = []  # Initialize list for data
    in_data_section = False  # Initialize flag for data section

    # Loop through each line in the file
    for line in lines:
        if line.strip() == '-' * len(line.strip()):  # Check for separator lines
            sectionID = sectionID + 1  # Increment section identifier
            continue

        if sectionID == 0:  # First section: Simulation header
            if '=' in line:
                key, value = line.split(' =')  # Split line into key and value
                simulation_header[key.strip()] = float(value.strip())  # Store in dictionary

        elif sectionID == 1:  # Second section: FFT header
            if '=' in line:
                key, value = line.split(' =')  # Split line into key and value
                try:
                    FFT_header[key.strip()] = float(value.strip())  # Try to convert value to float
                except ValueError:
                    FFT_header[key.strip()] = value.strip()  # Store as string if conversion fails

        elif sectionID == 2:  # Third section: Units header
            units_info = line.split()  # Split line into parts
            for i in range(0, len(units_info), 2):
                units_header[units_info[i]] = units_info[i+1].replace("(", "").replace(")", "")  # Store units

        elif sectionID == 3:  # Fourth section: Data
            data.append(line.strip())  # Add data line to list

    # Convert data section to a DataFrame or numpy array
    data = [line.split() for line in data]  # Split each data line into parts
    if (use_pandas):
        import pandas as pd  # Import pandas if use_pandas is True
        df_data = pd.DataFrame(data, columns=list(units_header.keys()))  # Create DataFrame with column names
        df_data = df_data.apply(pd.to_numeric)  # Convert all data to numeric types
    else:
        import numpy as np
        df_data = np.array(data, dtype=np.float)  # Create numpy array with float type

    return simulation_header, FFT_header, units_header, df_data  # Return all headers and data
#########################################################

#########################################################
def extract_model_name(filename):
   """Extracts the model name from the filename."""
   if ("FableDM" in filename):
      return "DM"
   return filename.split('_')[1]
#########################################################

#########################################################
def files_quickplot(files_list, read_path="", req_yquantity="P", req_xquantity="k"):
   """Generates quick plots for the provided list of files."""
   plt.figure()
   for file in files_list:
      simheader, FFT_header, units_header, df_data = read_FFT_file(read_path+file)
      # Get model name and capitalize first letter
      model_name = extract_model_name(file)
      model_name = model_name[0].upper() + model_name[1:]
      # Get model redshift rounded to 2 decimal places 
      model_redshift = "z = "+str(round(1/simheader['aexp'] - 1, 2))
      plt.loglog(df_data[req_xquantity], df_data[req_yquantity], label=model_name+" ("+model_redshift+")")
   plt.xlabel(req_xquantity); plt.ylabel(req_yquantity)
   plt.legend()
   print ("Saving plot: ", req_yquantity+'_'+req_xquantity+'.png')
   plt.savefig(req_yquantity+'_'+req_xquantity+'.png')
   plt.close()
#########################################################

# Main function (direct file execution) #######################################
if __name__ == "__main__":
   import matplotlib.pyplot as plt
   do_redshift=False
   read_folder="data/"
   model_names=["fiducial","RadioStrong","RadioWeak","Quasar","NoFeedback"]
   model_choice=0
   if (do_redshift):
      # Fiducial model, all redshifts
      filenames=["Fable_"+model_names[model_choice]+"_totalmass_overdensity_z0.0_ng_01024.dat"
         , "Fable_"+model_names[model_choice]+"_totalmass_overdensity_z0.4_ng_01024.dat"
         , "Fable_"+model_names[model_choice]+"_totalmass_overdensity_z1.0_ng_01024.dat"
         , "Fable_"+model_names[model_choice]+"_totalmass_overdensity_z2.0_ng_01024.dat"
         ]
   else:
      # Redshift z = 0; all models
      filenames=["Fable_fiducial_totalmass_overdensity_z0.0_ng_01024.dat"
         , "Fable_RadioStrong_totalmass_overdensity_z0.0_ng_01024.dat"
         , "Fable_RadioWeak_totalmass_overdensity_z0.0_ng_01024.dat"
         , "Fable_Quasar_totalmass_overdensity_z0.0_ng_01024.dat"
         , "Fable_NoFeedback_totalmass_overdensity_z0.0_ng_01024.dat"
         , "FableDM_totalmass_overdensity_z0.0_ng_01024.dat"
         ]
   print("Running read_test.py")

   # Read separate single files ------- Example 1
   model_choice = min(model_choice,len(filenames)-1)
   print ("Reading file: ", filenames[model_choice])
   simheader, FFT_header, units_header, df_data = read_FFT_file(read_folder+filenames[model_choice], use_pandas=True)
   print ("Reading file: ", filenames[-1])
   simheader, FFT_header, units_header, df_data_last = read_FFT_file(read_folder+filenames[-1], use_pandas=True)
   # Sample plot ratio of P in df_data vs P in df_data_last (e.g. fractional impact of baryonic physics on the MPS)
   plt.figure()
   plt.plot(df_data['k'], df_data['P']/df_data_last['P'])
   # Add horizontal line at ratio = 1
   plt.axhline(y=1, color='gray', linestyle='--')
   plt.xscale('log')
   plt.xlabel('k')
   plt.ylabel('P/P_last')
   figname="P_P_last.png"
   print ("Saving plot:",figname)
   plt.savefig(figname)

   # Quick graph sample for various models ------- Example 2
   print("Reading files: ", filenames)
   files_quickplot(filenames, read_path=read_folder)
   files_quickplot(filenames, read_path=read_folder, req_yquantity="P-NoCorr", req_xquantity="k")


