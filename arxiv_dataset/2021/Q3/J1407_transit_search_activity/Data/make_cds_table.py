import cdspyreadme
print(cdspyreadme.__version__)

tablemaker = cdspyreadme.CDSTablesMaker()

# add a table
table = tablemaker.addTable("Final_Combined_Data_Accepted.csv", description="Photometry corrected for stellar activity")
# write table in CDS-ASCII aligned format (required)
table.name="j1407.dat"
column = table.get_column("MJD")
column.set_format("F11.5")
column.description="Epoch [Modified Julian Date]"
column.unit='d'

column2 = table.get_column("Non_Corr_Flux")
column2.name="flux"
column2.set_format("F6.4")
column2.description="Normalised Uncorrected Flux"


column6 = table.get_column("Corr_Flux")
column6.name="fluxcorr"
column6.set_format("F6.4")
column6.description="Normalised Corrected Flux"

column3 = table.get_column("Eflux")
column3.name = 'fluxerr'
column3.set_format("F6.4")
column3.description="Error on Flux"

column4 = table.get_column("Telescope")
column4.name = 'telnum'
column4.set_format("F1.0")
column4.description="Telescope number"

tablemaker.writeCDSTables()

# Print ReadMe
tablemaker.makeReadMe()

print('Warning: ReadMe is output to ReadMe.old and edited manually to produce ReadMe')
# Save ReadMe into a file
with open("ReadMe.old", "w") as fd:
    tablemaker.makeReadMe(out=fd)
