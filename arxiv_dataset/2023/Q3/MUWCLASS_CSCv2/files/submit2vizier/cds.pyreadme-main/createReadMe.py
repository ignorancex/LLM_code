import cdspyreadme

tablemaker = cdspyreadme.CDSTablesMaker()

# add a table
table = tablemaker.addTable("table.csv", description="my CSV table")
# write table in CDS-ASCII aligned format (required)
tablemaker.writeCDSTables()

# Customize ReadMe output
tablemaker.title = "catalogue title"
tablemaker.author = 'G.Landais'
tablemaker.date = 2020
tablemaker.abstract = "This is my abstract..."
tablemaker.more_description = "Additional information of the data context."
tablemaker.putRef("II/246", "2mass catalogue")
tablemaker.putRef("http://...", "external link")

# Print ReadMe
tablemaker.makeReadMe()

# Save ReadMe into a file
with open("ReadMe", "w") as fd:
    tablemaker.makeReadMe(out=fd)
