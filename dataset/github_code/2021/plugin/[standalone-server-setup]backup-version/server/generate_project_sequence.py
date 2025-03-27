# Add the GitHub project URLs in the file "urls" in new lines 
# GitHub Project URL should be in the form "https://github.com/abc/def"
# And then run this python file to generate auto numbered sequence ids for all the projects 

# Step-1: Cleaning the contents of the sequence file 
github_project_sequence_fp = open('sequence','w')
github_project_sequence_fp.write('')
github_project_sequence_fp.close()


# Step-2: Adding the ids for the sequence file 
github_project_sequence_fp = open('sequence','a')

github_project_urls = open('urls','r')

adder = 1
stream = ""
for x in github_project_urls:
    stream +=  str(adder) + "\n"
    adder+=  1

github_project_sequence_fp.write(stream[:len(stream)-1])
github_project_urls.close()
github_project_sequence_fp.close()