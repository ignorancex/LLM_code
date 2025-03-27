# The results from the GitQ will be stored in a global database, and a local copy of the results
# for the projects executed by the user will be stored locally to help the user export the results
# if needed. The metric values in GitQ Assessment report are indexed by commit_id. 

# This file initiates a sqlite database file
import os
import sqlite3
def Re_initialize_database():
    conn = sqlite3.connect('gitq_metric_evaluation.db')
    conn.cursor().execute("DROP TABLE IF EXISTS `assessment_report`")
    conn.execute('''CREATE TABLE `assessment_report` (
      `project_sequence` int(11) NOT NULL,
      `commit_id` varchar(255) NOT NULL,
      `url` varchar(255) DEFAULT NULL,
      `N_Pack` varchar(255) NULL,
      `D_Pack` varchar(255) NULL,
      `N_Methods` varchar(255) NULL,
      `O_Methods` varchar(255) NULL,
      `N_Class` varchar(255) NULL,
      `I_Class` varchar(255) NULL,
      `Code_Impact` varchar(255) NULL,
      `Code_Churn` varchar(255) NULL,
      `Open_Bugs` varchar(255) NULL,
      `Closed_Bugs` varchar(255) NULL,
      `Active_Contributors` varchar(255) NULL,
      `Percentage_Active_Contributors` varchar(255) NULL,
       PRIMARY KEY (`project_sequence`)
    );''')
    print("Database `gitq_metric_evaluation.db` Re-initialized, Local Database Created!")
    conn.close()

def initialize_database():
    conn = sqlite3.connect('gitq_metric_evaluation.db')
    conn.execute('''CREATE TABLE `assessment_report` (
      `project_sequence` int(11) NOT NULL,
      `commit_id` varchar(255) NOT NULL,
      `url` varchar(255) DEFAULT NULL,
      `N_Pack` varchar(255) NULL,
      `D_Pack` varchar(255) NULL,
      `N_Methods` varchar(255) NULL,
      `O_Methods` varchar(255) NULL,
      `N_Class` varchar(255) NULL,
      `I_Class` varchar(255) NULL,
      `Code_Impact` varchar(255) NULL,
      `Code_Churn` varchar(255) NULL,
      `Open_Bugs` varchar(255) NULL,
      `Closed_Bugs` varchar(255) NULL,
      `Active_Contributors` varchar(255) NULL,
      `Percentage_Active_Contributors` varchar(255) NULL,
       PRIMARY KEY (`project_sequence`)
    );''')
    print("Local Database Created!")
    conn.close()

def database_file_exists():
    if("gitq_metric_evaluation.db" in os.listdir()):
        return True 
    return False
remove_database = False 
if(database_file_exists()):
    user_confirmation = input("Do you want to remove the existing local sqlite database file?\nYou might want to export results to csv/json before deleting the contents?\nEnter Y/y for yes and any other key otherwise: ")
    if user_confirmation == ("Y") or user_confirmation == ("y"):
        remove_database = True 
        print("Removing the file `gitq_metric_evaluation.db`...")
        Re_initialize_database()
    else:
        print("Re-initializing the database is skipped. Local database contents will remain unchanged")
else:
    initialize_database()