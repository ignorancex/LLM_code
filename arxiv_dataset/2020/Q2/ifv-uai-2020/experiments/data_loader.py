# Author: Philips George John

import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from ibv.model import InputVariable

	
def load_data(ds_name, mask_prot_attrs=False):
	DS_NAMES = ['german', 'adult', 'fraud', 'credit']
	
	class_attr = 'fubar'
	prot_attrs, input_vars = [], []
	train_x, train_y, test_x, test_y = None, None, None, None	
	df, df_test = None, None
	
	if ds_name == 'german':
		class_attr = 'credit_rating'
		prot_attrs = ['sex_marital_status']
		
		input_vars = [
			InputVariable('salary_account_range', 'integer', 1, 4),
			InputVariable('duration_months', 'integer', 1, 1200),
			InputVariable('credit_history_status', 'integer', 0, 4),
			InputVariable('purpose', 'integer', 0, 10),
			InputVariable('credit_amount', 'real', 1, 10000000),
			InputVariable('savings_account_range', 'integer', 1, 5),
			InputVariable('employment_range', 'integer', 1, 5),
			InputVariable('installment_fraction', 'real', 0.0, 100.0),
			InputVariable('sex_marital_status', 'integer', 1, 5, protected=True),
			InputVariable('codebtors_status', 'integer', 1, 3),
			InputVariable('present_residence_period', 'real', 0.0, 1200.0),
			InputVariable('property_type', 'integer', 1, 4),
			InputVariable('age', 'integer', 1, 100),
			InputVariable('other_installment_status', 'integer', 1, 3),
			InputVariable('housing_type', 'integer', 1, 3),
			InputVariable('num_existing_credits', 'integer', 1, 20),
			InputVariable('job_status', 'integer', 1, 4),
			InputVariable('num_dependents', 'integer', 1, 10),
			InputVariable('has_phone', 'integer', 0, 1),
			InputVariable('is_foreign_worker', 'integer', 0, 1)
		]
		
		with open('../datasets/german-credit/german.data.csv') as f:
			df = pd.read_csv(f, header=0)
	elif ds_name == 'adult':
		class_attr = 'income'
		prot_attrs = ['race']
		
		input_vars = [
			InputVariable('age', 'integer', 15, 120),
			InputVariable('workclass', 'integer', 0, 3),
			InputVariable('education', 'integer', 1, 8),
			InputVariable('marital_status', 'integer', 0, 4),
			InputVariable('occupation', 'integer', 0, 5),
			InputVariable('race', 'integer', 0, 1, protected=True),
			InputVariable('sex', 'integer', 0, 1),
			InputVariable('hours_per_week', 'integer', 0, 168),
		]
		
		# with open('../datasets/adult/adult.data.csv') as f:
		#	df = pd.read_csv(f, header=0)
		# with open('../datasets/adult/adult.test.csv') as f:
		#	df_test  = pd.read_csv(f, header=0)
		
		with open('../datasets/adult/adult_mod.csv') as f:
			df = pd.read_csv(f, header=0)
	elif ds_name == 'fraud':
		class_attr = 'fraudulent'
		prot_attrs = ['ethnicity']
		
		input_vars = [
			InputVariable('age', 'integer', 18, 90),
			InputVariable('gender', 'integer', 0, 1),
			InputVariable('ethnicity', 'integer', 0, 2, protected=True),
			InputVariable('incident_cause', 'integer', 1, 5),
			InputVariable('days_to_incident', 'integer', 1, 20000),
			InputVariable('claim_area', 'integer', 1, 2),
			InputVariable('police_report', 'integer', 0, 2),
			InputVariable('claim_type', 'integer', 1, 3),
			InputVariable('total_policy_claims', 'integer', 1, 10)
		]
		
		with open('../datasets/fraud/fraud.data.csv') as f:
			df = pd.read_csv(f, header=0)
	elif ds_name == 'credit':
		class_attr = 'High-Balance'
		prot_attrs = ['Gender', 'Ethnicity']
		
		input_vars = [
			InputVariable('Income', 'real', 10.0, 200.0),
			InputVariable('Limit', 'integer', 800, 15000),
			InputVariable('Rating', 'integer', 80, 1000),
			InputVariable('Cards', 'integer', 1, 10),
			InputVariable('Age', 'integer', 18, 100), 
			InputVariable('Education', 'integer', 4, 20),
			InputVariable('Gender', 'integer', 0, 1, protected = True),
			InputVariable('Student', 'integer', 0, 1),
			InputVariable('Married', 'integer', 0, 1),
			InputVariable('Ethnicity', 'integer', 1, 3, protected = True),
		]
		
		with open('../datasets/credit/credit.csv') as f:
			df = pd.read_csv(f, header=0)
	else:
		print('Error :: load_data: dataset-name has to be in', DS_NAMES)
	# End if (ds_name)

	# Process data and create train and test sets.
	df_X = df.drop([class_attr], axis=1)
	column_names = df_X.columns.values.tolist()

	print('Loading dataset: %s... Feature columns:' % ds_name, column_names)
	print('Class attr: %s, Protected attrs:' % class_attr, prot_attrs)
	
	df_test_X = None
	if df_test is not None:
		df_test_X  = df_test.drop([class_attr], axis=1)

	if mask_prot_attrs:
		print('Masking protected attributes:', prot_attrs)
		for prot_attr in prot_attrs:
			df_X[prot_attr] = 0 # Mask protected attr
			if df_test is not None:
				df_test_X[prot_attr] = 0
	# End if

	if df_test is not None:
		train_x = df_X.values.astype(np.float64)
		train_y = df[class_attr].values.astype(np.int64)	
		test_x  = df_test_X.values.astype(np.float64)
		test_y  = df_test[class_attr].values.astype(np.int64)
	else: # Do a stratified 75-25 split
		X = df_X.values.astype(np.float64)
		y = df[class_attr].values.astype(np.int64)
		train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.25, stratify = y)
	
	return class_attr, prot_attrs, input_vars, train_x, train_y, test_x, test_y
# End fn load_data
