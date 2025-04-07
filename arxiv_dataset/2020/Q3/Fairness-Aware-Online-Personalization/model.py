import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

#############
## Parameters:
##	 x : floating point number
##
## Return Type: floating point number
##
## This function computes the standard sigmoid funtion (1/(1 + e^(-x)))
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#############
## Parameters:
##	 n : Number of feature vectors to be generated
##	 p : probability of feature vectors to be generated from group 1
##	 mean : mean of normal distributions used to generate the feature vectors (array of length m-1 x 2, for m attributes)
##	 sigma : variance of normal distributions used to generate the feature vectors (array of length m-1 x 2, for m attributes)
##
## Return Type: List of features, x.
##				features is a 2D numpy array of n rows and m columns.
##				x is a numpy array of length n of 0s and 1s which indicates which group the ith feature comes from
##
## This function generates data according to the Bayesian network described
## here: https://iwww.corp.linkedin.com/wiki/cf/display/ENGS/Fairness-aware+Online+Learning#Fairness-awareOnlineLearning-DataModel:
## First a coin is flipped which lands 1 with probability p and 0 with probability 1-p.
## The result of this coin flip is used as the group label
## For each coin flip, m random samples are drawn from m different distributions to get the feature vector.
## The first attribute of the feature vector is drawn from a uniform distribution btw 0 and 1 and is independent of the group that the feature belongs to.
## The second, third, .... upto the mth attribute are sampled from a Normal distribution whose parameters are taken from mu and sigma (m-1 rows one each for one distribution, 2 columns for 2 groups).
def data_generation(n, p, mean, sigma):
	s = np.random.random(n)

	x = np.zeros(n)
	for i in range(len(s)):
		if(s[i] >= p):
			x[i] = 0
		else:
			x[i] = 1

	m = len(mean) + 1
	features = np.zeros((n,m))
	for i in range(n):
		for j in range(m):
			if(j == 0):
				features[i][j] = np.random.uniform()
			else:
				features[i][j] = np.random.normal(mean[j - 1][int(x[i])], sigma[j - 1][int(x[i])])
	return [features, x]

#############
## Parameters:
##	 n : Number of feature vectors to be labelled
##	 p : probability of user being biassed
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 weights : weights used for scoring the features. weights = [w d]. Is Dot-Product(w, feature-vector) - d > 0
##	 x : numpy array of length n of 0s and 1s which indicates which group the ith feature comes from
##
## Return Type: List of y, mood.
##				y is a numpy array of length n of 0s and 1s which indicates whether the ith feature vector is rejected or selected
##				mood is a numpy array of length n of 0s and 1s which indicates whether the decision on the ith feature vector was made by user being fair or unfair
##
## This function generates data according to the Bayesian network described
## here: https://iwww.corp.linkedin.com/wiki/cf/display/ENGS/Fairness-aware+Online+Learning#Fairness-awareOnlineLearning-RecruiterModel:
## First a coin is flipped which lands 1 with probability p and 0 with probability 1-p.
## The result of this coin flip is used as the mood of user. 1 stands for unfair 0 and stands for fair.
## When the user is fair, he makes his decisions using a linear combination of features attributes.
## weights = [w d]. If Dot-Product(w, feature-vector) - d > 0, then the user gives the feature vector a label of 1 (meaning selected) else 0 (rejected)
## When the user is unfair, he makes his decisions based on which group the feature vector belongs to.
## If feature vector comes from group 1, then the user gives the feature vector a label of 1 (meaning selected) else 0 (rejected)
def userFeedback(n, p, features, weights, x):
	mood = np.random.random(n)

	#print(weights)
	y = np.zeros(n)
	for i in range(len(mood)):
		if(mood[i] >= p):
			mood[i] = 0
		else:
			mood[i] = 1
		if(mood[i] == 0):
			if( np.dot(weights[:-1], features[i][:]) - weights[-1] > 0):
				y[i] = 1
			else:
				y[i] = 0
		else:
			if(x[i] == 1):
				y[i] = 1
			else:
				y[i] = 0

	return [y, mood]

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 output : numpy array of length number of features of 0s and 1s which indicates which class the ith data features comes from in training data
##
## Return Type: List of sigma, sigma0, sigma1.
##				sigma is a numpy array of m X m. It is correlation matrix of features.
##				sigma0 is a numpy array of m X m. It is correlation matrix of features given the ouput class is 0.
##				sigma1 is a numpy array of m X m. It is correlation matrix of features given the output class is 1.
##
## This function computes the correlation matrix of features and the conditional correlation matrix of features given the output.
def getCovariance(features, output):
	sigma = np.cov(features.T)
	mean = np.mean(features.T, axis = 1)

	f0 = []
	f1 = []
	for i in range(len(features)):
		if(output[i] < 0.5):
			f0.append(features[i])
		if(output[i] > 0.5):
			f1.append(features[i])
	f0 = np.array(f0)
	f1 = np.array(f1)

	sigma0 = np.cov(f0.T)
	sigma1 = np.cov(f1.T)

	mean0 = np.mean(f0.T, axis = 1)
	mean1 = np.mean(f1.T, axis = 1)

	return [sigma, sigma0, sigma1]

#############
## Parameters:
##	 features : 2D numpy array of features. Every row corresponds to one feature vector
##	 output : numpy array of length n of 0s and 1s which indicates whether the ith feature vector is rejected or selected
##	 title : string which records the title of the 3D plot
##
## Return Type: None
##
## This function assumes that each feature vector has exactly 3 components. It ignores all components except the first 3. It generates a 3D scatter plot of feature vectors
## with postively labelled feature vectors shown in green and the feature vectors with negatively labelled feature vectors shown in red
def show_attributes(features, output, title):
	r = np.reshape(1 - output,(-1,1))
	g = np.reshape(output,(-1,1))
	b = np.reshape(0 * output,(-1,1))
	c = np.htack((r,g,b))

	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(features[:,0], features[:,1], features[:,2], color=c)
	ax.set_xlabel('Unbiassed Attribute')
	ax.set_ylabel('Biassed Attribute-1')
	ax.set_zlabel('Biassed Attribute-2')
	ax.set_title(title)
	plt.show()
	return

if(__name__ == "__main__"):
	mu = [[0.35, 0.65],[0.65, 0.35]]
	sigma = [[0.12, 0.12],[0.12, 0.12]]

	#n = 1000000
	n = 10000
	#p_groups = 0.9999999
	#p_groups = 0.0000001
	p_groups = 0.5

	[features, protected_attribute] = data_generation(n, p_groups, mu, sigma)
	p_bias = 0.00


	weights = np.random.uniform(0, 1, 4)
	weights[0:3] = weights[0:3] / np.sum(weights[0:3])
	weights = np.array([0.35, 0.28, 0.37, 0.48])

	[output , bias] = userFeedback(n, p_bias, features, weights, protected_attribute)

	plt.figure()
	plt.plot(np.sign(np.dot(features, np.array([0, 3.57, -3.57])) -   0.0), color='b')
	plt.plot(np.sign(np.dot(features, np.array([-0.16639281,  0.74013601, -0.5929042])) + 0.27014205), color='g')
	plt.plot(2*protected_attribute-1, color='r')
	plt.ylim([-2, 2])
	plt.show()

	plt.figure()
	plt.hist(features[:,0], bins=50, label='0', alpha = 0.75, histtype = 'step')
	plt.hist(features[:,1], bins=50, label='1', alpha = 0.75, histtype = 'step')
	plt.hist(features[:,2], bins=50, label='2', alpha = 0.75, histtype = 'step')
	plt.legend()
	plt.show()

	r = np.reshape(1 - output,(-1,1))
	g = np.reshape(output,(-1,1))
	b = np.reshape(0 * output,(-1,1))
	c = np.hstack((r,g,b))
	plt.figure()
	plt.scatter(x = features[:,0], y = features[:,1] + features[:,2] , color = c)
	#plt.scatter(x = features[:,0], y = features[:,1], color = c)
	plt.xlabel("Unbiassed Attribute")
	plt.ylabel("Sum of implicitly biassed Attribute")
	plt.show()

	plt.figure()
	#plt.scatter(x = features[:,0], y = features[:,0] + features[:,1] - features[:,2] , color = c)
	plt.scatter(x = features[:,0], y = features[:,2], color = c)
	plt.xlabel("Unbiassed Attribute")
	plt.ylabel("Biassed Attribute")
	plt.show()

	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(features[:,0], features[:,1], features[:,2], color=c)
	ax.set_xlabel('Unbiassed Attribute')
	ax.set_ylabel('Biassed Attribute-1')
	ax.set_zlabel('Biassed Attribute-2')
	plt.show()