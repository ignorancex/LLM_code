import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
f1 = open('', 'w')
f2 = open('', 'w')

def feature(root):
    load = np.loadtxt(root,delimiter=',')
    fea = load[:]
    r = np.max(fea) - np.min(fea)
    return (fea - np.min(fea)) / r



ID_a = ''
ID_b = ''
ID_l = ''
ID_g = ''

OOD_a = ''
OOD_b = ''
OOD_l = ''
OOD_g = ''

TEST_a = ''
TEST_b = ''
TEST_l = ''
TEST_g = ''


X1 = feature(ID_a)
X2 = feature(ID_b)
X3 = feature(ID_l)
X4 = feature(ID_g)

Y1 = feature(OOD_a)
Y2 = feature(OOD_b)
Y3 = feature(OOD_l)
Y4 = feature(OOD_g)

T1 = feature(TEST_a)
T2 = feature(TEST_b)
T3 = feature(TEST_l)
T4 = feature(TEST_g)

id_feature = np.vstack((X1,X2,X3,X4)).T
ood_feature = np.vstack((Y1,Y2,Y3,Y4)).T
test_feature = np.vstack((T1,T2,T3,T4)).T

feature = np.vstack((id_feature,ood_feature))


X0=np.zeros((10000,1))
Y0=np.ones((26032,1))
label = np.vstack((X0,Y0))

clf = SVC(C=5,kernel='rbf',probability=True)
clf.fit(feature,label)


id_prob=clf.predict_proba(id_feature)

ood_prob = clf.predict_proba(test_feature)

for i in range(id_prob.shape[0]):
    f1.write("{}\n".format(id_prob[i,0]))
for i in range(ood_prob.shape[0]):
    f2.write("{}\n".format(ood_prob[i,0]))
