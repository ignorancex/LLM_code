'''
signal here are only 4-pronged samples

This code implements Logistic Regression
while optimization is done with tensorflow v1.4.0

Based on https://arxiv.org/abs/2002.12320
'''


import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import sys
from scipy.integrate import simps
from scipy import interpolate

class LogReg(object):

    def __init__(self,D,K,optimizer):
        self.D = D
        self.K = K
        self.optimizer = optimizer

        self.sess = tf.InteractiveSession()

        self.X = tf.placeholder(tf.float32, [None, self.D], name='X')
        self.y = tf.placeholder(tf.int64, [None], name='y')

        # building the model
        '''
        self.Z1 = self.nn_layer(self.X, self.D, 512, "H1",act=tf.nn.relu)
        self.Z2 = self.nn_layer(self.Z1, 512, 32, "H2",act=tf.nn.relu)
        self.y_pred = self.nn_layer(self.Z2, 32, self.K, "O")
        '''
        self.y_pred = self.nn_layer(self.X, self.D, self.K,"out")

        # defining loss function
        self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=self.y,
                                                                    logits=self.y_pred)
        #self.cross_entropy = self.CE_loss(self.y, self.y_pred)
        
        # optimizer
        self.train_step = self.optimizer.minimize(self.cross_entropy)

        # measures of performance
        self.correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                                               tf.float32))
        

        # initializing variables
        tf.global_variables_initializer().run()
        

    def weight_variable(self, shape,Lname):
        #initial = tf.truncated_normal(shape, stddev=0.1)
        initializer = tf.contrib.layers.xavier_initializer()
        #return tf.Variable(initial)
        return tf.get_variable(Lname,shape,initializer=initializer)
        
    def nn_layer(self, input_tensor, input_dim, output_dim, Lname,
                 act=tf.nn.softmax):
        self.weights = self.weight_variable([input_dim, output_dim],Lname)
        preactivate = tf.matmul(input_tensor, self.weights)
        activations = act(preactivate)
        return activations

    def CE_loss(self, t, Y):
        '''
        this is the cross-entropy loss-function
        '''
        T = tf.one_hot(t, self.K)
        tot = tf.math.multiply(T,tf.math.log(Y))
        return -tf.reduce_sum(tot)


    def fit_and_predict(self, X_train, t_train, X_test, t_test, N_epochs, batch_size):

        n_batches = int(np.ceil(float(X_train.shape[0]) / batch_size))
        for e in range(N_epochs):
            if e % 10 == 0:
                acc = self.sess.run(self.accuracy,feed_dict={self.X: X_test,
                                                             self.y: t_test})
                print('Accuracy at step %s: %s' % (e, acc))
                #y_test = self.sess.run(self.y_pred,feed_dict={self.X: X_test})
                #print('first preds: ',y_test[:10])
            else:
                permut =  np.random.permutation(len(X_train))
                X_batch = X_train[permut][:batch_size]
                t_batch = t_train[permut][:batch_size]
                self.sess.run(self.train_step,feed_dict={self.X: X_batch, self.y: t_batch})
            

    def get_ROC(self,X_test,t_test):
        #compute the ROC curve
        y_pred_fin = self.sess.run(self.y_pred,feed_dict={self.X:X_test})
        y_pred_fin = y_pred_fin[:,0]
        fpr, tpr, thresholds = metrics.roc_curve(t_test, y_pred_fin, pos_label=0)
        return fpr, tpr, thresholds
                

    def get_weights(self):
        return self.sess.run(self.weights)

    def get_ypred(self,X_test):
        return self.sess.run(self.y_pred,feed_dict={self.X:X_test})

def read_and_preprocess(M):
    
    print('reading signal train data')
    directory='taus_ungroomed/std1000/'
    signal_train_1 = np.loadtxt(directory+'train_M80/R2200-M80-2P_b.all.dat')
    signal_train_2 = np.loadtxt(directory+'train_M80/R2200-M80-2P_u.all.dat')
    
    print('reading signal test data')
    signal_test_1 = np.loadtxt(directory+'test_M80/R2200-M80-2P_g.all.dat')
    signal_test_2 = np.loadtxt(directory+'test_M80/R2200-M80-AA15_b.all.dat')
    signal_test_3 = np.loadtxt(directory+'test_M80/R2200-M80-AA30_b.all.dat')
    signal_test_4 = np.loadtxt(directory+'test_M80/R2200-M80-AA30_u.all.dat')
    signal_test_5 = np.loadtxt(directory+'test_M80/R2200-Whad.all.dat')
    signal_test_6 = np.loadtxt(directory+'test_M80/R2200-M80-N3P.all.dat')
    signal_test_7 = np.loadtxt(directory+'test_M80/R2200-M80-bb.all.dat')

    print('reading background data')
    bckg_q = np.loadtxt(directory+'train_M80/Zq-PT1000.all.dat')
    bckg_g = np.loadtxt(directory+'train_M80/Zg-PT1000.all.dat')


    print('balancing sub-classes of background and (train) signal')
    np.random.seed(123)
    delind = np.random.permutation(len(bckg_q) - len(bckg_g))
    bckg_q = np.delete(bckg_q,delind,axis=0)
    print("   final length of bckg gluon samples: ",len(bckg_g))
    print("   final length of bckg quark samples: ",len(bckg_q))
    bckg = np.vstack((bckg_q,bckg_g))
    np.random.seed(11)
    bckg = bckg[ np.random.permutation(len(bckg_q) + len(bckg_g)) ]

    np.random.seed(321) 
    delind1 = np.random.permutation(len(signal_train_1) - len(signal_train_2))
    signal_train_1 = np.delete(signal_train_1,delind1,axis=0)

    print('   final lengths for signal train sub-samples:',len(signal_train_1),
          len(signal_train_2))
    
    signal_train = np.vstack((signal_train_1,signal_train_2))
    np.random.seed(1)
    signal_train = signal_train[np.random.permutation(len(signal_train_1)\
                                                      +len(signal_train_2))]
    
    
    print('remove first 2 columns from data')
    bckg=bckg[:,2:]
    signal_train=signal_train[:,2:]
    signal_test_1=signal_test_1[:,2:]
    signal_test_2=signal_test_2[:,2:]
    signal_test_3=signal_test_3[:,2:]
    signal_test_4=signal_test_4[:,2:]
    signal_test_5=signal_test_5[:,2:]
    signal_test_6=signal_test_6[:,2:]
    signal_test_7=signal_test_7[:,2:]

    
    print('using only taus corresponding to phase space of M particles')
    M0=9 # initial size of phase space in original data
    # selecting corresponding columns
    orind = np.arange(0,3*M0-4) # original indices
    print('all indices:',orind)
    M0m1block= orind[:M0-1] # first block
    sel_ind = M0m1block[:M-1]
    M0m1block= orind[M0-1:2*(M0-1)] # second block
    sel_ind= np.hstack((sel_ind,M0m1block[:M-1]))
    M0m1block= orind[2*(M0-1):3*(M0-1)] # third block
    sel_ind= np.hstack((sel_ind,M0m1block[:M-2]))
    print(sel_ind)

    # reducing the dimension of the data
    bckg=bckg[:,sel_ind]
    signal_train=signal_train[:,sel_ind]
    signal_test_1=signal_test_1[:,sel_ind]
    signal_test_2=signal_test_2[:,sel_ind]
    signal_test_3=signal_test_3[:,sel_ind]
    signal_test_4=signal_test_4[:,sel_ind]
    signal_test_5=signal_test_5[:,sel_ind]
    signal_test_6=signal_test_6[:,sel_ind]
    signal_test_7=signal_test_7[:,sel_ind]
    

    print('balancing train & test classes')
    np.random.seed(33333) # change here for different random train samples
    bckg = bckg[np.random.permutation(len(bckg))]
    bckg_train = bckg[:len(signal_train)]
    bckg  = bckg[len(signal_train):]
    bckg_test_1 = bckg[:len(signal_test_1)]
    np.random.seed(4)
    bckg = bckg[np.random.permutation(len(bckg))]
    bckg_test_2 = bckg[:len(signal_test_2)]
    np.random.seed(5)
    bckg = bckg[np.random.permutation(len(bckg))]
    bckg_test_3 = bckg[:len(signal_test_3)]
    np.random.seed(6)
    bckg = bckg[np.random.permutation(len(bckg))]
    bckg_test_4 = bckg[:len(signal_test_4)]
    np.random.seed(7)
    bckg = bckg[np.random.permutation(len(bckg))]
    bckg_test_5 = bckg[:len(signal_test_5)]
    np.random.seed(8)
    bckg = bckg[np.random.permutation(len(bckg))]
    bckg_test_6 = bckg[:len(signal_test_6)]
    np.random.seed(9)
    bckg = bckg[np.random.permutation(len(bckg))]
    bckg_test_7 = bckg[:len(signal_test_7)]
    

    X = np.vstack((signal_train, bckg_train))
    X_test_1 = np.vstack((signal_test_1, bckg_test_1))
    X_test_2 = np.vstack((signal_test_2, bckg_test_2))
    X_test_3 = np.vstack((signal_test_3, bckg_test_3))
    X_test_4 = np.vstack((signal_test_4, bckg_test_4))
    X_test_5 = np.vstack((signal_test_5, bckg_test_5))
    X_test_6 = np.vstack((signal_test_6, bckg_test_6))
    X_test_7 = np.vstack((signal_test_7, bckg_test_7))

    print('creating output labels')
    t = np.array([0]*len(signal_train) + [1]*len(bckg_train))
    t_test_1 = np.array([0]*len(signal_test_1) + [1]*len(bckg_test_1))
    t_test_2 = np.array([0]*len(signal_test_2) + [1]*len(bckg_test_2))
    t_test_3 = np.array([0]*len(signal_test_3) + [1]*len(bckg_test_3))
    t_test_4 = np.array([0]*len(signal_test_4) + [1]*len(bckg_test_4))
    t_test_5 = np.array([0]*len(signal_test_5) + [1]*len(bckg_test_5))
    t_test_6 = np.array([0]*len(signal_test_6) + [1]*len(bckg_test_6))
    t_test_7 = np.array([0]*len(signal_test_7) + [1]*len(bckg_test_7))

    print('randomizing test samples')
    np.random.seed(8)
    perm1 = np.random.permutation(len(X_test_1))
    X_test_1 = X_test_1[perm1]
    t_test_1 = t_test_1[perm1]
    np.random.seed(9)
    perm2 = np.random.permutation(len(X_test_2))
    X_test_2 = X_test_2[perm2]
    t_test_2 = t_test_2[perm2]
    np.random.seed(10)
    perm3 = np.random.permutation(len(X_test_3))
    X_test_3 = X_test_3[perm3]
    t_test_3 = t_test_3[perm3]
    np.random.seed(14)
    perm4 = np.random.permutation(len(X_test_4))
    X_test_4 = X_test_4[perm4]
    t_test_4 = t_test_4[perm4]
    np.random.seed(15)
    perm5 = np.random.permutation(len(X_test_5))
    X_test_5 = X_test_5[perm5]
    t_test_5 = t_test_5[perm5]
    np.random.seed(16)
    perm6 = np.random.permutation(len(X_test_6))
    X_test_6 = X_test_6[perm6]
    t_test_6 = t_test_6[perm6]
    np.random.seed(17)
    perm7 = np.random.permutation(len(X_test_7))
    X_test_7 = X_test_7[perm7]
    t_test_7 = t_test_7[perm7]


    # getting log of the input (not done when reproducing JAAS paper)
    t = np.delete(t,np.unique(np.where(X==0)[0]),axis=0)
    X = np.delete(X,np.unique(np.where(X==0)[0]),axis=0)
    X = np.log10(X)
    t_test_1 = np.delete(t_test_1,np.unique(np.where(X_test_1==0)[0]),axis=0)
    X_test_1 = np.delete(X_test_1,np.unique(np.where(X_test_1==0)[0]),axis=0)
    X_test_1 = np.log10(X_test_1)
    t_test_2 = np.delete(t_test_2,np.unique(np.where(X_test_2==0)[0]),axis=0)
    X_test_2 = np.delete(X_test_2,np.unique(np.where(X_test_2==0)[0]),axis=0)
    X_test_2 = np.log10(X_test_2)
    t_test_3 = np.delete(t_test_3,np.unique(np.where(X_test_3==0)[0]),axis=0)
    X_test_3 = np.delete(X_test_3,np.unique(np.where(X_test_3==0)[0]),axis=0)
    X_test_3 = np.log10(X_test_3)
    t_test_4 = np.delete(t_test_4,np.unique(np.where(X_test_4==0)[0]),axis=0)
    X_test_4 = np.delete(X_test_4,np.unique(np.where(X_test_4==0)[0]),axis=0)
    X_test_4 = np.log10(X_test_4)
    t_test_5 = np.delete(t_test_5,np.unique(np.where(X_test_5==0)[0]),axis=0)
    X_test_5 = np.delete(X_test_5,np.unique(np.where(X_test_5==0)[0]),axis=0)
    X_test_5 = np.log10(X_test_5)
    t_test_6 = np.delete(t_test_6,np.unique(np.where(X_test_6==0)[0]),axis=0)
    X_test_6 = np.delete(X_test_6,np.unique(np.where(X_test_6==0)[0]),axis=0)
    X_test_6 = np.log10(X_test_6)
    t_test_7 = np.delete(t_test_7,np.unique(np.where(X_test_7==0)[0]),axis=0)
    X_test_7 = np.delete(X_test_7,np.unique(np.where(X_test_7==0)[0]),axis=0)
    X_test_7 = np.log10(X_test_7)
    
    
    
    print('splitting train+validation')
    np.random.seed(16)
    perm = np.random.permutation(len(X))
    X = X[perm]
    t = t[perm]
    val_size = int(0.2*len(X))
    X_val = X[:val_size]
    X_train = X[val_size:]
    t_val = t[:val_size]
    t_train = t[val_size:]

    return X_train,t_train,X_val,t_val,\
        X_test_1,t_test_1,X_test_2,t_test_2,X_test_3,t_test_3,\
        X_test_4,t_test_4,X_test_5,t_test_5,X_test_6,t_test_6,\
        X_test_7,t_test_7,sel_ind


def main():
    task_learn = False
    task_roc = False

    task = int(sys.argv[1])
    if task==1:
        task_learn = True
    else:
        task_roc = True

    if(task_learn):
        M = int(sys.argv[2]) # phase space (out of M0=9 possible)

        X_train,t_train,X_val,t_val,\
            X_test_1,t_test_1,X_test_2,t_test_2,X_test_3,t_test_3,\
            X_test_4,t_test_4,X_test_5,t_test_5,X_test_6,t_test_6,\
            X_test_7,t_test_7,sel_taus \
            = read_and_preprocess(M)

        D = X_train.shape[1]
        K = 2
        
        lrate = 0.01 #0.05 #0.25 # learning rate
        optimizer=tf.train.AdamOptimizer(lrate)
        model = LogReg(D,K,optimizer)
        N_epochs = 1000 #1000
        batch_size=200 #200
        print('training the model')
        model.fit_and_predict(X_train,t_train,X_val,t_val,N_epochs,batch_size)

        print(' collecting and writing the weights')
        weights = model.get_weights()
        print('all weights for signal=',weights[:,0])
        all_taus_list = np.array(['tau1_b1','tau2_b1','tau3_b1','tau4_b1','tau5_b1',
                         'tau6_b1','tau7_b1','tau8_b1',
                         'tau1_b2','tau2_b2','tau3_b2','tau4_b2','tau5_b2',
                         'tau6_b2','tau7_b2','tau8_b2',
                         'tau1_b0','tau2_b0','tau3_b0','tau4_b0','tau5_b0',
                         'tau6_b0','tau7_b0'])
        selected_taus = all_taus_list[np.array(sel_taus)]

        directorio='taus_ungroomed/std1000/results_M80/'
        os.system('rm '+directorio+'/weights_signal_LR_2P_M80.dat')
        fileow=open(directorio+'/weights_signal_LR_2P_M80.dat',"a")
        for i in range(len(weights)):
            fileow.write(" ".join([selected_taus[i],str(weights[i,0]),"\n"]))
        fileow.close()
        quit()
        
        print('getting ROC points')
        print(' test sample 1 ')
        fpr1,tpr1,thresholds=model.get_ROC(X_test_1,t_test_1)
        print(' test sample 2')
        fpr2,tpr2,thresholds=model.get_ROC(X_test_2,t_test_2)
        print(' test sample 3')
        fpr3,tpr3,thresholds=model.get_ROC(X_test_3,t_test_3)
        print(' test sample 4')
        fpr4,tpr4,thresholds=model.get_ROC(X_test_4,t_test_4)
        print(' test sample 5')
        fpr5,tpr5,thresholds=model.get_ROC(X_test_5,t_test_5)
        print(' test sample 6')
        fpr6,tpr6,thresholds=model.get_ROC(X_test_6,t_test_6)

        
        #os.system('rm results_LogReg/roc_points_LogReg_*.dat')
        fileo1=open('results_LogReg/roc_points_LogReg_2P_g_M'+str(M)+'_.dat',"a")
        fileo2=open('results_LogReg/roc_points_LogReg_AA15b_M'+str(M)+'_.dat',"a")
        fileo3=open('results_LogReg/roc_points_LogReg_AA30b_M'+str(M)+'_.dat',"a")
        fileo4=open('results_LogReg/roc_points_LogReg_AA30u_M'+str(M)+'_.dat',"a")
        fileo5=open('results_LogReg/roc_points_LogReg_W_M'+str(M)+'_.dat',"a")
        fileo6=open('results_LogReg/roc_points_LogReg_N3P_M'+str(M)+'_.dat',"a")
        
        for i in range(len(fpr1)):
            fileo1.write(" ".join([str(fpr1[i]),str(tpr1[i]),"\n"]) )
        for i in range(len(fpr2)):
            fileo2.write(" ".join([str(fpr2[i]),str(tpr2[i]),"\n"]) )
        for i in range(len(fpr3)):
            fileo3.write(" ".join([str(fpr3[i]),str(tpr3[i]),"\n"]) )
        for i in range(len(fpr4)):
            fileo4.write(" ".join([str(fpr4[i]),str(tpr4[i]),"\n"]) )
        for i in range(len(fpr5)):
            fileo5.write(" ".join([str(fpr5[i]),str(tpr5[i]),"\n"]) )
        for i in range(len(fpr6)):
            fileo6.write(" ".join([str(fpr6[i]),str(tpr6[i]),"\n"]) )

        fileo1.close()
        fileo2.close()
        fileo3.close()
        fileo4.close()
        fileo5.close()
        fileo6.close()
        

    if(task_roc):

        sample = '2P_g'
        color_dict={'AA15b':'green','AA30b':'red','AA30u':'orange',
                    '2P_g':'brown','W':'blue'}
        os.system('ls results_LogReg/roc_points_LogReg_'+sample+'_M9_*.dat > \
        results_LogReg/roc_files_'+sample+'.dat')
        roc_files1=np.loadtxt('results_LogReg/roc_files_'+sample+'.dat',dtype=str)
        roc_files2=np.loadtxt('results_2HLNN/roc_files_'+sample+'.dat',dtype=str)
        fig, ax = plt.subplots()
        c=0
        for f in roc_files1:
            print('file=',f)
            fpr,tpr = np.loadtxt(f,unpack=True)
            ax.plot(tpr,1/fpr,lw=2,c=color_dict.get(sample),
                    label='LR#'+str(c+1))
            c+=1
        c=0
        for f in roc_files2:
            print('file=',f)
            fpr,tpr = np.loadtxt(f,unpack=True)
            ax.plot(tpr,1/fpr,lw=2,ls='dashed',c=color_dict.get(sample),
                    alpha=0.5,label='NN#'+str(c+1))
            c+=1
        plt.yscale("log")
        plt.ylim(1,10000)
        plt.xticks([0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.grid(which='both')
        plt.title('test sample: '+sample)
        plt.xlabel('true positive rate',fontsize=11)
        plt.ylabel('(false positive rate)$^{-1}$',fontsize=11)
        plt.legend()
        plt.savefig('fig1right_LR-vs-NN_'+sample+'.pdf')
        
        
    

if __name__ == '__main__':
    main()
