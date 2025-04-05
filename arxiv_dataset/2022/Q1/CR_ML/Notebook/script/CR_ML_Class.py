#!/usr/bin/python3

import numpy as np
from sklearn.model_selection import train_test_split
import random
import time
from scipy import interpolate
import importlib
from tqdm import tqdm
import logging
importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

"""
Mock_Data_to_NumpyArray
"""
class Mock_Data_to_NumpyArray:
    def __init__(self, mock_data_file):
        """
        Usage: 
            Mock_Data_to_NumpyArray("mock_data_file.txt")
        Return:
            Null
        Item:
            self.parameter, self.spectrum, self.chisq 
        """
        import time
        logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        logging.info("Now loading...")
        para_smoothly_raw = []
        data_smoothly_raw = []
        chisq = []
        with open(mock_data_file,"r") as fname1:
            for i,t in enumerate(fname1.readlines()):
                if i%86==0:
                    try:
                        para_smoothly_raw.append(np.array(t.split()[1:]).astype(np.float)) ### take away the word "Para:"\
                    except ValueError:
                        pass
                elif i%86==85:  ##this item is chisq
                    chisq.append(float(t)) 
                else:
                    try:
                        data_smoothly_raw.append(np.array(t.split()).astype(np.float)) 
                    except ValueError:
                        pass

        parameter = np.array(para_smoothly_raw)
        spectrum = np.array(data_smoothly_raw)
        chisq = np.array(chisq)
        logging.info("Total data: {}".format(parameter.shape[0]))

        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))
        
        self.parameter, self.spectrum, self.chisq = parameter, spectrum, chisq
#=========================================================================================================#
        
        
"""
Mock_Data_Rescale
"""        
class Mock_Data_Rescale:
    def __init__(self, origin_parameter=[], new_parameter=[], spectrum=[], usedata = False):
        """
        Usage: 
            Mock_Data_Rescale(origin_parameter=origin_parameter, new_parameter=new_parameter, spectrum=spectrum, usedata = False)
        Return:
            Null
        Item:
            self.data
        """
        self.len = len(origin_parameter)

        self.origin_parameter = origin_parameter
        self.new_parameter = new_parameter
        self.origin_Ap = origin_parameter[:,5]
        self.new_Ap = new_parameter[:,5]
        self.origin_NLi, self.origin_NBe, self.origin_NO = origin_parameter[:,11], origin_parameter[:,12], origin_parameter[:,13]
        self.new_NLi, self.new_NBe, self.new_NO = new_parameter[:,11], new_parameter[:,12], new_parameter[:,13]
        
        if usedata == False:
            self.E = np.reshape(spectrum[:,0],(self.len,84))
            self.Li = np.reshape(spectrum[:,6],(self.len,84))
            self.Be = np.reshape(spectrum[:,7],(self.len,84))
            self.B = np.reshape(spectrum[:,8],(self.len,84))
            self.C = np.reshape(spectrum[:,9],(self.len,84))
            self.O = np.reshape(spectrum[:,10],(self.len,84))
        
        
        if usedata == True:
            self.E = spectrum[:,:,0]
            self.Li = spectrum[:,:,1]
            self.Be = spectrum[:,:,2]
            self.B = spectrum[:,:,3]
            self.C = spectrum[:,:,4]
            self.O = spectrum[:,:,5]
        
        self.newLi = self.Li/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))/np.reshape(self.origin_NLi,(self.len,1))*np.reshape(self.new_NLi,(self.len,1))
        self.newBe = self.Be/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))/np.reshape(self.origin_NBe,(self.len,1))*np.reshape(self.new_NBe,(self.len,1))
        self.newB = self.B/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))
        self.newC = self.C/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))
        self.newO = self.O/np.reshape(self.origin_Ap,(self.len,1))*np.reshape(self.new_Ap,(self.len,1))/np.reshape(self.origin_NO,(self.len,1))*np.reshape(self.new_NO,(self.len,1))
    
        data = np.zeros((self.len,84,6))
        spectrumlist = [self.E, self.newLi, self.newBe, self.newB, self.newC, self.newO]
        for i in range(6):
            data[:,:,i] = spectrumlist[i]

        self.data = data
#=========================================================================================================#        
   
"""
Mock_Data_Processing
"""
class Mock_Data_Processing:
    def __init__(self, parameter=[], E=0, Li=0, Be=0, B=0, C=0, O=0, data=0, usedata = False):
        """
        Usage: 
            Mock_Data_Processing(parameter, E, Li, Be, B, C, O, data, usedata = False)
        Return:
            Null
        Item:
            Null
        """
        self.len = len(parameter)
        self.E, self.Li, self.Be, self.B, self.C, self.O = E, Li, Be, B, C, O
        self.parameter = parameter
        self.NLi, self.NBe, self.NO = parameter[:,11], parameter[:,12], parameter[:,13]
        self.LiC, self.BeC, self.BC = 0, 0, 0
        self.LiO, self.BeO, self.BO = 0, 0, 0
        self.CC_110, self.OC = 0, 0
        self.input_train, self.input_test, self.source_train, self.source_test = 0, 0, 0, 0
        self.total_data_in_ratio = 0
        if usedata == True:
            self.len = len(data)
            self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        
            
    def spectrum_ratio(self, mock = True):
        """
        Usage: 
            Mock_Data_Processing(parameter, E, Li, Be, B, C, O, data, usedata = False).spectrum_ratio(mock = True)
        Return:
            Null
        Item:
            self.total_data_in_ratio
        """
        import time
        logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        '''
        Whitening
        '''
        logging.info("Whitening")
        logging.info("=====START=====")
        t1 = time.time()
        
        total_data_in_ratio = np.zeros((self.len, 84, 8))
        C_109_5 = self.C[:,52].reshape(self.len,1)
        N_Li = self.NLi.reshape(self.len,1)
        N_Be = self.NBe.reshape(self.len,1)
        N_O = self.NO.reshape(self.len,1)
        
        total_data_in_ratio[:,:,0] = (self.Li/N_Li)/(self.C/C_109_5) # (Li/N_Li)/(C/C_109.5)
        total_data_in_ratio[:,:,1] = (self.Be/N_Be)/(self.C/C_109_5) # (Be/N_Be)/(C/C_109.5)
        total_data_in_ratio[:,:,2] = self.B/(self.C/C_109_5) # B/(C/C_109.5)

        total_data_in_ratio[:,:,3] = (self.Li/N_Li)/(self.O/N_O) # (Li/N_Li)/(O/N_O)
        total_data_in_ratio[:,:,4] = (self.Be/N_Be)/(self.O/N_O) # (Be/N_Be)/(O/N_O)
        total_data_in_ratio[:,:,5] = self.B/(self.O/N_O) # B/(O/N_O)

        if mock:
            total_data_in_ratio[:,:,6] = self.C/C_109_5 # C/C_109.5
            total_data_in_ratio[:,:,7] = (self.O/N_O)/C_109_5  # (O/N_O)/C_109.5
        """
        Needo to Check
        """
        if not mock:
            """for pseudo data"""
            total_data_in_ratio[i,:,6] = self.C[i,:] # C
            total_data_in_ratio[i,:,7] = (self.O[:,:]/N_O)/(self.C[:,:]/C_109_5) # (O/N_O)/(C/C_109.5)


        
        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")


        self.LiC, self.BeC, self.BC = total_data_in_ratio[:,:,0], total_data_in_ratio[:,:,1], total_data_in_ratio[:,:,2]
        self.LiO, self.BeO, self.BO = total_data_in_ratio[:,:,3], total_data_in_ratio[:,:,4], total_data_in_ratio[:,:,5]
        self.CC_110, self.OC = total_data_in_ratio[:,:,6], total_data_in_ratio[:,:,7]
        self.total_data_in_ratio = total_data_in_ratio
        
        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        
        
        
        
    def Train_Test_split(self, splitrate = 0.1, split = True):
        """
        Usage: 
            Mock_Data_Processing(parameter, E, Li, Be, B, C, O, data, usedata = False).Train_Test_split(splitrate = 0.1, split = True)
        Return:
            Null
        Item:
            self.input_train, self.input_test, self.source_train, self.source_test
        """
        from sklearn.model_selection import train_test_split
        import random
        import time
        def Apply_Norm(xin, xmin, sc_factor):
            xout=(xin[:]-xmin)/sc_factor
            return xout
        logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        logging.info("\033[3;33m{}\033[0;m".format("Prepare Ratio"))
        self.spectrum_ratio()
        """     
        random split traning sample and test sample, 10% for test
        """
        logging.info("random split traning sample and test sample, 10% for test")
        logging.info("=====START=====")
        t1 = time.time()
        if split == True:
            ran = random.randint(0,100000)
            splitting = train_test_split(self.total_data_in_ratio, self.parameter, test_size=splitrate, random_state = np.random.seed(ran), shuffle = True )
            train_raw, predict_raw = splitting[0], splitting[1]  
            train_raw_para, predict_raw_para = splitting[2], splitting[3]
            
            
            
        elif split == False:
            train_raw = self.total_data_in_ratio
            train_raw_para = self.parameter
            predict_raw = np.zeros((0, 84, 8))
            predict_raw_para = np.zeros((0, 14))
        
        train_height, predict_height = train_raw.shape[0], predict_raw.shape[0]
        
        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")

        """
        Normolize every spectrum and take log_{10}
        """
        input_num = 8
        input_min    = [-10.25695965, -9.41842931, -8.77788068, -6.26408656, -5.42555622, -4.78500758, -9.43388299, -9.332034187]
        input_factor = [ 5.58788693,  4.50063727,  4.23359167,  5.95765576,  4.86602089,  4.59962876 , 16.58187362, 16.472289475]

        input_train = np.zeros((train_height, input_num, 84))
        input_test = np.zeros((predict_height, input_num, 84))
        ####### train_input 
        for i in range(train_height):
            for j in range(input_num):
                input_train[i,j,:] = Apply_Norm(np.log10(train_raw[i,:,j]), input_min[j], input_factor[j])
        ####### test_input
        for i in range(predict_height):
            for j in range(input_num):
                input_test[i,j,:] = Apply_Norm(np.log10(predict_raw[i,:,j]), input_min[j], input_factor[j])

        D_min, D_max = [1.0, 15.0]
        delta_min, delta_max = [0.20, 0.72]
        zh_min, zh_max=[1.0, 20.0]
        va_min, va_max=[1.0, 60.0]
        eta_min, eta_max=[-4.0, 2.0]
        # Ap_min, Ap_max=[]
        nu1_min, nu1_max=[0.0, 3.0]
        nu2_min, nu2_max=[2.0, 2.8]
        R1_min, R1_max=[2.3, 4.3]
        nu3_min, nu3_max=[1.8, 2.7]
        R2_min, R2_max=[5.0, 6.0]
        para_min    = [D_min, delta_min, zh_min, va_min, eta_min, nu1_min, nu2_min, R1_min, nu3_min, R2_min]
        para_factor = [D_max-D_min, delta_max-delta_min, zh_max-zh_min, va_max-va_min, eta_max-eta_min, 
                       nu1_max-nu1_min, np.round(nu2_max-nu2_min,2), R1_max-R1_min, 
                       np.round(nu3_max-nu3_min,2), R2_max-R2_min]

        """
        Fitting target -> $D_0$, $\delta$, $z_h$, $v_A$, $\eta$, $\nu_1$, $\nu_2$, $log_{10}(R_{br,1})$, $\nu_3$, $log_{10}(R_{br,2})$
        """
        source_train = np.zeros((train_height, 10))
        source_test = np.zeros((predict_height, 10))
        for i in range(source_train.shape[1]):
            if i >= 5:
                j = i + 1
            else:
                j = i
            source_train[:, i] = Apply_Norm(train_raw_para[:,j], para_min[i], para_factor[i])   
            source_test[:, i] = Apply_Norm(predict_raw_para[:,j], para_min[i], para_factor[i])
        

        logging.info("Shape for training Input: {}".format(input_train.shape))
        logging.info("Shape for  testing Input: {}".format(input_test.shape))
        logging.info("Shape for training Target: {}".format(source_train.shape))
        logging.info("Shape for  testing Target: {}".format(source_test.shape))
        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        
        self.input_train, self.input_test = input_train, input_test
        self.source_train, self.source_test = source_train, source_test
#=========================================================================================================#        
   
    
        
"""
Select_Sample
"""
class Select_Sample:
    def __init__(self, parameter=[], data=[], total_chisq_list=[], sigma=[]):
        """
        Usage: 
            Select_Sample(parameter, data, total_chisq_list, sigma)
        Return:
            Null
        Item:
            Null
        """
        self.len = len(parameter)
        self.parameter = parameter
        self.data = data
        self.total_chisq_list = total_chisq_list
        self.sigma = sigma
        
    def Sample(self):
        """
        Usage: 
            Select_Sample(parameter, data, total_chisq_list, sigma).Sample()
        Return:
            para_sigma, data_sigma, chi_sigma
        Item:
            Null
        """
        import time
        ticks_1 = time.time()
        #######################################################################################################
        # 1 sigma: min_chi_square + 15.94 
        # 3 sigma: min_chi_square + 33.20
        # 6 sigma: min_chi_square + 70
        if self.sigma == 1:
            deltachi = 15.94 
        elif self.sigma == 2:
            deltachi = 24.03
        elif self.sigma == 3:
            deltachi = 33.20
        elif self.sigma == 4:
            deltachi = 43.82
        elif self.sigma == 5:
            deltachi = 56.04
        elif self.sigma == 6:
            deltachi = 70
        else:
            logging.info(" ONLY 1 σ, 2 σ, 3 σ, 6 σ are available.")
            sys.exit(1)
        total_chisq_list = self.total_chisq_list
        
#         MIN = 347.73266
#         MIN = 347.33
#         MIN = min(min(total_chisq_list),347.73266)
#         logging.info(MIN)
        MIN = min(total_chisq_list)
        
        
        length = np.count_nonzero(total_chisq_list < MIN + deltachi )
#         length = np.count_nonzero(total_chisq_list < min(total_chisq_list) + deltachi )
        data_sigma = np.zeros((length,84,6))
        para_sigma = np.zeros((length, 14))
        chi_sigma = []
        count = 0

        for i in range(len(total_chisq_list)):
            if total_chisq_list[i] < MIN + deltachi:
                para_sigma[count] = self.parameter[i]
                chi_sigma.append(total_chisq_list[i])
                for j in range(6):
                    data_sigma[count,:,j] = self.data[i,:,j]
                count +=1
        logging.info("There are {} data in the {} σ region.".format(count,self.sigma))
        #######################################################################################################
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))

        return para_sigma, data_sigma, np.array(chi_sigma)
#=========================================================================================================#        
       
    

    
"""
Calculate_Chi_Square
"""
class Calculate_Chi_Square:
    def __init__(self, E=[0], Li=0, Be=0, B=0, C=0, O=0, data=0, usedata = False):
        """
        Usage: 
            Calculate_Chi_Square(E, Li, Be, B, C, O, data, usedata = False)
        Return:
            Null
        Item:
            Null
        """
        self.len = len(E)
        self.E, self.Li, self.Be, self.B, self.C, self.O = E, Li, Be, B, C, O
        if usedata == True:
            self.len = len(data)
            self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        
    def chi_square(self):
        """
        Usage: 
            Calculate_Chi_Square(E, Li, Be, B, C, O, data, usedata = False).chi_square()
        Return:
            total_chisq
        Item:
            Null
        """
        from scipy import interpolate
        import time
        
        def interpolate_chisq(P,F,S):
            return np.round(np.sum(((P-np.array(F))**2)/(np.array(S)**2), axis=1), 5)
        
        
        logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        ######################################################################################################

        """
        Load Experimental Data
        """
        ####################################################################################
        exp_data_path = "../Data/Exp_Data/"

        LiAMS, LiV = np.load(exp_data_path + "Li_AMS2.npy"),np.load(exp_data_path + "Li_Voyager.npy")
        Li_Eams, Li_Fams, Li_Sams = LiAMS[0], LiAMS[1], LiAMS[2]
        Li_Evy, Li_Fvy, Li_Svy = LiV[0], LiV[1], LiV[2]

        BeAMS, BeV = np.load(exp_data_path + "Be_AMS2.npy"), np.load(exp_data_path + "Be_Voyager.npy")
        Be_Eams, Be_Fams, Be_Sams = BeAMS[0], BeAMS[1], BeAMS[2]
        Be_Evy, Be_Fvy, Be_Svy = BeV[0], BeV[1], BeV[2]

        BAMS, BV, BA = np.load(exp_data_path + "B_AMS2.npy"), np.load(exp_data_path + "B_Voyager.npy"), np.load(exp_data_path + "B_ACE.npy")
        B_Eams, B_Fams, B_Sams = BAMS[0], BAMS[1], BAMS[2]
        B_Evy, B_Fvy, B_Svy = BV[0], BV[1], BV[2]
        B_Eace,B_Face,B_Sace = BA[0], BA[1], BA[2]

        CAMS, CV, CA = np.load(exp_data_path + "C_AMS2.npy"), np.load(exp_data_path + "C_Voyager.npy"), np.load(exp_data_path + "C_ACE.npy")
        C_Eams, C_Fams, C_Sams = CAMS[0], CAMS[1], CAMS[2]
        C_Evy, C_Fvy, C_Svy = CV[0], CV[1], CV[2]
        C_Eace,C_Face,C_Sace = CA[0], CA[1], CA[2]

        OAMS, OV, OA = np.load(exp_data_path + "O_AMS2.npy"), np.load(exp_data_path + "O_Voyager.npy"), np.load(exp_data_path + "O_ACE.npy")
        O_Eams, O_Fams, O_Sams = OAMS[0], OAMS[1], OAMS[2]
        O_Evy, O_Fvy, O_Svy = OV[0], OV[1], OV[2]
        O_Eace,O_Face,O_Sace = OA[0], OA[1], OA[2]
        ####################################################################################
        # Fit the Spectrum
        logging.info("Fit the Spectrum.")
        logging.info("=====START=====")
        t1 = time.time()
        
        Li_interpolate = interpolate.interp1d((self.E[0]), self.Li, kind='cubic')
        Be_interpolate = interpolate.interp1d((self.E[0]), self.Be, kind='cubic')
        B_interpolate = interpolate.interp1d((self.E[0]), self.B, kind='cubic')
        C_interpolate = interpolate.interp1d((self.E[0]), self.C, kind='cubic')
        O_interpolate = interpolate.interp1d((self.E[0]), self.O, kind='cubic')
        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")
        
        
        # Calculate Chi-Square
        logging.info("Calculate Chi-Square.")
        logging.info("=====START=====")
        t1 = time.time()
        total_chisq = np.zeros((self.len))
        total_chisq = (
            interpolate_chisq(P=Li_interpolate(Li_Eams), F=Li_Fams, S=Li_Sams)+
            interpolate_chisq(P=Li_interpolate(Li_Evy), F=Li_Fvy, S=Li_Svy)+

            interpolate_chisq(P=Be_interpolate(Be_Eams), F=Be_Fams, S=Be_Sams)+
            interpolate_chisq(P=Be_interpolate(Be_Evy), F=Be_Fvy, S=Be_Svy)+

            interpolate_chisq(P=B_interpolate(B_Eams), F=B_Fams, S=B_Sams)+
            interpolate_chisq(P=B_interpolate(B_Evy), F=B_Fvy, S=B_Svy)+
            interpolate_chisq(P=B_interpolate(B_Eace), F=B_Face, S=B_Sace)+

            interpolate_chisq(P=C_interpolate(C_Eams), F=C_Fams, S=C_Sams)+
            interpolate_chisq(P=C_interpolate(C_Evy), F=C_Fvy, S=C_Svy)+
            interpolate_chisq(P=C_interpolate(C_Eace), F=C_Face, S=C_Sace)+

            interpolate_chisq(P=O_interpolate(O_Eams), F=O_Fams, S=O_Sams)+
            interpolate_chisq(P=O_interpolate(O_Evy), F=O_Fvy, S=O_Svy)+
            interpolate_chisq(P=O_interpolate(O_Eace), F=O_Face, S=O_Sace)
        )
        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")
        logging.info("\n")
        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33m Total Time Consumption : {:.4f} min\033[0;m".format(totaltime/60.))
        return total_chisq
#=========================================================================================================#        
       
    
    
"""
ReCalculateAp
"""
class ReCalculateAp:
    def __init__(self, data=[]):
        """
        Usage: 
            ReCalculateAp(data)
        Return:
            Null
        Item:
            Null
        """
        self.length = len(data)
        self.data = data
        self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        self.ap_5, self.minchi = 0, 0

    def GetBestAp(self):
        """
        Usage: 
            ReCalculateAp(data).GetBestAp()
        Return:
            Null
        Item:
            self.ap_5, self.minchi
        """
        import time
        from scipy import interpolate
        
        logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################

        """
        Load Experimental Data
        """
        ####################################################################################
        exp_data_path = "../Data/Exp_Data/"

        LiAMS, LiV = np.load(exp_data_path + "Li_AMS2.npy"),np.load(exp_data_path + "Li_Voyager.npy")
        Li_Eams, Li_Fams, Li_Sams = LiAMS[0], LiAMS[1], LiAMS[2]
        Li_Evy, Li_Fvy, Li_Svy = LiV[0], LiV[1], LiV[2]

        BeAMS, BeV = np.load(exp_data_path + "Be_AMS2.npy"), np.load(exp_data_path + "Be_Voyager.npy")
        Be_Eams, Be_Fams, Be_Sams = BeAMS[0], BeAMS[1], BeAMS[2]
        Be_Evy, Be_Fvy, Be_Svy = BeV[0], BeV[1], BeV[2]

        BAMS, BV, BA = np.load(exp_data_path + "B_AMS2.npy"), np.load(exp_data_path + "B_Voyager.npy"), np.load(exp_data_path + "B_ACE.npy")
        B_Eams, B_Fams, B_Sams = BAMS[0], BAMS[1], BAMS[2]
        B_Evy, B_Fvy, B_Svy = BV[0], BV[1], BV[2]
        B_Eace,B_Face,B_Sace = BA[0], BA[1], BA[2]

        CAMS, CV, CA = np.load(exp_data_path + "C_AMS2.npy"), np.load(exp_data_path + "C_Voyager.npy"), np.load(exp_data_path + "C_ACE.npy")
        C_Eams, C_Fams, C_Sams = CAMS[0], CAMS[1], CAMS[2]
        C_Evy, C_Fvy, C_Svy = CV[0], CV[1], CV[2]
        C_Eace,C_Face,C_Sace = CA[0], CA[1], CA[2]

        OAMS, OV, OA = np.load(exp_data_path + "O_AMS2.npy"), np.load(exp_data_path + "O_Voyager.npy"), np.load(exp_data_path + "O_ACE.npy")
        O_Eams, O_Fams, O_Sams = OAMS[0], OAMS[1], OAMS[2]
        O_Evy, O_Fvy, O_Svy = OV[0], OV[1], OV[2]
        O_Eace,O_Face,O_Sace = OA[0], OA[1], OA[2]
        ####################################################################################

        def findbestAp(E,Li,Be,B,C,O):
            
            def interpolate_chisq(P,F,S):
                return np.round(np.sum(((P-np.array(F))**2)/(np.array(S)**2), axis=1), 10)
            
            aplist = np.linspace(1.9135/5.,7.0114/5.,500).reshape(500,1)
            norm_interpolate_Li = interpolate.interp1d((E), (Li*aplist), kind='cubic')
            norm_interpolate_Be = interpolate.interp1d((E), (Be*aplist), kind='cubic')
            pred_interpolate_B  = interpolate.interp1d((E), (B*aplist), kind='cubic')
            pred_interpolate_C  = interpolate.interp1d((E), (C*aplist), kind='cubic')
            norm_interpolate_O  = interpolate.interp1d((E), (O*aplist), kind='cubic')

            chisq = (interpolate_chisq(norm_interpolate_Li(Li_Eams),Li_Fams,Li_Sams)+
                     interpolate_chisq(norm_interpolate_Li(Li_Evy),Li_Fvy,Li_Svy)+

                     interpolate_chisq(norm_interpolate_Be(Be_Eams),Be_Fams,Be_Sams)+
                     interpolate_chisq(norm_interpolate_Be(Be_Evy),Be_Fvy,Be_Svy)+

                     interpolate_chisq(pred_interpolate_B(B_Eams),B_Fams,B_Sams)+
                     interpolate_chisq(pred_interpolate_B(B_Evy),B_Fvy,B_Svy)+
                     interpolate_chisq(pred_interpolate_B(B_Eace),B_Face,B_Sace)+

                     interpolate_chisq(pred_interpolate_C(C_Eams),C_Fams,C_Sams)+
                     interpolate_chisq(pred_interpolate_C(C_Evy),C_Fvy,C_Svy)+
                     interpolate_chisq(pred_interpolate_C(C_Eace),C_Face,C_Sace)+

                     interpolate_chisq(norm_interpolate_O(O_Eams),O_Fams,O_Sams)+
                     interpolate_chisq(norm_interpolate_O(O_Evy),O_Fvy,O_Svy)+
                     interpolate_chisq(norm_interpolate_O(O_Eace),O_Face,O_Sace)
                    )
            
            return np.where(chisq == min(chisq))[0][0], chisq

        logging.info("Finding best Ap")
        logging.info("=====START=====")
        t1 = time.time()

        aplist = np.linspace(1.9135/5.,7.0114/5.,500)
        ap_5, minchi = [], []
        for i in tqdm(range(self.length)):
            index, chisq = findbestAp(self.E[i],self.Li[i],self.Be[i],self.B[i],self.C[i],self.O[i])
            minchi.append(min(chisq))
            ap_5.append(aplist[index])
                
        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")
        
        ap_5 = np.array(ap_5)
        minchi = np.array(minchi)
        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        
        self.ap_5, self.minchi = ap_5, minchi
#=========================================================================================================#        


"""
ReCalculateN
"""        
class ReCalculateN:
    import time
    def __init__(self, parameter=[], data=[], ap=1, ap_is1=False):
        """
        Usage: 
            ReCalculateN(parameter,data, ap=1, ap_is1=False)
        Return:
            Null
        Item:
            Null
        """
        self.length = len(data)
        self.data = data
        self.parameter = parameter
        self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        self.ap = ap
        self.new_factor, self.new_chi = 0, 0
        self.ap_is1 = ap_is1
    
    def GetBestN(self):
        """
        Usage: 
            ReCalculateN(parameter,data, ap=1, ap_is1=False).GetBestN()
        Return:
            Null
        Item:
            self.new_factor, self.new_chi
        """
        import time
        logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################

        """
        Load Experimental Data
        """
        ####################################################################################
        exp_data_path = "../Data/Exp_Data/"

        LiAMS, LiV = np.load(exp_data_path + "Li_AMS2.npy"),np.load(exp_data_path + "Li_Voyager.npy")
        Li_Eams, Li_Fams, Li_Sams = LiAMS[0], LiAMS[1], LiAMS[2]
        Li_Evy, Li_Fvy, Li_Svy = LiV[0], LiV[1], LiV[2]

        BeAMS, BeV = np.load(exp_data_path + "Be_AMS2.npy"), np.load(exp_data_path + "Be_Voyager.npy")
        Be_Eams, Be_Fams, Be_Sams = BeAMS[0], BeAMS[1], BeAMS[2]
        Be_Evy, Be_Fvy, Be_Svy = BeV[0], BeV[1], BeV[2]

        BAMS, BV, BA = np.load(exp_data_path + "B_AMS2.npy"), np.load(exp_data_path + "B_Voyager.npy"), np.load(exp_data_path + "B_ACE.npy")
        B_Eams, B_Fams, B_Sams = BAMS[0], BAMS[1], BAMS[2]
        B_Evy, B_Fvy, B_Svy = BV[0], BV[1], BV[2]
        B_Eace,B_Face,B_Sace = BA[0], BA[1], BA[2]

        CAMS, CV, CA = np.load(exp_data_path + "C_AMS2.npy"), np.load(exp_data_path + "C_Voyager.npy"), np.load(exp_data_path + "C_ACE.npy")
        C_Eams, C_Fams, C_Sams = CAMS[0], CAMS[1], CAMS[2]
        C_Evy, C_Fvy, C_Svy = CV[0], CV[1], CV[2]
        C_Eace,C_Face,C_Sace = CA[0], CA[1], CA[2]

        OAMS, OV, OA = np.load(exp_data_path + "O_AMS2.npy"), np.load(exp_data_path + "O_Voyager.npy"), np.load(exp_data_path + "O_ACE.npy")
        O_Eams, O_Fams, O_Sams = OAMS[0], OAMS[1], OAMS[2]
        O_Evy, O_Fvy, O_Svy = OV[0], OV[1], OV[2]
        O_Eace,O_Face,O_Sace = OA[0], OA[1], OA[2]
        ####################################################################################
        def findbestN(E,Li,Be,B,C,O,parameter,ap=1):

            def interpolate_chisq(P,F,S):
                return np.round(np.sum(((P-np.array(F))**2)/(np.array(S)**2), axis=1), 10)

            N_Li = parameter[11]
            N_Be = parameter[12]
            N_O = parameter[13]
            new_N = np.linspace(0.8,1.2,1000).reshape(1000,1)
            
        
            pred_interpolate_Li = interpolate.interp1d((E), (Li/N_Li*new_N*ap), kind='cubic')
            pred_interpolate_Be = interpolate.interp1d((E), (Be/N_Be*new_N*ap), kind='cubic')
            pred_interpolate_O  = interpolate.interp1d((E), (O/N_O*new_N*ap), kind='cubic')
            chisq_scan_Li = (interpolate_chisq(P=pred_interpolate_Li(Li_Eams), F=Li_Fams, S=Li_Sams)+
                             interpolate_chisq(P=pred_interpolate_Li(Li_Evy), F=Li_Fvy, S=Li_Svy))
            
            chisq_scan_Be = (interpolate_chisq(P=pred_interpolate_Be(Be_Eams), F=Be_Fams, S=Be_Sams)+
                             interpolate_chisq(P=pred_interpolate_Be(Be_Evy), F=Be_Fvy, S=Be_Svy))
            chisq_scan_O = (interpolate_chisq(P=pred_interpolate_O(O_Eams), F=O_Fams, S=O_Sams)+
                            interpolate_chisq(P=pred_interpolate_O(O_Evy), F=O_Fvy, S=O_Svy)+
                            interpolate_chisq(P=pred_interpolate_O(O_Eace), F=O_Face, S=O_Sace))
            
            new_normal_factor = np.zeros(3)
            new_normal_factor[0] = new_N[np.where(chisq_scan_Li == min(chisq_scan_Li))[0]]
            new_normal_factor[1] = new_N[np.where(chisq_scan_Be == min(chisq_scan_Be))[0]]
            new_normal_factor[2] = new_N[np.where(chisq_scan_O == min(chisq_scan_O))[0]]

            '''
            Calculate New chisquare
            '''
            # Change "axis" from 1 to 0
            def interpolate_chisq(P,F,S):
                return np.round(np.sum(((P-np.array(F))**2)/(np.array(S)**2), axis=0), 10)
            
            Li_interpolate = interpolate.interp1d(E, Li/N_Li*new_normal_factor[0]*ap, kind='cubic')
            Be_interpolate = interpolate.interp1d(E, Be/N_Be*new_normal_factor[1]*ap, kind='cubic')
            B_interpolate = interpolate.interp1d(E, B*ap, kind='cubic')
            C_interpolate = interpolate.interp1d(E, C*ap, kind='cubic')
            O_interpolate = interpolate.interp1d(E, O/N_O*new_normal_factor[2]*ap, kind='cubic')

            total_chisq = (
                interpolate_chisq(P=Li_interpolate(Li_Eams), F=Li_Fams, S=Li_Sams)+
                interpolate_chisq(P=Li_interpolate(Li_Evy), F=Li_Fvy, S=Li_Svy)+

                interpolate_chisq(P=Be_interpolate(Be_Eams), F=Be_Fams, S=Be_Sams)+
                interpolate_chisq(P=Be_interpolate(Be_Evy), F=Be_Fvy, S=Be_Svy)+

                interpolate_chisq(P=B_interpolate(B_Eams), F=B_Fams, S=B_Sams)+
                interpolate_chisq(P=B_interpolate(B_Evy), F=B_Fvy, S=B_Svy)+
                interpolate_chisq(P=B_interpolate(B_Eace), F=B_Face, S=B_Sace)+

                interpolate_chisq(P=C_interpolate(C_Eams), F=C_Fams, S=C_Sams)+
                interpolate_chisq(P=C_interpolate(C_Evy), F=C_Fvy, S=C_Svy)+
                interpolate_chisq(P=C_interpolate(C_Eace), F=C_Face, S=C_Sace)+

                interpolate_chisq(P=O_interpolate(O_Eams), F=O_Fams, S=O_Sams)+
                interpolate_chisq(P=O_interpolate(O_Evy), F=O_Fvy, S=O_Svy)+
                interpolate_chisq(P=O_interpolate(O_Eace), F=O_Face, S=O_Sace)
            )


            return new_normal_factor, total_chisq


        """
        Finding New Normalized Factor
        """   
        logging.info("Finding New Normalized Factor")
        logging.info("=====START=====")
        t1 = time.time()
        
        ap_5 = self.ap
        new_factor = np.zeros((self.length,3))
        new_chi = np.zeros(self.length)
        
        if self.ap_is1 == True:
            for i in tqdm(range(self.length)):
                new_factor[i,:], new_chi[i] = findbestN(self.E[i], self.Li[i], self.Be[i], self.B[i], self.C[i], self.O[i], self.parameter[i,:])
                
        elif self.ap_is1 == False:
            for i in tqdm(range(self.length)):
                new_factor[i,:], new_chi[i] = findbestN(self.E[i], self.Li[i], self.Be[i], self.B[i], self.C[i], self.O[i], self.parameter[i,:],ap=ap_5[i])
                
        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")
        
        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        self.new_factor, self.new_chi = new_factor, new_chi
        
#=========================================================================================================#    
          
          
"""
New_Parameter
"""      
class  New_Parameter:
    import time
    
    def __init__(self, parameter, new_factor, ap_5=1, ap_is1=False):
        """
        Usage: 
            New_Parameter(parameter, new_factor, ap_5=1, ap_is1=False)
        Return:
            Null
        Item:
            self.new_parameter
        """
        self.length = len(parameter)
        self.factor = new_factor
        self.parameter = parameter
        
        logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        if ap_is1 == True :
            new_parameter = np.zeros((parameter.shape[0],14))
            new_parameter[:,0:11] = parameter[:,0:11]
            new_parameter[:,11:14] = new_factor
        elif ap_is1 == False:
            new_parameter = np.zeros((parameter.shape[0],14))
            new_parameter[:,0:5] = parameter[:,0:5]
            new_parameter[:,5] = ap_5*5
            new_parameter[:,6:11] = parameter[:,6:11]
            new_parameter[:,11:14] = new_factor

        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
        
        self.new_parameter = new_parameter      
#=========================================================================================================#  


"""
Recovery
"""
class Recovery:
    def __init__(self, prediction):
        self.length = len(prediction)
            
        D_min, D_max=[1.0, 15.0]
        delta_min, delta_max=[0.20, 0.72]
        zh_min, zh_max=[1.0, 20.0]
        va_min, va_max=[1.0, 60.0]
        eta_min, eta_max=[-4.0, 2.0]
        # Ap_min, Ap_max=[]
        nu1_min, nu1_max=[0.0, 3.0]
        nu2_min, nu2_max=[2.0, 2.8]
        R1_min, R1_max=[2.3, 4.3]
        nu3_min, nu3_max=[1.8, 2.7]
        R2_min, R2_max=[5.0, 6.0]
        para_min    = [D_min, delta_min, zh_min, va_min, eta_min, nu1_min, nu2_min, R1_min, nu3_min, R2_min]
        para_factor = [D_max-D_min, delta_max-delta_min, zh_max-zh_min, va_max-va_min, eta_max-eta_min, 
                       nu1_max-nu1_min, np.round(nu2_max-nu2_min,2), R1_max-R1_min, 
                       np.round(nu3_max-nu3_min,2), R2_max-R2_min]

        para_recovery = np.zeros((len(prediction),10))
        for i in range(10):
            para_recovery[:,i] = prediction[:,i]*para_factor[i]+para_min[i]
            
            
        self.para_recovery = para_recovery  
        self.D,  self.delta, self.Zh, self.va, self.eta = para_recovery[:,0],para_recovery[:,1],para_recovery[:,2],para_recovery[:,3],para_recovery[:,4]
        self.nu1, self.nu2, self.R1, self.nu3, self.R2 = para_recovery[:,5],para_recovery[:,6],para_recovery[:,7],para_recovery[:,8],para_recovery[:,9]
#=========================================================================================================#  

               
            
            
            

                    
"""
Create_Pseudodata
"""
class Create_Pseudodata:
    def __init__(self, parameter=[], data=[], chi=[], LOW = 10 , HIGH = 30 , number = 100, index=0):
        """
        Usage: 
            Create_Pseudodata(parameter, data, chi, LOW = 10 , HIGH = 30 , number = 100, index=0)
        Return:
            Null
        Item:
            Null
        """
        self.length = len(data)
        self.data = data
        self.parameter = parameter
        self.chi = chi
        self.LOW = LOW
        self.HIGH = HIGH
        self.number = number
        
        self.E, self.Li, self.Be, self.B, self.C, self.O = data[:,:,0], data[:,:,1],data[:,:,2],data[:,:,3],data[:,:,4],data[:,:,5]
        self.index = index
        
    def Create_Pseudodata(self):
        """
        Usage: 
            Create_Pseudodata(parameter, data, chi, LOW = 10 , HIGH = 30 , number = 100, index=0).Create_Pseudodata()
        Return:
            normalfactor, pseudodata
        Item:
            Null
        """
        data = self.data
        parameter = self.parameter
        chi = self.chi
        LOW = self.LOW
        HIGH = self.HIGH
        number = self.number
        index = self.index
        
        def interpolate_chisq(P,F,S):
            return np.round(np.sum(((P-np.array(F))**2)/(np.array(S)**2), axis=1), 10)
        
        logging.info(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
        ticks_1 = time.time()
        #######################################################################################################
        """
        Load Experimental Data
        """
        logging.info("Experimental data are loading.")
        logging.info("=====START=====")
        t1 = time.time()
        ####################################################################################
        exp_data_path = "../Data/Exp_Data/"

        LiAMS, LiV = np.load(exp_data_path + "Li_AMS2.npy"),np.load(exp_data_path + "Li_Voyager.npy")
        Li_Eams, Li_Fams, Li_Sams = LiAMS[0], LiAMS[1], LiAMS[2]
        Li_Evy, Li_Fvy, Li_Svy = LiV[0], LiV[1], LiV[2]

        BeAMS, BeV = np.load(exp_data_path + "Be_AMS2.npy"), np.load(exp_data_path + "Be_Voyager.npy")
        Be_Eams, Be_Fams, Be_Sams = BeAMS[0], BeAMS[1], BeAMS[2]
        Be_Evy, Be_Fvy, Be_Svy = BeV[0], BeV[1], BeV[2]

        BAMS, BV, BA = np.load(exp_data_path + "B_AMS2.npy"), np.load(exp_data_path + "B_Voyager.npy"), np.load(exp_data_path + "B_ACE.npy")
        B_Eams, B_Fams, B_Sams = BAMS[0], BAMS[1], BAMS[2]
        B_Evy, B_Fvy, B_Svy = BV[0], BV[1], BV[2]
        B_Eace,B_Face,B_Sace = BA[0], BA[1], BA[2]

        CAMS, CV, CA = np.load(exp_data_path + "C_AMS2.npy"), np.load(exp_data_path + "C_Voyager.npy"), np.load(exp_data_path + "C_ACE.npy")
        C_Eams, C_Fams, C_Sams = CAMS[0], CAMS[1], CAMS[2]
        C_Evy, C_Fvy, C_Svy = CV[0], CV[1], CV[2]
        C_Eace,C_Face,C_Sace = CA[0], CA[1], CA[2]

        OAMS, OV, OA = np.load(exp_data_path + "O_AMS2.npy"), np.load(exp_data_path + "O_Voyager.npy"), np.load(exp_data_path + "O_ACE.npy")
        O_Eams, O_Fams, O_Sams = OAMS[0], OAMS[1], OAMS[2]
        O_Evy, O_Fvy, O_Svy = OV[0], OV[1], OV[2]
        O_Eace,O_Face,O_Sace = OA[0], OA[1], OA[2]
        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")
        ####################################################################################

        '''
        Using whitening data to make pseudodata
        '''
        ####################################################################################
        logging.info("Using whitening data to make pseudodata")
        logging.info("=====START=====")
        t1 = time.time()
        
        total_data_divAp = np.zeros((data.shape[0], 84, 8))
        
        C_109_5 = data[:,52,4].reshape(data.shape[0],1)
        N_Li = parameter[:,11].reshape(data.shape[0],1)
        N_Be = parameter[:,12].reshape(data.shape[0],1)
        N_O = parameter[:,13].reshape(data.shape[0],1)
        
        total_data_divAp[:,:,0] = (data[:,:,1]/N_Li)/(data[:,:,4]/C_109_5) # (Li/N_Li)/(C/C_109.5)
        total_data_divAp[:,:,1] = (data[:,:,2]/N_Be)/(data[:,:,4]/C_109_5) # (Be/N_Be)/(C/C_109.5)
        total_data_divAp[:,:,2] = data[:,:,3]/(data[:,:,4]/C_109_5) # B/(C/C_109.5)

        total_data_divAp[:,:,3] = (data[:,:,1]/N_Li)/(data[:,:,5]/N_O) # (Li/N_Li)/(O/N_O)
        total_data_divAp[:,:,4] = (data[:,:,2]/N_Be)/(data[:,:,5]/N_O) # (Be/N_Be)/(O/N_O)
        total_data_divAp[:,:,5] = data[:,:,3]/(data[:,:,5]/N_O) # B/(O/N_O)

        total_data_divAp[:,:,6] = data[:,:,4] # C
        total_data_divAp[:,:,7] = (data[:,:,5]/N_O)/(data[:,:,4]/C_109_5)  # (O/N_O)/(C/C_109.5)
        
        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")
        ####################################################################################
        '''
        Search The Pack of Spectrum
        '''
        logging.info("Search The Pack of Spectrum")
        logging.info("=====START=====")
        t1 = time.time()
        divAp_pack = np.zeros((2, 84, 8))
        divAp_pack[0,:,:] = np.max(total_data_divAp[:,:,:], axis=0)
        divAp_pack[1,:,:] = np.min(total_data_divAp[:,:,:], axis=0) 
    
                
        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")

        
        
        '''
        Create Pseudodata
        '''
        logging.info("Create Pseudodata")
        logging.info("=====START=====")
        t1 = time.time()
        
        spec_data_LiC = np.zeros((number,84))
        spec_data_BeC = np.zeros((number,84))
        spec_data_BC = np.zeros((number,84))
        spec_data_LiO = np.zeros((number,84))
        spec_data_BeO = np.zeros((number,84))
        spec_data_BO = np.zeros((number,84))
        spec_data_C = np.zeros((number,84))
        spec_data_OC = np.zeros((number,84))


        for i in range(number):
            rand = np.random.randint(data.shape[0])
#             rand = i

            data_delta = np.zeros((84, 8))
            delta = float(np.random.uniform(LOW, HIGH, size=None)) #15, 30
            data_delta = (np.max(total_data_divAp[:,:,:], axis=0) - np.min(total_data_divAp[:,:,:], axis=0))/delta
            
            '''
            arround one line + Delta_data
            '''      
            spec_data_LiC[i,:] = np.random.uniform(total_data_divAp[rand,:,0]-data_delta[:,0], total_data_divAp[rand,:,0]+data_delta[:,0], size=None)
            spec_data_BeC[i,:] = np.random.uniform(total_data_divAp[rand,:,1]-data_delta[:,1], total_data_divAp[rand,:,1]+data_delta[:,1], size=None)
            spec_data_BC[i,:]  = np.random.uniform(total_data_divAp[rand,:,2]-data_delta[:,2], total_data_divAp[rand,:,2]+data_delta[:,2], size=None)
            spec_data_C[i,:]  = np.random.uniform(total_data_divAp[rand,:,6]-data_delta[:,6], total_data_divAp[rand,:,6]+data_delta[:,6], size=None)
            spec_data_OC[i,:]  = np.random.uniform(total_data_divAp[rand,:,7]-data_delta[:,7], total_data_divAp[rand,:,7]+data_delta[:,7], size=None)
            
            
#             '''
#             pack
#             '''
#             spec_data_LiC[i,:] = np.random.uniform(divAp_pack[1,:,0], divAp_pack[0,:,0], size=None)
#             spec_data_BeC[i,:] = np.random.uniform(divAp_pack[1,:,1], divAp_pack[0,:,1], size=None)
#             spec_data_BC[i,:]  = np.random.uniform(divAp_pack[1,:,2], divAp_pack[0,:,2], size=None)
#             spec_data_C[i,:]  = np.random.uniform(divAp_pack[1,:,6], divAp_pack[0,:,6], size=None)
#             spec_data_OC[i,:]  = np.random.uniform(divAp_pack[1,:,7], divAp_pack[0,:,7], size=None)



#             '''
#             arround one line 
#             '''
#             spec_data_LiC[i,:] = np.random.uniform(total_data_divAp[rand,:,0], total_data_divAp[rand,:,0], size=None)
#             spec_data_BeC[i,:] = np.random.uniform(total_data_divAp[rand,:,1], total_data_divAp[rand,:,1], size=None)
#             spec_data_BC[i,:]  = np.random.uniform(total_data_divAp[rand,:,2], total_data_divAp[rand,:,2], size=None)
#             spec_data_C[i,:]  = np.random.uniform(total_data_divAp[rand,:,6], total_data_divAp[rand,:,6], size=None)
#             spec_data_OC[i,:]  = np.random.uniform(total_data_divAp[rand,:,7], total_data_divAp[rand,:,7], size=None)


# NEED TO REWRITE THIS PART
#     #             if spec_data_LiC[i,j] < divAp_pack[1,j,0] or spec_data_LiC[i,j] > divAp_pack[0,j,0]:
#     #                 j -= 1
#     #             elif spec_data_BeC[i,j] < divAp_pack[1,j,1] or spec_data_BeC[i,j] > divAp_pack[0,j,1]:
#     #                 j -= 1
#     #             elif spec_data_BC[i,j] < divAp_pack[1,j,2] or spec_data_BC[i,j] > divAp_pack[0,j,2]:
#     #                 j -= 1
#     #             elif spec_data_C[i,j] < divAp_pack[1,j,6] or spec_data_C[i,j] > divAp_pack[0,j,6]:
#     #                 j -= 1
#     #             elif spec_data_OC[i,j] < divAp_pack[1,j,7] or spec_data_OC[i,j] > divAp_pack[0,j,7]:
#     #                 j -= 1


#     #     Li_norm = np.random.uniform(0.8,1.2,100)
#     #     Be_norm = np.random.uniform(0.8,1.2,100)
#     #     O_norm = np.random.uniform(0.8,1.2,100)

        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")
        

        '''
        Recorver to (Spectrum)/(Normal Factor)
        '''
        logging.info("Recorver to (Spectrum)/(Normal Factor)")
        logging.info("=====START=====")
        t1 = time.time()


        Li_norm = np.random.uniform(min(parameter[:,11]),max(parameter[:,11]),100)
        Be_norm = np.random.uniform(min(parameter[:,12]),max(parameter[:,12]),100)
        O_norm = np.random.uniform(min(parameter[:,13]),max(parameter[:,13]),100)

        spectrum_E = np.zeros((number,84))
        for i in range(number):
            spectrum_E[i,:] = data[0,:,0]
        spectrum_Li = spec_data_LiC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))
        spectrum_Be = spec_data_BeC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))
        spectrum_B = spec_data_BC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))
        spectrum_C = spec_data_C[:,:]
        spectrum_O = spec_data_OC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))

        '''
        Determine the Normal Factor for each Pseudodata
        '''
        norm_factor = np.zeros((number,14))
        for i in range(number):
            chisq_scan_Li = np.zeros((1000))
            chisq_scan_Be = np.zeros((1000))
            chisq_scan_O = np.zeros((1000))

            pred_interpolate_Li = interpolate.interp1d((spectrum_E[0,:]), (spectrum_Li[i,:]*Li_norm.reshape(100,1)), kind='cubic')
            pred_interpolate_Be = interpolate.interp1d((spectrum_E[0,:]), (spectrum_Be[i,:]*Be_norm.reshape(100,1)), kind='cubic')
            pred_interpolate_O = interpolate.interp1d((spectrum_E[0,:]), (spectrum_O[i,:]*O_norm.reshape(100,1)), kind='cubic')

            chisq_scan_Li = (interpolate_chisq(pred_interpolate_Li(Li_Eams),Li_Fams,Li_Sams)+
                                interpolate_chisq(pred_interpolate_Li(Li_Evy),Li_Fvy,Li_Svy)
                               )

            chisq_scan_Be = (interpolate_chisq(pred_interpolate_Be(Be_Eams),Be_Fams,Be_Sams)+
                                interpolate_chisq(pred_interpolate_Be(Be_Evy),Be_Fvy,Be_Svy)
                               )

            chisq_scan_O = (interpolate_chisq(pred_interpolate_O(O_Eams),O_Fams,O_Sams)+
                               interpolate_chisq(pred_interpolate_O(O_Evy),O_Fvy,O_Svy)+
                               interpolate_chisq(pred_interpolate_O(O_Eace),O_Face,O_Sace)
                              )

            norm_factor[i,11] = Li_norm[np.where(chisq_scan_Li == min(chisq_scan_Li))]
            norm_factor[i,12] = Be_norm[np.where(chisq_scan_Be == min(chisq_scan_Be))]
            norm_factor[i,13] = O_norm[np.where(chisq_scan_O == min(chisq_scan_O))]

        logging.info("Pseudo Normal Factor")
        logging.info(" {:^4} {:^6} {:^6} {:^6}".format("","Li","Be","O"))
        logging.info(" {:^4} {:^6} {:^6} {:^6}".format("#",len(norm_factor[:,11]),len(norm_factor[:,12]),len(norm_factor[:,13])))
        logging.info(" {:^4} {:^4.4f} {:^4.4f} {:^4.4f}".format("Max",max(norm_factor[:,11]),max(norm_factor[:,12]),max(norm_factor[:,13])))
        logging.info(" {:^4} {:^4.4f} {:^4.4f} {:^4.4f}".format("Min",min(norm_factor[:,11]),min(norm_factor[:,12]),min(norm_factor[:,13])))
        logging.info(" {:^4} {:^4.4f} {:^4.4f} {:^4.4f}".format("Ave.",np.average(norm_factor[:,11]),np.average(norm_factor[:,12]),np.average(norm_factor[:,13])))


        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")

        
        '''
        Recorver to Spectrum
        '''
        logging.info("Recorver to Spectrum")
        logging.info("=====START=====")
        t1 = time.time()
        
        
        spectrum_E = np.zeros((number,84))
        for i in range(number):
            spectrum_E[i,:] = data[0,:,0] 
        spectrum_Li = spec_data_LiC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))*np.reshape(norm_factor[:,11],(number,1))
        spectrum_Be = spec_data_BeC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))*np.reshape(norm_factor[:,12],(number,1))
        spectrum_B = spec_data_BC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))
        spectrum_C = spec_data_C[:,:]
        spectrum_O = spec_data_OC[:,:]*(spec_data_C[:,]/np.reshape(spec_data_C[:,52],(number,1)))*np.reshape(norm_factor[:,13],(number,1))

        '''
        Select 
        '''
        pseudodata_tmp = []
        norm_chisq = np.zeros((number))
        normalfactor = []
        norm_interpolate_Li = interpolate.interp1d((spectrum_E[0,:]), (spectrum_Li[:,:]), kind='cubic')
        norm_interpolate_Be = interpolate.interp1d((spectrum_E[0,:]), (spectrum_Be[:,:]), kind='cubic')
        pred_interpolate_B  = interpolate.interp1d((spectrum_E[0,:]), (spectrum_B[:,:]), kind='cubic')
        pred_interpolate_C  = interpolate.interp1d((spectrum_E[0,:]), (spectrum_C[:,:]), kind='cubic')
        norm_interpolate_O  = interpolate.interp1d((spectrum_E[0,:]), (spectrum_O[:,:]), kind='cubic')
        norm_chisq = (
                            interpolate_chisq(P=norm_interpolate_Li(Li_Eams), F=Li_Fams, S=Li_Sams)+
                            interpolate_chisq(P=norm_interpolate_Li(Li_Evy), F=Li_Fvy, S=Li_Svy)+

                            interpolate_chisq(P=norm_interpolate_Be(Be_Eams), F=Be_Fams, S=Be_Sams)+
                            interpolate_chisq(P=norm_interpolate_Be(Be_Evy), F=Be_Fvy, S=Be_Svy)+

                            interpolate_chisq(P=pred_interpolate_B(B_Eams), F=B_Fams, S=B_Sams)+
                            interpolate_chisq(P=pred_interpolate_B(B_Evy), F=B_Fvy, S=B_Svy)+
                            interpolate_chisq(P=pred_interpolate_B(B_Eace), F=B_Face, S=B_Sace)+

                            interpolate_chisq(P=pred_interpolate_C(C_Eams), F=C_Fams, S=C_Sams)+
                            interpolate_chisq(P=pred_interpolate_C(C_Evy), F=C_Fvy, S=C_Svy)+
                            interpolate_chisq(P=pred_interpolate_C(C_Eace), F=C_Face, S=C_Sace)+

                            interpolate_chisq(P=norm_interpolate_O(O_Eams), F=O_Fams, S=O_Sams)+
                            interpolate_chisq(P=norm_interpolate_O(O_Evy), F=O_Fvy, S=O_Svy)+
                            interpolate_chisq(P=norm_interpolate_O(O_Eace), F=O_Face, S=O_Sace)
                         )
            
        normalfactor = []
        for i in range(number):
            
            if index == 0:
    #             if norm_chisq[i] < min(chi)+15.94: # 1sigma
    #             if norm_chisq[i] < min(chi)+24.03: # 2sigma
    #             if norm_chisq[i] < min(chi)+33.20: # 3sigma
    #               if norm_chisq[i] < min(chi)+43.82: # 4sigma
#                 if norm_chisq[i] < min(chi)+70:  # 6 sigma
                if True:
                    pseudodata_tmp.append([data[0,:,0],spectrum_Li[i,:],
                                           spectrum_Be[i,:],spectrum_B[i,:],
                                           spectrum_C[i,:],spectrum_O[i,:]]
                                         )
                    normalfactor.append(norm_factor[i,:])



            if index == 1:
                pseudodata_tmp.append([data[0,:,0],spectrum_Li[i,:],
                                           spectrum_Be[i,:],spectrum_B[i,:],
                                           spectrum_C[i,:],spectrum_O[i,:]]
                                         )
                normalfactor.append(norm_factor[i,:])

    #         pseudodata_tmp.append([data[0,:,0],spectrum_Li[i,:],
    #                                            spectrum_Be[i,:],spectrum_B[i,:],
    #                                            spectrum_C[i,:],spectrum_O[i,:]]
    #                                          )
    #         normalfactor.append(norm_factor[i,:])

        normalfactor = np.array(normalfactor)     
        pseudodata = np.zeros((len(pseudodata_tmp),84,6))
        for i in range(6):
            pseudodata[:,:,i] = np.array(pseudodata_tmp)[:,i,:]

        t2 = time.time()
        logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2-t1)/60.))
        logging.info("=====Finish=====")
        logging.info("\n")
        #######################################################################################################    
        ticks_2 = time.time()
        totaltime =  ticks_2 - ticks_1
        logging.info("\033[3;33m Total Time consumption : {:.4f} min\033[0;m".format(totaltime/60.))
        return normalfactor, pseudodata
        
#     self.normalfactor = normalfactor
#     self.pseudodata = pseudodata
    
#=========================================================================================================#



