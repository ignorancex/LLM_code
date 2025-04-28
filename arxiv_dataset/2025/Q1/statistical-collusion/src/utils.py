import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split

pd.set_option('future.no_silent_downcasting', True)

col_names = ['Model Type', 'Safety Rating', 'Fuel Type', 'Interior Material',
       'Infotainment System', 'Country of Manufacture', 'Warranty Length',
       'Number of Doors', 'Number of Seats', 'Air Conditioning',
       'Navigation System', 'Tire Type', 'Sunroof', 'Sound System',
       'Cruise Control', 'Bluetooth Connectivity', 'Transmission Type',
       'Drive Type', 'Car Evaluation']

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)
    
def R(delta,k):
    return np.sqrt(np.log(1/delta)/(2*k))


class SignalPlanting:
    def __init__(self, dataframe: pd.DataFrame, N, n, Ntest, g: dict, y_star, delta, epsilon):
        """
        :param dataframe: the original dataframe to sample from
        :param N: number of samples for the dN dataset: dataset of the N consumers
        :param n: number of samples for the dn dataset: subdataset of dN that forms the collective
        :param Ntest: number of test samples
        :param g: transformation X -> X; here we assume that g is a function that sets features to some fixed constants
        :param y_star: target label
        :param delta: results hold with proba at least 1-delta
        :param epsilon: "robustness" of the classifier
        """
        self.dataframe = dataframe

        self.n = n
        self.N = N
        self.Ntest = Ntest

        self.g = g
        self.y_star = y_star

        self.delta = delta 
        self.epsilon = epsilon

        # generate datasets
        dtrain, dtest = train_test_split(self.dataframe, train_size=self.N, random_state=42)
        self.dtest, _ = train_test_split(dtest, train_size=self.Ntest, random_state=42)
        self.dn, self.dN = train_test_split(dtrain, train_size=self.n, random_state=42)

        self.Delta = {} # stores Delta for each \tilde{x} \in X^*
        self.Proba = {} # stores Pr(\tilde{x}) for each \tilde{x} \in X^*, taken on D_tilde^{(n)}
    
    def compute_proba_x(self,dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive):
        """
        get the empirical probability of a given x in some dataframe
        """
        condition = (dataframe['Model Type']==model) & (dataframe['Safety Rating']==safety) & (dataframe['Fuel Type']==fuel) & (dataframe['Interior Material']==interior) & (dataframe['Infotainment System']==infotainment) & (dataframe['Country of Manufacture']==country)&(dataframe['Warranty Length']==warranty)&(dataframe['Number of Doors']==doors)&(dataframe['Number of Seats']==seats)&(dataframe['Air Conditioning']==aircon)&(dataframe['Navigation System']==navig)&(dataframe['Tire Type']==tire)&(dataframe['Sunroof']==sunroof)&(dataframe['Sound System']==sound)&(dataframe['Cruise Control']==cruise)&(dataframe['Bluetooth Connectivity']==bluetooth)&(dataframe['Transmission Type']==transmission)&(dataframe['Drive Type']==drive)
        count_conditions_met = dataframe[condition].shape[0]
        proba = count_conditions_met / len(dataframe)
        return proba
    
    def compute_proba_xy(self,dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive, evaluation):
        """
        get the empirical probability of a given (x,y) in some dataframe
        """
        condition = (dataframe['Model Type']==model) & (dataframe['Safety Rating']==safety) & (dataframe['Fuel Type']==fuel) & (dataframe['Interior Material']==interior) & (dataframe['Infotainment System']==infotainment) & (dataframe['Country of Manufacture']==country)&(dataframe['Warranty Length']==warranty)&(dataframe['Number of Doors']==doors)&(dataframe['Number of Seats']==seats)&(dataframe['Air Conditioning']==aircon)&(dataframe['Navigation System']==navig)&(dataframe['Tire Type']==tire)&(dataframe['Sunroof']==sunroof)&(dataframe['Sound System']==sound)&(dataframe['Cruise Control']==cruise)&(dataframe['Bluetooth Connectivity']==bluetooth)&(dataframe['Transmission Type']==transmission)&(dataframe['Drive Type']==drive)&(dataframe['Car Evaluation']==evaluation)
        count_conditions_met = dataframe[condition].shape[0]
        proba = count_conditions_met / len(dataframe)
        return proba

    def compute_Delta(self, dataframe):
        """
        get Delta for each \tilde{x} \in X^*
        """
        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        list_y_prime = ['Excellent','Good','Average','Poor'] # or more generally: list_y_prime = self.get_possible_values('Car Evaluation')
        list_y_prime.remove(self.y_star)

        # generate all combinations of feature values
        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))

        # compute delta for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination

            # compute proba for y' != y^*
            proba = 0
            for y in list_y_prime:
                proba_candidate = self.compute_proba_xy(dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,y)
                proba = max(proba, proba_candidate)

            # compute proba for y^*
            proba_star = self.compute_proba_xy(dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,self.y_star)
            
            # store
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            self.Delta[key] = proba - proba_star
    
    def compute_Proba(self, dataframe):
        """
        get Pr(\tilde{x}) for each \tilde{x} \in X^*, taken on D_tilde^{(n)}
        """
        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        # generate all combinations of feature values
        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))

        # compute Proba for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination

            # compute proba
            proba = self.compute_proba_x(self.apply_g(dataframe),model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive)
            
            # store
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            self.Proba[key] = proba
            
    def get_possible_values(self, column):
        return list(set(self.dataframe[column].values))
    
    def apply_g(self,dataframe):
        df_copy = dataframe.copy()
        for key in self.g.keys():
            df_copy[key] = self.g[key] 
        return df_copy
    
    def get_card_x_star(self):
        card_x_star = 1
        features = col_names.copy()
        features.remove('Car Evaluation')
        for key in self.g.keys():
            features.remove(key)
        for feature in features:
            card_x_star *= len(self.get_possible_values(feature))
        return card_x_star
    
    def get_card_y(self):
        return len(self.get_possible_values('Car Evaluation'))
    
    def get_delta_tilde(self):
        x_star = self.get_card_x_star()
        y = self.get_card_y()
        return self.delta / (2 + 2*x_star +2*x_star*y)

    def compute_lower_bound(self):
        S = 0
        dn_tilde = self.apply_g(self.dn)
        delta_tilde = self.get_delta_tilde()

        for i in range(len(dn_tilde)):
            model = dn_tilde.iloc[i,0]
            safety = dn_tilde.iloc[i,1]
            fuel = dn_tilde.iloc[i,2]
            interior = dn_tilde.iloc[i,3]
            infotainment = dn_tilde.iloc[i,4]
            country = dn_tilde.iloc[i,5]
            warranty= dn_tilde.iloc[i,6]
            doors= dn_tilde.iloc[i,7]
            seats=dn_tilde.iloc[i,8]
            aircon=dn_tilde.iloc[i,9]
            navig=dn_tilde.iloc[i,10]
            tire=dn_tilde.iloc[i,11]
            sunroof=dn_tilde.iloc[i,12]
            sound=dn_tilde.iloc[i,13]
            cruise=dn_tilde.iloc[i,14]
            bluetooth=dn_tilde.iloc[i,15]
            transmission=dn_tilde.iloc[i,16]
            drive=dn_tilde.iloc[i,17]

            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            Delta = self.Delta[key]
            Proba = self.Proba[key]

            if self.n/self.N*(Proba-2*R(delta_tilde,self.n)) - ((self.N-self.n)/self.N)*(Delta+2*R(delta_tilde,self.n)+2*R(delta_tilde,self.N-self.n)) - self.epsilon/(1-self.epsilon) > 0:
                S += 1

        S = S/self.n

        return S - R(delta_tilde,self.n) - R(delta_tilde,self.Ntest) 

    def compute_success(self):
        S = 0
        dn_tilde = self.apply_g(self.dn)
        dn_tilde['Car Evaluation'] = self.y_star
        mixed_df = pd.concat([dn_tilde, self.dN], ignore_index=True)
        list_y = ['Good','Average','Poor','Excellent'] # list_y = self.get_possible_values('Car Evaluation')

        # compute all argmax_{y'} Pr(x=g(x_test),y=y') once 
        argmax = {} # store the argmax values

        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        # generate all combinations of feature values
        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))

        # compute argmax for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination

            proba = 0
            y_candidate = list_y[0]
            for y in list_y:
                proba_candidate = self.compute_proba_xy(mixed_df,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,y)
                if proba_candidate > proba:
                    proba = proba_candidate
                    y_candidate = y
            
            # store
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            argmax[key] = y_candidate

        for i in range(len(self.dtest)):
            model = self.g['Model Type'] if 'Model Type' in self.g.keys() else self.dtest.iloc[i,0]
            safety = self.g['Safety Rating'] if 'Safety Rating' in self.g.keys() else self.dtest.iloc[i,1]
            fuel = self.g['Fuel Type'] if 'Fuel Type' in self.g.keys() else self.dtest.iloc[i,2]
            interior = self.g['Interior Material'] if 'Interior Material' in self.g.keys() else self.dtest.iloc[i,3]
            infotainment = self.g['Infotainment System'] if 'Infotainment System' in self.g.keys() else self.dtest.iloc[i,4]
            country = self.g['Country of Manufacture'] if 'Country of Manufacture' in self.g.keys() else self.dtest.iloc[i,5]
            warranty = self.g['Warranty Length'] if 'Warranty Length' in self.g.keys() else self.dtest.iloc[i,6]
            doors = self.g['Number of Doors'] if 'Number of Doors' in self.g.keys() else self.dtest.iloc[i,7]
            seats = self.g['Number of Seats'] if 'Number of Seats' in self.g.keys() else self.dtest.iloc[i,8]
            aircon = self.g['Air Conditioning'] if 'Air Conditioning' in self.g.keys() else self.dtest.iloc[i,9]
            navig = self.g['Navigation System'] if 'Navigation System' in self.g.keys() else self.dtest.iloc[i,10]
            tire = self.g['Tire Type'] if 'Tire Type' in self.g.keys() else self.dtest.iloc[i,11]
            sunroof = self.g['Sunroof'] if 'Sunroof' in self.g.keys() else self.dtest.iloc[i,12]
            sound = self.g['Sound System'] if 'Sound System' in self.g.keys() else self.dtest.iloc[i,13]
            cruise = self.g['Cruise Control'] if 'Cruise Control' in self.g.keys() else self.dtest.iloc[i,14]
            bluetooth = self.g['Bluetooth Connectivity'] if 'Bluetooth Connectivity' in self.g.keys() else self.dtest.iloc[i,15]
            transmission = self.g['Transmission Type'] if 'Transmission Type' in self.g.keys() else self.dtest.iloc[i,16]
            drive =  self.g['Drive Type'] if 'Drive Type' in self.g.keys() else self.dtest.iloc[i,17]

            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            y = argmax[key]

            if y == self.y_star:
                S += 1

        return S/self.Ntest
    

class SignalUnplanting:
    def __init__(self, dataframe: pd.DataFrame, N, n, ne, Ntest, g: dict, y_star, delta, epsilon):
        """
        :param dataframe: the original dataframe to sample from
        :param N: number of samples for the dN dataset: dataset of the N consumers
        :param n: number of samples for the dn+de dataset: subdataset of dN that forms the collective
        :param ne: number of samples for the de dataset: subdataset of the collective that estimates y_hat
        :param Ntest: number of test samples
        :param g: transformation X -> X; here we assume that g is a function that sets features to some fixed constants
        :param y_star: target label
        :param delta: results hold with proba at least 1-delta
        :param epsilon: "robustness" of the classifier
        """
        self.dataframe = dataframe

        self.n = n
        self.N = N
        self.ne = ne
        self.Ntest = Ntest

        self.g = g
        self.y_star = y_star

        self.delta = delta
        self.epsilon = epsilon 

        # generate datasets
        dtrain, dtest = train_test_split(self.dataframe, train_size=self.N, random_state=42)
        self.dtest, _ = train_test_split(dtest, train_size=self.Ntest, random_state=42)
        dtrain_i, self.dN = train_test_split(dtrain, train_size=self.n, random_state=42)
        self.de, self.dn = train_test_split(dtrain_i, train_size=self.ne, random_state=42)

        # /!\ careful here dn is D^{(n-n_{est})} and not D^{(n)}

        self.Delta = {} # stores Delta for each \tilde{x} \in X^*
        self.Proba = {} # stores Pr(\tilde{x}) for each \tilde{x} \in X^*, taken on D_tilde^{(n)}
        self.yhat = {} # stores yhat(\tilde{x}) for each \tilde{x} \in X^*

    def compute_proba_x(self,dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive):
        """
        get the empirical probability of a given x in some dataframe
        """
        condition = (dataframe['Model Type']==model) & (dataframe['Safety Rating']==safety) & (dataframe['Fuel Type']==fuel) & (dataframe['Interior Material']==interior) & (dataframe['Infotainment System']==infotainment) & (dataframe['Country of Manufacture']==country)&(dataframe['Warranty Length']==warranty)&(dataframe['Number of Doors']==doors)&(dataframe['Number of Seats']==seats)&(dataframe['Air Conditioning']==aircon)&(dataframe['Navigation System']==navig)&(dataframe['Tire Type']==tire)&(dataframe['Sunroof']==sunroof)&(dataframe['Sound System']==sound)&(dataframe['Cruise Control']==cruise)&(dataframe['Bluetooth Connectivity']==bluetooth)&(dataframe['Transmission Type']==transmission)&(dataframe['Drive Type']==drive)
        count_conditions_met = dataframe[condition].shape[0]
        proba = count_conditions_met / len(dataframe)
        return proba
    
    def compute_proba_xy(self,dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive, evaluation):
        """
        get the empirical probability of a given (x,y) in some dataframe
        """
        condition = (dataframe['Model Type']==model) & (dataframe['Safety Rating']==safety) & (dataframe['Fuel Type']==fuel) & (dataframe['Interior Material']==interior) & (dataframe['Infotainment System']==infotainment) & (dataframe['Country of Manufacture']==country)&(dataframe['Warranty Length']==warranty)&(dataframe['Number of Doors']==doors)&(dataframe['Number of Seats']==seats)&(dataframe['Air Conditioning']==aircon)&(dataframe['Navigation System']==navig)&(dataframe['Tire Type']==tire)&(dataframe['Sunroof']==sunroof)&(dataframe['Sound System']==sound)&(dataframe['Cruise Control']==cruise)&(dataframe['Bluetooth Connectivity']==bluetooth)&(dataframe['Transmission Type']==transmission)&(dataframe['Drive Type']==drive)&(dataframe['Car Evaluation']==evaluation)
        count_conditions_met = dataframe[condition].shape[0]
        proba = count_conditions_met / len(dataframe)
        return proba
    
    def compute_yhat(self, dataframe):
        """
        get yhat(\tilde{x}) for each \tilde{x} \in X^*
        """

        list_y_prime = ['Excellent','Good','Average','Poor'] # list_y_prime = self.get_possible_values('Car Evaluation')
        list_y_prime.remove(self.y_star)

        # generate all possible values of \tilde{x}
        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))

        # compute yhat for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination

            # compute argmax
            y = list_y_prime[0] # candidate
            proba = self.compute_proba_xy(dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,y)
            for yc in list_y_prime[1::]:
                proba_candidate = self.compute_proba_xy(dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,yc)
                if proba_candidate > proba:
                    y = yc
                    proba = proba_candidate
            
            # store
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            self.yhat[key] = y

    def compute_Delta(self, dataframe1, dataframe2):
        """
        get Delta for each \tilde{x} \in X^*
        :param dataframe1: dataframe used for the computation of the first term of Delta
        :param dataframe2: dataframe used for the computation of the second term of Delta
        """

        list_y_prime = ['Excellent','Good','Average','Poor'] # list_y_prime = self.get_possible_values('Car Evaluation')
        list_y_prime.remove(self.y_star)

        # generate all possible values of \tilde{x}
        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))

        # compute delta for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination

            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            yhat = self.yhat[key]

            delta1 = self.compute_proba_xy(dataframe1,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,self.y_star)
            delta2 = self.compute_proba_xy(dataframe2,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,yhat)

            self.Delta[key] = delta1 - delta2
    
    def compute_Proba(self, dataframe):
        """
        get Pr(\tilde{x}) for each \tilde{x} \in X^*, taken on D_tilde^{(n)}
        """

        # generate all possible values of \tilde{x}
        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))

        # compute Proba for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination

            proba = self.compute_proba_x(self.apply_g(dataframe),model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive)
            
            # store
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            self.Proba[key] = proba
            
    def get_possible_values(self, column):
        return list(set(self.dataframe[column].values))

    def apply_g(self,dataframe):
        df_copy = dataframe.copy()
        for key in self.g.keys():
            df_copy[key] = self.g[key] 
        return df_copy
    
    def get_card_x_star(self):
        card_x_star = 0
        features = col_names.copy()
        features.remove('Car Evaluation')
        for key in self.g.keys():
            features.remove(key)
        for feature in features:
            card_x_star += len(self.get_possible_values(feature))
        return card_x_star
    
    def get_card_y(self):
        return len(self.get_possible_values('Car Evaluation'))
    
    def get_delta_tilde(self):
        x_star = self.get_card_x_star()
        return self.delta / (2 + 6*x_star)

    def compute_lower_bound(self):
        S = 0
        dn_tilde = self.apply_g(self.dn)
        delta_tilde = self.get_delta_tilde()

        for i in range(len(dn_tilde)):
            model = dn_tilde.iloc[i,0]
            safety = dn_tilde.iloc[i,1]
            fuel = dn_tilde.iloc[i,2]
            interior = dn_tilde.iloc[i,3]
            infotainment = dn_tilde.iloc[i,4]
            country = dn_tilde.iloc[i,5]
            warranty= dn_tilde.iloc[i,6]
            doors= dn_tilde.iloc[i,7]
            seats=dn_tilde.iloc[i,8]
            aircon=dn_tilde.iloc[i,9]
            navig=dn_tilde.iloc[i,10]
            tire=dn_tilde.iloc[i,11]
            sunroof=dn_tilde.iloc[i,12]
            sound=dn_tilde.iloc[i,13]
            cruise=dn_tilde.iloc[i,14]
            bluetooth=dn_tilde.iloc[i,15]
            transmission=dn_tilde.iloc[i,16]
            drive=dn_tilde.iloc[i,17]

            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            Delta = self.Delta[key]
            Proba = self.Proba[key]

            if self.n/self.N*(Proba-2*R(delta_tilde,self.n)) - ((self.N-self.n)/self.N)*(Delta+R(delta_tilde,self.n)+R(delta_tilde,self.n-self.ne)+2*R(delta_tilde,self.N-self.n)) - self.epsilon/(1-self.epsilon) > 0:
                S += 1

        S = S/self.n

        return S - R(delta_tilde,self.n) - R(delta_tilde,self.Ntest)   

    def compute_success(self):
        S = 0
        # apply g on features
        dn_tilde = self.apply_g(self.dn)
        # apply yhat(\tilde{x})
        for i in range(len(dn_tilde)):
            # get \tilde{x}
            model = dn_tilde.iloc[i,0]
            safety = dn_tilde.iloc[i,1]
            fuel = dn_tilde.iloc[i,2]
            interior = dn_tilde.iloc[i,3]
            infotainment = dn_tilde.iloc[i,4]
            country = dn_tilde.iloc[i,5]
            warranty= dn_tilde.iloc[i,6]
            doors= dn_tilde.iloc[i,7]
            seats=dn_tilde.iloc[i,8]
            aircon=dn_tilde.iloc[i,9]
            navig=dn_tilde.iloc[i,10]
            tire=dn_tilde.iloc[i,11]
            sunroof=dn_tilde.iloc[i,12]
            sound=dn_tilde.iloc[i,13]
            cruise=dn_tilde.iloc[i,14]
            bluetooth=dn_tilde.iloc[i,15]
            transmission=dn_tilde.iloc[i,16]
            drive=dn_tilde.iloc[i,17]
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            # change label
            dn_tilde.iloc[i,18] = self.yhat[key]

        mixed_df = pd.concat([dn_tilde, self.dN], ignore_index=True)
        list_y = ['Good','Average','Poor','Excellent'] # list_y = self.get_possible_values('Car Evaluation')

        # compute all argmax_{y'} Pr(x=g(x_test),y=y') once 
        argmax = {} #store the argmax values

        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        # generate all combinations of feature values
        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))

        # compute argmax for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination
            y = list_y[0]
            proba = self.compute_proba_xy(mixed_df,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,y)
            for yc in list_y[1::]:
                proba_candidate = self.compute_proba_xy(mixed_df,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,yc)
                if proba_candidate > proba:
                    proba = proba_candidate
                    y = yc
            
            # store
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            argmax[key] = y

        for i in range(len(self.dtest)):
            model = self.g['Model Type'] if 'Model Type' in self.g.keys() else self.dtest.iloc[i,0]
            safety = self.g['Safety Rating'] if 'Safety Rating' in self.g.keys() else self.dtest.iloc[i,1]
            fuel = self.g['Fuel Type'] if 'Fuel Type' in self.g.keys() else self.dtest.iloc[i,2]
            interior = self.g['Interior Material'] if 'Interior Material' in self.g.keys() else self.dtest.iloc[i,3]
            infotainment = self.g['Infotainment System'] if 'Infotainment System' in self.g.keys() else self.dtest.iloc[i,4]
            country = self.g['Country of Manufacture'] if 'Country of Manufacture' in self.g.keys() else self.dtest.iloc[i,5]
            warranty = self.g['Warranty Length'] if 'Warranty Length' in self.g.keys() else self.dtest.iloc[i,6]
            doors = self.g['Number of Doors'] if 'Number of Doors' in self.g.keys() else self.dtest.iloc[i,7]
            seats = self.g['Number of Seats'] if 'Number of Seats' in self.g.keys() else self.dtest.iloc[i,8]
            aircon = self.g['Air Conditioning'] if 'Air Conditioning' in self.g.keys() else self.dtest.iloc[i,9]
            navig = self.g['Navigation System'] if 'Navigation System' in self.g.keys() else self.dtest.iloc[i,10]
            tire = self.g['Tire Type'] if 'Tire Type' in self.g.keys() else self.dtest.iloc[i,11]
            sunroof = self.g['Sunroof'] if 'Sunroof' in self.g.keys() else self.dtest.iloc[i,12]
            sound = self.g['Sound System'] if 'Sound System' in self.g.keys() else self.dtest.iloc[i,13]
            cruise = self.g['Cruise Control'] if 'Cruise Control' in self.g.keys() else self.dtest.iloc[i,14]
            bluetooth = self.g['Bluetooth Connectivity'] if 'Bluetooth Connectivity' in self.g.keys() else self.dtest.iloc[i,15]
            transmission = self.g['Transmission Type'] if 'Transmission Type' in self.g.keys() else self.dtest.iloc[i,16]
            drive =  self.g['Drive Type'] if 'Drive Type' in self.g.keys() else self.dtest.iloc[i,17]

            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            y = argmax[key]

            if y != self.y_star:
                S += 1

        return S/self.Ntest
    

class SignalPlantingFO: # features-only
    def __init__(self, dataframe: pd.DataFrame, N, n, Ntest, g: dict, x0: dict, y_star, delta, epsilon):
        """
        :param dataframe: the original dataframe to sample from
        :param N: number of samples for the dN dataset: dataset of the N consumers
        :param n: number of samples for the dn dataset: subdataset of dN that forms the collective
        :param Ntest: number of test samples
        :param g: transformation X -> X; here we assume that g is a function that sets features to some fixed constants
        :param x_0: feature used in the feature-only strategy 
        :param y_star: target label
        :param delta: results hold with proba at least 1-delta
        :param epsilon: "robustness" of the classifier
        """
        self.dataframe = dataframe

        self.n = n
        self.N = N
        self.Ntest = Ntest

        self.g = g
        self.x0 = x0
        self.y_star = y_star

        self.delta = delta
        self.epsilon = epsilon

        # generate datasets
        dtrain, dtest = train_test_split(self.dataframe, train_size=self.N, random_state=42)
        self.dtest, _ = train_test_split(dtest, train_size=self.Ntest, random_state=42)
        self.dn, self.dN = train_test_split(dtrain, train_size=self.n, random_state=42)

        self.Delta = {} # stores Delta for each \tilde{x} \in X^*
        self.Proba = {} # stores Pr(\tilde{x},y^*) for each \tilde{x} \in X^* 

    def compute_proba_x(self,dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive):
        """
        get the empirical probability of a given x in some dataframe
        """
        condition = (dataframe['Model Type']==model) & (dataframe['Safety Rating']==safety) & (dataframe['Fuel Type']==fuel) & (dataframe['Interior Material']==interior) & (dataframe['Infotainment System']==infotainment) & (dataframe['Country of Manufacture']==country)&(dataframe['Warranty Length']==warranty)&(dataframe['Number of Doors']==doors)&(dataframe['Number of Seats']==seats)&(dataframe['Air Conditioning']==aircon)&(dataframe['Navigation System']==navig)&(dataframe['Tire Type']==tire)&(dataframe['Sunroof']==sunroof)&(dataframe['Sound System']==sound)&(dataframe['Cruise Control']==cruise)&(dataframe['Bluetooth Connectivity']==bluetooth)&(dataframe['Transmission Type']==transmission)&(dataframe['Drive Type']==drive)
        count_conditions_met = dataframe[condition].shape[0]
        proba = count_conditions_met / len(dataframe)
        return proba
    
    def compute_proba_xy(self,dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive, evaluation):
        """
        get the empirical probability of a given (x,y) in some dataframe
        """
        condition = (dataframe['Model Type']==model) & (dataframe['Safety Rating']==safety) & (dataframe['Fuel Type']==fuel) & (dataframe['Interior Material']==interior) & (dataframe['Infotainment System']==infotainment) & (dataframe['Country of Manufacture']==country)&(dataframe['Warranty Length']==warranty)&(dataframe['Number of Doors']==doors)&(dataframe['Number of Seats']==seats)&(dataframe['Air Conditioning']==aircon)&(dataframe['Navigation System']==navig)&(dataframe['Tire Type']==tire)&(dataframe['Sunroof']==sunroof)&(dataframe['Sound System']==sound)&(dataframe['Cruise Control']==cruise)&(dataframe['Bluetooth Connectivity']==bluetooth)&(dataframe['Transmission Type']==transmission)&(dataframe['Drive Type']==drive)&(dataframe['Car Evaluation']==evaluation)
        count_conditions_met = dataframe[condition].shape[0]
        proba = count_conditions_met / len(dataframe)
        return proba

    def compute_Delta(self, dataframe):
        """
        get Delta for each \tilde{x} \in X^*
        """
        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        list_y_prime = ['Excellent','Good','Average','Poor'] # list_y_prime = self.get_possible_values('Car Evaluation')
        list_y_prime.remove(self.y_star)

        # generate all combinations of feature values
        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))

        # compute Delta for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination

            # compute proba for y' != y^*
            proba = 0
            for y in list_y_prime:
                proba_candidate = self.compute_proba_xy(dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,y)
                proba = max(proba, proba_candidate)

            # compute proba for y^*
            proba_star = self.compute_proba_xy(dataframe,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,self.y_star)
            
            # store
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            self.Delta[key] = proba - proba_star
    
    def compute_Proba(self, dataframe):
        """
        get Pr(\tilde{x},y^*) taken on D_tilde^{(n)}
        """
        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        # generate all combinations of feature values
        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))
        
        df_tilde = dataframe.copy()
        for index, row in df_tilde.iterrows():
            if row['Car Evaluation'] == self.y_star:
                for feature in self.g:
                    df_tilde.at[index, feature] = self.g[feature]
            else:
                for feature in self.x0:
                    df_tilde.at[index, feature] = self.x0[feature]

        # compute Proba for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination

            proba = self.compute_proba_xy(df_tilde,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,self.y_star)
            
            # store
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            self.Proba[key] = proba
    
    def get_possible_values(self, column):
        return list(set(self.dataframe[column].values))

    def apply_g(self,dataframe):
        df_copy = dataframe.copy()
        for key in self.g.keys():
            df_copy[key] = self.g[key] 
        return df_copy

    def get_card_x_star(self):
        card_x_star = 1
        features = col_names.copy()
        features.remove('Car Evaluation')
        for key in self.g.keys():
            features.remove(key)
        for feature in features:
            card_x_star *= len(self.get_possible_values(feature))
        return card_x_star

    def get_card_y(self):
        return len(self.get_possible_values('Car Evaluation'))

    def get_delta_tilde(self):
        x_star = self.get_card_x_star()
        y = self.get_card_y()
        return self.delta / (2 + 2*x_star*y + 2*x_star)

    def compute_lower_bound(self):
        S = 0
        delta_tilde = self.get_delta_tilde()

        for i in range(len(self.dn)):
            model_= self.g['Model Type'] if 'Model Type' in self.g.keys() else self.dn.iloc[i,0]
            safety_= self.g['Safety Rating'] if 'Safety Rating' in self.g.keys() else self.dn.iloc[i,1]
            fuel_= self.g['Fuel Type'] if 'Fuel Type' in self.g.keys() else self.dn.iloc[i,2]
            interior_= self.g['Interior Material'] if 'Interior Material' in self.g.keys() else self.dn.iloc[i,3]
            infotainment_= self.g['Infotainment System'] if 'Infotainment System' in self.g.keys() else self.dn.iloc[i,4]
            country_= self.g['Country of Manufacture'] if 'Country of Manufacture' in self.g.keys() else self.dn.iloc[i,5]
            warranty_= self.g['Warranty Length'] if 'Warranty Length' in self.g.keys() else self.dn.iloc[i,6]
            doors_= self.g['Number of Doors'] if 'Number of Doors' in self.g.keys() else self.dn.iloc[i,7]
            seats_= self.g['Number of Seats'] if 'Number of Seats' in self.g.keys() else self.dn.iloc[i,8]
            aircon_= self.g['Air Conditioning'] if 'Air Conditioning' in self.g.keys() else self.dn.iloc[i,9]
            navig_= self.g['Navigation System'] if 'Navigation System' in self.g.keys() else self.dn.iloc[i,10]
            tire_= self.g['Tire Type'] if 'Tire Type' in self.g.keys() else self.dn.iloc[i,11]
            sunroof_= self.g['Sunroof'] if 'Sunroof' in self.g.keys() else self.dn.iloc[i,12]
            sound_= self.g['Sound System'] if 'Sound System' in self.g.keys() else self.dn.iloc[i,13]
            cruise_= self.g['Cruise Control'] if 'Cruise Control' in self.g.keys() else self.dn.iloc[i,14]
            bluetooth_= self.g['Bluetooth Connectivity'] if 'Bluetooth Connectivity' in self.g.keys() else self.dn.iloc[i,15]
            transmission_= self.g['Transmission Type'] if 'Transmission Type' in self.g.keys() else self.dn.iloc[i,16]
            drive_= self.g['Drive Type'] if 'Drive Type' in self.g.keys() else self.dn.iloc[i,17]

            key = f"{model_}{safety_}{fuel_}{interior_}{infotainment_}{country_}{warranty_}{doors_}{seats_}{aircon_}{navig_}{tire_}{sunroof_}{sound_}{cruise_}{bluetooth_}{transmission_}{drive_}"
            Delta = self.Delta[key]
            Proba = self.Proba[key]

            if self.n/self.N*(Proba-2*R(delta_tilde,self.n)) - ((self.N-self.n)/self.N)*(Delta+2*R(delta_tilde,self.n)+2*R(delta_tilde,self.N-self.n)) - self.epsilon/(1-self.epsilon) > 0:
                S += 1

        S = S/self.n

        return S - R(delta_tilde,self.n) - R(delta_tilde,self.Ntest)
    
    def compute_success(self):
        S = 0
        dn_tilde = self.dn.copy()
        for index, row in dn_tilde.iterrows():
            if row['Car Evaluation'] == self.y_star:
                for feature in self.g:
                    dn_tilde.at[index, feature] = self.g[feature]
            else:
                for feature in self.x0:
                    dn_tilde.at[index, feature] = self.x0[feature]
        mixed_df = pd.concat([dn_tilde, self.dN], ignore_index=True)
        list_y = ['Good','Average','Poor','Excellent'] # list_y = self.get_possible_values('Car Evaluation')

        # compute all argmax_{y'} Pr(x=g(x_test),y=y') once 
        argmax = {} #store the argmax values

        feature_values = {
            'Model Type': [self.g['Model Type']] if 'Model Type' in self.g.keys() else self.get_possible_values('Model Type'),
            'Safety Rating': [self.g['Safety Rating']] if 'Safety Rating' in self.g.keys() else self.get_possible_values('Safety Rating'),
            'Fuel Type': [self.g['Fuel Type']] if 'Fuel Type' in self.g.keys() else self.get_possible_values('Fuel Type'),
            'Interior Material': [self.g['Interior Material']] if 'Interior Material' in self.g.keys() else self.get_possible_values('Interior Material'),
            'Infotainment System': [self.g['Infotainment System']] if 'Infotainment System' in self.g.keys() else self.get_possible_values('Infotainment System'),
            'Country of Manufacture': [self.g['Country of Manufacture']] if 'Country of Manufacture' in self.g.keys() else self.get_possible_values('Country of Manufacture'),
            'Warranty Length': [self.g['Warranty Length']] if 'Warranty Length' in self.g.keys() else self.get_possible_values('Warranty Length'),
            'Number of Doors': [self.g['Number of Doors']] if 'Number of Doors' in self.g.keys() else self.get_possible_values('Number of Doors'),
            'Number of Seats': [self.g['Number of Seats']] if 'Number of Seats' in self.g.keys() else self.get_possible_values('Number of Seats'),
            'Air Conditioning': [self.g['Air Conditioning']] if 'Air Conditioning' in self.g.keys() else self.get_possible_values('Air Conditioning'),
            'Navigation System': [self.g['Navigation System']] if 'Navigation System' in self.g.keys() else self.get_possible_values('Navigation System'),
            'Tire Type': [self.g['Tire Type']] if 'Tire Type' in self.g.keys() else self.get_possible_values('Tire Type'),
            'Sunroof': [self.g['Sunroof']] if 'Sunroof' in self.g.keys() else self.get_possible_values('Sunroof'),
            'Sound System': [self.g['Sound System']] if 'Sound System' in self.g.keys() else self.get_possible_values('Sound System'),
            'Cruise Control': [self.g['Cruise Control']] if 'Cruise Control' in self.g.keys() else self.get_possible_values('Cruise Control'),
            'Bluetooth Connectivity': [self.g['Bluetooth Connectivity']] if 'Bluetooth Connectivity' in self.g.keys() else self.get_possible_values('Bluetooth Connectivity'),
            'Transmission Type': [self.g['Transmission Type']] if 'Transmission Type' in self.g.keys() else self.get_possible_values('Transmission Type'),
            'Drive Type': [self.g['Drive Type']] if 'Drive Type' in self.g.keys() else self.get_possible_values('Drive Type')
        }

        # generate all combinations of feature values
        combinations = list(itertools.product(
            feature_values['Model Type'],
            feature_values['Safety Rating'],
            feature_values['Fuel Type'],
            feature_values['Interior Material'],
            feature_values['Infotainment System'],
            feature_values['Country of Manufacture'],
            feature_values['Warranty Length'],
            feature_values['Number of Doors'],
            feature_values['Number of Seats'],
            feature_values['Air Conditioning'],
            feature_values['Navigation System'],                    
            feature_values['Tire Type'],                    
            feature_values['Sunroof'],                    
            feature_values['Sound System'],                    
            feature_values['Cruise Control'],                    
            feature_values['Bluetooth Connectivity'],                    
            feature_values['Transmission Type'],                    
            feature_values['Drive Type'],                    
                            ))

        # compute argmax for each combination
        for combination in combinations:
            model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive = combination

            proba = 0
            y_candidate = list_y[0]
            for y in list_y:
                proba_candidate = self.compute_proba_xy(mixed_df,model,safety,fuel,interior,infotainment,country,warranty,doors,seats,aircon,navig,tire,sunroof,sound,cruise,bluetooth,transmission,drive,y)
                if proba_candidate > proba:
                    proba = proba_candidate
                    y_candidate = y
            
            # store
            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            argmax[key] = y_candidate

        for i in range(len(self.dtest)):
            model = self.g['Model Type'] if 'Model Type' in self.g.keys() else self.dtest.iloc[i,0]
            safety = self.g['Safety Rating'] if 'Safety Rating' in self.g.keys() else self.dtest.iloc[i,1]
            fuel = self.g['Fuel Type'] if 'Fuel Type' in self.g.keys() else self.dtest.iloc[i,2]
            interior = self.g['Interior Material'] if 'Interior Material' in self.g.keys() else self.dtest.iloc[i,3]
            infotainment = self.g['Infotainment System'] if 'Infotainment System' in self.g.keys() else self.dtest.iloc[i,4]
            country = self.g['Country of Manufacture'] if 'Country of Manufacture' in self.g.keys() else self.dtest.iloc[i,5]
            warranty = self.g['Warranty Length'] if 'Warranty Length' in self.g.keys() else self.dtest.iloc[i,6]
            doors = self.g['Number of Doors'] if 'Number of Doors' in self.g.keys() else self.dtest.iloc[i,7]
            seats = self.g['Number of Seats'] if 'Number of Seats' in self.g.keys() else self.dtest.iloc[i,8]
            aircon = self.g['Air Conditioning'] if 'Air Conditioning' in self.g.keys() else self.dtest.iloc[i,9]
            navig = self.g['Navigation System'] if 'Navigation System' in self.g.keys() else self.dtest.iloc[i,10]
            tire = self.g['Tire Type'] if 'Tire Type' in self.g.keys() else self.dtest.iloc[i,11]
            sunroof = self.g['Sunroof'] if 'Sunroof' in self.g.keys() else self.dtest.iloc[i,12]
            sound = self.g['Sound System'] if 'Sound System' in self.g.keys() else self.dtest.iloc[i,13]
            cruise = self.g['Cruise Control'] if 'Cruise Control' in self.g.keys() else self.dtest.iloc[i,14]
            bluetooth = self.g['Bluetooth Connectivity'] if 'Bluetooth Connectivity' in self.g.keys() else self.dtest.iloc[i,15]
            transmission = self.g['Transmission Type'] if 'Transmission Type' in self.g.keys() else self.dtest.iloc[i,16]
            drive =  self.g['Drive Type'] if 'Drive Type' in self.g.keys() else self.dtest.iloc[i,17]

            key = f"{model}{safety}{fuel}{interior}{infotainment}{country}{warranty}{doors}{seats}{aircon}{navig}{tire}{sunroof}{sound}{cruise}{bluetooth}{transmission}{drive}"
            y = argmax[key]

            if y == self.y_star:
                S += 1

        return S/self.Ntest