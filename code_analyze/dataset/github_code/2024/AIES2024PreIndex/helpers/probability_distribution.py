import numpy as np
import random
import warnings
import gc

warnings.filterwarnings('ignore')

np.random.seed(1)
random.seed(1)

#np.set_printoptions(linewidth=np.inf)

class GenerateProbabilityDistributions:

    def __init__(self, data_dict, bins):
        self.data_dict = data_dict
        self.bins = bins

    def generate_probability(self, array_data):
        prob, _ = np.histogram(array_data, bins=self.bins, density=True)
        prob = prob/np.sum(prob)
        return prob
    
    
    def average_probability_distributions(self):
        """
        Compute the averaged layer probability distribution for the entire data dictionary.
        Output: Averaged layer wise probability as a dictionary, each data point represents a layer_name and it's averaged probability distribution
        """
        print("\n\n\nGenerating average probability distributions")
        averaged_prob_data = {} #Stores key value pair of entire result. Key:Layer name. Value:Layer's averaged probability distribution

        #Iterate through each layer and it's activation values for all data points
        for layer, layer_data in self.data_dict.items(): 
            layer_prob_sum = np.zeros(self.bins) #To add all probability distributions
            
            filter_count = 0
            print(f"\nProcessing layer: {layer} data")
            #Iterate through each filter's vector. Each vector contains the flattended activation output produced by one filter for the entire dataset
            for each_filter_data in layer_data: 
                filter_probability = self.generate_probability(each_filter_data) #Generate a probability distribution for each filter
                if (np.isnan(filter_probability).any()):
                    print(f"NAN VALUE Layer:{layer}") 
                    filter_probability[np.isnan(filter_probability)] = 0  #In instances where bin is empty. To not lose bin data in layer_prob_sum when aggregating data
                    if np.isnan(filter_probability).any():
                        print(f"Filter data included with NAN")
                layer_prob_sum = np.add(layer_prob_sum, filter_probability) #Sum it to the final aggregated probability array
                filter_count += 1
            #Divide probability array for the given layer and divide by number of filters
            #Store data as key-vakye pair    
            averaged_prob_data[layer] = layer_prob_sum/len(layer_data) 

            print(f"Processed layer: {layer}, computed/averaged {filter_count} probability distributions")
        return averaged_prob_data
    

    def combine_activation_distribution(self):
        """
        Compute the layer aggregated probability distribution for the entire data dictionary.
        Output: Layer wise aggregated probability as a dictionary, each data point represents a layer_name and it's aggregated probability distribution
        """
        print("\n\n\nCombined activation probability distributions")
        layer_prob_data = {} #Stores key value pair of entire result. Key:Layer name. Value:Layer's averaged probability distribution

        #Iterate through each layer and it's activation values for all data points
        for layer, layer_data in self.data_dict.items(): 
            print(f"\nProcessing layer: {layer} data")
            aggregated_layer_data = layer_data.reshape(-1) #Flatten entire layer data. Each filter's activation value is combined into one single array
            layer_prob_data[layer] = self.generate_probability(aggregated_layer_data) #Generate a probability distribution for entire layer
            print(f"Processed layer: {layer}, reshaped data from {layer_data.shape} to {aggregated_layer_data.shape} to create distribution")
        return layer_prob_data
             

    def __del__(self):
        del self.data_dict
        gc.collect()