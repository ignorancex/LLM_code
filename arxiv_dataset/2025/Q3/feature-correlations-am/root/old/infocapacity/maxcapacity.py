"""
This code is based on Krotov and Hopfield 2016 paper in Neurips

This program calculates and plots the maximum number of memories a dense associatve Hopfield network can store
without retrieval error depending on two input factors:

N = Number of neurons in Hopfield net
n = degree of the polynomial smoothing function in the update function of the network
"""
import math

from scipy.special import factorial2
import matplotlib.pyplot as plt

#calc capacity
def calc_k (num_neurons, poly_degree):
    factorial = factorial2(2*poly_degree-3)
    
    one = 1/(2*factorial)
    two = (num_neurons**(poly_degree-1))/(math.log(num_neurons))

    return one*two

def plot_graph(start_neurons, end_neurons):
    degrees = range(1,50) #polynomial degree of function

    for neuron in range(start_neurons, end_neurons+1):
        y_values = []
        for degree in degrees:
            y_values.append(calc_k(neuron, degree))

        plt.plot(degrees, y_values)
    plt.show()


def main():
    print(calc_k(25,2))
    print(calc_k(25,3))
    print(calc_k(25,4))
    print(calc_k(25,5))

    plot_graph(25,25)

if __name__ == "__main__":
    main()
