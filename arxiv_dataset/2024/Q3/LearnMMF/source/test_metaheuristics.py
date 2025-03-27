import torch
import time
import random
import unittest
import matplotlib.pyplot as plt
from metaheuristics import get_cost, evolutionary_algorithm, generate_single_random_weighted_graph_laplacian, directed_evolution

class TestGenerateSingleRandomWeightedGraphLaplacian(unittest.TestCase):
    def test_output_shape(self):
        matrix_size = 5
        edge_probability = 0.5
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        self.assertEqual(laplacian_matrix.shape, (matrix_size, matrix_size))

    def test_symmetric(self):
        matrix_size = 5
        edge_probability = 0.5
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        self.assertTrue(torch.allclose(laplacian_matrix, laplacian_matrix.t()))

    def test_matrix_content(self):
        matrix_size = 5
        edge_probability = 0.6
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        print(f'The matrix is {laplacian_matrix}, size {laplacian_matrix.size()}')


class TestMetaheuristics(unittest.TestCase):
    def test_get_cost(self):
        # Define dummy input parameters
        matrix_size = 10
        edge_probability = 0.8
        device = 'cpu'

        # Call the function
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        L = 4
        K = 3

        wavelet_indices = torch.tensor(random.sample(range(matrix_size), L))
        rest_indices = torch.zeros(L, K - 1, dtype=torch.int64)
        for j in range(L):
            values = list(range(matrix_size))
            values.remove(wavelet_indices[j].item())
            rest_indices[j] = torch.tensor(random.sample(values, K - 1))

        cost = get_cost(laplacian_matrix, wavelet_indices, rest_indices, L, K)
        print(f'The cost is {cost}')

    def test_evolutionary_algorithm(self):
        # Define dummy input parameters
        matrix_size = 60
        edge_probability = 0.6
        device = 'cpu'

        # Call the function
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        L = 40
        K = 8

        wavelet_indices = torch.tensor(random.sample(range(matrix_size), L))
        rest_indices = torch.zeros(L, K - 1, dtype=torch.int64)
        for j in range(L):
            values = list(range(matrix_size))
            values.remove(wavelet_indices[j].item())
            rest_indices[j] = torch.tensor(random.sample(values, K - 1))

        cost = get_cost(laplacian_matrix, wavelet_indices, rest_indices, L, K)
        print(f'The random cost is {cost}')
        start_time = time.time()

        wavelet_indices, rest_indices, ea_cost, min_cost_per_gen, mean_cost_per_gen, all_time_min_cost_per_gen = evolutionary_algorithm(get_cost, laplacian_matrix, L, K, population_size=100, generations=100, mutation_rate=0.2)

        # Record the end time
        end_time = time.time()

        # Calculate the runtime
        runtime = end_time - start_time

        print("Runtime:", runtime, "seconds")

        print(f'The best EA cost is {ea_cost}')
        print(f'The wavelets is {wavelet_indices}, rest_indices {rest_indices}')

        # Sample data

        # Plotting
        # plt.plot(range(len(min_cost_per_gen)), min_cost_per_gen, label='Min evaluation')
        plt.plot(range(len(mean_cost_per_gen)), mean_cost_per_gen, label='Mean evaluation')
        plt.plot(range(len(all_time_min_cost_per_gen)), all_time_min_cost_per_gen, label='Best evaluation')

        # Adding labels and title
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.title('Best and mean evaluation values of EA')
        plt.legend()  # Adding legend

        # Display plot
        plt.show()

        # Keep the plot window open until the user exits
        while True:
            plt.pause(0.05)  # Pause for 0.05 seconds

    def test_directed_evolution(self):
        # Define dummy input parameters
        matrix_size = 60
        edge_probability = 0.6
        device = 'cpu'

        # Call the function
        laplacian_matrix = generate_single_random_weighted_graph_laplacian(matrix_size, edge_probability)
        L = 40
        K = 8

        wavelet_indices = torch.tensor(random.sample(range(matrix_size), L))
        rest_indices = torch.zeros(L, K - 1, dtype=torch.int64)
        for j in range(L):
            values = list(range(matrix_size))
            values.remove(wavelet_indices[j].item())
            rest_indices[j] = torch.tensor(random.sample(values, K - 1))

        cost = get_cost(laplacian_matrix, wavelet_indices, rest_indices, L, K)
        print(f'The random cost is {cost}')
        start_time = time.time()

        wavelet_indices, rest_indices, de_cost, min_cost_per_gen, mean_cost_per_gen, all_time_min_cost_per_gen = directed_evolution(get_cost, laplacian_matrix, L, K, population_size=100, generations=100, sample_kept_rate=0.3)

        # Record the end time
        end_time = time.time()

        # Calculate the runtime
        runtime = end_time - start_time

        print("Runtime:", runtime, "seconds")

        print(f'The best DE cost is {de_cost}')
        print(f'The wavelets is {wavelet_indices}, rest_indices {rest_indices}')

        # Sample data

        # Plotting
        # plt.plot(range(len(min_cost_per_gen)), min_cost_per_gen, label='Min evaluation')
        plt.plot(range(len(mean_cost_per_gen)), mean_cost_per_gen, label='Mean evaluation')
        plt.plot(range(len(all_time_min_cost_per_gen)), all_time_min_cost_per_gen, label='Best evaluation')

        # Adding labels and title
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.title('Best and mean evaluation values of DE')
        plt.legend()  # Adding legend

        # Display plot
        plt.show()

        # Keep the plot window open until the user exits
        while True:
            plt.pause(0.05)  # Pause for 0.05 seconds


if __name__ == '__main__':
    unittest.main()