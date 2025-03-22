from tune import tune
import igraph as ig
import itertools

graph_ = r"Data/Network/less_Tclin_Signalink_PIN_graph.graphml"
graph = ig.Graph.Load(graph_, format='graphml')

def grid_search(graph, param_grid):
    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    trial = 0
    for params in combinations:
        trial +=1
        # Update the hype dictionary with the current combination of hyperparameters
        # Run the pipeline with the current hyperparameters
        met = tune(graph, negsamplesize=params['negsamplesize'], dimension=params['dimension'], batchsize=params['batchsize'], epochs=params['epochs'], learning_rate=params['learning_rate'], negativepower=params['negativepower'])

        # Store the result
        print(f"trial: {trial}: score: {met}")
        results.append({'params': params, 'score': met})

    # Sort results by score in descending order
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    best_params = results[0]['params']
    best_score = results[0]['score']

    return best_params, best_score, results


param_grid = {
    'negsamplesize': [5],
    'dimension': [128],
    'batchsize': [5],
    'epochs': [30],
    'learning_rate': [0.01],
    'negativepower': [0.75]
}

best_params, best_score, results = grid_search(graph, param_grid)
print(f"Best Params: {best_params}")
print(f"Best Score: {best_score}")
print("All Results:", results)

