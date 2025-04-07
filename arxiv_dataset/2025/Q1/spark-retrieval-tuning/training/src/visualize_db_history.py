import plotly.graph_objects as go
import sqlite3
import numpy as np
import sys
import argparse


def plot_optimization_history(db_path):

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)  # Specify the correct path to your SQLite database file
    c = conn.cursor()

    # Fetch the data from the trials table
    c.execute('SELECT * FROM trials')
    data = c.fetchall()

    # Extract x (iteration) and y (objective value) values from the data
    iterations = []
    objective_values = []
    for i,row in enumerate(data):
        iterations.append(i)  # Assuming the first column is the iteration number
        objective_values.append(row[2])  # Assuming the third column is the objective value

    # Close the database connection
    conn.close()

    # Create a Plotly figure
    fig = go.Figure()
    coefficients = np.polyfit(iterations, objective_values, 1)
    line_of_best_fit = np.polyval(coefficients, iterations)

    # Add a scatter plot trace for the optimization history
    fig.add_trace(go.Scatter(x=iterations, y=objective_values, mode='markers', name='Optimization History'))

    # Add a line of best fit trace
    fig.add_trace(go.Scatter(x=iterations, y=line_of_best_fit, mode='lines', name='Line of Best Fit'))


    # Update layout
    fig.update_layout(title='Optimization History',
                    xaxis_title='Iteration',
                    yaxis_title='Time (Seconds)')

    # Show the figure in a browser window
    fig.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize optimization history from a SQLite database')
    parser.add_argument('--db_path', type=str, help='Path to the SQLite database file')
    args = parser.parse_args()

    db_path = args.db_path
    plot_optimization_history(db_path)