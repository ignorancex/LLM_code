#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from bibliobanana import compute_yearly_citations, load_results_from_file, \
    plot_yearly_count
from matplotlib import pyplot


# Define the search terms of interest.
search_terms = [ \
    "k-means", \
    "hierarchical clustering", \
    "c-means", \
    "latent class analysis", \
    "latent profile analysis", \
    ]

# Define the comperison terms.
comparison_terms = ["t-test"]
# Define the search range.
start_date = 1990
end_date = 2022

# Construct the name of the file to which we should save the data.
save_file_name = "bibliobanana_clustering_{}-{}".format(start_date, end_date)

# Files and folders.
this_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(this_dir, "output_bibliobanana")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
save_file = os.path.join(out_dir, save_file_name)

# Get the results from PubMed.
if not os.path.isfile(save_file+".csv"):
    print("Getting data from PubMed...")
    result = compute_yearly_citations(search_terms, start_date, end_date, \
        comparison_terms=comparison_terms, database="pubmed", \
        exact_phrase=True, pause=0.5, verbose=True, \
        save_to_file=save_file+".csv", plot_to_file=None)
# Load from an existing file.
else:
    print("Loading data from file...")
    result = load_results_from_file(save_file+".csv")

print("Plotting results...")

# Plot the results.
fig, ax = plot_yearly_count(result, plot_ratio=False, \
    plot_average_comparison=False, scale_to_max=False, \
    figsize=(8.0,6.0), dpi=100.0)
fig.savefig(save_file+".png")
pyplot.close(fig)

# Plot the results as ratios of the comparison terms.
fig, ax = plot_yearly_count(result, plot_ratio=True, \
    plot_average_comparison=False, scale_to_max=False, \
    figsize=(8.0,6.0), dpi=900.0)
# Move the legend to the top-left.
ax.legend(loc="upper left", fontsize=14)
# Save the figure.
fig.savefig(save_file+"_ratios.png")
pyplot.close(fig)

print("All done!")
