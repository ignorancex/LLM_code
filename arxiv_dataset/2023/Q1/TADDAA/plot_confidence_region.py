#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yu Wang
"""
from matplotlib.patches import Ellipse
from scipy.stats import f
from utils import mean_diff_ci, variance_ratio_ci
import matplotlib.pyplot as plt
import numpy as np
import scipy



def plot_confidence_interval(x, sample, mu_0, true_mean, true_std,
                             color='#2187bb', horizontal_line_width=0.25, 
                             coordinate=1, error_color = 'olive',
                             label = None, margin = 0.0015):
    ci=mean_diff_ci(x=sample, mu=mu_0, k=coordinate)
    left = x - horizontal_line_width / 2
    top = (mu_0[coordinate-1]+ci[1]-true_mean[coordinate-1])/true_std[coordinate-1]
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2
    bottom = (mu_0[coordinate-1]+ci[0]-true_mean[coordinate-1])/true_std[coordinate-1]
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    approx = (mu_0[coordinate-1]-true_mean[coordinate-1])/true_std[coordinate-1]
    if ci[1] < 0:
            plt.plot([x, x], [top+margin, approx-margin], color=error_color,
                     linewidth=6.0, label =label, alpha=0.5)
    elif ci[0] > 0:
            plt.plot([x, x], [approx+margin, bottom-margin], color=error_color,
                     linewidth=6.0, label =label, alpha=0.5)
    plt.plot(x, (np.mean(sample[:,coordinate-1])-true_mean[coordinate-1])/true_std[coordinate-1], 
             '*', color='#2187bb')
    


def plot_var_confidence_interval(x, sample, var_0, true_var,
                                 color='#2187bb', horizontal_line_width=0.25, 
                                 coordinate=1, error_color = 'olive',
                                 label = None, margin = 0.02):
    ci=variance_ratio_ci(sample, var_0, k=coordinate)
    left = x - horizontal_line_width / 2
    top = np.log10(var_0[coordinate-1]*ci[1]/true_var[coordinate-1])
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2
    bottom = np.log10(var_0[coordinate-1]*ci[0]/true_var[coordinate-1])
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    approx = np.log10(var_0[coordinate-1]/true_var[coordinate-1])
    if ci[0] > 1:
            plt.plot([x, x], [approx+margin, bottom-margin], color=error_color,
                     linewidth=6.0, label =label, alpha=0.5)
    elif ci[1] < 1:
            plt.plot([x, x], [top+margin, approx-margin], color=error_color,
                     linewidth=6.0, label =label, alpha=0.5)
    plt.plot(x, np.log10(np.var(sample[:,coordinate-1])/true_var[coordinate-1]), 
             '*', color=color)


def plot_quantile_confidence_interval(x, sample, mu_0, var_0, true_mu, true_var,
                                 color='#2187bb', horizontal_line_width=0.25, 
                                 coordinate=1, error_color = 'olive',
                                 label = None, margin = 0.02, p = 0.5):
    zp = scipy.stats.norm.ppf(p)
    approx = (zp*var_0[coordinate-1]**0.5+mu_0[coordinate-1])-(zp*true_var[coordinate-1]**0.5+true_mu[coordinate-1])
    ci=quantile_diff_ci(sample, mu=mu_0, diag_sigma=var_0,
                        p = p, alpha = 0.05, k=coordinate)
    left = x - horizontal_line_width / 2
    top = ci[1] + approx
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2
    bottom = ci[0] + approx
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    if ci[0] > 0:
            plt.plot([x, x], [approx+margin, bottom-margin], color=error_color,
                     linewidth=6.0, label =label, alpha=0.5)
    elif ci[1] < 0:
            plt.plot([x, x], [top+margin, approx-margin], color=error_color,
                     linewidth=6.0, label =label, alpha=0.5)
    interval_mid = np.std(sample[:, coordinate-1])*zp+np.mean(sample[:, coordinate-1])-(zp*(true_var[coordinate-1]**0.5)+true_mu[coordinate-1])
    plt.plot(x, interval_mid, '*', color=color)








def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)


def confidence_region(x, color, alpha = 0.05):
    """
    plot confidence region
    """
    mean = np. mean(x, 0)
    covariance = np.cov(x.T)
    n = x.shape[0]
    p = x.shape[1]
    scale = np.sqrt(p * (n-1) * f.ppf(1-alpha, p, n-p)/(n * (n-p)))
    return get_cov_ellipse(cov = covariance, centre = mean, 
                           nstd = scale, fc = color, alpha = 0.3)
























