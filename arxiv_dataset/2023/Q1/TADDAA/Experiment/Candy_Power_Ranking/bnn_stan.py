# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:31:16 2022

@author: Aawangyu2799
"""

nn_code = """
data {
  int<lower=1> N; // num train instances
  int<lower=1> M; // num train predictors
  matrix[N, M] x; // train predictors
  int<lower=2> K; // num categories
  int<lower=0,upper=K> y[N];// train category
  int<lower=1> J; // num hidden units on layer 1
  int<lower=1> H; // num hidden units on layer 2
}
parameters {
  matrix[M, J] alpha;
  matrix[J, H] lambda;
  vector[H] beta;
}
model {
  vector[N] v = tanh(tanh(x * alpha)*lambda) * beta;
  
  // priors
  to_vector(alpha) ~ normal(0, 2);
  to_vector(lambda) ~ normal(0, 2);
  beta ~ normal(0, 2);
  
  // likelihood
    y ~ bernoulli_logit(tanh(tanh(x * alpha)*lambda) * beta);
}
"""