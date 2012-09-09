# SGD Algorithms in Julia

This repository contains very basic implementations of SGD algorithms that can be used to fit OLS linear regression, ridge regression and logistic regression to large data sets that are stored as CSV files on disk. We've chosen to implement these algorithms in Julia to show how Julia's speed can make it the perfect tool for doing large-scale data analysis.

For those unfamiliar with the concept of SGD methods for fitting models, we'll start by pointing out that SGD stands for "Stochastic Gradient Descent". In a stochastic gradient descent algorithm, we follow a gradient that is defined by a small minibatch of data in the hopes of iteratively improving the parameters of a model until they converge upon a globally good set of parameters for the data.

In standard SGD algorithms, each minibatch consists of a single row, so that every row suggests a new direction to move in to improve model fit. But it is also possible to use minibatches of arbitrary size; at the furthest extreme, we can set up an SGD algorithm to use a minibatch that is as large as the entire data set. At that point, an SGD algorithm is identical to a standard gradient descent algorithm. Our codebase allows you to explore this continuum and we've provided one example that can help you to build an intuition for what happens when you change the size of the minibatches you use.

# Current Status
At the moment, the core functionality of this project can be found in `sgd.jl`. Inside, you'll find:
  * Three basic model types: `LinearModel`,`RidgeModel` and `LogisticModel`.
  * A `LinearModel` is a composite type defined by a set of coefficients and a count of the number of rows of data that have been seen in the past.
  * A `RidgeModel` is a composite type defined by a set of coefficients,a regularization hyperparameter that controls how strongly the model tries to push the coefficients back towards 0, and a count of the number of rows of data that have been seen in the past.
  * A `LogisticModel` is a composite type defined by a set of coefficients and a count of the number of rows of data that have been seen in the past.

All of these three models implement the following behaviors:
  * `predict(Model, Inputs, Outputs)`: What prediction does the model make for the current set of inputs?
  * `residual(Model, Inputs, Outputs)`: What residual does the model leave after predicting the current set of inputs?
  * `cost(Model, Inputs, Outputs)`: What is the cost for the current model parameters on the current set of inputs?
  * `gradient(Model, Inputs, Outputs)`: What is the direction of the negative gradient for the current set of inputs?
  * `fit(Filename, Model)`: Call this function to fit the model to a CSV data set in a specific file.
  * `predict(Filename, Model)`: Call this function to make predictions on a CSV data set.
  * `rmse(Filename, Model)`: Call this function to calculate the RMSE of a model on a CSV data set.

# Configuration Settings

  * Minimize: Should the model be fit to minimize a cost or maximize a likelihood?
  * Epoches: How many epochs of training should be performed?
  * Mini-Batch Size: How many rows should go into each minibatch?
  * Learning Rate: What is the hard-coded learning rate?
  * Learning Rule: Use a constant learning rate? Or a decreasing learning rate?
  * Averaging: Should the learned parameters be averaged over time?
  * Logging: Show the model at regular intervals.
  * Trace: Show model's performance on other data set
  * Interval Length: How often should logging and tracing occur?

# Tips and Tricks

* Because the SGD operates very locally based on the current row of data and recent rows of data, it is important that the rows of data be relatively randomly ordered. The worst case scenario is to apply the SGD to data in which the optimal model gradually changes over rows; in this case, the model will never settle down in a reasonable region of parameter space.
* It is also important to insure that the algorithm not move too far along the gradient at each step. This requires setting a low learning rate, so that the model never makes a step so large that the next required step will have to be even larger to make up for the last step.

# Fitting Considerations

In order to move along the gradient, a distance must be chosen at each step. This distance is often controlled by a learning rate, which may be constant or diminish over time. In this code, we've chosen a different approach and instead decided to do a simple search over distances along the gradient for the distance that approximately minimizes the cost function. This method isn't ideal, but is better than using a constant learning rate or using a learning rate that varies using a simple schedule over time. The real parameter is that the problems we face are often not well-scaled and the use of a single step-size in all directions is problematic.

# To Do

* Add better monitoring functions, including dumping to a file and to STDOUT during model fitting.
* Fitted parameters are not optimal. Need to find ways to get better results out from a single pass through the data.
* Need to automate making multiple passes through the data.
* Need to be able to use minibatches of larger size than a single row.

# Walkthrough

To try this code, start with our basic examples. Run the following scripts in the following order:

* ols_generate_data.jl
* ols_example.jl
* ols_example.R

* logistic_generate_data.jl
* logistic_example.jl
* logistic_example.R

After that, you'll want to try working on a real problem. To do that, you'll first need to download the MSD data set from the UCI ML repository. You can find that data set at http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD

After that, you'll want to run:

* split_data.jl
* real_example.jl
* real_example.R

# Notes

* At present, it takes longer for R for read the MSD data set into memory than it takes for Julia to run an SGD regression on the data. R also uses > 1.5 GB to represent the data set during the loading phase.
* But R finds better final estimates of the model parameters.
* This leaves one big questions: wan we maintain the superiority of Julia's speed while approaching R's accuracy?
