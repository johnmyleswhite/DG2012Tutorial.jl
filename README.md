# SGD Algorithms in Julia

This repository contains very basic implementations of SGD algorithms that can be used to fit OLS regression, ridge regression and logistic regression to large data sets that are stored as CSV files on disk. We've chosen to implement these algorithms in Julia to show how Julia's speed can make it the perfect tool for doing large-scale data analysis.

For those unfamiliar with the use of SGD methods for fitting models, we note that SGD stands for "Stochastic Gradient Descent". In a stochastic gradient descent algorithm, we follow a gradient that is defined by a small minibatch of data in the hopes of iteratively improving the parameters of a model until they converge upon a globally good set of parameters for the data.

In standard SGD algorithms, each minibatch consists of a single randomly selected row, so that every row suggests a new direction to move in to improve model fit. But it is also possible to use minibatches of arbitrary size; at the furthest extreme, we can set up an SGD algorithm to use a minibatch that is as large as the entire data set. At that point, an SGD algorithm is identical to a standard gradient descent algorithm run in batch mode on your entire data set.

The functions in this repository allows you to explore this continuum. To get your started, we've provided one example that can help you to build an intuition for what happens when you change the size of the minibatches you use. We've also explored several of the other hyperparameters that affect performance of SGD. As you'll see, SGD can be finicky. When it works, it works beautifully; but it can easily fail to find any useful parameter estimates for your problem. Caution and cross-validation are strongly encouraged.

# Current Status

At the moment, the core functionality of the code in this repository can be found in `sgd.jl`. Inside, you'll find:

  * Three basic model types: `LinearModel`,`RidgeModel` and `LogisticModel`.
  * A `LinearModel` is a composite type defined by a set of coefficients and a count of the number of rows of data that have been seen in the past.
  * A `RidgeModel` is a composite type defined by a set of coefficients, a regularization hyperparameter that controls how strongly the model tries to push the coefficients back towards 0, and a count of the number of rows of data that have been seen in the past.
  * A `LogisticModel` is a composite type defined by a set of coefficients and a count of the number of rows of data that have been seen in the past.

All of these three models implement the following behaviors:

  * `predict(Model, Inputs, Outputs)`: What prediction does the model make for the current set of inputs?
  * `residual(Model, Inputs, Outputs)`: What residual does the model leave after predicting the current set of inputs?
  * `cost(Model, Inputs, Outputs)`: What is the cost for the current model parameters on the current set of inputs?
  * `gradient(Model, Inputs, Outputs)`: What is the direction of the negative gradient for the current set of inputs?
  * `fit(Model, Filename)`: Call this function to fit a model to a data set stored in a specific CSV file.
  * `predict(Model, Filename)`: Call this function to make predictions on a specific CSV data set.
  * `rmse(Model, Filename)`: Call this function to calculate the RMSE of a model on a CSV data set.

# Configuration Settings

  * Minimize: Should the model be fit to minimize a cost or maximize a likelihood? For `LinearModel` and `RidgeModel`, this should be set to `true`. For a `LogisticModel`, this should be set to false.
  * Epoches: How many epochs of training should be performed? Defaults to `1`.
  * Mini-Batch Size: How many rows should go into each minibatch? Defaults to `1`.
  * Learning Rate: What is the hard-coded learning rate? Defaults to `0.01`.
  * Learning Rule: Should SGD use a constant learning rate? Or a decreasing learning rate? Defaults to `:constant`.
  * Averaging: Should the learned parameters be averaged over time? This averaging will only occur in the second epoch, so it may have no effect if you will only have epoch of training for your model. Defaults to `false`.
  * Logging: Prints the model's coefficients to STDOUT at regular intervals. Defaults to `false`.
  * Trace: Prints the model's performance on the entire training data set. Defaults to `false`.
  * Interval Length: How often should logging and/or tracing occur? Defaults to `1_000`.

# Walkthrough

To try this code, start with our basic examples. Run the following scripts in the following order:

### OLS Regression

* ols_generate_data.jl
* ols_example.jl
* ols_example.R

### Logistic Regression

* logistic_generate_data.jl
* logistic_example.jl
* logistic_example.R

# Tips and Tricks

* Because the SGD operates very locally based on the current row of data and recent rows of data, it is important that the rows of data be as close to randomly ordered as possible. The worst case scenario is to apply the SGD to data in which the optimal model gradually changes over rows; in this case, the model will never settle down in a reasonable region of parameter space. In practice, haphazard orderings instead of intentionally randomized orderings often work well.
* It is also important to insure that the algorithm not move too far along the gradient at each step. This requires setting a low learning rate, so that the model never makes a step so large that the next required step will have to be even larger to make up for the last step.

# To Do

* Add better monitoring functions, including dumping to a file and to STDOUT during model fitting.
* Fitted parameters are not optimal. Need to find ways to get better results out from a single pass through the data.
* Need to automate making multiple passes through the data.
* Need to be able to use minibatches of larger size than a single row.

