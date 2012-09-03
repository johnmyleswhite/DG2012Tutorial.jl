# SGD Algorithms in Julia

This repository contains very basic implementations of SGD algorithms for fitting OLS and ridge regression to large data sets stored as CSV files. We've chosen to implement these algorithms in Julia to suggest why Julia's speed may make it the perfect tool for doing large-scale data analysis.

For those unfamiliar with the concept of SGD methods for fitting models, we can start by pointing out that SGD stands for Stochastic Gradient Descent. In a stochastic gradient descent algorithm, we follow a gradient that is defined by a small minibatch of data in the hopes of iteratively improving the parameters of a model. In these two examples, the minibatches we've used consist of a single row, so that every row suggests a new direction to move in to improve model fit.

In order to keep this movement reasonable, a line search is done along the direction of the gradient that insures that the new model really does improve the model fit on the current row. This is done over and over again in hopes of finding the global minimum.

# Current STatus
At the moment, the core functionality can be found in `sgd.jl`. Inside, you'll find:
  * Two basic model types: `LinearModel` and `RidgeModel`.
  * A `LinearModel` is a composite type defined by a set of coefficients.
  * A `RidgeModel` is a composite type defined by a set of coefficients and a regularization hyperparameter that controls how strongly the model tries to push the coefficients back towards 0.

Both of these two models implement behaviors:
  * `predict(Model, Inputs, Outputs)`: What prediction does the model make for the current set of inputs?
  * `residual(Model, Inputs, Outputs)`: What residual does the model leave after predicting the current set of inputs?
  * `cost(Model, Inputs, Outputs)`: What is the cost for the current model parameters on the current set of inputs?
  * `gradient(Model, Inputs, Outputs)`: What is the direction of the negative gradient for the current set of inputs?
  * `fit(Filename, Model)`: Call this function to fit the model to a CSV data set in a specific file.
  * `predict(Filename, Model)`: Call this function to make predictions on a CSV data set.
  * `rmse(Filename, Model)`: Call this function to calculate the RMSE of a model on a CSV data set.

# Fitting Considerations

In order to move along the gradient, a distance must be chosen at each step. This distance is often controlled by a learning rate, which may be constant or diminish over time. In this code, we've chosen a different approach and instead decided to do a simple search over distances along the gradient for the distance that approximately minimizes the cost function. This method isn't ideal, but is better than using a constant learning rate or using a learning rate that varies using a simple schedule over time. The real parameter is that the problems we face are often not well-scaled and the use of a single step-size in all directions is problematic.

# To Do

* Add better monitoring functions, including dumping to a file and to STDOUT during model fitting.
* Fitted parameters are not optimal. Need to find ways to get better results out from a single pass through the data.
* Need to automate making multiple passes through the data.
* Need to be able to use minibatches of larger size than a single row.

# Walkthrough

To try this code, start with our toy example. Run the following scripts in the following order:

* toy_generate_data.jl
* toy_example.jl
* toy_example.R

After that, you'll want to try working on a real problem. To do that, you'll first need to download the MSD data set from the UCI ML repository. You can find that data set at http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD

After that, you'll want to run:

* split_data.jl
* real_example.jl
* real_example.R

# Notes

* At present, it takes longer for R for read the MSD data set into memory than it takes for Julia to run an SGD regression on the data. R also uses > 1.5 GB to represent the data set during the loading phase.
* But R finds better final estimates of the model parameters.
* This leaves one big questions: wan we maintain the superiority of Julia's speed while approaching R's accuracy?
