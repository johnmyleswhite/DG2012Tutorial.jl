* We want to showcase Julia's speed.
* One place where fast code matters a lot is in fitting simple linear models to very large data sets.
* This type of model fitting is typically done using Stochastic Gradient Descent or SGD for short.
* I've made a very rough first pass at implementing SGD code for OLS regression and ridge regression. Both are found in `sgd.jl`.
* There are two model types: `LinearModel` and `RidgeModel`. A `LinearModel` is defined by a set of coefficients. A `RidgeModel` is defined by a set of coefficients and a regularization hyperparameter that controls how strongly the model tries to push the coefficients back towards 0.
* These models implement the following behaviors:
  * predict(Model, Inputs, Outputs)
  * residual(Model, Inputs, Outputs)
  * cost(Model, Inputs, Outputs)
  * gradient(Model, Inputs, Outputs)
  * fit(Filename, Model)
  * predict(Filename, Model)
  * rmse(Filename, Model)

* In order to fit these models, we estimate the gradient and take a step in that direction using a grid-based line search that moves somewhere along the negative gradient of the cost function. This method isn't ideal, but is better than using a constant learning rate or using a learning rate that varies using a simple schedule over time. The real parameter is that the problems we face are often not well-scaled and the use of a single step-size in all directions is problematic.

* Need better monitoring functions, including dumping to a file and to STDOUT.

* Need to figure out how to change algorithm to give more precise answers. May need something smarter than SGD or may need to use minibatches or larger size than a single row.

* Run in order:
* toy_generate_data.jl
* toy_example.jl
* toy_example.R
* Gather data from UCI ML repository
* split_data.jl
* real_example.jl
* real_example.R

* At present, it takes longer for R for read the MSD data set into memory than it takes Julia to run an SGD regression on the data.
* R also uses > 1.5 GB to represent the data set during the loading phase.
* Can we maintain the superiority of Julia's speed while approaching R's accuracy?


