# Load a library of types and functions.
load("sgd.jl")

###
#
# OLS SGD
#
###

# Initialize the model. Have to know that there 90 predictors + 1 intercept.
lm = LinearModel(zeros(91))

# See how a null model performs.
rmse("data/training.csv", lm)
rmse("data/validation.csv", lm)

# Fit the model.
fit("data/training.csv", lm)

# See how the fitted model performs.
rmse("data/training.csv", lm)
rmse("data/validation.csv", lm)

# Multiple passes through the data do help because the SGD algorithm does
# not run to convergence by default.
fit("data/training.csv", lm)

rmse("data/training.csv", lm)
rmse("data/validation.csv", lm)

fit("data/training.csv", lm)

rmse("data/training.csv", lm)
rmse("data/validation.csv", lm)

# Can substantially improve performance by setting the intercept term
# manually to the mode of the data set.
lm.w[1] = 2007.0

fit("data/training.csv", lm)

rmse("data/training.csv", lm)
rmse("data/validation.csv", lm)

###
#
# Ridge SGD
#
###

# Initialize the model. Have to know that there 90 predictors + 1 intercept.
rm = RidgeModel(zeros(91), 10e-2)

# See how a null model performs.
rmse("data/training.csv", rm)
rmse("data/validation.csv", rm)

# Fit the model.
fit("data/training.csv", rm)

# See how the fitted model performs.
rmse("data/training.csv", rm)
rmse("data/validation.csv", rm)
