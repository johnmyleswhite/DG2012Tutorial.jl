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
println(join([0, rmse("data/training.csv", lm)], "\t"))
println(join([0, rmse("data/validation.csv", lm)], "\t"))

# Iteratively fit the model.
# See how the fitted model performs.
for i = 1:10
  fit("data/training.csv", lm)
  println(join([i, rmse("data/training.csv", lm)], "\t"))
  println(join([i, rmse("data/validation.csv", lm)], "\t"))
end

# Can substantially improve performance by setting the intercept term
# manually to the mode of the data set or some other reasonable initial
# value.
lm.w = zeros(91)
lm.w[1] = 2007.0

# See how an improved null model performs.
println(join([0, rmse("data/training.csv", lm)], "\t"))
println(join([0, rmse("data/validation.csv", lm)], "\t"))

# Iteratively fit the model.
# See how the fitted model performs.
for i = 1:10
  fit("data/training.csv", lm)
  println(join([i, rmse("data/training.csv", lm)], "\t"))
  println(join([i, rmse("data/validation.csv", lm)], "\t"))
end

# Consider optimum reached by R.
weights = csvread("real_target.csv")

lm.w = weights[:, 1]

# See how an improved null model performs.
println(join([0, rmse("data/training.csv", lm)], "\t"))
println(join([0, rmse("data/validation.csv", lm)], "\t"))

# Iteratively fit the model.
# See how the fitted model performs.
for i = 1:10
  fit("data/training.csv", lm)
  println(join([i, rmse("data/training.csv", lm)], "\t"))
  println(join([i, rmse("data/validation.csv", lm)], "\t"))
end

# This is very strange. Model fitting moves away from the optimum.
# Need to understand why. Broken gradient?

###
#
# Ridge SGD
#
###

# Initialize the model. Have to know that there 90 predictors + 1 intercept.
rm = RidgeModel(zeros(91), 10e-2)

# See how a null model performs.
println(join([0, rmse("data/training.csv", rm)], "\t"))
println(join([0, rmse("data/validation.csv", rm)], "\t"))

# Iteratively fit the model.
# See how the fitted model performs.
for i = 1:10
  fit("data/training.csv", lm)
  println(join([i, rmse("data/training.csv", rm)], "\t"))
  println(join([i, rmse("data/validation.csv", rm)], "\t"))
end
