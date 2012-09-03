# Load libraries.
load("sgd.jl")

# Fit a simple linear model.
lm = LinearModel(zeros(2))
fit("data/toy.csv", lm)

# Look at the results a bit.
lm
