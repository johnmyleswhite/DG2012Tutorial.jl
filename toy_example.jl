# Load libraries.
load("sgd.jl")

# Set up a linear model.
lm = LinearModel(zeros(2))

# Fit a simple linear model with one-pass through the data.
fit(lm, "data/toy.csv")

# Look at the results a bit.
lm

# Make four more passes.
fit(lm, "data/toy.csv", 4)

# Make five more passes.
fit(lm, "data/toy.csv", 5)

# Make ninety more passes.
fit(lm, "data/toy.csv", 90)

# Exploit all options.
fit(lm, "data/toy.csv", 1, 1, 0.01, :constant, true, true, true, 10)

# Exploit all options.
fit(lm, "data/toy.csv", 1, 1, 0.01, :constant, true, true, false, 1)

# Exploit all options.
fit(lm, "data/toy.csv", 1, 25, 0.01, :constant, true, true, true, 25)
