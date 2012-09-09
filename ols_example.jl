# Load libraries.
load("sgd.jl")

# Set up a linear model.
lm = LinearModel(zeros(2))

# Fit a simple linear model with one-pass through the data.
fit(lm, "data/ols.csv")

# Look at the results a bit.
lm

# Make four more passes.
fit(lm, "data/ols.csv", 4)

# Look at the results again.
lm

# Make five more passes.
fit(lm, "data/ols.csv", 5)

# Look at the results one last time.
lm

# Now we'll start again and exploit the many configurable options.
lm = LinearModel(zeros(2))
fit(lm, "data/ols.csv", true, 1, 25, 0.1, :constant, true, true, true, 50)
