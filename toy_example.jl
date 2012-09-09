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

# Look at the results again.
lm

# Make five more passes.
fit(lm, "data/toy.csv", 5)

# Look at the results one last time.
lm

# Now we'll start again and exploit the many configurable options.
lm = LinearModel(zeros(2))
fit(lm, "data/toy.csv", true, 1, 25, 0.1, :constant, true, true, true, 50)

# Minimize: Should the model be fit to minimize a cost or maximize a likelihood?
# Epoches: How many epochs of training should be performed?
# Mini-Batch Size: How many rows should go into each minibatch?
# Learning Rate: What is the hard-coded learning rate?
# Learning Rule: Use a constant learning rate? Or a decreasing learning rate?
# Averaging: Should the learned parameters be averaged over time?
# Logging: Show the model at regular intervals.
# Trace: Show model's performance on other data set
# Interval Length: How often should logging and tracing occur?
