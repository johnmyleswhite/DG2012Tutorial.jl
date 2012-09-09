# Load libraries.
load("sgd.jl")

# QUESTION:
# Which configuration setting is best?
for epochs = [1, 2, 3]
  for learning_rate = [0.9, 0.5, 0.001, 0.0001]
    for minibatch_size = 1:10
      lm = LinearModel(zeros(2))
      fit(lm, "data/ols.csv", true, epochs, minibatch_size, learning_rate, :constant, true, false, false, 100)
      println(join([epochs, learning_rate, minibatch_size, rmse(lm, "data/ols.csv")], "\t"))
    end
  end
end
