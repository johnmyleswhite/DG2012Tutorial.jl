# Generate an example data set to test on.
load("distributions.jl")
load("sgd.jl")

file = open("data/logistic.csv", "w")
for i = 1:100_000
  x = randn()
  z = 1.7 * x + 0.29
  p = invlogit(z)
  y = rand(Bernoulli(p))
  row = [y, x]
  println(file, join(row, ","))
end
close(file)
