load("sgd.jl")

lm = LinearModel(zeros(2))
fit("data/toy.csv", lm)

lm
