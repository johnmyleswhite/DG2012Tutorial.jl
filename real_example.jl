load("sgd.jl")

# OLS SGD
lm = LinearModel(zeros(91))

rmse("data/training.csv", lm)
rmse("data/validation.csv", lm)

fit("data/training.csv", lm)

rmse("data/training.csv", lm)
rmse("data/validation.csv", lm)

# Ridge SGD
rm = RidgeModel(zeros(91), 10e-2)

rmse("data/training.csv", rm)
rmse("data/validation.csv", rm)

fit("data/training.csv", rm)

rmse("data/training.csv", rm)
rmse("data/validation.csv", rm)
