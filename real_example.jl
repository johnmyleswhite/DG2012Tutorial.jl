load("sgd.jl")

# OLS SGD
lm = LinearModel(zeros(91))

#predict("data/training.csv", lm)
rmse("data/training.csv", lm)

lm.w[1] = 2007.0
fit("data/training.csv", lm)

#predict("data/training.csv", lm)
rmse("data/training.csv", lm)

# Ridge SGD

rm = RidgeModel(zeros(91), 10e-2)

#predict("data/training.csv", rm)
rmse("data/training.csv", rm)

rm.w[1] = 2007.0
fit("data/training.csv", rm)

#predict("data/training.csv", rm)
rmse("data/training.csv", rm)
