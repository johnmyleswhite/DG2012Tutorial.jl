load("sgd.jl")

lambda_values = (10.0) .^ -(3:12)

for lambda = lambda_values
  rm = RidgeModel(zeros(91), lambda)
  fit(rm, "data/training.csv", true, 1, 1, 10e-12, :constant, true, false, false, 2500)
  cv_error = rmse(rm, "data/validation.csv")
  println("$(lambda)\t$(cv_error)")
end
