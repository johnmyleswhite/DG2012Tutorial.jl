load("sgd.jl")

lambda_values = (10.0) .^ -(1:12)

lowest_error = Inf
best_lambda = lambda_values[1]

for lambda = lambda_values
  rm = RidgeModel(zeros(91), lambda)
  fit("data/training.csv", rm)
  cv_error = rmse("data/validation.csv", rm)
  if cv_error <= lowest_error
    best_lambda = lambda
    lowest_error = cv_error
  end
  println("$(lambda)\t$(cv_error)")
end

rm = RidgeModel(zeros(91), best_lambda)
fit("data/training.csv", rm)
test_error = rmse("data/test.csv", rm)
