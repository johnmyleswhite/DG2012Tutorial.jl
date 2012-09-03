load("sgd.jl")

lm = LinearModel(zeros(91))

for year = 1930:2010
  lm.w[1] = year
  year_rmse = rmse("data/training.csv", lm)
  println("$(year)\t$(year_rmse)")
end
