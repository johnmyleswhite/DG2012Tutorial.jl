abstract Model

type LinearModel <: Model
  w::Vector{Float64}
end

type RidgeModel <: Model
  w::Vector{Float64}
  lambda::Float64
end

# Parse fields from a row of the CSV file into floats.
# Appends an intercept term.
# Assumes y is the first field of the row.
# Assumes x is defined by all the other fields of the row.
# Assumes everything is a floating point number.
function parse_fields(row::String)
  fields = split(row, ",")
  y = float(shift(fields))
  x = map(field -> float(field), fields)
  unshift(x, 1.0)
  (x, y)
end

predict(m::Model, x::Vector{Float64}, y::Float64) = dot(x,m.w)
residual(m::Model, x::Vector{Float64}, y::Float64) = y-predict(m,x,y)

cost(m::LinearModel, x::Vector{Float64}, y::Float64) = 0.5*residual(m,x,y)^2
cost(m::RidgeModel, x::Vector{Float64}, y::Float64) = 0.5*(residual(m,x,y)^2 + m.lambda*sum(m.w[2:end].^2))

gradient(m::LinearModel, x::Vector{Float64}, y::Float64) = residual(m,x,y)*x
function gradient(m::RidgeModel, x::Vector{Float64}, y::Float64)
  r = residual(m,x,y)
  dw = r*x - m.lambda*m.w
  dw[1] = r*x[1] # Don't regularize intercept.
  return dw
end

# Update coefficients.
# Select a step-size using grid-style line search.
function update(lm::LinearModel, x::Vector{Float64}, y::Float64)
  current_cost = cost(lm, x, y)
  dw = gradient(lm, x, y)

  alphas = (10.0).^-(0:32)
  alpha_star = alphas[1]
  least_cost = current_cost

  original_w = copy(lm.w)

  for alpha = alphas
    lm.w = original_w + alpha * dw
    proposed_cost = cost(lm, x, y)

    if proposed_cost <= least_cost
      alpha_star = alpha
      least_cost = proposed_cost
    end
  end

  lm.w = original_w + alpha_star * dw

  if any(isnan(lm.w))
    error("NaN's produced as coefficients")
  end
end

# Update coefficients.
# Select a step-size using grid-style line search.
function update(rm::RidgeModel, x::Vector{Float64}, y::Float64)
  current_cost = cost(rm, x, y)
  dw = gradient(rm, x, y)

  alphas = (10.0).^-(0:32)
  alpha_star = alphas[1]
  least_cost = current_cost

  original_w = copy(rm.w)

  for alpha = alphas
    rm.w = original_w + alpha * dw
    proposed_cost = cost(rm, x, y)

    if proposed_cost <= least_cost
      alpha_star = alpha
      least_cost = proposed_cost
    end
  end

  rm.w = original_w + alpha_star * dw

  if any(isnan(rm.w))
    error("NaN's produced as coefficients")
  end
end

function fit(filename::String, lm::LinearModel)
  # Open the data set
  data_file = open(filename, "r")

  # Loop over the rows of the data set
  row = readline(data_file)
  while length(row) != 0
    (x, y) = parse_fields(chomp(row))
    update(lm, x, y)
    # f = open("logs/lm.tsv", "a")
    # println(f, "Intercept\t$(lm.w[1])")
    # close(f)
    row = readline(data_file)
  end

  close(data_file)
end

function fit(filename::String, rm::RidgeModel)
  # Open the data set
  data_file = open(filename, "r")

  # Loop over the rows of the data set
  row = readline(data_file)
  while length(row) != 0
    (x, y) = parse_fields(chomp(row))
    update(rm, x, y)
    # f = open("logs/rm.tsv", "a")
    # println(f, "Intercept\t$(rm.w[1])")
    # close(f)
    row = readline(data_file)
  end

  close(data_file)
end

function rmse(filename::String, lm::LinearModel)
  # Open the data set
  data_file = open(filename, "r")

  # Keep track of mean squared error.
  mse = 0.0
  i = 1

  # Loop over the entries of the data
  row = readline(data_file)

  while length(row) != 0
    (x, y) = parse_fields(chomp(row))
    se = residual(lm, x, y)^2
    mse = ((i - 1.0) / i) * mse + (1.0 / i) * se
    row = readline(data_file)
    i = i + 1
  end

  close(data_file)

  # Return the RMSE of the model
  sqrt(mse)
end

function rmse(filename::String, rm::RidgeModel)
  # Open the data set
  data_file = open(filename, "r")

  # Keep track of mean squared error.
  mse = 0.0
  i = 1

  # Loop over the entries of the data
  row = readline(data_file)

  while length(row) != 0
    (x, y) = parse_fields(chomp(row))
    se = residual(rm, x, y)^2
    mse = ((i - 1.0) / i) * mse + (1.0 / i) * se
    row = readline(data_file)
    i = i + 1
  end

  close(data_file)

  # Return the RMSE of the model
  sqrt(mse)
end

function predict(filename::String, lm::LinearModel)
  # Open the data set
  data_file = open(filename, "r")

  # Keep track of mean squared error.
  predictions = Array(Float64, 0)

  # Loop over the entries of the data
  row = readline(data_file)

  while length(row) != 0
    (x, y) = parse_fields(chomp(row))
    push(predictions, predict(lm, x, y))
    row = readline(data_file)
  end

  close(data_file)

  predictions
end

function predict(filename::String, rm::RidgeModel)
  # Open the data set
  data_file = open(filename, "r")

  # Keep track of mean squared error.
  predictions = Array(Float64, 0)

  # Loop over the entries of the data
  row = readline(data_file)

  while length(row) != 0
    (x, y) = parse_fields(chomp(row))
    push(predictions, predict(rm, x, y))
    row = readline(data_file)
  end

  close(data_file)

  predictions
end
