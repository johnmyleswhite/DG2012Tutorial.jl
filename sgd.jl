## Generic Model Functionality ##

abstract Model

# NOTE: Models are assumed to have a .w "weights" field
# that holds a vector of Float64 weight coefficients

predict(m::Model, x::Vector{Float64}, y::Float64) = dot(x,m.w)
residual(m::Model, x::Vector{Float64}, y::Float64) = y-predict(m,x,y)

# Update coefficients.
# Select a step-size using grid-style line search.
function update(m::Model, x::Vector{Float64}, y::Float64, log_state::Bool)
  m.n += 1
  current_cost = cost(m,x,y)
  dw = gradient(m,x,y)

  alphas = (10.0).^-(0:32)
  push(alphas, 0.0)
  alpha_star = alphas[1]
  least_cost = current_cost

  original_w = copy(m.w)

  for alpha = alphas
    # TODO: faster in-place update code
    m.w = original_w + alpha * dw
    proposed_cost = cost(m,x,y)

    if proposed_cost <= least_cost
      alpha_star = alpha
      least_cost = proposed_cost
    end
  end

  m.w = original_w + alpha_star * dw

  if any(isnan(m.w))
    error("NaN's produced as coefficients")
  end
  if log_state
    open("logs/model.tsv", "a") do f
      println(f, join([current_cost, m.w], "\t"))
    end
  end
end
update(m::Model, x::Vector{Float64}, y::Float64) = update(m,x,y,false)

# Parse fields from a row of the CSV file into floats.
# Appends an intercept term.
# Assumes y is the first field of the row.
# Assumes x is defined by all the other fields of the row.
# Assumes everything is a floating point number.
function parse_fields(row::String)
  x = float(split(row, ","))
  y, x[1] = x[1], 1.0
  (x, y)
end

function fit(filename::String, m::Model)
  function average(history)
    sum(history) ./ length(history)
  end
  open(filename, "r") do data_file
    history_length = 25_000
    history = Array(Vector{Float64}, history_length)
    i = 0
    row = readline(data_file)
    while length(row) != 0
      i += 1
      x, y = parse_fields(chomp(row))
      update(m,x,y)
      # Added weight averaging
      index = i % history_length
      if index == 0
        index = history_length
      end
      history[index] = lm.w
      if i > 2 * history_length && index == history_length
        println("Average: $(average(history))")
        lm.w = average(history)
      end
      # End weight averaging.
      row = readline(data_file)
    end
  end
end

function rmse(filename::String, m::Model)
  mse = 0.0
  open(filename, "r") do data_file
    i = 1
    row = readline(data_file)
    while length(row) != 0
      x, y = parse_fields(chomp(row))
      se = residual(m,x,y)^2
      mse = ((i-1.0)/i)*mse+(1.0/i)*se
      row = readline(data_file)
      i += 1
    end
  end
  sqrt(mse)
end

function predict(filename::String, m::Model)
  predictions = Array(Float64, 0)
  open(filename, "r") do data_file
    row = readline(data_file)
    while length(row) != 0
      x, y = parse_fields(chomp(row))
      push(predictions, predict(m,x,y))
      row = readline(data_file)
    end
  end
  predictions
end

## Specific Model Implementations ##

type LinearModel <: Model
  w::Vector{Float64}
  n::Int64
end

LinearModel(w::Vector{Float64}) = LinearModel(w, 0)

type RidgeModel <: Model
  w::Vector{Float64}
  lambda::Float64
  n::Int64
end

RidgeModel(w::Vector{Float64}, lambda::Float64) = RidgeModel(w, lambda, 0)

cost(m::LinearModel, x::Vector{Float64}, y::Float64) = 0.5*residual(m,x,y)^2
cost(m::RidgeModel, x::Vector{Float64}, y::Float64) =
    0.5*(residual(m,x,y)^2 + m.lambda*sum(m.w[2:end].^2))

gradient(m::LinearModel, x::Vector{Float64}, y::Float64) = residual(m,x,y)*x
function gradient(m::RidgeModel, x::Vector{Float64}, y::Float64)
  r = residual(m,x,y)
  dw = r*x - m.lambda*m.w
  dw[1] = r*x[1] # Don't regularize intercept.
  return dw
end
