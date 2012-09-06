## Generic Model Functionality ##

abstract Model

# NOTE: Models are assumed to have a .w "weights" field
# that holds a vector of Float64 coefficients.

predict(m::Model, x::Vector{Float64}) = dot(x, m.w)

function predict(m::Model, x::Matrix{Float64})
  n = size(x, 1)
  p = size(x, 2)
  predictions = zeros(n)
  for i = 1:n
    predictions[i] = predict(m, reshape(x[i, :], p))
  end
  return predictions
end

residuals(m::Model, x::Vector{Float64}, y::Float64) = y - predict(m, x)

function residuals(m::Model, x::Matrix{Float64}, y::Vector{Float64})
  n = size(x, 1)
  p = size(x, 2)
  # if n != length(y)
  #   error("x and y must have the same length")
  # end
  res = zeros(n)
  for i = 1:n
    res[i] = y[i] - predict(m, reshape(x[i, :], p))
  end
  return res
end

# Update coefficients.
# Modify to handle multiple examples at once.
function update(m::Model,
                x::Union(Vector{Float64}, Matrix{Float64}),
                y::Union(Float64, Vector{Float64}),
                learning_rate::Float64,
                learning_rule::Symbol,
                averaging::Bool)
  # Increment the number of examples we've seen.
  if typeof(x) == Vector{Float64}
    m.n += 1
  else
    m.n += size(x, 1)
  end

  # Evaluate the gradient on these examples.
  # This needs to be customized for vector vs. matrix inputs.
  dw = -gradient(m, x, y)

  # Use Scikits-Learning style learning rate schedules.
  if learning_rule == :constant
    alpha = learning_rate
  elseif learning_rule == :optimal
    alpha = 1.0 / (m.n + 1.0) # Make 1.0 configurable in the future.
  elseif learning_rule == :inverse_scaling
    alpha = learning_rate / m.n ^ 0.5 # Make 0.5 configurable in the future.
  end

  # Use ASGD to smooth weights over many examples.
  # Only start averaging during second epoch.
  if averaging && m.epoch > 1
    new_w = m.w + alpha * dw
    m.w = ((m.n - 1.0) / (m.n)) * m.w + (1.0 / m.n) * new_w
  else
    m.w = m.w + alpha * dw
  end

  # Stop processing if weights become corrupted.
  if any(isnan(m.w))
    error("NaN's produced as coefficients during update step")
  end
end

function update(m::Model,
                x::Union(Vector{Float64}, Matrix{Float64}),
                y::Union(Float64, Vector{Float64}))
  update(m, x, y, 0.01, :constant, false)
end

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

function fit(m::Model,
             filename::String,
             epochs::Int64,
             minibatch_size::Int64,
             learning_rate::Float64,
             learning_rule::Symbol,
             averaging::Bool,
             logging::Bool,
             trace::Bool,
             interval_length::Int64)
  for epoch = 1:epochs
    m.epoch += 1
    open(filename, "r") do data_file
      while true
        if minibatch_size == 1
          row = readline(data_file)
          if length(row) == 0
            break
          end
          x, y = parse_fields(chomp(row))
        else
          row = readline(data_file)
          if length(row) == 0
            break
          end
          tmp_x, tmp_y = parse_fields(chomp(row))
          x, y = (tmp_x', [tmp_y])
          for i = 2:minibatch_size
            row = readline(data_file)
            if length(row) == 0
              continue
            end
            tmp_x, tmp_y = parse_fields(chomp(row))
            x = vcat(x, tmp_x')
            push(y, tmp_y)
          end
        end
        update(m, x, y, learning_rate, learning_rule, averaging)
        if logging && m.n % interval_length == 0
          println(join({"Iteration: $(m.n)", m.w}, "\t"))
        end
        if trace && m.n % interval_length == 0
          # Need to add * ASCIIString, Float64 to Julia.
          println("RMSE: " * string(rmse(m, filename)))
        end
      end
    end
  end
end

function fit(m::Model, filename::String, epochs::Int64)
  fit(m, filename, epochs, 1, 0.001, :constant, true, false, false, 0)
end

# Run for 1 epoch by default.
function fit(m::Model, filename::String)
  fit(m, filename, 1, 1, 0.001, :constant, true, false, false, 0)
end

function rmse(m::Model, filename::String)
  mse = 0.0
  open(filename, "r") do data_file
    i = 1
    row = readline(data_file)
    while length(row) != 0
      x, y = parse_fields(chomp(row))
      se = residuals(m,x,y)^2
      mse = ((i-1.0)/i)*mse+(1.0/i)*se
      row = readline(data_file)
      i += 1
    end
  end
  sqrt(mse)
end

# Change to not use in-memory storage.
# Default to printing predictions to STDOUT, but allow IOStream to be specified.
# Allow forcing to use in-memory storage.
function predict(m::Model, filename::String)
  predictions = Array(Float64, 0)
  open(filename, "r") do data_file
    row = readline(data_file)
    while length(row) != 0
      x, y = parse_fields(chomp(row))
      push(predictions, predict(m,x))
      row = readline(data_file)
    end
  end
  predictions
end

## Specific Model Implementations ##

type LinearModel <: Model
  w::Vector{Float64}
  n::Int64
  epoch::Int64
end

LinearModel(w::Vector{Float64}) = LinearModel(w, 0, 0)

type RidgeModel <: Model
  w::Vector{Float64}
  lambda::Float64
  n::Int64
  epoch::Int64
end

RidgeModel(w::Vector{Float64}, lambda::Float64) = RidgeModel(w, lambda, 0, 0)

cost(m::LinearModel, x::Vector{Float64}, y::Float64) = 0.5*residuals(m,x,y)^2
cost(m::RidgeModel, x::Vector{Float64}, y::Float64) =
    0.5*(residual(m,x,y)^2 + m.lambda*sum(m.w[2:end].^2))
function cost(m::LinearModel, x::Matrix{Float64}, y::Vector{Float64})
  n = size(x, 1)
  p = size(x, 2)
  total_cost = 0.0
  for i = 1:n
    total_cost += cost(m, reshape(x[i, :], p), y[i])
  end
  return total_cost
end
function cost(m::RidgeModel, x::Matrix{Float64}, y::Vector{Float64})
  n = size(x, 1)
  p = size(x, 2)
  total_cost = 0.0
  for i = 1:n
    total_cost += 0.5*(residual(m, reshape(x[i, :], p), y))^2
  end
  total_cost += m.lambda * sum(m.w[2:end].^2)
  return total_cost
end

gradient(m::LinearModel, x::Vector{Float64}, y::Float64) = residuals(m,x,y)*(-x)
function gradient(m::RidgeModel, x::Vector{Float64}, y::Float64)
  r = residuals(m,x,y)
  dw = r*(-x) + m.lambda*m.w
  dw[1] = r*(-x[1])
  return dw
end
function gradient(m::LinearModel, x::Matrix{Float64}, y::Vector{Float64})
  n = size(x, 1)
  p = size(x, 2)
  summed_gradient = zeros(p)
  for i = 1:n
    summed_gradient += gradient(m, reshape(x[i, :], p), y[i])
  end
  return summed_gradient
end
function gradient(m::RidgeModel, x::Matrix{Float64}, y::Vector{Float64})
  n = size(x, 1)
  p = size(x, 2)
  summed_gradient = zeros(p)
  for i = 1:n
    local_x = reshape(x[i, :], p)
    r = residuals(m, local_x, y[i])
    summed_gradient += r * (-local_x)
  end
  for j = 2:p
    summed_gradient[j] += m.lambda * m.w[j]
  end
  return summed_gradient
end
