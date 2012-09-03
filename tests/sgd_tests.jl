load("sgd.jl")

#
# OLS SGD
#

lm = LinearModel([19.0])
x = [1.0]
y = 20.0

@assert abs(predict(lm, x, y) - 19.0) < 10e-16
@assert abs(residual(lm, x, y) - 1.0) < 10e-16
@assert abs(cost(lm, x, y) - 0.5) < 10e-16
@assert norm(gradient(lm, x, y) - [1.0]) < 10e-16

update(lm, x, y)

@assert abs(predict(lm, x, y) - 20.0) < 10e-16
@assert abs(residual(lm, x, y) - 0.0) < 10e-16
@assert abs(cost(lm, x, y) - 0.0) < 10e-16
@assert norm(gradient(lm, x, y) - [0.0]) < 10e-16

#
# Ridge SGD
#

rm = RidgeModel([19.0, 19.0], 10e8)
x = [1.0, 0.0]
y = 20.0

@assert abs(predict(rm, x, y) - 19.0) < 10e-16
@assert abs(residual(rm, x, y) - 1.0) < 10e-16
@assert abs(cost(rm, x, y) - (0.5 + (rm.lambda / 2.0) * rm.w[2]^2)) < 10e-16
@assert norm(gradient(rm, x, y) - [1.0, -rm.lambda * rm.w[2]]) < 10e-16

update(rm, x, y)

# Need to do this calculation by hand.

# @assert abs(predict(rm, x, y) - 19.0) < 10e-16
# @assert abs(residual(rm, x, y) - 1.0) < 10e-16
# @assert abs(cost(rm, x, y) - (0.5 + (rm.lambda / 2.0) * rm.w[2]^2)) < 10e-16
# @assert norm(gradient(rm, x, y) - [1.0, -rm.lambda * rm.w[2]]) < 10e-16
