load("sgd.jl")

lm = LinearModel([19.0])
x = [1.0]
y = 20.0


lm

predict(lm, x, y)
residual(lm, x, y)
cost(lm, x, y)
gradient(lm, x, y)
update(lm, x, y)

lm
