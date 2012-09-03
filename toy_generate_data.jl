# Generate a toy data set to test on.
f = open("data/toy.csv", "w")
for i = 1:10_000_000
  x = randn()
  y = 9.9 * x + 1729.0 + randn()
  row = [y, x]
  println(f, join(row, ","))
end
close(f)
