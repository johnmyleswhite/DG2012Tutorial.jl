filenames = ["data/training.csv", "data/validation.csv", "data/test.csv"]
for filename = filenames
  f = open(filename, "r")
  lines = readlines(f)
  close(f)

  f = open(filename, "w")
  indices = randperm(length(lines))
  for index = indices
    print(f, lines[index])
  end
  close(f)
end
