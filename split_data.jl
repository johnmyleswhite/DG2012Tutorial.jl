# According to the UCI Archive:
# The training set is the first 463,715 rows
# The test set is the last 51,630 rows

# We'll split the data into three parts:

# Training: Rows 1-347,786
# Validation: Rows 347,787 - 463,715
# Test: 463,716 - 515,345

# This is not done randomly.
# A better implementation would split the training and validation rows at random.

total_bound = 515_345
training_bound = 347_786
validation_bound = 463_715
test_bound = 515_345

input_file = open("data/YearPredictionMSD.txt", "r")
training_file = open("data/training.csv", "w")
validation_file = open("data/validation.csv", "w")
test_file = open("data/test.csv", "w")

for i = 1:total_bound
  if 1 <= i <= training_bound
    print(training_file, readline(input_file))
  elseif training_bound < i <= validation_bound
    print(validation_file, readline(input_file))
  elseif validation_bound < i <= test_bound
    print(test_file, readline(input_file))
  else
    error("Index went outside of bounds on data size")
  end
end

close(training_file)
close(validation_file)
close(test_file)
