results <- read.csv("results/minibatch.tsv",
                    header = FALSE,
                    sep = "\t")
names(results) <- c("Epochs",
                    "LearningRate",
                    "MinibatchSize",
                    "RMSE")

ggplot(results,
       aes(x = MinibatchSize,
           y = RMSE,
           color = LearningRate,
           group = LearningRate)) +
  geom_line() +
  facet_grid(Epochs ~ LearningRate)

ggplot(subset(results, LearningRate != 0.0001),
       aes(x = MinibatchSize,
           y = RMSE,
           color = LearningRate,
           group = LearningRate)) +
  geom_line() +
  facet_grid(Epochs ~ LearningRate)

summary(lm(RMSE ~ LearningRate + MinibatchSize + Epochs, data = results))

ggplot(results, aes(x = LearningRate, y = RMSE)) + geom_point()
