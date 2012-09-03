results <- read.csv("logs/lm.tsv", header = FALSE, sep = "\t")

names(results) <- c("Cost", "Intercept", paste("Covariate", 1:90, sep = ""))

results <- transform(results, Iteration = 1:nrow(results))

ggplot(results, aes(x = Iteration, y = Cost)) + geom_smooth()
ggplot(results, aes(x = Iteration, y = Intercept)) + geom_smooth()

ggplot(results, aes(x = Iteration, y = Covariate1)) + geom_smooth()
ggplot(results, aes(x = Iteration, y = Covariate2)) + geom_smooth()
#...
ggplot(results, aes(x = Iteration, y = Covariate89)) + geom_smooth()
ggplot(results, aes(x = Iteration, y = Covariate90)) + geom_smooth()
