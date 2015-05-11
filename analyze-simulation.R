# Reads simulation.txt and plots the results using ggplot

library(ggplot2)

d <- read.table("simulation.txt")
names(d) <- c("replication", "ndata", "type", "type.name", "model.p")

plt <- ggplot(d, aes(x=ndata, y=model.p)) + 
    stat_summary(fun.y=mean, geom="line") + 
    facet_wrap(~type.name)

plt # show it
