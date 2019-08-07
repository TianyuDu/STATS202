# Aug. 7 2019
# Determine the optimal value of clusters
library(factoextra)
library(NbClust)
setwd("/Users/tianyudu/Documents/UToronto/2019 Summer Exchange/Stanford Summer Session/STATS202/STATS202/cluster/")

df <-  read.csv("./zero_day_standardized.csv")

fviz_nbclust(df, kmeans, method = "wss") +
    geom_vline(xintercept = 4, linetype = 2)+
    labs(subtitle = "Elbow method")

# Silhouette method
fviz_nbclust(df, kmeans, method = "silhouette")+
    labs(subtitle = "Silhouette method")

set.seed(123)
fviz_nbclust(df, kmeans, nstart = 25,  method = "gap_stat", nboot = 50)+
    labs(subtitle = "Gap statistic method")

NbClust(data=df, diss=NULL, distance="euclidean", min.nc=2, max.nc=7, method="ward.D")

