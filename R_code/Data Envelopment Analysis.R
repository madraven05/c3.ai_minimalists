install.packages("Benchmarking")
library(Benchmarking)

health_data = read.csv("Data_input/health_input.csv")
class(health_data)
str(health_data)

#input Data
x<-with(health_data,cbind(health_data$IB_S,health_data$AB_S))
#output Data
y<-matrix(health_data$TDC_S,health_data$TC_S)

#ccr model of DEA analysis
ccr<-dea(x,y, RTS="crs",ORIENTATION = "in")
ccr
summary(ccr)
shapiro.test(ccr$eff)
eff(ccr)
data.frame(ccr$eff)
dataframe1<-data.frame(health_data$NAME,ccr$eff)
dataframe1
