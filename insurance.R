library(tidyverse)
library(cowplot)
library(corrplot)
library(modelr)

df <- read.csv(file = '/Users/adit0106/Desktop/Boston University/SEM 1/MET CS 555 Foundations of Machine Learning/Final Project/insurance.csv')
head(df)
attach(df)
summary(df)

#Checking for outliers
par(mfrow=c(1,3))
boxplot(age, col='steelblue', xlab='Age')
boxplot(bmi, col='steelblue', xlab='BMI')
boxplot(charges, col='steelblue', xlab='Insurance Premium')

#treat Outliers
library('ExPanDaR')
df$bmi <- treat_outliers(bmi, 0.05)
df$charges <- treat_outliers(charges, 0.05)

attach(df)

par(mfrow=c(1,3))
boxplot(age, col='steelblue', xlab='Age')
boxplot(bmi, col='steelblue', xlab='BMI (After treating Outliers)')
boxplot(charges, col='steelblue', xlab='Insurance Premium (After treating Outliers)')


#Continuous - age, bmi, charges.
#Discrete - sex, children, smoker, region.

plot_scatter <- function(data, x){
  ggplot(data, aes_string(x = x , y = data$charges))+
    geom_point()
}

plot_box <- function(data, x){
  ggplot(data, aes_string(x = x , y = data$charges, fill = x, group = x))+
    geom_boxplot()
}

ggplot(df) + geom_bar(aes(smoker)) + theme_bw() + ggtitle("SMOKER VS NON-SMOKERS")
ggplot(df) + geom_bar(aes(sex)) + theme_bw() + ggtitle("FEMALES VS MALES")
ggplot(df) + geom_bar(aes(children)) + theme_bw() + ggtitle("CHILDREN DISTRIBUTION")
ggplot(df) + geom_bar(aes(region)) + theme_bw() + ggtitle("COUNT BASED ON REGION")

##Continuous variables in relation to the insurance premium
plots.scatterplot <- lapply(colnames(df[,c(1,3)]),plot_scatter, data = df)
plot_grid(plotlist = plots.scatterplot)

df_cor <- cor(df[,c(1,3,7)])
df_cor
par(mfrow=c(1,1))
corrplot(df_cor, method="number")

#We can see that age does play a role in determining the insurance premium. 
#The bmi does also a positive relationship, however it is less clear given that many of the data point are near zero.

#Discrete variables in relation to the insurance premium
par(mfrow=c(1,1))
boxpl <- lapply(colnames(df[,c(-1,-3, -7)]),plot_box, data = df)
plot_grid(plotlist = boxpl)
title(main = "Boxplots for discrete variables to insurance premium ")

#We clearly see that people who smoker are charged with a higher insurance premium. 
#Like wise, we see a little increase in insurance premium when the number of children increases. 
#But,decreases when there are 3 and more children. 
#The sex and region variables seem to show no effect on the insurance premium.  

one_hot_encoding = function(df, columns="xyz"){
  # create a copy of the original data.frame for not modifying the original
  df = cbind(df)
  # convert the columns to vector in case it is a string
  columns = c(columns)
  # for each variable perform the One hot encoding
  for (column in columns){
    unique_values = sort(unique(df[column])[,column])
    non_reference_values  = unique_values[c(-1)] # the first element is going 
    # to be the reference by default
    for (value in non_reference_values){
      # the new dummy column name
      new_col_name = paste0(column,'.',value)
      # create new dummy column for each value of the non_reference_values
      df[new_col_name] <- with(df, ifelse(df[,column] == value, 1, 0))
    }
    # delete the one hot encoded column
    df[column] = NULL
    
  }
  return(df)
}

df.modeling <- one_hot_encoding(df, columns = c("sex","smoker","children","region"))
head(df.modeling)
attach(df.modeling)
# Shifting charges column to the end
charges <- df.modeling$charges
df.modeling$charges <- NULL
df.modeling$charges <- charges
head(df.modeling)

#Checking for null values
df.modeling %>% is.na() %>% sum()

#We will consider female, nonsmoker, southwest, and no children as our reference for the dummy encoding.

#Data Splitting
#use 70% of dataset as training set and 30% as test set
library(caTools)
set.seed(123)
split = sample.split(df.modeling$charges, SplitRatio = 0.7)
df.train = subset(df.modeling, split == TRUE)
df.test = subset(df.modeling, split == FALSE)

#Multiple Linear Regression

model1 <- lm(charges ~ ., data = df.train)
summary(model1)
print(model1)
#As expected, the sex and region variable shows no significance to the model. 
#Even though the child1 and child5 shows no significance, we will keep it in the model given that majority of the variables show sigificance.

model2 <- lm(charges ~ . -sex.male -region.southeast -region.northwest -region.southwest, data = df.train)
summary(model2)

#SIMPLE LINEAR REGRESSION ON SIGNIFICANT VARIABLES

par(mfrow=c(2,1))
slr1 <- lm(charges ~ age, data = df.train)
summary(slr1)
plot(age,charges)
abline(slr1)

slr2 <- lm(charges ~ bmi, data = df.train)
summary(slr2)
plot(bmi,charges)
abline(slr2)

#After removing the sex and region variable, the adjusted R2 is nearly equal to first model. This further shows these variable are useless when predicting the insurance premium.

#Prediction
df.pred <- df.test%>%
  add_predictions(model2)%>%
  mutate(
    residual = charges - pred,
    residual_sq = (residual)^2
  )
head(df.pred, 10)

cat("The RMSE for the test data is",sqrt(sum(df.pred$residual_sq)/nrow(df.pred)))

# Predicting the Test set results
y_pred = predict(model2, newdata = df.test)
MSE <- mean((y_pred - df.test$charges)^2)
totalss <- sum((df.test$charges - mean(df.test$charges))^2)

# Regression and Residual Sum of the Squered.
regss <- sum((y_pred - mean(df.test$charges))^2)
resiss <- sum((df.test$charges - y_pred)^2)

# Calculate R squared.
R2 <- regss/totalss
cat("The R2 value for MODEL 2 is:", R2)

####### HYPOTHESIS TESTING #######

### T-TEST ###
#calculating the p-value
library(MASS)                 
### T-TEST HYPOTHESIS TESTING ###
# H0: Mean insurance premium = 13011.48
# H1: Mean insurance premium ≠ 13011.48
# Two-tailed sample test 
#Since n <= 30, so we take the t-distribution
t <- df[sample(nrow(df), size=25), ]
alpha = 0.05
u =  mean(df$charges, na.rm=TRUE)
n = length(t$charges)
xbar = mean(t$charges)
s = sd(t$charges)
t_val = abs((xbar - u)/(s/sqrt(n)))
p_val = 2 * pt(t_val, df = (n - 1), lower.tail = FALSE)

cat('The t-value for the distribution is ', t_val)
cat('The p-value for the distribution is ', p_val)
if(p_val <= alpha){
  print('Reject the NULL hypothesis')
} else{
  print('Fail to reject NULL hypothesis')
}
ci = 0.9
alpha = 1 - ci
aplha = alpha/2
mu = mean(df$charges, na.rm=TRUE) 
n = length(t$charges)
xbar = mean(t$charges)
s = sd(t$charges)
t_val = qt(0.95,df=(n-1))

lower <- xbar-(t_val*(s/sqrt(n)))
upper <- xbar+(t_val*(s/sqrt(n)))
cat('The lower bound of confidence interval is ', lower)
cat('The upper bound of confidence interval is ', upper)

### Z-TEST ###
z <- df[sample(nrow(df), size=100), ]
alpha = 0.05
u = mean(df$charges, na.rm=TRUE)
n = nrow(z)
xbar = mean(z$charges)
s = sd(z$charges)
z_val = abs((xbar - u)/(s/sqrt(n)))
p_val = 2 * pnorm(z_val, lower.tail = FALSE)

cat('The z-value for the distribution is ', z_val)
cat('The p-value for the distribution is ', p_val)
if(p_val <= alpha){
  print('Reject the NULL hypothesis')
} else{
  print('Fail to reject NULL hypothesis')
}

z_val <- qnorm(0.975)
lower <- xbar-(z_val*(s/sqrt(n)))
upper <- xbar+(z_val*(s/sqrt(n)))
cat('The lower bound of confidence interval is ', lower)
cat('The upper bound of confidence interval is ', upper)
