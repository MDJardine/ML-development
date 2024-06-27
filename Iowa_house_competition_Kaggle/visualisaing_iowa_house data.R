#################### data exploration for kaggle competition
library(corrplot)
library(Hmisc)
library("PerformanceAnalytics")
library(dplyr)
library(ggplot2) 
library(GGally) 


tempdir()
# [1] "C:\Users\XYZ~1\AppData\Local\Temp\Rtmp86bEoJ\Rtxt32dcef24de2"
dir.create(tempdir())

### read in the data 

train_data <- read.csv("C:/Users/user/Documents/GitHub/ML-development/Iowa_house_competition_Kaggle/train.csv")
head(train_data)
### check data
summary(train_data)
str(train_data)
## which ones are continuous variables?
## c[1,2,4,5]

## find all numeric ones
just_num <- train_data %>% dplyr::select(where(is.numeric))
## summaries
summary(just_num)
str(just_num)
head(just_num)


## write a loop that does a simple linear model for each variable 
## then extracts those there the p value is less than 0.0001 and R2 is > 10%
p_values <- vector()
imp_values <- vector()
for(i in 1:ncol(just_num[-1])){
  model <- lm(just_num$SalePrice ~ just_num[,i], data=just_num)
  p <- coef(summary(model))[2,4]
  if(p<0.001 && summary(model)$r.squared > 0.1){
    name <- colnames(just_num[i])
    imp_values <- append(imp_values, name)
    p_values <- c(p_values, p)
  }
}
imp_nums <- cbind.data.frame(imp_values, p_values)
imp_nums

### not sure why but are there issues in repeating this with the non-numeric variables?
## find all the categories
not_num <- train_data %>% dplyr::select(where(is.character))
## add sale price
SalePrice <- train_data$SalePrice
not_num <- cbind(not_num, SalePrice)
head(not_num)
## run loop to pull out all variables than have p value <0.001 and R2 >10% for sale price
p_values_chr <- vector()
imp_values_chr <- vector()
for(i in 1:ncol(not_num[-1])){
  model <- lm(not_num$SalePrice ~ not_num[,i], data=not_num)
  p <- coef(summary(model))[2,4]
  if(p<0.001 && summary(model)$r.squared > 0.1){
    name <- colnames(not_num[i])
    imp_values_chr <- append(imp_values_chr, name)
    p_values_chr <- c(p_values_chr, p)
  }
}
imp_chr <- cbind.data.frame(imp_values_chr, p_values_chr)
imp_chr

# save this to use in python script
write.csv(imp_chr, "C:/Users/user/Documents/GitHub/ML-development/Iowa_house_competition_Kaggle/important_characters.csv", row.names = F)
## so nine factors have a low p-value and and R2 of >0.1
## so we'll use these 9

############################################
##### pairwise correlations of important variables ####

## select just those columns which predict price 
var_of_interest <- just_num %>% dplyr::select(any_of(imp_values))

## correlation matrix with significance
var_of_interest_NA <- na.omit(var_of_interest)
## remove NAs
summary(var_of_interest_NA)
str(var_of_interest_NA)
res2 <- rcorr(as.matrix(var_of_interest_NA))


#####
# Insignificant correlations are left blank
corrplot(res2$r, type="upper", order="hclust", 
         p.mat = res2$P, sig.level = 0.05, insig = "blank")
## some error here  -not sure what its affecting - still creates plot

###
#par(mfrow=c(1,1))
#chart.Correlation(var_of_interest_NA, histogram=TRUE, pch=19)
## this is actually difficult to see

## another approach with plotting
# create pairs plot 
ggpairs( var_of_interest_NA )
## after a few iterations I'm removing a few that seen to correlate very well
final_vars <- subset(var_of_interest_NA, select = -c(YearBuilt, GarageYrBlt, TotalBsmtSF, X1stFlrSF, X2ndFlrSF, BsmtFinSF1))
ggpairs(final_vars)

### so the 20 features I'm using are the 9 characters and the 11 numbers that don't correlate so much
colnames(final_vars)
features = c(colnames(final_vars), imp_chr$imp_values_chr)
features
