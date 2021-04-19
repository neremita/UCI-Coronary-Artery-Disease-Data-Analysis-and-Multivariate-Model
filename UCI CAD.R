'''
UCI Coronary Artery Disease Data Analysis and Multivariate Model


Introduction:
In this project, I have focused on creating a multivariate machine learning 
model K-Nearest Neighbors model based off of UCI Heart Disease Data Set in the 
Machine Learning Repository. The dataset was donated in July 1988 and has been 
used for multiple testing projects regarding data analysis. Locations for data 
collection include Cleveland, Hungary, Switzerland, and the VA Long Beach. Though
many use only the Cleveland dataset for their projects, I intend to both perform
an analysis and machine learning model on all four sets. Here is to a good first
project!

For starters, I am utilizing the processed datasets, given the simplicity of the
formatting and relevant data.

'''

library(dplyr)
library(stringr)
library(tidyr)
library(tidyverse)
library(broom)
library(readxl)
library(ggplot2)
library(readr)
library(caret)
library(ggpubr)

#Initial Upload and Combining of Datasets

cleveland_df <- read_csv("~/Google Drive/Work/Med/Medical Machine Learning/UCI CAD/processed.cleveland.data",
                         col_names = c('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                                       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'),
                         na = c('?', NA))
switz_df <- read_csv("~/Google Drive/Work/Med/Medical Machine Learning/UCI CAD/processed.switzerland.data",
                     col_names = c('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                                   'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'),
                     na = c('?', NA))
hungary_df <- read_csv("~/Google Drive/Work/Med/Medical Machine Learning/UCI CAD/processed.hungarian.data",
                       col_names = c('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                                     'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'),
                       na = c('?', NA))
va_df <- read_csv("~/Google Drive/Work/Med/Medical Machine Learning/UCI CAD/processed.va.data",
                  col_names = c('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'),
                  na = c('?', NA))


cad_df <- bind_rows(cleveland_df, switz_df, hungary_df, va_df)

'''
Upon further inspection of the datasets, we see that there are a few amount of 
incomplete entries depending on the region. Along with this, the info given
is hard to use without the given .names file provided. To give a brief summary,
I have provided the relevant info from that file:

sex: sex (1 = male; 0 = female)
cp: chest pain type
        -- Value 1: typical angina
        -- Value 2: atypical angina
        -- Value 3: non-anginal pain
        -- Value 4: asymptomatic
trestbps: resting blood pressure (in mm Hg on admission to the hospital)
chol: serum cholestoral in mg/dl
fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
restecg: resting electrocardiographic results
        -- Value 0: normal
        -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST 
                    elevation or depression of > 0.05 mV)
        -- Value 2: showing probable or definite left ventricular hypertrophy
                    by Estes criteria
thalach: maximum heart rate achieved
exang: exercise induced angina (1 = yes; 0 = no)
oldpeak = ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
        -- Value 1: upsloping
        -- Value 2: flat
        -- Value 3: downsloping
ca: number of major vessels (0-3) colored by flourosopy
thal: thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
num: diagnosis of heart disease (angiographic disease status)
        -- Value 0: < 50% diameter narrowing
        -- Value 1: > 50% diameter narrowing
        (in any major vessel: attributes 59 through 68 are vessels)
'''

#Data Cleaning

'''
Let us consider the missing data first.
'''
na <- c()
i = 0
for (col in 1:ncol(cad_df)) {
   for (row in 1:nrow(cad_df)) {
      if (is.na(cad_df[row, col])) {
        i = i + 1
      } else {}
   }
  na <- c(na, i)
  i = 0
}

na_df <- data.frame(Category = c('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                          'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'), number_NA = na)

na <- ggplot(data = na_df) +
  aes(x = Category, y = number_NA, fill = number_NA) +
  geom_bar(stat = "identity")

'''
age: 0
sex: 0
cp: 0
trestbps: 59
chol: 30
fbs: 90
restecg: 2
thalach: 55
exang: 55
oldpeak: 62
slope: 309
ca: 611
thal: 486
num: 0

We see from the NA table and bar graph created that majority of our data would 
be lost if we consider selecting the `ca`, `slope`, and `thal` columns.
Let us wait to see what goes on with the model and see how the relationships 
come around.

To begin cleaning, we see that there are different levels of diagnosed CAD,
resulting in info that may make the model too complex to create. Instead, for
any value greater than or equal to 1, we will change them to 1 since that is the
minimum value for diagnosis.

'''

for (row in 1:nrow(cad_df)) {
  if (cad_df$num[row] > 1) {
    cad_df$num[row] = 1
  }
}

i = 0
j = 0
for (row in 1:nrow(cad_df)) {
    if (cad_df$num[row] == 1) {
      i = i + 1
    } else {
      j = j + 1
    }
}
one_or_zero <- c(i, j)
num_df <- data.frame(have_cad = c('yes', 'no'), num = one_or_zero)

'''
yes: 509, no: 411
'''

#Data Exploration

'''
Let us have a look at the data trends to see if there are any variables we are
potentially interested in.
'''

# for (row in 1:nrow(cad_df)) {
#   if (cad_df$num[row] == 1) {
#     cad_df$num[row] <- 'yes'
#   } else {
#     cad_df$num[row] <- 'no'
#   }
# }

cad_df <- cad_df %>%
  mutate(cad = ifelse(cad_df$num == 1, 'Yes', 'No')) %>%
  mutate(restecg2 = ifelse(cad_df$restecg == 0, 'Normal', 
                           ifelse(cad_df$restecg == 1, 'ST-T wave abnormality', 
                                  ifelse(cad_df$restecg == 2, 'Probable or Definite LVH', NA)))) %>%
  mutate(cp2 = ifelse(cad_df$cp == 1, 'Typical Angina', 
                           ifelse(cad_df$cp == 2, 'Atypical Angina', 
                                  ifelse(cad_df$cp == 3, 'Non-Anginal Pain', 
                                         ifelse(cad_df$cp == 4, 'Asymptomatic', NA))))) %>%
  mutate(slope2 = ifelse(cad_df$slope == 1, 'Upsloping', 
                           ifelse(cad_df$slope == 2, 'Flat', 
                                  ifelse(cad_df$slope == 3, 'Downsloping', NA)))) %>%
  mutate(fbs2 = ifelse(cad_df$fbs == 1, '> 120 mg/dl', '<= 120 mg/dl'))

write.csv(cad_df,"~/Google Drive/Work/Med/Medical Machine Learning/UCI CAD/processed_cad.csv", row.names = TRUE)

'''
Quantitative Variables
'''

cp <- ggplot(cad_df, aes(x = cp2, fill = cad)) +
  geom_bar(stat = 'count') +
  labs(title = 'Dx of CAD Based on Chest Pain from UCI Dataset',
       x = 'Chest Pain Rating',
       y = 'Count',
       fill = "Dx w/ CAD") +
  theme(axis.text.x = element_text (angle = 90))

restecg <- ggplot(cad_df, aes(x = restecg2, fill = cad)) +
  geom_bar(stat = 'count') +
  labs(title = 'Dx of CAD Based on Resting ECG Results from UCI Dataset',
       x = 'Resting ECG Rating',
       y = 'Count',
       fill = "Dx w/ CAD") +
  theme(axis.text.x = element_text (angle = 90))

slope <- ggplot(cad_df, aes(x = slope2, fill = cad)) +
  geom_bar(stat = 'count') +
  labs(title = 'Dx of CAD Based on Slope at Peak-Exercise ST from UCI Dataset',
       x = 'Slope Quality',
       y = 'Count',
       fill = "Dx w/ CAD") +
  theme(axis.text.x = element_text (angle = 90))

fbs <- ggplot(cad_df, aes(x = fbs2, fill = cad)) +
  geom_bar(stat = 'count') +
  labs(title = 'Dx of CAD Based on Fasting Blood Glucose from UCI Dataset',
       x = 'Fasting Blood Glucose',
       y = 'Count',
       fill = "Dx w/ CAD") +
  theme(axis.text.x = element_text (angle = 90))

'''
Quantitative Variables
'''

chol <- ggplot(cad_df, aes(x = chol, fill = cad)) +
  geom_histogram(bins = 30) +
  labs(title = 'Dx of CAD Based on Cholesterol from UCI Dataset',
       x = 'Cholestrol (mg/dl)',
       y = 'Count',
       fill = "Dx w/ CAD")

age <- ggplot(cad_df, aes(x = age, fill = cad)) +
  geom_histogram(bins = 30) +
  labs(title = 'Dx of CAD Based on Age from UCI Dataset',
       x = 'Age (yrs)',
       y = 'Count',
       fill = "Dx w/ CAD")

figure_qual <- ggarrange(cp, restecg, slope, fbs,
                    ncol = 2,
                    nrow = 2,
                    common.legend = TRUE, legend = 'right')

figure_quant <- ggarrange(chol, age,
                         ncol = 2,
                         nrow = 2,
                         common.legend = TRUE, legend = 'right')

figure <- ggarrange(figure_quant, figure_qual,
                    nrow = 2,
                    common.legend = TRUE, legend = 'right')
# figure
# figure_quant
figure_qual
# na

i = 0
j = 0
for (row in 1:nrow(cad_df)) {
  if (is.na(cad_df$fbs[row])) {
    
  } else if (cad_df$fbs[row] == 0) {
    if (cad_df$num[row] == 1) {
      i = i + 1
    } else {
      j = j + 1
    }
  } else {
    
  }
}
fbs_count <- c(i, j)
fbs_df <- data.frame(have_cad_and_low_bg = c('yes', 'no'), count = fbs_count)

count(cad_df, '<= 120 mg/dl')
count(cad_df, '> 120 mg/dl')

'''
We can see that all of the asymptomatic patient (without chest pain or 4) had CAD, as
compared to, say atypical or typical anginas (2 and 1 respectively)
'''

#Model Training

'''
Now that the data is set up appropriately to get meaning data, it is now time to
train the the model. 
'''


set.seed(1)

train_indices <- createDataPartition(y = cad_df[["num"]],
                                     p = 0.7,
                                     list = FALSE)
train_cad <- cad_df[train_indices,]
test_cad <- cad_df[-train_indices,]
train_control <- trainControl(method = 'cv',
                              number = 9)

knn_model <- train(num ~ cp,
                   data = train_cad,
                   method = 'knn',
                   trControl = train_control,
                   preProcess = c("center", "scale"))

knn_model
