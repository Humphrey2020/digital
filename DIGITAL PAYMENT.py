#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libaries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE


# INSPECTING THE DATA  AND ALSO CHECK THE FIRST 5 DATA

# In[2]:



df=pd.read_excel(r'C:\Users\Humphery\Desktop\digital_loan_data.xlsx')
df.head()


# print the size of the data

# In[3]:


print('the length of the data:',df.shape)


# Rename the name of the feature

# In[4]:


df.rename(columns={'Timestamp':'Time','1. Please indicate your age range:':'age','2. What is your gender?':'gend',
            '3. How would you rate your level of education?':'loe' ,   '4. What is your current employment status?':'ces',
            '5. Do you use digital payment platforms such as OPay, Palmpay, Paga etc. for payment transactions?':'dpp' ,
    '6. How likely do you expect that utilising digital payment methods (e.g., mobile payments, online transfers) would be more convenient than using traditional payment methods (e.g., cash, cheques)?':'conv',
    '7. How confident are you that digital payment methods, when compared to traditional payment methods, will give you more control over your financial transactions?':'contft',
       '8. How likely do you believe that adopting digital payment methods will give you with more security and fraud protection than traditional payment methods?':' secu','9. How simple do you think it is to understand and use digital payment methods compared to traditional payment methods?':'unds',
'10. How likely do you believe adopting digital payment methods will necessitate significant adjustments in your current financial routines and habits compared to traditional payment methods? ':'habi','11. How much influence do recommendations and pleasant experiences from friends and family have on your decision to choose digital payment methods compared to traditional payment methods?  ':'frnfa',
        
      '12. How satisfied are you with the availability and reliability of digital payment infrastructure and platforms compared to traditional payment methods?':'infr' ,
      
    '13. How probable do you believe it is that having the appropriate financial and technical support services in place will facilitate your use of digital payment methods rather than traditional payment methods?':'tecsu', '14. Are you willing to switch from traditional payment methods to digital payment platforms in the near future?':'switdp',
    
    
     '15. Do you use digital lending platforms such as OKash, Palmcredit, Branch etc. to secure loan for your financial needs?':' dlp',
                   
                   
    '16. How likely do you believe that using digital lending platforms (e.g., peer-to-peer lending, online loan applications) would offer you faster access to credit than traditional lending institutions (e.g., banks, microfinance institutions)?':'fasac',      

    '17. How likely do you believe that using digital lending platforms (e.g., peer-to-peer lending, online loan applications) would offer you faster access to credit than traditional lending institutions (e.g., banks, microfinance institutions)?' :'fascr', ' 18. How much do you believe digital lending platforms demand less paperwork and documentation than traditional lending institutions?':'ppwrk',
    
    '19. How comfortable and secure are you with sharing personal and financial information through digital platforms for loan applications compared to dealing with traditional lending institutions?':'shapi','20. How much influence do recommendations and favourable experiences from friends and family have on your decision to use digital lending platforms rather than traditional lending institutions? ':'favex',
     '21. How likely are you to use online lending platforms if you believe a sizable portion of the people you know is currently using them instead of traditional lending institutions? ':'sizpp',
      '22. To what extent do you believe the regulatory framework in Nigeria promotes and encourages the usage of online lending platforms in comparison to traditional lending institutions? ':'regfw' ,
                    ' 23.Are you willing to switch from traditional lending methods to digital lending platforms in the near future?':'switdl',
                   
     
                   '24. How accessible are fintech services to you in terms of technology requirements (e.g., smartphone, internet connectivity, cost of data)?':'smicd',
                   
                   '25. Would you consider switching to using more digital financial platform (e.g., cryptocurrencies, robo adviser) for financial transactions instead of traditional financial platform?\n':'mordf'
                   
                  },inplace=True)


# In[5]:


df.head()


# checking for missing values

# In[6]:


df.isnull().sum()


# Drop missing values

# In[7]:



df.dropna(inplace=True)


# Inspecting the data for DIGITAL PAYMENT DATA

# In[8]:


digital_payment=df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,25]]
digital_payment


#  check if digital payment as a missing value

# In[9]:


digital_payment.isnull().sum()


# In[10]:


digital_payment.age


# describe the digital payment data

# LABEL ENCODER FOR GENDER,level of education,current employment status,(CATIGORICAL VARIABLES)
# USING THE AVERAGE TO REPLACE THE AGE

# In[11]:


digital_payment['age']=digital_payment['age'].replace({'36-45':41, '26-35':31,'18-25':22,'46-55':51,'36?45':41,'56-65':61, 'Prefer not to say':18,'Option 2':31, '36â€\x9045':41})
encoder=LabelEncoder()
digital_payment.gend=encoder.fit_transform(digital_payment.gend)
digital_payment.loe=encoder.fit_transform(digital_payment.loe)
digital_payment.ces=encoder.fit_transform(digital_payment.ces)
digital_payment.dpp=encoder.fit_transform(digital_payment.dpp)
digital_payment.switdp=encoder.fit_transform(digital_payment.switdp)
digital_payment.head()


# Describe the Data

# In[12]:


digital_payment.describe()


# check for columns

# In[13]:


digital_payment.columns


# THE AVERAGE AGE OF THE SURVEY WAS AROUND 35 YEARS OLD
# this chart shows that the ages with time are going to extinction

# In[14]:


sns.kdeplot(digital_payment['age'],data=digital_payment,shade=True)
print('print the skewness of age:',digital_payment.age.skew())


# the count plot chart show that the number of ages that was consider the most is 30 years for the survey

# In[15]:


sns.countplot(x='age', data=digital_payment)
plt.title(' Age  Distribution on Digital Payment')
plt.show()


#  percentage of the gender that was surveyed  for digital payment

# the chart below shows that the number of males are more compare to the number of female that was survyed

# In[16]:


digital_payment.gend.value_counts()
x=[143,72,1]
label=['Male','Female','Prefer not to say']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('percentage of gender in digital payment ')
plt.show()  


# the chart below shows that the number of  "Bachelor's degree holders are more compare to other  of level of education that was survyed
# 

# In[17]:


digital_payment['loe'].value_counts()
x=[129,74,9,4]
label=["Bachelor's degree","Master's degree",'Secondary school','PhD or equivalent']
plt.pie(x,labels=label,autopct='%1.2f%%')
plt.title('the percentage of Level of Education in digital payment')
plt.show()   #the chart below shows that the number of  "Bachelor's degree holders are more compare to the number of level of education that was survyed


# In[18]:


df.conv.value_counts()
x=[125,54,17,15,5]
label=["Strongly Agree",'Agree','Strongly Disagree','Neutral','Disagree']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('the percentage of people that want  the utilisation of digital payment to be used .')
plt.show() 


# In[19]:


sns.countplot(x='dpp', data=digital_payment)
plt.title('Use Digital Payment Platforms')
plt.xlabel('the number of people who use digital payment')
plt.show()
#the number of people who prefer  to Use Digital Payment Platforms are  higher than the people who do not expect it.


#  THIS PIE CHART SHOWS THAT THE PEOPLE THAT ARE willingness to switch to digital payment platforms is more
#  than the number of people that are not willing to switch  (IMBALANCED DATA)

# In[20]:


#percentage  willingness to switch to digital payment platforms
digital_payment.switdp.value_counts()
x=[200,16]
label=['people that will like to switch to digital payment','people that will not  like to switch to digital payment']
plt.pie(x,labels=label,autopct='%1.1f%%')
plt.title('percentage willingness to switch to digital payment platforms')
plt.show()   #the chart below shows that the number of males are more compare to the number of female that was survyed


# In[21]:


digital_payment.frnfa.value_counts()
x=[76,78,19,33,10]
label=["Strongly Agree",'Agree','Strongly Disagree','Neutral','Disagree']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('recommendations and pleasant experiences from friends and family to use digital payment')
plt.show()
 


# In[22]:


digital_payment.habi.value_counts()
x=[66,73,20,45,12]
label=["Strongly Agree",'Agree','Strongly Disagree','Neutral','Disagree']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('Percentage of people who believe adopting digital payment methods will necessitate significant adjustments in your current financial routines and habits compared to traditional payment methods')
plt.show()
 


# RESEARCH HYPOTHESES

# IT IS CLEARLY SHOWN IN THE AGE DISTRIBUTION ABOVE THAT, AS  AGE INCREASES(>60YEARS), THE CONFIDENT LEVEL OF OLDER 
# PEOPLE IN USING DIGITAL PAYMENT  methods, when compared to traditional payment methods, will give you more control over your financial transactions also Increases.(tecsu)

# In[23]:


digital_payment.corrwith(digital_payment['age']).sort_values(ascending=False)


# there is a positive relationship between conv and unds

# In[24]:


digital_payment.corrwith(digital_payment['conv']).sort_values(ascending=False)


# there is a positive relation between secu  , smicd and conv

# In[25]:


digital_payment.corrwith(digital_payment[' secu']).sort_values(ascending=False)


# There is a positive relationship between frnfa  and habi

# In[26]:


digital_payment.corrwith(digital_payment['frnfa']).sort_values(ascending=False)


# THE PROBABILTY VALUE BELOW SHOWS THE IMPORTANCE OF switching from traditional payment methods to digital payment platforms in the near future by choosing the feature that are likely to predict it.

# In[27]:


# Features Selections on digital payment platforms such as OPay, Palmpay, Paga etc. for payment transactions.
x_digital_payment = digital_payment.drop(['switdp'], axis=1)
y_digital_payment = digital_payment['switdp']
digital_pay_selection=SelectKBest(score_func=chi2,k=3).fit(x_digital_payment,y_digital_payment)
p_value=digital_pay_selection.pvalues_
p_value
#Important featurs are:contft,secu, frnfa, infr,tecsu,smicd


# In[28]:


names=digital_pay_selection.feature_names_in_
names


# In[29]:


for p , name in zip(p_value,names):
    print(name,p)


# In[30]:


#Important featurs are:contft,secu, infr,tecsu,smicd


# In[31]:


df.contft.value_counts()
x=[101,55,22,27,11]
label=["Strongly Agree",'Agree','Strongly Disagree','Neutral','Disagree']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('confident level of the use of digital payment method')
plt.show()


# In[32]:


digital_payment[' secu'].value_counts()
x=[52,42,23,70,29]
label=["Strongly Agree",'Agree','Strongly Disagree','Neutral','Disagree']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('confident level of adopting digital payment methods will give people with more security and fraud protection than traditional payment')
plt.show()


# In[33]:


digital_payment['infr'].value_counts()

digital_payment.infr.value_counts()
x=[84,70,14,37,11]
label=["Strongly Agree",'Agree','Strongly Disagree','Neutral','Disagree']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('percentage of people that are satisfied  with the availability and reliability of digital payment infrastructure')
plt.show()


# In[34]:


digital_payment.tecsu.value_counts()
x=[87,75,13,32,9]
label=["Strongly Agree",'Agree','Strongly Disagree','Neutral','Disagree']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('financial and technical support services in place will facilitate the use of digital payment methods rather than traditional payment methods')
plt.show()


# In[35]:


digital_payment.smicd.value_counts()
x=[99,60,22,29,6]
label=["Strongly Agree",'Agree','Strongly Disagree','Neutral','Disagree']
plt.pie(x,labels=label,autopct='%1.0f%%')
plt.title('accessible are fintech services to you in terms of technology requirements"(how fast it is)" ')
plt.show()


# MODEL BUILDING UPON THE TRAINING DATA

# In[36]:


x_payment=digital_payment[['contft',' secu','infr','tecsu','smicd' ]]
y_payment=digital_payment['switdp']
y_payment.unique()


# COUNT THE Y_PAYMENT(NUMBER  willingness to switch to digital payment platforms)

# In[37]:


y_payment.value_counts()


# In[38]:


#Check if the digital payment features as outliers
x_payment.describe()


# In[39]:


for i in x_payment:
    plt.figure(figsize=(3,4))
    sns.kdeplot(x_payment[i],shade=True)
#THE density graph below shows that lot of people are willing to switch to digital payment platfrom


# HISTOGRAPH OF PEOPLE WHO CHOSE TO SWITICH FROM TRADITIONAL TO DIGITAL MODE OF TRANSACTIONS

# In[40]:


digital_payment['cat_switdp']=pd.cut(digital_payment['switdp'],bins=2,labels=[0,1])
digital_payment['cat_switdp'].hist() # we have more people who prefer using digital payment than tranditional


# In[41]:


#This create digital payment category in ther to cover all the data according to the original
sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in sss.split(digital_payment,digital_payment['cat_switdp']):
    strat_train=digital_payment.iloc[train_index]
    strat_test=digital_payment.iloc[test_index]
    


# Using The stratified method to check the percentage of people switching FROM TRADITIONAL TO DIGITAL MODE OF TRANSACTIONS

# In[42]:


strat_train['switdp'].value_counts()/len(strat_train)*100 # this clearly stratified equallly with the original data
#i.e it captures all the targets variables.


# USING The stratified

# In[43]:


x_trains=strat_train.drop(columns='switdp')
y_trains=strat_train['switdp']


# SELECT TRAINING MODELS
# 

# In[44]:


model1=LogisticRegression()
model1.fit(x_trains,y_trains)
predictions1=model1.predict(x_trains)


# CHECKING FOR THE ACTUAL VALUES AND THE PREDICTED VALUES FOR EACH MODELS
# USING The stratified method  UNPON THE TRAINING DATA

# In[45]:


diff1={'actual':y_trains,
      'prediction1':predictions1}
pd.DataFrame(diff1).head()


# In[46]:


model2=RandomForestClassifier()
model2.fit(x_trains,y_trains)
predictions2=model2.predict(x_trains)
diff2={'actual':y_trains,
      'prediction2':predictions2}
pd.DataFrame(diff2).head()


# In[47]:


model_train3=DecisionTreeClassifier(max_depth=2)
model_train3.fit(x_trains,y_trains)
predictions3=model_train3.predict(x_trains)
diff3={'actual':y_trains,
      'prediction3':predictions3}
pd.DataFrame(diff3).head()


# #CHECKING THE PERFORMANCE OF THE MODELS

# In[48]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv1=cross_val_score(model1,x_trains,y_trains,cv=3,scoring='accuracy')
print('the accuracy of the logistic model is:',cv1.mean())


# In[49]:


cv2=cross_val_score(model2,x_trains,y_trains,cv=3,scoring='accuracy')
print('the accuracy of the randomforest model is:',cv2.mean())


# In[50]:


cv3=cross_val_score(model_train3,x_trains,y_trains,cv=3,scoring='accuracy')
print('the accuracy of the decision model is:',cv3.mean())


# CHART OF THE DECISION TREE ON THE  STRATIFIED METHOD
# Since the gini is pure (zero gini) this classsifies properly that people who prefer to use di

# In[51]:


fig, ax=plt.subplots(figsize=(6,7))
tree.plot_tree(model_train3,fontsize=10)
plt.show()


# check the confusion matrix 
# USING The stratified method  UNPON THE TRAINING DATA

# In[52]:



conf1=confusion_matrix(y_trains,predictions1)
print(conf1)


# In[53]:


# check the confusion matrix
conf2=confusion_matrix(y_trains,predictions2)
print(conf2)


# In[54]:


# check the confusion matrix
conf3=confusion_matrix(y_trains,predictions3)
print(conf3)


# CLASSIFICATION REPORT ON STRATIFIED METHOD

# In[55]:


print(classification_report(y_trains,predictions1))


# In[56]:


print(classification_report(y_trains,predictions2))


# In[57]:


print(classification_report(y_trains,predictions3))


# Classification report shows that since the data is uneven , the the F1-score is a good way to measure how
# our model was able to predict well. 
# this clearly show that Randomforest classifer, Decision tree classifyer and the logistic regression model are a good model for prediction.
# with 1.00 confident that the prediction was correctly done with help of this 2 models.(ON STRATIFIED METHOD)

# In[58]:


sns.kdeplot(digital_payment['loe'],shade=True) # level of education density chart


# # Using Random sample  crossvalidation (train_test _split)

# In[59]:



x_train,x_test,y_train,y_test=train_test_split(x_payment,y_payment,test_size=0.2,random_state=42)


# In[60]:


y_train.value_counts()/len(y_train)*100  


# In[61]:


model1=LogisticRegression()
model1.fit(x_train,y_train)
prediction1=model1.predict(x_train)
diff1={'actual':y_train,
      'prediction1':prediction1}
pd.DataFrame(diff1).head()


# In[62]:


model2=RandomForestClassifier()
model2.fit(x_train,y_train)
prediction2=model2.predict(x_train)
diff2={'actual':y_train,
      'prediction2':prediction2}
pd.DataFrame(diff2).head()


# In[63]:


model3=DecisionTreeClassifier(max_depth=2)
model3.fit(x_train,y_train)
prediction3=model3.predict(x_train)
diff3={'actual':y_train,
      'prediction3':prediction3}
pd.DataFrame(diff3).head()


# CHART OF THE DECISION TREE ON THE  TRAIN-TEST SPLIT METHOD
# Since the gini is pure (zero gini)

# In[64]:


fig, ax=plt.subplots(figsize=(6,7))
tree.plot_tree(model3,fontsize=10)
plt.show()


# CHECKING THE PERFORMANCE OF THE MODEL USING RANDOM SAMPLE (TRAIN_TEST_SPLIT METHODE)

# In[65]:


print(y_train.shape)


# In[66]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv1=cross_val_score(model1,x_train,y_train,cv=5,scoring='accuracy')
print('the accuracy of the logistic model is:',cv1.mean())


# In[67]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv2=cross_val_score(model2,x_train,y_train,cv=5,scoring='accuracy')
print('the accuracy of the Randomforest model is:',cv2.mean())


# In[68]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv3=cross_val_score(model3,x_train,y_train,cv=5,scoring='accuracy')
print('the accuracy of the Decision model is:',cv3.mean())


# check the confusion matrix

# In[69]:



conf1=confusion_matrix(y_train,prediction1)
print(conf1)


# In[70]:


# check the confusion matrix
conf2=confusion_matrix(y_train,prediction2)
print(conf2)


# In[71]:


# check the confusion matrix
conf3=confusion_matrix(y_train,prediction3)
print(conf3)


# In[72]:


print(classification_report(y_train,prediction1))


# In[73]:


print(classification_report(y_train,prediction2))


# In[74]:


print(classification_report(y_train,prediction3))


# Classification report shows that since the data is uneven , the the F1-score is a good way to measure how
# our model was able to predict well. 
# this clearly show that Randomforest classifer  are a good model for prediction.
# with 0.99 confident that the prediction was correctly done with help of the models.

# TESTING THE MODELS
# 

# In[75]:


model2=RandomForestClassifier()
model2.fit(x_test,y_test)
prediction_test_2=model2.predict(x_test)


# In[76]:


model_test3=DecisionTreeClassifier()
model_test3.fit(x_test,y_test)
prediction_test_3=model_test3.predict(x_test)


# In[77]:


#CHECKING THE PERFORMANCE OF THE MODELS ON TEST  DATA
cv1=cross_val_score(model1,x_test,y_test,cv=5,scoring='accuracy')
print('the accuracy of the logistic model is:',cv1.mean())


# In[78]:


#CHECKING THE PERFORMANCE OF THE MODELS ON TEST  DATA
cv2=cross_val_score(model2,x_test,y_test,cv=5,scoring='accuracy')
print('the accuracy of the random forest model is:',cv2.mean())


# In[79]:


#CHECKING THE PERFORMANCE OF THE MODELS ON TEST  DATA
cv3=cross_val_score(model_test3,x_test,y_test,cv=5,scoring='accuracy')
print('the accuracy of the decision model is:',cv3.mean())


# # check the confusion matrix

# In[80]:


# check the confusion matrix
conf_test_2=confusion_matrix(y_test,prediction_test_2)
print(conf_test_2)


# In[81]:


# check the confusion matrix
conf_test_3=confusion_matrix(y_test,prediction_test_3)
print(conf_test_3)


# In[82]:


print(classification_report(y_test,prediction_test_2))


# In[83]:


print(classification_report(y_test,prediction_test_3))


# Create a FactorAnalysis object(FEATURE EXTRACTION)

# In[84]:


# Create a FactorAnalysis object
fa = FactorAnalysis(n_components=3,rotation='varimax')

# Fit the FactorAnalysis model to the data
transformed_features = fa.fit_transform(digital_payment)
transformed_features


# FACTOR 3 SHOWS A GOOD ANALYSIS OF THE FEATURES TO BE USED
# 

# In[85]:


# Access the factor loadings
factor_loadings = fa.components_

# Print the transformed features
#print(transformed_features)

# Print the factor loadings
print(factor_loadings*100)

# 


# In[86]:


digital_payment.columns


# In[87]:


new_x=digital_payment[['age','ces','conv', 'contft', ' secu', 'unds',
       'habi', 'frnfa', 'infr', 'tecsu', 'smicd','mordf']]
new_x
new_y=digital_payment['switdp']


# In[88]:


#Using Random sample  crossvalidation (train_test _split)
x_train_new,x_test_new,y_train_new,y_test_new=train_test_split(new_x,new_y,test_size=0.2,random_state=42)


# In[89]:


y_train_new.value_counts()/len(y_train_new)*100  


# In[90]:


model_new_1=LogisticRegression()
model_new_1.fit(x_train_new,y_train_new)
prediction_new_1=model_new_1.predict(x_train_new)


# In[91]:


model_new_2=RandomForestClassifier()
model_new_2.fit(x_train_new,y_train_new)
prediction_new_2=model_new_2.predict(x_train_new)


# In[92]:


# check the confusion matrix
conf1=confusion_matrix(y_train_new,prediction_new_1)
print(conf1)


# In[93]:


# check the confusion matrix
conf2=confusion_matrix(y_train_new,prediction_new_2)
print(conf2)


# In[94]:


print(classification_report(y_train_new,prediction_new_1))


# THIS IS THE BEST MODEL CLASSIFYER FOR THIS PROJECT
# BECAUSE THE F1 SCORE IS 1
# 

# In[95]:


print(classification_report(y_train_new,prediction_new_2)) 


# TEST THE MODEL FINETUNING

# In[96]:


model2=RandomForestClassifier()
model2.fit(x_test_new,y_test_new)
prediction_new_test=model2.predict(x_test_new)


# In[97]:


print(classification_report(y_test_new,prediction_new_test))


# EXAMINE THE DATA (IS ITS AN IMBALANCE DATA SET)
# SINCE WE HAVE LESS DATA, WE ARE GOING TO USE OVERSAMPLING

# In[98]:


x_train,x_test,y_train,y_test=train_test_split(new_x,new_y,test_size=0.3,random_state=0)
print('the shape of the x_training:',x_train.shape)
print(y_train.shape)
print('the shape of the x_test:',x_test.shape)


# the training data

# In[99]:


print(y_train.value_counts())


# SHOW THE COUNTS OF THE UNBALANCED DATA

# In[100]:


new_y.value_counts()


# THE BARCHART OF THE UNBALANCED DATA

# In[101]:


new_y.value_counts().plot(kind='bar',color='r')
plt.xlabel('the classes for the people who like to switch from Traditional to Digital')
plt.show()


# the test data value

# In[102]:


print(y_test.value_counts())


# OVERSAMPLING THE DATA using SMOTE Technique

# In[103]:


smote=SMOTE(random_state=1)
x_train_balanced,y_train_balanced=smote.fit_resample(x_train,y_train)


# Showing the Training shape  of the Oversampling

# In[104]:


print('the shape of the x_training:',x_train_balanced.shape)


# In[105]:


print(y_train_balanced.value_counts())


# THE BARCHART OF THE BALANCED DATA 

# In[106]:


y_train_balanced.value_counts().plot(kind='bar')


# TRAINGING THE BALANCED DATA

# In[107]:


model1=RandomForestClassifier()
model1.fit(x_train_balanced,y_train_balanced)
prediction1=model2.predict(x_train_balanced)


# In[108]:


model2=RandomForestClassifier()
model2.fit(x_train_balanced,y_train_balanced)
prediction2=model2.predict(x_train_balanced)


# In[109]:


model3=DecisionTreeClassifier(max_depth=2)
model3.fit(x_train_balanced,y_train_balanced)
prediction3=model3.predict(x_train_balanced)


# CHART OF THE DECISION TREE ON THE  TRAIN-TEST SPLIT METHOD
# Since the gini is pure (zero gini)

# In[110]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv1=cross_val_score(model1,x_train_balanced,y_train_balanced,cv=3,scoring='accuracy')
print('the accuracy of the logistic model is:',cv1.mean())


# In[111]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv2=cross_val_score(model2,x_train_balanced,y_train_balanced,cv=3,scoring='accuracy')
print('the accuracy of the Randomforest model is:',cv2.mean())


# In[112]:


#CHECKING THE PERFORMANCE OF THE MODELS
cv3=cross_val_score(model3,x_train_balanced,y_train_balanced,cv=3,scoring='accuracy')
print('the accuracy of the Decision model is:',cv3.mean())


# check the confusion matrix

# In[113]:


conf1=confusion_matrix(y_train_balanced,prediction1)
print(conf1)


# In[114]:


conf2=confusion_matrix(y_train_balanced,prediction2)
print(conf2)


# In[115]:


conf3=confusion_matrix(y_train_balanced,prediction3)
print(conf3)


# TESTING THE MODELS FOR THE RANDOMFOREST(BEST MODEL)

# In[116]:


model2=RandomForestClassifier()
model2.fit(x_test,y_test)
prediction_test_2=model2.predict(x_test)


# In[117]:


# check the confusion matrix
conf_test_2=confusion_matrix(y_test,prediction_test_2)
print(conf_test_2)


# In[118]:


print(classification_report(y_test,prediction_test_2))


# #CHECKING THE PERFORMANCE OF THE RANDOMFOREST  MODELS ON TEST THE DATA

# In[119]:



cv2=cross_val_score(model2,x_test,y_test,cv=5,scoring='accuracy')
print('the accuracy of the Randomforest model is:',cv2.mean())


# 
# IT'S CLEARLY SHOWN THAT RANDOM FOREST PERFORM BETTER WHEN COMPARED TO OTHER MODELS
# AS THE F1-SCORE RESULT TO 1

# In[ ]:





# In[ ]:





# In[ ]:




