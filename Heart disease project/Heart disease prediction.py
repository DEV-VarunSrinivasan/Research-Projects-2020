#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np    #ALL OF THESE ARE PACKAGES IN PYTHON HAVING VARIOUS FEATURES
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')
hdis_df = pd.read_csv("framingham.csv") #reading the data from the file


# In[52]:


hdis_df.drop(['education'],axis=1,inplace=True) #to exclude the person's educational background cuz education has no effect on diseas


# In[53]:


hdis_df.head() #To give sample data as output to check if we've read the data or not.


# In[54]:


hdis_df.rename(columns={'male':'Sex_male'},inplace=True)
hdis_df.isnull().sum() #to sum up the values 


# In[55]:


hdis_df.dropna(axis=0,inplace=True)


# In[56]:


count=0
for i in hdis_df.isnull().sum(axis=1):
    if (i>0):
        count=count+1
print('Total number of rows with missing values is ', count)
print('since it is only',round((count/len(hdis_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')


# In[57]:


def Graph(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='black')
        ax.set_title(feature+" Distribution",color='darkred')
        
    fig.tight_layout()  
    plt.show()
Graph(hdis_df,hdis_df.columns,6,3)


# In[58]:


hdis_df.TenYearCHD.value_counts()


# In[59]:


sn.countplot(x='TenYearCHD',data=hdis_df)  #plots graph those likely to have CHD based on Gender


# In[60]:


hdis_df.describe() #calculates a summary of statistics related to the DataFrame columns


# In[61]:


sn.pairplot(data=hdis_df) #pairplot is just scatter plot graph


# In[62]:


from statsmodels.tools import add_constant as add_constant #adds a column of ones to an array
hdis_df_constant = add_constant(hdis_df)
hdis_df_constant.head()


# In[63]:


st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)#lambda is a small anonymous function which can have only one expresion
cols=hdis_df_constant.columns[:-1]
model=sm.Logit(hdis_df.TenYearCHD,hdis_df_constant[cols])
result=model.fit()
result.summary()


# In[64]:


def back_feature_elem (data_frame,dep_var,col_list):
   #takes data frame, independent var and column names
   #keeps repeating algorithm to eliminate features with P value more than alpha one at a time
    #p value testing is a method of hypothesis testing to find likely or unlikely.
    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(hdis_df_constant,hdis_df.TenYearCHD,cols)


# In[65]:


result.summary() #creates a summary table


# In[66]:


params = np.exp(result.params)
conf = np.exp(result.conf_int()) #exp is exponential
conf['OR'] = params
pvalue=round(result.pvalues,3) #rounds off the given number and returns naerest floating point value
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))


# In[67]:


import sklearn
new_features=hdis_df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
x=new_features.iloc[:,:-1] #iloc returns a panda series
y=new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)


# In[68]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


# In[69]:


sklearn.metrics.accuracy_score(y_test,y_pred) #testing accuracy of prediction


# In[70]:


from sklearn.metrics import confusion_matrix#table that's usedd to describe the performance of a classification model
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)
print('The acuuracy of the model= ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = ',TN/float(TN+FP),'\n',

'Positive Predictive value = ',TP/float(TP+FP),'\n',

'Negative predictive Value = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = ',(1-sensitivity)/specificity)


# In[71]:


y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])
y_pred_prob_df.head()


# In[72]:


from sklearn.preprocessing import binarize #discretization of continuous feature values.
for i in range(1,5):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')


# In[73]:


from sklearn.metrics import roc_curve #receiver operating curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[74]:


sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])

