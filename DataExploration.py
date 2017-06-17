import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
print(df.head())
print(df.describe())
print(df['Property_Area'].value_counts())
print(df['ApplicantIncome'].hist(bins = 10))
df.boxplot(column = 'ApplicantIncome')
df.boxplot(column = 'ApplicantIncome',by = 'Education')
temp1 = df['Credit_History'].value_counts(ascending = True)
print(temp1)
temp2 = df.pivot_table(values = 'Loan_Status',index = ['Credit_History'],aggfunc = lambda x : x.map({'Y':1,'N':0}).mean())
print (temp2)
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


temp4 = df['Married'].value_counts(ascending = True)
print(temp4)
temp5 = df.pivot_table(values= 'Loan_Status',index = ['Married']), aggfunc = lambda x : x.map({'Y':1,'N':0}).mean())
print(temp5)


fig2 = plt.figure(figure size=(8,4))
ax1 = fig2.add_subplot(121)
ax1.set_xlabel()

 
