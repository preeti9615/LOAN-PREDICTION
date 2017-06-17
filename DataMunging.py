import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace = True)
df['Self_Employed'].fillna('No', inplace = True)
df['Gender'].fillna('Female',inplace = True)
df['Married'].fillna('Yes',inplace = True)
df['Dependents'].fillna(0,inplace = True)
df['Loan_Amount_Term'].fillna(360,inplace = True)
df['Credit_History'].fillna(1,inplace = True)
df['LoanAmount_Log']= np.log(df['LoanAmount'])
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Total_Income_log'] = np.log(df ['Total_Income'])
df['Total_Income_log'].hist(bins = 20)
print(df.apply(lambda x : sum(x.isnull()),axis = 0))
