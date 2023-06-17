import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df=pd.read_csv(r'C:\Users\Acer\Desktop\HACKATHON\Train.csv')
df2=pd.read_csv(r'C:\Users\Acer\Desktop\HACKATHON\Test.csv')

df.fillna(0, inplace=True)
df2.fillna(0,inplace=True)

var="Q46"

df3=df[['Q1','Q2','Q5','Q6','Q27','Q46','Q47','Q48','Q49','Q50','Q51','Q54','Q90','Q106','Q108','Q119','Q120','Q121','Q131','Q142','Q143','Q158','Q160','Q164','Q173','Q176','Q199','Q240','Q253','Q260','Q273']]
df4=df2[['Q1','Q2','Q5','Q6','Q27','Q46','Q47','Q48','Q49','Q50','Q51','Q54','Q90','Q106','Q108','Q119','Q120','Q121','Q131','Q142','Q143','Q158','Q160','Q164','Q173','Q176','Q199','Q240','Q253','Q260','Q273']]

train_labels = pd.DataFrame(df3[var])
train_labels = np.array(df3[var])
train_features= df3.drop(var, axis = 1)
feature_list = list(train_features.columns)
train_features = np.array(train_features)

test_labels = pd.DataFrame(df4[var])
test_labels = np.array(df4[var])
test_features= df4.drop(var, axis = 1)
test_features = np.array(test_features)

rf=RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =42,max_features = "auto", min_samples_leaf = 12)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
print(metrics.accuracy_score(test_labels, predictions))
