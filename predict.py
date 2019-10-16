import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# read data
data = pd.read_csv("training_data.csv")
predict_data = pd.read_csv("test_data.csv")

# drop training data rows that miss data
data = data.drop(['Instance'], axis=1)
print(data.shape)
for c in data:
    data = data[np.invert(data[c].isnull())]
    if data[c].dtype.kind not in 'bifu':
        data[c] = data[c].str.lower()
        data = data[np.invert(data[c].isin(["0", "unknown"]))]
data = data[data["Income in EUR"] > 0]
print(data.shape)

#unify categorical data in data to predict
predict_data = predict_data.drop(['Income'], axis=1)
for c in predict_data:
    if c == "Instance":
        continue
    if predict_data[c].dtype.kind not in 'bifu':
        predict_data[c] = predict_data[c].str.lower()
    else:
        predict_data[c][predict_data[c].isnull()] = data[c].value_counts().keys()[0]

# calc mean value for cat cols
cat_cols = ['Gender','Country','Profession','University Degree','Wears Glasses','Hair Color']
for c in cat_cols:
    mean_map = data.groupby(c)['Income in EUR'].mean()
    data.loc[:, c] = data[c].map(mean_map)
    predict_data.loc[:, c] = predict_data[c].map(mean_map)
    predict_data[c][predict_data[c].isnull()|predict_data[c].isin(["0", "unknown"])] = data[c].value_counts().keys()[0]

# label encoding does not work well
#labelencoder = preprocessing.LabelEncoder()
#data["Gender"] = labelencoder.fit_transform(data["Gender"])
#data["University Degree"] = labelencoder.fit_transform(data["University Degree"])
#data["Country"] = labelencoder.fit_transform(data["Country"])
#data["Profession"] = labelencoder.fit_transform(data["Profession"])

# one hot encode, hard to deal with because it changed dimension of matrix, give this up
#enc = preprocessing.OneHotEncoder(categories=[gender])
#cat_data = pd.DataFrame({'Gender':data["Gender"]})
#data["Gender"] = enc.fit_transform(cat_data).toarray()
#enc = preprocessing.OneHotEncoder(categories=[degree])
#cat_data = pd.DataFrame({'University Degree':data["University Degree"]})
#data["University Degree"] = enc.fit_transform(cat_data).toarray()
#enc = preprocessing.OneHotEncoder(categories=[country])
#cat_data = pd.DataFrame({'Country':data["Country"]})
#data["Country"] = enc.fit_transform(cat_data).toarray()
#enc = preprocessing.OneHotEncoder(categories=[profession])
#cat_data = pd.DataFrame({'Profession':data["Profession"]})
#data["Profession"] = enc.fit_transform(cat_data).toarray()

X_train, X_test, y_train, y_test = train_test_split(data.drop(['Income in EUR'], axis=1), data['Income in EUR'], test_size=0.20, random_state=42)

regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train, y_train)
print(regr.score(X_test,y_test))

pred_data_pred = regr.predict(predict_data.drop(["Instance"], axis=1))
with open("tcd ml 2019-20 income prediction submission file.csv", 'w') as output_file:
    for index, row in predict_data.iterrows():
        output_file.write(str(row["Instance"]).split(".")[0] + "," + str(pred_data_pred[index]) + "\n")
