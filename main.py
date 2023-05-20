print("EDA on Sample Super store")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Dataset/SampleSuperstore.csv", delimiter=",")

#print(data.head(5))

print(data.columns)
print(data.shape)

print(data.isna().sum())

print(data.dtypes)
objectData = data
LabelEncoder = LabelEncoder()
encodedData = objectData.apply(LabelEncoder.fit_transform)
print(encodedData.dtypes)
print(encodedData.head(5))

SalesAndQuantity = pd.DataFrame(data[["Sales", "Quantity"]])
print(SalesAndQuantity.head(5))

plt.plot(SalesAndQuantity)
plt.xlabel("Sales")
plt.ylabel("Quantity")
plt.show()


CategoryAndSale = encodedData[["Category", "Sales"]]
print(CategoryAndSale.head(5))

plt.plot(CategoryAndSale)
plt.xlabel("Category")
plt.ylabel("Sales")
plt.show()

CategoryAndSale = encodedData[["Category", "Sales"]]
print(CategoryAndSale.head(5))

plt.scatter(encodedData["Category"], encodedData["Sales"])

