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
objectData = data[["Country", "City", "State", "Region", "Category", "Sub-Category"]]
LabelEncoder = LabelEncoder()
encodedData = objectData.apply(LabelEncoder.fit_transform)
print(encodedData.dtypes)
print(encodedData.head(5))

SalesAndQuantity = pd.DataFrame(data[["Sales", "Quantity"]])
print(SalesAndQuantity.head(5))

# plt.plot(SalesAndQuantity)
# plt.xlabel("Sales")
# plt.ylabel("Quantity")
# plt.show()
# Graph buy Category and sale

CategoryAndSale = pd.concat([encodedData["Category"], data["Sales"]], axis=1)

print(CategoryAndSale.head(5))

plt.plot(CategoryAndSale)
plt.xlabel("Category")
plt.ylabel("Sales")
plt.show()

plt.scatter(encodedData["Category"], data["Sales"])
plt.show()

# salesAccordingToRegion = encodedData[["Region", "Sales"]]
# plt.scatter(encodedData["Region"], data["Sales"].astype(int))
# plt.show()
