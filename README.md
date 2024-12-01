## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
        from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/DATA/

import pandas as pd
import numpy as np

file_path = '/content/drive/MyDrive/Colab Notebooks/Encoding Data.csv'

df = pd.read_csv(file_path)

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

unique_values = df['ord_2'].unique()

pm = ["Hot", "Warm" , "Cold"]

pm = ["Hot", "Warm" , "Cold"] + [val for val in unique_values if val not in pm]

e1 = OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])
![Screenshot 2024-12-01 155929](https://github.com/user-attachments/assets/a4d4b8aa-5011-4b3e-8caf-a9241dfb557b)



df["bo2"] = e1.fit_transform(df[["ord_2"]])

df

![Screenshot 2024-12-01 155941](https://github.com/user-attachments/assets/15bb0251-8be4-441b-a85b-82386b90493d)


le = LabelEncoder()

dfc = df.copy()

dfc["ord_2"] =le.fit_transform(df["ord_2"])

dfc
![Screenshot 2024-12-01 155950](https://github.com/user-attachments/assets/4fe1b6b6-feda-46dc-8ab9-544641fb42c0)



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

df2 = df.copy()

enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

df2 = pd.concat([df2, enc], axis=1)

pd.get_dummies(df2, columns=["nom_0"])

![Screenshot 2024-12-01 155958](https://github.com/user-attachments/assets/7d698f1c-e9c3-4257-bf41-0ff7462aa5a7)


pip install --upgrade category_encoders

![Screenshot 2024-12-01 160015](https://github.com/user-attachments/assets/e72d0501-17f9-40cd-b6ff-48cc2628d686)


from category_encoders import BinaryEncoder





df= pd.read_csv(file_path)

be = BinaryEncoder()

nd = be.fit_transform(df["ord_2"])

dfb = pd.concat([df, nd], axis=1)

dfb1 = df.copy()

dfb

![Screenshot 2024-12-01 160028](https://github.com/user-attachments/assets/f1436b14-1b7f-418b-9b5d-71e5e8bd076d)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats


from sklearn.preprocessing import QuantileTransformer

file_path = '/content/drive/MyDrive/Colab Notebooks/Data_to_Transform.csv'

df = pd.read_csv(file_path)

df.skew()

![Screenshot 2024-12-01 160042](https://github.com/user-attachments/assets/592dba15-bab5-4583-a837-f4ee8abcc4a8)


np.log(df["Highly Positive Skew"])

![Screenshot 2024-12-01 160053](https://github.com/user-attachments/assets/de4aa0bd-35de-463c-9e10-7a14af5ea3e7)

np.reciprocal(df["Moderate Positive Skew"])

![Screenshot 2024-12-01 160104](https://github.com/user-attachments/assets/0b5f0b83-9bb7-496e-8677-b5917904dfe6)

np.sqrt(df["Highly Positive Skew"])

![Screenshot 2024-12-01 160112](https://github.com/user-attachments/assets/c32d3493-a27a-4809-89dc-b3115d08a5f4)


np.square(df["Highly Positive Skew"])

![Screenshot 2024-12-01 160120](https://github.com/user-attachments/assets/87613200-5e9e-484e-888f-752a96d3091a)


df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"], lmbda = stats.yeojohnson(df["Moderate Negative Skew"])

df.skew()

![Screenshot 2024-12-01 160129](https://github.com/user-attachments/assets/5d01f9d2-a094-4570-8657-91a8a01bcb05)


df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

![Screenshot 2024-12-01 160302](https://github.com/user-attachments/assets/f4f1e765-d763-4777-9678-1513419f4710)


sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

![Screenshot 2024-12-01 160311](https://github.com/user-attachments/assets/b9761bb7-d4a7-492d-9e8d-8af27e74dc0b)


from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew]])



sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![Screenshot 2024-12-01 160319](https://github.com/user-attachments/assets/8ed4fd7e-cda3-4e93-a9ed-3b8b622425e9)


df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()

![Screenshot 2024-12-01 160325](https://github.com/user-attachments/assets/21d54292-c114-4547-8ea0-c20c9290f090)


sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

![Screenshot 2024-12-01 160333](https://github.com/user-attachments/assets/5d6d7764-2753-4836-86f7-8c7764e8a580)


dt=pd.read_csv("drive/MyDrive/Colab Notebooks/titanic_dataset.csv")

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()


![Screenshot 2024-12-01 160341](https://github.com/user-attachments/assets/d65b3234-2bc7-4c85-bd3e-c50818513b76)

sm.qqplot(dt['Age_1'],line='45')
plt.show()

![Screenshot 2024-12-01 160349](https://github.com/user-attachments/assets/163cb0fb-a0a1-44c6-939b-dae799d226b1)

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
