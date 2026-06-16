import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:/Users/cc/Downloads/bengaluru_house_prices.csv")
print(df.head())
print(df['area_type'].value_counts())
print(df.shape)

# removing features which are not important
df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
print(df2.shape)

print(df2.isna().sum())
df2['bath'].fillna(df2['bath'].mean(), inplace=True)  # others are not numerical
print(df2.isna().sum())

def convert_bhk(x):
    try:
        return int(x.split(' ')[0])
    except:
        return None

df2['bhk'] = df2['size'].apply(convert_bhk)
print(df2)

def convert_range(x):
    no = x.split('-')
    if len(no) == 2:
        return (float(no[0]) + float(no[1])) / 2
    try:
        return float(x)
    except:
        return None

df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_range)
print(df3.head())
print(df3.loc[30])

# price per square.ft
df4 = df3.copy()
df4['price_per_sqft'] = df4['price'] * 100000 / df4['total_sqft']
df4.head()

df4 = df4[df4['location'].notna()]  # drop rows with missing location
df4.location = df4.location.apply(lambda x: x.strip())
location_stats = df4['location'].value_counts(ascending=False)

location_stats_less_than_10 = location_stats[location_stats <= 10]
df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df4.location.unique()))
print(df4.shape)

# Removing outliers
df5 = df4[~((df4.total_sqft / df4.bhk) < 300)]
print(df5.shape)

# removing strange prices per sqr.ft locations
def remove(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[
            (subdf.price_per_sqft > (m - st)) &
            (subdf.price_per_sqft <= (m + st))
        ]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df6 = remove(df5)
print(df6.shape)


def remove_bhk(df):
    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):
        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }

        for bhk, bhk_df in location_df.groupby('bhk'):
            if bhk - 1 in bhk_stats and bhk_stats[bhk - 1]['count'] > 5:
                lower_bhk_mean = bhk_stats[bhk - 1]['mean']
                outliers = bhk_df[
                    bhk_df.price_per_sqft < lower_bhk_mean
                ].index
                exclude_indices = np.concatenate((exclude_indices, outliers))

    return df.drop(exclude_indices)

df7 = remove_bhk(df6)
print(df7.shape)

# bathroom should be less than bedrooms
df8 = df7[df7.bath < df7.bhk + 2].copy()
df8.drop(['size', 'price_per_sqft'], axis='columns', inplace=True)
print(df8.shape)

dummies = pd.get_dummies(df8.location)
df9 = pd.concat([df8, dummies.drop('other', axis='columns')], axis='columns')
print(df9.shape)

df9.drop('location', axis='columns', inplace=True)
print(df9.head())

inputs = df9.drop('price', axis='columns')
target = df9.price

X_train, X_test, y_train, y_test = train_test_split(
    inputs, target, test_size=0.2, random_state=10
)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

print(cross_val_score(LinearRegression(), inputs, target, cv=cv))

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(inputs.columns))

    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in inputs.columns:
        loc_index = np.where(inputs.columns == location)[0][0]
        x[loc_index] = 1

    return model.predict([x])[0]

print(predict_price('Indira Nagar', 1000, 2, 2))
