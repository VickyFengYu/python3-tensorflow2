#!/usr/bin/env python
# coding: utf-8

## Making Recommendations Based on Correlation

import numpy as np
import pandas as pd

frame = pd.read_csv('rating_final.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')
geodata = pd.read_csv('geoplaces2.csv', encoding='ISO-8859-1')

frame.head()

geodata.head()

places = geodata[['placeID', 'name']]
places.head()

cuisine.head()

# ## Grouping and Ranking Data
rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean())
rating.head()

rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())
rating.head()

rating.describe()

rating.sort_values('rating_count', ascending=False).head()

places[places['placeID'] == 135085]
cuisine[cuisine['placeID'] == 135085]

# Preparing Data For Analysis
places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
places_crosstab.head()

Tortas_ratings = places_crosstab[135085]
Tortas_ratings[Tortas_ratings >= 0]

# Evaluating Similarity Based on Correlation
similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)

corr_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR'])
corr_Tortas.dropna(inplace=True)
corr_Tortas.head()

Tortas_corr_summary = corr_Tortas.join(rating['rating_count'])

Tortas_corr_summary[Tortas_corr_summary['rating_count'] >= 10].sort_values('PearsonR', ascending=False).head(10)

places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index=np.arange(7),
                                  columns=['placeID'])
summary = pd.merge(places_corr_Tortas, cuisine, on='placeID')
print(summary)
print("\n")

print(places[places['placeID'] == 135046])
print("\n")

print(cuisine['Rcuisine'].describe())
