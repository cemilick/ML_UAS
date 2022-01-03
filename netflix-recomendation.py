#!/usr/bin/env python
# coding: utf-8

# # <center> 📺Netflix EDA and Movie Recommendation System😎🍿

# ![](https://www.extremetech.com/wp-content/uploads/2016/03/Netflix-Feature.jpg)

# Netflix is the world's leading streaming entertainment service with 208 million paid memberships in over 190 countries enjoying TV series, documentaries and feature films across a wide variety of genres and languages. Members can watch as much as they want, anytime, anywhere, on any internet-connected screen. Members can play, pause and resume watching, all without commercials or commitments.

# ### Here I have done a detailed analysis of netflix content data with awesome visualizations and built a Recommendation System.

# ## 1. Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb

# ### Reading Data

# In[ ]:


df = pd.read_csv("netflix_titles.csv")


# ## 2. Data Exploration

# In[ ]:


# In[ ]:


# - There are missing values in column director,cast,country and date_added.
# - We can't randomly fill the missing values in columns of director and cast, so we can drop them.
# - For minimal number of missing values in country and date_added,rating, we can fill them using mode(most common value) and mean.

# ### --> Handling missing values

# In[ ]:


df['country'] = df['country'].fillna(df['country'].mode()[0])
df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
df['rating'] = df['rating'].fillna(df['country'].mode()[0])


# In[ ]:


df = df.dropna( how='any',subset=['cast', 'director'])



# - dataset has 0 duplicated values.

# ### --> Cleaning the data

# Adding some new columns:
# - listed_in - Genre
# * Year Added - year_add
# * Month Added - month_add
# * Princial Country - country_main 

# In[ ]:


#Rename the 'listed_in' column as 'Genre' for easy understanding
df = df.rename(columns={"listed_in":"Genre"})
df['Genre'] = df['Genre'].apply(lambda x: x.split(",")[0])
df['year_add'] = df['date_added'].apply(lambda x: x.split(" ")[-1])
df['month_add'] = df['date_added'].apply(lambda x: x.split(" ")[0])
df['country_main'] = df['country'].apply(lambda x: x.split(",")[0])

# ## 4. Netflix Recommendation System

# ## Content Based Filtering

# - For this recommender system the content of the movie (cast, description, director,genre etc) is used to find its similarity with other movies. Then the movies that are most likely to be similar are recommended.

# ![](https://miro.medium.com/max/998/1*O_GU8xLVlFx8WweIzKNCNw.png)

# ## Plot description based Recommender

# - We will calculate similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score. The plot description is given in the **description** feature of our dataset.


# In[ ]:


features=['Genre','director','cast','description','title']
filters = df[features]


# In[ ]:


#Cleaning the data by making all the words in lower case.
def clean_data(x):
        return str.lower(x.replace(" ", ""))


# In[ ]:


for feature in features:
    filters[feature] = filters[feature].apply(clean_data)
    
filters.head()


# - We can now create our "metadata soup", which is a string that contains all the metadata that we want to feed to our vectorizer.

# In[ ]:


def create_soup(x):
    return x['director'] + ' ' + x['cast'] + ' ' +x['Genre']+' '+ x['description']


# In[ ]:


filters['soup'] = filters.apply(create_soup, axis=1)


# The next steps are the same as what we did with our plot description based recommender. One important difference is that we use the **CountVectorizer()** instead of TF-IDF.

# In[ ]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filters['soup'])


# In[ ]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[ ]:


filters


# In[ ]:


# Reset index of our main DataFrame and construct reverse mapping as before
filters=filters.reset_index()
indices = pd.Series(filters.index, index=filters['title'])


# In[ ]:


def get_recommendations_new(title, cosine_sim=cosine_sim):
    title=title.replace(' ','').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]


# In[ ]:


#In[ ]:
from flask import request
from flask import Flask, render_template

app = Flask(__name__)
@app.route("/")
def hello():
    list_film = df['title'][0:100].tolist()
    return render_template('index.html',film = list_film)
 
@app.route("/submit", methods=['POST'])
def submit():
    list_film = df['title'][0:100].tolist()
    title = request.form['film']
    result = get_recommendations_new(title, cosine_sim2).to_list()

    return render_template('index.html', result = result[0:10], title = title, film = list_film)
 

if __name__ == "__main__":
    app.run()
    

# %%
