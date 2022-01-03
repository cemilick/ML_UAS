#!/usr/bin/env python
# coding: utf-8

# In[20]:
import pandas as pd
import numpy as np


# In[2]:
data = pd.read_csv('netflix_titles.csv')


# In[3]:


data.head()


# In[4]:


import matplotlib.pyplot as plt
import networkx as nx


# In[5]:


data.describe()


# In[6]:


data["date_added"] = pd.to_datetime(data['date_added'])
data['year'] = data['date_added'].dt.year
data['month'] = data['date_added'].dt.month
data['day'] = data['date_added'].dt.day


# In[7]:


data['directors']=data['director'].apply(lambda x: [] if pd.isna(x) else [i.strip() for i in x.split(',')])
data['actors']=data['cast'].apply(lambda x: [] if pd.isna(x) else [i.strip() for i in x.split(',')])
data['categories']=data['listed_in'].apply(lambda x: [] if pd.isna(x) else [i.strip() for i in x.split(',')])
data['countries']=data['country'].apply(lambda x: [] if pd.isna(x) else [i.strip() for i in x.split(',')])
data.head()


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
text_content = data['description']
vector = TfidfVectorizer(max_df=0.3,        
                             min_df=1,     
                             stop_words='english', 
                             lowercase=True, 
                             use_idf=True, 
                             norm=u'l2',
                             smooth_idf=True
                            )
tfidf = vector.fit_transform(text_content)
kmeans = MiniBatchKMeans(n_clusters = 200)
kmeans.fit(tfidf)
centers = kmeans.cluster_centers_.argsort()[:,::-1]
terms = vector.get_feature_names()   
request_transform = vector.transform(data['description'])
data['cluster'] = kmeans.predict(request_transform) 
data['cluster'].value_counts().head()


# In[9]:


print(request_transform)


# In[10]:


print(data['cluster'])


# In[11]:


def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n] 


# In[12]:


G=nx.Graph(label='NETFLIX')
for i,row in data.iterrows():
    G.add_node(row['title'],key=row['show_id'],label='MOVIE',mtype=row['type'],rating=row['rating'])
    for j in row['actors']:
        G.add_node(j,label='PERSON')
        G.add_edge(row['title'],j,label='ACTED_IN')
    for j in row['directors']:
        G.add_node(j,label='PERSON')
        G.add_edge(row['title'],j,label='DIRECTED')
    for j in row['categories']:
        G.add_node(j,label='CAT')
        G.add_edge(row['title'],j,label='CAT_IN')
    for j in row['countries']:
        G.add_node(j,label='COUNTRY')
        G.add_edge(row['title'],j,label='COUNTRY_IN')
for i,row in data.iterrows():
    similar=find_similar(tfidf,i,top_n=5)
    for e in similar:
        G.add_edge(row['title'],data['title'].loc[e],label='SIMILAR_TO')
    


# In[13]:


G.number_of_nodes()


# In[14]:


G.number_of_edges()


# In[15]:


import math as math
def get_recommendation(root):
    commons_dict = {}
    for e in G.neighbors(root):
        for e2 in G.neighbors(e):
            if e2==root:
                continue
            if G.nodes[e2]['label']=="MOVIE":
                commons = commons_dict.get(e2)
                if commons==None:
                    commons_dict.update({e2 : [e]})
                else:
                    commons.append(e)
                    commons_dict.update({e2 : commons})
    movies=[]
    weight=[]
    for key, values in commons_dict.items():
        w=0.0
        for e in values:
            w=w+1/math.log(G.degree(e))
        movies.append(key) 
        weight.append(w)

    result = np.vstack((movies, weight)).T
    result = result.tolist()
    result = sorted(result, key=lambda x: x[1], reverse=True)
    
    return result

# In[ ]:
title = "Transformers Prime"
result = get_recommendation(title)
print("*"*40+"\n Recommendation for '" + title +"'\n"+"*"*40)
print(result[:5])

#In[ ]:
from flask import request
from flask import Flask, render_template

app = Flask(__name__)
@app.route("/")
def hello():
    list_film = data['title'][0:700].tolist()
    return render_template('index.html',film = list_film)
 
@app.route("/submit", methods=['POST'])
def submit():
    list_film = data['title'][0:700].tolist()
    title = request.form['film']
    result = get_recommendation(title)

    return render_template('index.html', result = result[0:10], title = title, film = list_film)
 

if __name__ == "__main__":
    app.run()
    

# %%
