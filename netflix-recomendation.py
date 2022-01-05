#!/usr/bin/env python
# coding: utf-8


# In[ ]:
# --------Importing Library---------
import numpy as np
import pandas as pd
import seaborn as sb

# In[ ]:
# --------Read Dataset-------------
df = pd.read_csv("netflix_titles.csv")

# In[ ]:

#-------Data Cleaning--------
#mengganti data country,date_added, dan rating yang kosong dengan nilai yang paling sering muncul (menggunakan fungsi mode())

df['country'] = df['country'].fillna(df['country'].mode()[0])
df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
df['rating'] = df['rating'].fillna(df['rating'].mode()[0])

# In[ ]:

#-------Data Cleaning--------
#menghapus data yang cast dan director nya kosong karena hasil akan melenceng jauh apabila kolom tersebut diganti dengan nilai modus

df = df.dropna( how='any',subset=['cast', 'director'])

# In[ ]:


# In[ ]:
#---------Data Reduction---------
#Mengambil kolom yang dibutuhkan untuk perhitungan saja

features=['listed_in','director','cast','description','title']
filters = df[features]

# In[ ]
#------Data Understanding---------
#Mengubah kolom listed_in menjadi Genre agar mudah dalam proses nantinya
filters = filters.rename(columns={"listed_in":"Genre"})

features=['Genre','director','cast','description','title']
#Memastikan apakah masih ada data NULL atau tidak
filters.isnull().sum()

# In[ ]:
#-------Data Preparation---------
#mendefinisikan fungsi untuk mengubah data menjadi huruf kecil semua dan menghapus spasi
def to_lower(x):
        return str.lower(x.replace(" ", ""))


# In[ ]:

#-------Data Preparation---------
#menggunakan fungsi to_lower pada data yang sudah direduksi sebelumnya
for feature in features:
    filters[feature] = filters[feature].apply(to_lower)

# In[ ]:
#-------Data Transformation--------
#mendefinisikan fungsi untuk membuat metadata yang berisi director, cast, genre, dan description
def metadata(x):
    return x['director'] + ' ' + x['cast'] + ' ' +x['Genre']+' '+ x['description']


# In[ ]:
#-------Data Transformation---------
#menggunakan fungsi metadata pada filters kemudian disimpan pada filters['metadata']
#axis=1 berguna agar data yang diambil adalah untuk setiap baris dari filters
filters['metadata'] = filters.apply(metadata, axis=1)

# In[ ]:
# ------Modeling---------
# menggunakan countVectorizer untuk menghitung frekuensi setiap kata pada metadata dari masing-masing film
# hasil perhitungan frekuensi disimpan pada variable count_matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filters['metadata'])

# In[ ]:

from sklearn.metrics.pairwise import cosine_similarity

# Menghitung cosine similarity berdasarkan nilai frekuensi setiap kata pada variable count_matrix sebelumnya
# sehingga akan menghasilkan matrix cosine similarity dari seluruh film yang akan disimpan pada variable cosine_sim
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[ ]:

#---------Evaluation-----------
# Mengembalikan index agar diawali dari index 0 
filters=filters.reset_index()

#In[ ]:
#---------Evaluation-----------
#mengubah index menjadi title untuk memudahkan dalam mengetahui index dari judul film yg pernah ditonton nantinya
indices = pd.Series(filters.index, index=filters['title'])


# In[ ]:
# ----------Evaluation---------
def get_recommendations(title, cosine_sim):
    # menghapus spasi dan mengubah jadi huruf kecil
    title=title.replace(' ','').lower()
    # mengambil index dari judul film yang pernah ditonton
    idx = indices[title]

    # menghitung kemiripan film yang pernah ditonton dengan semua film yang ada
    sim_scores = list(enumerate(cosine_sim[idx]))

    # mengurutkan nilai berdasarkan terbesar ke terkecil
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # mengambil skor 10 film paling mirip
    # index 0 dilewati karena index 0 berisi film itu sendiri
    sim_scores = sim_scores[1:11]

    # mengambil index dari 10 film paling mirip
    movie_indices = [i[0] for i in sim_scores]

    # mengembalikan judul dari 10 film paling mirip berdasarkan index sebelumnya
    return df['title'].iloc[movie_indices][:10]


# In[ ]:


#In[ ]:
from flask import request
from flask import Flask, render_template
# ------Deployment---------
app = Flask(__name__)
@app.route("/")
def hello():
    list_film = df['title'][0:400].tolist()
    return render_template('index.html',film = list_film)
 
@app.route("/", methods=['POST'])
def submit():
    list_film = df['title'][0:400].tolist()
    title = request.form['film']
    result = get_recommendations(title, cosine_sim).to_list()

    return render_template('index.html', result = result, title = title, film = list_film)
 

if __name__ == "__main__":
    app.run()
    

# %%
