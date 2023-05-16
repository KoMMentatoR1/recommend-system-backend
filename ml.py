from pymongo import MongoClient
import numpy as np
import pickle 
from sklearn.neighbors import NearestNeighbors

def learnModel(books):
  knnPickle = open('model', 'wb') 
  data = books.find({})

  X = np.array([i['description_vektor'] for i in data])
  y = np.array([str(i['_id']) for i in data])

  knn = NearestNeighbors(n_neighbors=20)

  knn.fit(X, y)

  pickle.dump(knn, knnPickle) 

  knnPickle.close()