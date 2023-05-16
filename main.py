from datetime import datetime, timedelta
import jwt
from typing import Optional
from fastapi import FastAPI, HTTPException, status, Header, Path, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from passlib.context import CryptContext
from pydantic import BaseModel
from pymongo import MongoClient
from bson.json_util import  dumps
from bson.objectid import ObjectId
import json
import math
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import ASCENDING
from migration import migrate
from ml import learnModel

app = FastAPI()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

client = MongoClient('localhost', 27017)
db = client['library']
user = db['users']
book = db['books']
review = db['reviews']

user.create_index([('_id', ASCENDING)])
review.create_index([('user_id', ASCENDING)])
book.create_index([('_id', ASCENDING)])
review.create_index([('book_id', ASCENDING)])

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not book.find_one({}):
    migrate(book, model)

learnModel(book)    

#Data schemas
class LoginData(BaseModel):
    username: str
    password: str

class RegisterData(BaseModel):
    username: str
    password: str    


#methods
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "mysecretkey"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def decode_access_token(token: str):
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded_token
    except error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
def serialiseObj(obj):
   new_book = obj.copy()
   new_book["_id"] = str(obj["_id"])
   return new_book     
    
def get_recommend(Authorization: str):    
    token = Authorization.split()[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        current_user = user.find_one({'username': payload['sub']['username']})
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверный логин или пароль",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = ObjectId(current_user['_id'])
        
        books = book.aggregate([
            {
                '$lookup': {
                    'from': 'reviews',
                    'localField': '_id',
                    'foreignField': 'book_id',
                    'as': 'reviews'
                }
            },
            {
                '$match': {
                    'reviews.rating': {
                        '$gt': 4.0
                    },
                    'reviews.user_id': user_id
                }
            }
        ])

        indices = np.array([])
        distances = np.array([])
        file = open('model', 'rb')
        knn = pickle.load(file)
        file.close()
        for el in books:
            temp = np.array(el['description_vektor'])
            new_indices = knn.kneighbors(temp.reshape(1, -1))[1].flatten()
            indices = np.concatenate((indices, new_indices)).astype(int)
            new_distances = knn.kneighbors(temp.reshape(1, -1))[0].flatten()
            distances = np.concatenate((distances, new_distances)).astype(float)

        sorted_idx = np.argsort(distances)
        sorted_indices = indices[sorted_idx][1:]
        book_indexes = []

        for el in book.find({}, {"_id": 1}):
            book_indexes.append(el)

        need_book = [book_indexes[index]['_id'] for index in sorted_indices]

        recommend_books = book.find({"_id": {"$in": need_book}})

        if recommend_books:
            return json.loads(dumps([serialiseObj(book) for book in recommend_books])) 
        else: 
            return HTTPException(status_code=404, detail='Непредвиденная ошибка')
    except jwt.exceptions.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Invalid token')
    
def authMiddleware(Authorization: str):
    token = Authorization.split()[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        current_user = user.find_one({'username': payload['sub']['username']})
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Пользователь не авторизован",
                headers={"WWW-Authenticate": "Bearer"},
            )  
        return current_user
    except jwt.exceptions.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Пользователь не авторизован')     
#Api

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/login")
def login_user(data: LoginData):
    current_user = user.find_one({'username':data.username}) 
    
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(data.password, current_user['password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": {'username': current_user['username'], '_id': str(current_user['_id'])}},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"token": access_token, "username": current_user['username'], '_id': str(current_user['_id']),}

@app.post("/registration")
def register_user(data: RegisterData):
    current_user = user.find_one({'username': data.username}) 
    
    if current_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким именем уже существует",
        )

    newUser_id = user.insert_one({'username': data.username, 'password': hash_password(data.password)}).inserted_id
    newUser = user.find_one({'_id': newUser_id})
    access_token = create_access_token(
        data={"sub": {'username': newUser['username'], '_id': str(newUser_id)}},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"token": access_token, "username": newUser['username'], '_id': str(newUser_id),}

@app.get('/refresh')
def refresh(Authorization: str = Header()):
    current_user = authMiddleware(Authorization)
    new_token = jwt.encode({"sub": {'username': current_user['username'], 'id': str(current_user['_id'])}}, SECRET_KEY, algorithm='HS256')
    return {'username': current_user['username'], '_id': str(current_user['_id']), 'token': new_token}

@app.get("/book/{book_id}")
async def get_book_by_id(book_id: str = Path()):
    current_book = book.find_one({'_id': ObjectId(book_id)})
    if current_book is None:
        raise HTTPException(status_code=404, detail=f"Book with id={book_id} not found")
    return json.loads(dumps(serialiseObj(current_book)))  

@app.get("/books/{page}")
async def get_book_by_id(page: int = Path(), author = Query(), category = Query(), title = Query()):
    skip = (page - 1) * 15
    findParam = {}

    if not author == '' and not author == 'null':
        findParam['authors'] = author
    if not category == '' and not category == 'null':
        findParam['categories'] = category
    if not title == '':
        findParam['title'] =  {"$regex": title, "$options": "i"}      

    books = book.find(findParam).skip(skip).limit(15)
    if books is None:
        raise HTTPException(status_code=404, detail=f"Book with id={page} not found")
    return json.loads(dumps([serialiseObj(book) for book in books])) 

@app.get("/booksCount")
async def get_books_count(author = Query(), category = Query(), title = Query()):
    findParam = {}
    if not author == '' and not author == 'null':
        findParam['authors'] = author
    if not category == '' and not category == 'null':
        findParam['categories'] = category
    if not title == '':
        findParam['title'] =  {"$regex": title, "$options": "i"}     
    return math.ceil(book.count_documents(findParam) / 15)

@app.get("/authors")
async def get_authors():
    authors = book.distinct('authors', {"authors": {"$not": {"$type": 1}}}) 
    return [author for author in authors] 

@app.post('/review/{book_id}')
async def review_book(book_id: str = Path(), Authorization: str = Header(), data = Body()):
    current_book = book.find_one({'_id': ObjectId(book_id)})
    if current_book is None:
        raise HTTPException(status_code=404, detail=f"Book with id={book_id} not found")
    current_user = authMiddleware(Authorization)
    new_review = review.update_one({'book_id': ObjectId(book_id), 'user_id': ObjectId(current_user['_id'])}, {"$set": {"rating": data['review']}}, upsert=True)
    if new_review:
        return True
    else: 
        return HTTPException(status_code=404, detail='Непредвиденная ошибка')
    
@app.get('/review/{book_id}')
async def get_review_book(book_id: str = Path(), Authorization: str = Header()):
    current_book = book.find_one({'_id': ObjectId(book_id)})
    if current_book is None:
        raise HTTPException(status_code=404, detail=f"Book with id={book_id} not found")
    
    current_user = authMiddleware(Authorization)
    current_review = review.find_one({'book_id': ObjectId(book_id), 'user_id': ObjectId(current_user['_id'])})
    if current_review:
        return current_review['rating']
    else: 
        return HTTPException(status_code=404, detail='Непредвиденная ошибка')
    
@app.get('/my/review')
async def get_my_review(Authorization: str = Header()):
    current_user = authMiddleware(Authorization)
    pipeline = [
        {'$match': {'user_id': ObjectId(current_user['_id'])}},
        {'$lookup': {
            'from': 'books',
            'localField': 'book_id',
            'foreignField': '_id',
            'as': 'book'
        }}
    ]

    reviews = review.aggregate(pipeline)
    if reviews:
        return json.loads(dumps([{'rating': review['rating'], 'book': serialiseObj(review['book'][0])} for review in reviews])) 
    else: 
        return HTTPException(status_code=404, detail='Непредвиденная ошибка')
    
@app.get('/my/recommend')
async def get_my_recommend(Authorization: str = Header()):
    return get_recommend(Authorization)

@app.get('/my/recommend/{category}')
async def get_my_recommend_with_category(Authorization: str = Header(), category: str = Path()):
    books = get_recommend(Authorization)
    return [book for book in books if 'categories' in book and book['categories'] is not None and category in book['categories']]

@app.get('/categories')
async def get_all_genre():
    categories = book.distinct('categories', {"categories": {"$not": {"$type": 1}, "$ne": None}}) 
    return [categorie for categorie in categories]

@app.get('/my/categories')
async def get_my_genre(Authorization: str = Header()):
    current_user = authMiddleware(Authorization)
    user_id = ObjectId(current_user['_id'])
    books = book.aggregate([
        {
            '$lookup': {
                'from': 'reviews',
                'localField': '_id',
                'foreignField': 'book_id',
                'as': 'reviews'
            }
        },
        {
            '$match': {
                'reviews.rating': {
                    '$gt': 4.0
                },
                'reviews.user_id': user_id
            }
        }
    ])

    categories = book.distinct('categories', {"categories": {"$exists": True, "$ne": None}, '_id': {"$in": [book['_id'] for book in books]}})
    if categories:
        return [category for category in categories] 
    else: 
        return HTTPException(status_code=404, detail='Непредвиденная ошибка')
    
@app.post('/book')
async def create_book(Authorization: str = Header(), data = Body()):
    authMiddleware(Authorization)
    current_book = book.find_one({'title': data['title']})
    if current_book:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Книга с таким названием уже существует",
        )

    book_data = {}
    book_data['title'] = data['title']
    book_data['description'] = data['description']
    if(data['authors']):
        book_data['authors'] = data['authors'].split(',')
    else:
        book_data['authors'] = None
    if(data['image']):
        book_data['image'] = data['image']
    else:
        book_data['image'] = None
    if(data['categories']):
        book_data['categories'] = data['categories'].split(',')
    else:
        book_data['categories'] = None    
    if(data['previewLink']):
        book_data['previewLink'] = data['previewLink']
    else:
        book_data['previewLink'] = None    
    if(data['publisher']):
        book_data['publisher'] = data['publisher']
    else:
        book_data['publisher'] = None    
    description_vektor = model.encode(data['description'])
    book_data['description_vektor'] = description_vektor.tolist()
    book.insert_one(book_data)

    learnModel(book) 

    return True