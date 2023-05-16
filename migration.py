import pandas as pd

def parse_string(string):
    if(type(string) == str):
      string = string.strip('[]')
      string_list = eval(string)
      if(type(string_list) == str):
        return [string_list]
      else:
        return string_list
    else:
       return string
    

count = 0

def migrate(books, model):

  def transformToVector(x):
    global count
    count += 1
    if count % 1 == 0: 
      print(count / data.size * 100)

    encode = model.encode(x)  

    return encode.tolist()  

  def book_to_db(x):
    book_data = {
      'title': x.title,
      'description': x.description,
      'authors': x.authors,
      'image': x.image,
      'previewLink': x.previewLink,
      'publisher': x.publisher,
      'infoLink': x.infoLink,
      'categories': x.categories,
      'ratingsCount': x.ratingsCount,
      'description_vektor': x.description_vektor,
    }
    books.insert_one(book_data)  

  booksData = pd.read_csv('books_data.csv')
  booksData.dropna(subset=['description'], inplace=True)
  data = booksData.loc[0:10000]
  data['authors'] =  data['authors'].apply(parse_string)
  data['categories'] =  data['categories'].apply(parse_string)
  data['description_vektor'] =  data['description'].apply(transformToVector)
  data = data.rename(columns={
      "Title": 'title',
  })

  data.apply(book_to_db, axis=1)