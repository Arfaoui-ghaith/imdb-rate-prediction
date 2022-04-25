import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Preprocessing:

    def transformYear(ch):
        if(str(ch).find('(') > -1): 
            return str(ch[1:-1])
        return ch

    def transformDuration(ch):
        if(str(ch).find(' min') > -1): 
            return str(ch.replace(' min',''))
        return ch

    def transformVotes(ch):
        if(str(ch).upper() == "NAN"):
            return np.nan
        c=str(ch).replace(',','.')
        return c

    def transformGenre(ch):
        if(str(ch).upper() == "NAN"):
            return np.nan
        c=str(ch).split(',')[0]
        return c

    def transformRating(ch):
        if(str(ch).upper() == "NAN"):
            return np.nan
        if(float(ch) >= 7):
            return '1'
        return '0'

    def transformRatingLinear(ch):
        return float(ch)

    
    dataset = pd.read_csv("imdb.csv",encoding='unicode_escape', na_values=['NaN','nan',np.nan])
    dataset = dataset.drop_duplicates()
    dataset['Votes']=list(map(transformVotes,dataset['Votes']))
    dataset['Year']=list(map(transformYear,dataset['Year']))
    dataset['Duration']=list(map(transformDuration,dataset['Duration']))
    dataset['Genre']=list(map(transformGenre,dataset['Genre']))
    dataset['Linear_Rating']=list(map(transformRatingLinear,dataset['Rating']))
    dataset['Rating']=list(map(transformRating,dataset['Rating']))

    def __init__(self,dataset=dataset):
        
        self.data = dataset.drop(columns=['Actor 1','Actor 2','Actor 3','Director','Name']).fillna(method='bfill')
       

        self.data['Year'] = self.data['Year'].astype('float16')
        self.data['Duration'] = self.data['Duration'].astype('float16')
        self.data['Linear_Rating'] = self.data['Linear_Rating'].astype('float16')
        self.data['Rating'] = self.data['Rating'].astype('float16')
        self.data['Votes'] = self.data['Votes'].astype('float')

        genre = LabelEncoder()
        self.data['Genre']=genre.fit_transform(self.data['Genre'])

    def getData(self):
        return self.data

