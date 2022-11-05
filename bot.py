import pandas as pd
import numpy as np
import operator
from sklearn.feature_extraction.text import TfidfVectorizer

path = './sample_data.csv'

def bot_engine(query= ''):
    resp = ""

    csv_reader=pd.read_csv(path)
    
    question_list = csv_reader[csv_reader.columns[0]].values.tolist()
    answers_list  = csv_reader[csv_reader.columns[1]].values.tolist()
    
    # Creating TF_IDF Vectorizer
    vectorizer = TfidfVectorizer(min_df=0, ngram_range=(2, 4), strip_accents='unicode', norm='l2', encoding='ISO-8859-1')
    
    # Training the model
    ## array for training data set (questions)
    X_train = vectorizer.fit_transform(np.array([''.join(que) for que in question_list]))
    ## transforming the query sent by user to bot (test data)
    X_query=vectorizer.transform([query])
    
    # Processing the query
    ## we find out similarity of query with other questions 
    ## this is done by taking a dot product of the training data matrix with a transpose of query data
    XX_similarity=np.dot(X_train.todense(), X_query.transpose().todense())
    XX_sim_scores= np.array(XX_similarity).flatten().tolist() 

    # Ranking Results
    ## creating a sorted dictionary of similarities of a query
    dict_sim= dict(enumerate(XX_sim_scores))
    sorted_dict_sim = sorted(dict_sim.items(), key=operator.itemgetter(1), reverse =True)
    
    ## checking index of the most similar question and respond as the value at that index
    ## if nothing is found, we return a default response.
    if sorted_dict_sim[0][1]==0:
        print("Sorry I don't have an answer for your query, please try reframing your question :)")
        resp = "Sorry I don't have an answer for your query, please try reframing your question :)"
    
    elif sorted_dict_sim[0][1]>0:
        print(answers_list[sorted_dict_sim[0][0]])
        resp = answers_list[sorted_dict_sim[0][0]]
    
    return resp