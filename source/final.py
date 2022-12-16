from metaflow import FlowSpec, step, Parameter, IncludeFile, current
import requests
from datetime import datetime
import os

import pandas as pd
import seaborn as sns
import matplotlib 
import pickle

pd.set_option('display.max_columns',None)
pd.options.display.max_seq_items = 2000
# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import requests, re
import nltk
import string, itertools
from collections import Counter, defaultdict
from sklearn.cluster import KMeans

import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

# Set up COMET
os.environ['COMET_API_KEY'] = 'ZOPkYTijoybMWYFSgwURwDvQw'
os.environ['MY_PROJECT_NAME'] = 'Final Project'

assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']
print("Running experiment for project: {}".format(os.environ['MY_PROJECT_NAME']))

from comet_ml import Experiment






class MyClassificationFlow(FlowSpec):
    TEST_SPLIT = Parameter(name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20)


    Parameters  = [0.5,1,10,100]


    ## get dataset by category
    def get_dataset(self,restaurants_reviews,category):
        df = restaurants_reviews[['removed_punct_text','labels']][restaurants_reviews.category==category]
        df.reset_index(drop=True, inplace =True)
        df.rename(columns={'removed_punct_text':'text'}, inplace=True)
        return df
    ## only keep positive and negative words
 
    ## 得到重要性评分
    def get_polarity_score(self,dataset,positive_words,negative_words,parameter):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.svm import LinearSVC

        def filter_words(review):
            words = [word for word in review.split() if word in positive_words + negative_words]
            words = ' '.join(words)
            return words

        dataset.text = dataset.text.apply(filter_words)
        
        terms_train=list(dataset['text'])
        class_train=list(dataset['labels'])
        
        ## get bag of words
        vectorizer = CountVectorizer()
        feature_train_counts=vectorizer.fit_transform(terms_train)
        
        ## run model
        svm = LinearSVC(C = parameter)
        svm.fit(feature_train_counts, class_train)
        
        ## create dataframe for score of each word in a review calculated by svm model
        coeff = svm.coef_[0]
        cuisine_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})
        
        ## get frequency of each word in all reviews in specific category
        cuisine_reviews = pd.DataFrame(feature_train_counts.toarray(), columns=vectorizer.get_feature_names())
        cuisine_reviews['labels'] = class_train
        cuisine_frequency = cuisine_reviews[cuisine_reviews['labels'] =='positive'].sum()[:-1]
        
        cuisine_words_score.set_index('word', inplace=True)
        cuisine_polarity_score = cuisine_words_score
        cuisine_polarity_score['frequency'] = cuisine_frequency
        
        cuisine_polarity_score.score = cuisine_polarity_score.score.astype(float)
        cuisine_polarity_score.frequency = cuisine_polarity_score.frequency.astype(int)
        
        ## calculate polarity score 
        cuisine_polarity_score['polarity'] = cuisine_polarity_score.score * cuisine_polarity_score.frequency / cuisine_reviews.shape[0]
        
        cuisine_polarity_score.polarity = cuisine_polarity_score.polarity.astype(float)
        ## drop unnecessary words
        unuseful_positive_words = ['great','amazing','love','best','awesome','excellent','good',
                                                   'favorite','loved','perfect','gem','perfectly','wonderful',
                                                    'happy','enjoyed','nice','well','super','like','better','decent','fine',
                                                    'pretty','enough','excited','impressed','ready','fantastic','glad','right',
                                                    'fabulous','liked','incredible','outstanding','positive']
        unuseful_negative_words =  ['bad','disappointed','disappointing','horrible','disappoint','lacking','unfortunately','sorry']

        unuseful_words = unuseful_positive_words + unuseful_negative_words
        for word in cuisine_polarity_score.index:
            if word in unuseful_words:
                cuisine_polarity_score.drop(word, axis=0, inplace=True)
        
        #cuisine_polarity_score.drop(cuisine_polarity_score.loc[unuseful_words].index, axis=0, inplace=True)
        
        return cuisine_polarity_score,vectorizer,svm,feature_train_counts

    def get_top_words(self,dataset, label, number=20):
        if label == 'positive':
            df = dataset[dataset.polarity>0].sort_values('polarity',ascending = False)[:number]
        else:
            df = dataset[dataset.polarity<0].sort_values('polarity')[:number]
        return df
    def split_data(self,dataset, test_size):
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(dataset[['text','labels']],test_size=test_size)
        return df_train,df_test

    def test_data(self,dataset,transform,model):
        from sklearn.metrics import accuracy_score

        x_test=list(dataset['text'])
        y_test=list(dataset['labels'])
    
        ## get bag of words
    
        feature_train_counts=transform.transform(x_test)
        y_predict=model.predict(feature_train_counts)
    
    
        score=accuracy_score(y_predict,y_test)
    
        return score
    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_business_data)
    @step
    def load_business_data(self):
        import json
        import pandas as pd
        data_file_1 = open("../data/yelp_academic_dataset_business.json")
        data_1 = []
        for line in data_file_1:
            data_1.append(json.loads(line))
        self.business = pd.DataFrame(data_1)
        data_file_1.close()

        self.next(self.business_data_preprocessing)
    @step
    def business_data_preprocessing(self):
        ## drop unuseful column 'hours','attributes'
        import json
        import pandas as pd
        business=self.business
        business.drop(['hours','attributes'], axis=1, inplace=True)

        ## remove quotation marks in name and address column
        business.name=business.name.str.replace('"','')
        business.address=business.address.str.replace('"','')#把引号去掉

        ## 按洲名简写筛选数据，
        ## 并存入新的dataframe：use
        states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
        usa=business.loc[business['state'].isin(states)] #把不是美国州的去掉
        usa=usa.dropna(axis=0, subset=['categories'])

        ## 选出所有餐厅，
        ## 并存入新的dataframe：us_restaurants
        us_restaurants=usa[usa['categories'].str.contains('Restaurants')]

         ## select out 16 cuisine types of restaurants and rename the category
         ## 把 us_restaurants['category'] 中含有指定种类的餐厅筛选出来，
          ## 并新建column：'category'

         # us_restaurants.is_copy=False 版本过老
        us_restaurants['category']=pd.Series()
        us_restaurants.loc[us_restaurants.categories.str.contains('American'),'category'] = 'American'
        us_restaurants.loc[us_restaurants.categories.str.contains('Mexican'), 'category'] = 'Mexican'
        us_restaurants.loc[us_restaurants.categories.str.contains('Italian'), 'category'] = 'Italian'
        us_restaurants.loc[us_restaurants.categories.str.contains('Japanese'), 'category'] = 'Japanese'
        us_restaurants.loc[us_restaurants.categories.str.contains('Chinese'), 'category'] = 'Chinese'
        us_restaurants.loc[us_restaurants.categories.str.contains('Thai'), 'category'] = 'Thai'
        us_restaurants.loc[us_restaurants.categories.str.contains('Mediterranean'), 'category'] = 'Mediterranean'
        us_restaurants.loc[us_restaurants.categories.str.contains('French'), 'category'] = 'French'
        us_restaurants.loc[us_restaurants.categories.str.contains('Vietnamese'), 'category'] = 'Vietnamese'
        us_restaurants.loc[us_restaurants.categories.str.contains('Greek'),'category'] = 'Greek'
        us_restaurants.loc[us_restaurants.categories.str.contains('Indian'),'category'] = 'Indian'
        us_restaurants.loc[us_restaurants.categories.str.contains('Korean'),'category'] = 'Korean'
        us_restaurants.loc[us_restaurants.categories.str.contains('Hawaiian'),'category'] = 'Hawaiian'
        us_restaurants.loc[us_restaurants.categories.str.contains('African'),'category'] = 'African'
        us_restaurants.loc[us_restaurants.categories.str.contains('Spanish'),'category'] = 'Spanish'
        us_restaurants.loc[us_restaurants.categories.str.contains('Middle_eastern'),'category'] = 'Middle_eastern'
        us_restaurants[:20]

        
        ## drop null values in category, 
        us_restaurants=us_restaurants.dropna(axis=0, subset=['category'])

        del us_restaurants['categories']

        ## and reset the index
        self.us_restaurants=us_restaurants.reset_index(drop=True)

        self.review = pd.read_csv('../data/yelp_reviews_3.csv')
        self.next(self.join_two_dataset)
        

       
    @step
    def join_two_dataset(self):
        ## 以 'business_id' 为准合并两个df，得到 “restaurants_reviews”
        self.restaurants_reviews = pd.merge(self.us_restaurants, self.review, on = 'business_id')
        
        self.next(self.generate_labels_and_preprocessing)

    @step
    def generate_labels_and_preprocessing(self):

        restaurants_reviews=self.restaurants_reviews
        ## 更新 column names
        #restaurants_reviews.rename(columns={'stars_x':'avg_star','stars_y':'review_star'}, inplace=True)

        ## 把评论中所有的：特殊符号和换行符 全部用空格替换，然后计算每个评论的字数并存入 'num_words_review'
        restaurants_reviews['num_words_review'] = restaurants_reviews.text.str.replace('\n','').str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','').map(lambda x: len(x.split()))

        restaurants_reviews.reset_index(drop=True, inplace=True)

        pd.set_option('display.float_format', lambda x: '%.4f' % x)


        ## convert text to lower case 全部转化为小写
        restaurants_reviews.text = restaurants_reviews.text.str.lower()

        ## remove unnecessary punctuation（又重复操作了一遍）
        restaurants_reviews['removed_punct_text']= restaurants_reviews.text.str.replace('\n','').str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','')
        
        self.restaurants_reviews=restaurants_reviews
        self.next(self.load_pos_neg_words)

    

    @step
    def load_pos_neg_words(self):

        import sys
        positive_words=[]
        with open('../data/positive.txt','r') as p:
            for pline in p:
                positive_words.append(pline.strip('\n'))

        self.positive_words=positive_words
        print(self.positive_words)

        negative_words=[]
        with open('../data/negative2.txt','r') as n:
            for nline in n:
                negative_words.append(nline.strip('\n'))
        print(negative_words)
        self.negative_words = negative_words

        self.next(self.seperate_dataset)


        
    @step
    def seperate_dataset(self):
        restaurants_reviews=self.restaurants_reviews
        Dataset_reviews = self.get_dataset(restaurants_reviews,'American')
        self.Dataset_train,self.Dataset_test = self.split_data(Dataset_reviews, 0.9)
        print('Total %d number of reviews' % self.Dataset_train.shape[0])

        self.next(self.train_dataset, foreach="Parameters")


    @step
    def train_dataset(self):
        self.parameter = float(self.input)
        Dataset_polarity_score,self.vectorizer,self.svc_model,self.X_train = self.get_polarity_score(self.Dataset_train,self.positive_words,self.negative_words,self.parameter)

        self.test_score=self.test_data(self.Dataset_test,self.vectorizer,self.svc_model)
        print("The accurancy_score on Dataset test set is:{}".format(self.test_score))

        self.next(self.join)


    @step
    def join(self, inputs):
        self.all_vectorizer = [inp.vectorizer for inp in inputs]
        self.all_svc_model = [inp.svc_model for inp in inputs]
        self.all_test_score = [inp.test_score for inp in inputs]
        # list1.index(max(list1))
        self.best_vectorizer = self.all_vectorizer[self.all_test_score.index(max(self.all_test_score))]
        self.best_svc_model = self.all_svc_model[self.all_test_score.index(max(self.all_test_score))]

        self.merge_artifacts(inputs, include=['X_train'])
        self.next(self.save_to_comet_and_pickle)

    @step
    def save_to_comet_and_pickle(self):
        # log hash of your dataset to Comet.ml
        exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'], auto_param_logging=False)       
        # self.statistics = {"all_test_score": self.all_test_score}
        self.statistics = {"test_score_1": self.all_test_score[0],
                            "test_score_2": self.all_test_score[1],
                            "test_score_3": self.all_test_score[2],
                            "test_score_4": self.all_test_score[3]}
        exp.log_dataset_hash(self.X_train)        
        exp.log_metrics(self.statistics)


        with open('../models/svm.pkl', 'wb') as f:
            pickle.dump(self.best_svc_model, f)
        with open('../models/vec.pkl', 'wb') as f:
            pickle.dump(self.best_vectorizer, f)

        self.vectorizer=self.best_vectorizer
        self.svc_model=self.best_svc_model
        self.next(self.end)

    
    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))



if __name__ == '__main__':
    MyClassificationFlow()















    