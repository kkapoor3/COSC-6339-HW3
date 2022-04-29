from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import re
import nltk
import json
import csv
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import Word2Vec


def utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    # nltk.download()
    # clean (convert to lowercase and remove punctuations and characters and then strip)
    text = x.text
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    # Tokenize (convert from string to list)
    lst_text = text.split()
    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    # Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    # back to string from list
    text = " ".join(lst_text)

    return (text, x.y, x.helpful_votes, x.total_votes, x.vine, x.verified_purchase)

def trim_decimals(v):
    clean_vec = [float('%.4f'%(i)) for i in v]
    return clean_vec

class PreProcessing():
    def __init__(self):
        # create new spark session
        spark = SparkSession.builder.getOrCreate()

        # read tsv into dataframe
        df = spark.read.csv('amazon_reviews_grocery_100k.tsv', sep=r'\t', header=True, inferSchema=True)

        filtered_data = self.filter(df)
        df_train, df_test = self.split(filtered_data)
        self.save_csv('train', df_train)
        self.save_csv('test', df_test)
        self.tokenize(df_train)

        spark.stop()

    def filter(self, df):
        # filter columns & convert string to lower case
        df = df.select(lower('review_body'), 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase').withColumnRenamed('lower(review_body)', 'text').withColumnRenamed('star_rating', 'y')

        # drop rows with null values
        df.na.drop()

        # map rating to class (<=1 = 0 | >1 = 1)
        df = df.withColumn('y', when(df.y <= 1, 0).when(df.y > 1, 1))

        # map vine (N = 0 | Y = 1)
        df = df.withColumn('vine', when(df.vine == 'N', 0).when(df.vine == 'Y', 1))

        # map verified_purchase (N = 0 | Y = 1)
        df = df.withColumn('verified_purchase', when(df.verified_purchase == 'N', 0).when(df.verified_purchase == 'Y', 1))

        lst_stopwords = nltk.corpus.stopwords.words("english")

        # applying preprocessing to dataset
        rdd = df.rdd.map(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords))
        df = rdd.toDF(['text', 'y', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase'])
        # df.show(10, truncate = False)

        # remove non string data
        df = df.filter(df.text.rlike("[a-z]"))

        return df


    def split(self, df):
        # split train and test data
        df_train, df_test = df.randomSplit([0.8, 0.2])

        return df_train, df_test


    def save_csv(self, name, df):
        # save data to CSV
        row = df.head(df.count())
        data = [[i['text'], i['y'], i['helpful_votes'], i['total_votes'], i['vine'], i['verified_purchase']] for i in row]
        with open('data/' + name + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'y', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase'])
            writer.writerows(data)

    def tokenize(self, df_train):
        # get tokens
        tokenizer = Tokenizer(outputCol="tokens")
        tokenizer.setInputCol("text")
        tokens = tokenizer.transform(df_train)
        t = tokens.select("tokens")

        # get vectors  
        word2Vec = Word2Vec(vectorSize=300, minCount=1, inputCol="tokens", outputCol="w2c").fit(t)
        w2v = word2Vec.getVectors()

        # rdd = w2v.rdd.map(lambda x: trim_decimals(x))
        # w2v = rdd.toDF(['word', 'vector'])

        # convert dataframe to dict and save in JSON file
        list_vectors = map(lambda row: row, w2v.collect())
        # dict_vectors = {vector['word']: trim_decimals(vector['vector'].tolist()) for vector in list_vectors}

        # with open('data/words_emb.json', 'w') as f:
        #     json.dump(dict_vectors, f)

        vectors = [trim_decimals(vector['vector'].tolist()) for vector in list_vectors]

        with open('data/word_emb.csv', 'w') as f:
            writer = csv.writer(f)
            for vector in vectors:
                writer.writerow(vector)

if __name__ == "__main__":
    PreProcessing()