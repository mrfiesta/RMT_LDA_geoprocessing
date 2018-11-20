import json
import os
import operator
import imp
from nltk.tokenize import word_tokenize
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import bigrams
import re
from collections import Counter
import string
from collections import defaultdict
import sys
import vincent
import psycopg2
import os
os.chdir('/home/goldroger/Documents/Thesis/RicCode/PostGreConnection')
from config import config
import fileinput
import vincent
from gensim import corpora, models
from gensim.test.utils import common_corpus


####Load the tile or bucket

def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # Extract Database

        sql = "COPY (with t as (select tweet_id,tweet_text,tweet_created,user_id,tweet_lon,tweet_lat from s6036740.latlong WHERE tweet_lon BETWEEN 3.29 AND 7.31 AND tweet_lat BETWEEN 50.61 AND 53.69) select json_agg(t) from t) TO STDOUT"
        with open("/home/goldroger/Documents/Thesis/RicCode/PostGreConnection/completefile.txt", "w") as file:
            cur.copy_expert(sql, file)

        print('Extracting information from the database...')

        # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

     # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')
     # adaptation of the postgrefile to a regular json format
    for line in fileinput.input(['completefile.txt'], inplace=True):
        print(line.replace('}, \\n {','}\r{'), end='')
    for line in fileinput.input(['completefile.txt'], inplace=True):
        print(line.replace('\\"',''), end='')
    for line in fileinput.input(['completefile.txt'], inplace=True):
        print(line.replace('\\',''), end='')
    for line in fileinput.input(['completefile.txt'], inplace=True):
        print(line.replace('[',''), end='')
    for line in fileinput.input(['completefile.txt'], inplace=True):
        print(line.replace(']',''), end='')

if __name__ == '__main__':
    connect()


fname = 'completefile.txt'

########## Tweets per users

# TweetsUser = {}
#
# with open(fname, 'r') as f:
#     count_user = Counter()
#     for line in f:
#         tweet = json.loads(line)
#         terms_all = [term for term in preprocess(tweet['tweet_text'])if term not in stop and
#         not term.startswith(('#', '@'))]

#####Source:https://marcobonzanini.com/2015/03/17/mining-twitter-data-with-python-part-3-term-frequencies/

####Define the emoticons and regex

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    # r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]


###list punctuation
punctuation = list('️'+'•'+'—'+'…'+'–' + string.punctuation)
# stop = stopwords.words('english') + punctuation + ['RT', 'via']

################## Language detection,
###code based on http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/
###Calculate probability of given text to be written in several languages and return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}

def _calculate_languages_ratios(text):
    languages_ratios = {}
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        languages_ratios[language] = len(common_elements) # language "score"
    return languages_ratios


###Calculate probability of given text to be written in several languages and return the highest scored.
def detect_language(text):
    ratios = _calculate_languages_ratios(text)
    most_rated_language = max(ratios, key=ratios.get)
    return most_rated_language


##################Pre-processing
###Stop words compilation
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

###Function to tokenize and stemming

def preprocessEnglish(s, lowercase=False):
    #Remove emoticons, numbers etc. and returns list of cleaned tweets.
    stemmer = PorterStemmer()
    noURL = re.sub("(\w+:\/\/\S+)|^rt|http.+?|^\d+\s|\s\d+\s|\s\d+$|\b\d+\b", '',s)
    tokens = tokenize(noURL)
    lemma = WordNetLemmatizer()
    # print('tokens', tokens)
    lemmtoken = [lemma.lemmatize(i) for i in tokens]
    stemed = [stemmer.stem(i) for i in lemmtoken]
    # print('stemed',stemed)
    if lowercase:
        stemed = [token if emoticon_re.search(token) else token.lower() for token in stemed]
    return stemed

def preprocessDutch(s, lowercase=False):
    #Remove emoticons, numbers etc. and returns list of cleaned tweets.
    stemmer = SnowballStemmer("dutch")
    noURL = re.sub("(\w+:\/\/\S+)|^rt|http.+?|^\d+\s|\s\d+\s|\s\d+$|\b\d+\b", '',s)
    tokens = tokenize(noURL)
    # print('tokens', tokens)
    stemed = [stemmer.stem(i) for i in tokens]
    # print('stemed',stemed)
    if lowercase:
        stemed = [token if emoticon_re.search(token) else token.lower() for token in stemed]
    return stemed

# totalist = []
#
#
# with open(fname, 'r') as f:
#     for line in f:
#         tweet = json.loads(line)
#         tokens = [term for term in tweet['tweet_text'] if term.startswith(('@', '''#'''))]
#         totalist.extend(tokens)
#         totallist = ' '.join([text for text in tokens])

# print('total list: ', totallist)

##···Define dictionary for co-ocurrence matrix
comeng = defaultdict(lambda : defaultdict(int))
comdutch = defaultdict(lambda : defaultdict(int))
####Count the most frequent terms, bigrams term in the files

with open(fname, 'r') as f:
    count_all_eng = Counter()
    count_all_dutch = Counter()
    count_bigrams_eng = Counter()
    count_bigrams_dutch = Counter()
    count_user_eng = Counter()
    count_user_dutch = Counter()
    count_hashtag_eng = Counter()
    count_hashtag_dutch = Counter()
    for line in f:
        tweet = json.loads(line)
        # Create a list with all the terms
        if (detect_language(tweet['tweet_text']) == 'english'):
            stop = stopwords.words(detect_language(tweet['tweet_text'])) + punctuation + ['RT', 'via']
            terms_all = [term for term in preprocessEnglish(tweet['tweet_text'])if term not in stop and
            not term.startswith(('#', '@'))]
            # terms_stop = [term for term in preprocess(tweet['text']) if term not in stop]
            terms_bigram = bigrams(terms_all)
            terms_hash= [term for term in preprocessEnglish(tweet['tweet_text']) if term.startswith('''#''')]
            # Update the counter
            count_all_eng.update(terms_all)
            count_bigrams_eng.update(terms_bigram)
            count_hashtag_eng.update(terms_hash)
            count_user_eng.update(terms_hash)
            # count_all.update(terms_bigram)
            for i in range(len(terms_all)-1):
                  for j in range(i+1, len(terms_all)):
                      w1, w2 = sorted([terms_all[i], terms_all[j]])
                      if w1 != w2:
                          comeng[w1][w2] += 1
        elif (detect_language(tweet['tweet_text']) == 'dutch'):
            stop = stopwords.words(detect_language(tweet['tweet_text'])) + punctuation + ['RT', 'via']
            terms_all = [term for term in preprocessDutch(tweet['tweet_text'])if term not in stop and
            not term.startswith(('#', '@'))]
            terms_bigram = bigrams(terms_all)
            terms_hash= [term for term in preprocessDutch(tweet['tweet_text']) if term.startswith('''#''')]
            # Update the counter
            count_all_dutch.update(terms_all)
            count_bigrams_dutch.update(terms_bigram)
            count_hashtag_dutch.update(terms_hash)
            count_user_dutch.update(terms_hash)
            for i in range(len(terms_all)-1):
                  for j in range(i+1, len(terms_all)):
                      w1, w2 = sorted([terms_all[i], terms_all[j]])
                      if w1 != w2:
                          comdutch[w1][w2] += 1
    # Print the first 5 most frequent words
    print('Most common terms english: ',count_all_eng.most_common(10))
    print('Most common bigrams english: ',count_bigrams_eng.most_common(10))
    print('Hashtags counter english: ',count_hashtag_eng.most_common(10))
    print('Most common terms dutch: ',count_all_dutch.most_common(10))
    print('Most common bigrams dutch: ',count_bigrams_dutch.most_common(10))
    print('Hashtags counter dutch: ',count_hashtag_dutch.most_common(10))

    word_freq = count_all_eng.most_common(10)
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    bar.to_json('term_freq.json')

    word_freq = count_all_dutch.most_common(10)
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    bar.to_json('term_freq_dutch.json')

    word_freq = count_hashtag_eng.most_common(10)
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    bar.to_json('hashtag_freq.json')

    word_freq = count_hashtag_dutch.most_common(10)
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    bar.to_json('hashtag_freq_dutch.json')




#######Co-ocurrence matrix english

com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in comeng:
    t1_max_terms = sorted(comeng[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print('Co-ocurrence matrix english: ',terms_max[:6])

#######Co-ocurrence matrix dutch

com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in comdutch:
    t1_max_terms = sorted(comdutch[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print('Co-ocurrence matrix dutch: ',terms_max[:6])

######Co-ocurrence with one particular word

search_word = 'lekker' # pass a term as a command-line argument
count_search = Counter()
with open (fname, 'r') as f:
    for line in f:
        tweet = json.loads(line)
        terms_only = [term for term in preprocessEnglish(tweet['tweet_text'])
                      if term not in stop
                      and not term.startswith(('#', '@'))]
        if search_word in terms_only:
            count_search.update(terms_only)
    print("Co-occurrence for %s:" % search_word)
    print(count_search.most_common(15))


TRACK = 'teek,teken,tekenbeet,tekenbeten,lyme, \
    camping,wandeling, fietstocht,lopen, \
    kamperen,wandelen,fietsen,spelen'
