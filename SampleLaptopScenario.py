import os
import sys
###establish the files path
os.chdir('/home/goldroger/Documents/Thesis/RicCode/PostGreConnection')
import json
import operator
import imp
import fileinput
import string
import re
from collections import Counter
from collections import defaultdict
import langid
from nltk import bigrams
from gensim import corpora, models
from gensim.test.utils import common_corpus
# Import function scripts
from connectElephant import connect
from Leviathan import stopwords
from Leviathan import preprocessEnglish
from Leviathan import preprocessDutch
from PIL import Image
import matplotlib.pyplot as plt

###Connect to the database and extract the file
connect()
####Load the tile or bucket
fname = 'completefile.txt'
########## List of tweets per users and parameters to ignore tweets
with open(fname, 'r') as f:
    count_users = Counter()
    combinedTweet = Counter()
    #Create list
    for line in f:
        tweet = json.loads(line)
        # Claassify the language
        user = [(tweet['user_id'])]
        # Create a list with all the terms
        count_users.update(user)
        #The v represent the tweet limit per user
        d = dict((k, v) for k, v in count_users.items() if v >= 3000)
        c = list(d.keys())
        mix = [(tweet['user_id'],tweet['tweet_text'])]
        combinedTweet.update(mix)
        #The v represent the repeated tweets per user allowed
        e = dict((k, v) for k, v in combinedTweet.items() if v >= 3)
        f = [i[0] for i in list(e.keys())]
    g = list(set(c)|set(f))

#####Source:https://marcobonzanini.com/2015/03/17/mining-twitter-data-with-python-part-3-term-frequencies/

###list punctuation
punctuation = list('__'+'️'+'•'+'—'+'…'+'–' + string.punctuation)

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
    countmissed = 0
    englishcounter = 0
    dutchcounter = 0
    # load per line
    for line in f:
        tweet = json.loads(line)
        # Claassify the language
        lang = langid.classify(tweet['tweet_text'])
        # Skip possible bots
        if (tweet['user_id']) in g:
            # print(tweet['user_id'],tweet['tweet_text'])
            continue
        # Create a list with all the terms
        elif (lang[0] == 'en'):
            stop = stopwords.words('english') + punctuation + ['RT', 'via']
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
            englishcounter+=1
            # count_all.update(terms_bigram)
            for i in range(len(terms_all)-1):
                  for j in range(i+1, len(terms_all)):
                      w1, w2 = sorted([terms_all[i], terms_all[j]])
                      if w1 != w2:
                          comeng[w1][w2] += 1
        elif (lang[0] == 'nl'):
            stop = stopwords.words('dutch') + punctuation + ['RT', 'via']
            terms_all = [term for term in preprocessDutch(tweet['tweet_text'])if term not in stop and
            not term.startswith(('#', '@'))]
            terms_bigram = bigrams(terms_all)
            terms_hash= [term for term in preprocessDutch(tweet['tweet_text']) if term.startswith('''#''')]
            # Update the counter
            count_all_dutch.update(terms_all)
            count_bigrams_dutch.update(terms_bigram)
            count_hashtag_dutch.update(terms_hash)
            count_user_dutch.update(terms_hash)
            dutchcounter+=1
            for i in range(len(terms_all)-1):
                  for j in range(i+1, len(terms_all)):
                      w1, w2 = sorted([terms_all[i], terms_all[j]])
                      if w1 != w2:
                          comdutch[w1][w2] += 1
        else:
            countmissed+=1
    # Print the first 5 most frequent words
    print('English Tweets: ',englishcounter)
    print('Dutch Tweets: ',dutchcounter)
    print('Language not identified: ',countmissed)
    print('Most common terms english: ',count_all_eng.most_common(3))
    print('Most common bigrams english: ',count_bigrams_eng.most_common(3))
    print('Hashtags counter english: ',count_hashtag_eng.most_common(3))
    print('Most common terms dutch: ',count_all_dutch.most_common(3))
    print('Most common bigrams dutch: ',count_bigrams_dutch.most_common(3))
    print('Hashtags counter dutch: ',count_hashtag_dutch.most_common(3))

#######Co-ocurrence matrix english

com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in comeng:
    t1_max_terms = sorted(comeng[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print('Co-ocurrence matrix english: ',terms_max[:5])

#######Co-ocurrence matrix dutch

com_max = []
# For each term, look for the most common co-occurrent terms
for t1 in comdutch:
    t1_max_terms = sorted(comdutch[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
# Get the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print('Co-ocurrence matrix dutch: ',terms_max[:5])

#####Co-ocurrence with one particular word
terms = ['tek', 'tekenbet', 'lym', 'camping', 'wandel', 'fietstocht', 'lop', 'kamper', 'fiets', 'spel']

for i in terms:
    search_word = i # pass a term as a command-line argument
    count_search = Counter()
    with open (fname, 'r') as f:
        for line in f:
                tweet = json.loads(line)
                lang = langid.classify(tweet['tweet_text'])
                # Create a list with all the terms
                if (lang[0] == 'nl'):
                    terms_only = [term for term in preprocessDutch(tweet['tweet_text'])
                                  if term not in stop
                                  and not term.startswith(('#', '@'))]
                    if search_word in terms_only:
                        count_search.update(terms_only)
        print("Co-occurrence for %s:" % search_word)
        print(count_search.most_common(5))

# pass a term as a command-line argument
with open (fname, 'r') as f:
    count_search = {}
    Track = ['tek', 'tekenbet', 'lym', 'camping', 'wandel', 'fietstocht', 'lop', 'kamper', 'fiets', 'spel']
    for line in f:
        tweet = json.loads(line)
        stop = stopwords.words('dutch') + punctuation + ['RT', 'via']
        terms_only = [term for term in preprocessDutch(tweet['tweet_text'])
                      if term not in stop
                      and not term.startswith(('#', '@'))]
        # print(terms_only)
        #count_search = {}
        for i in Track:
            if i in count_search:
                count_search[i] = float(count_search[i]+terms_only.count(i))
            else:
                count_search[i]=terms_only.count(i)
    print(count_search)


plt.bar(range(len(count_search)), list(count_search.values()), align='center')
plt.xticks(range(len(count_search)), list(count_search.keys()),fontsize=12, rotation=90)
plt.title('Queried dutch words mentions',fontsize=30)
# plt.xlabel('Words',fontsize=20)
plt.ylabel('Mentions',fontsize=20)

plt.show()

labels = 'English', 'Dutch', 'Not Detected'
Totaltweets = englishcounter+dutchcounter+countmissed
sizes = [(englishcounter/Totaltweets*100),(dutchcounter/Totaltweets*100),(countmissed/Totaltweets*100)]
print(sizes)
explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,textprops={'fontsize': 17})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
