import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import re
from tqdm import tqdm
from time import time

# ################################# skip this chunk if you have cleaned data
# # # data clean
# raw_data = pd.read_csv('8k_labeled.csv')
# text = raw_data['text']
# processed_features = []

# for sentence in tqdm(range(0, len(text))):
#     # Remove all the special characters
#     processed_feature = re.sub(r'\W', ' ', str(text[sentence]))
#     # Converting to Lowercase
#     processed_feature = processed_feature.lower()
#     processed_features.append(processed_feature)
# df_text = pd.DataFrame({'text': processed_features})
# df_text.to_csv('processed_text.csv')
#######################################
# # vectorize
text = pd.read_csv('processed_text.csv') # load cleaned data
LM_dict = pd.read_csv('LM_dict.csv',keep_default_na=False)
#create vocab from LM dict
vocab = dict()
for index, row in LM_dict.iterrows():
    word = row['Word'].lower()
    vocab[word] = index
pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
                 ('tfid', TfidfTransformer())]).fit(text['text'])# tfidf on 86000 vocab
X = pipe.transform(text['text'])
X.shape
sort_index = np.argsort(X.toarray().sum(axis = 0))# ascending
new_vocab = pipe['count'].get_feature_names_out()[sort_index[-500:]]#take top 500 vocab

#############################################
# # training with new vocab list
vectorizer = TfidfVectorizer()
y = pd.read_csv('8k_labeled.csv')['label']
X_train, X_test, y_train, y_test = train_test_split(text['text'], y, test_size=0.25, random_state=42)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=1).fit(X_train, y_train)
print("test score: ", clf.score(X_test, y_test))
print("train score: ",clf.score(X_train, y_train))

##########################################
# # Filter in the 500 vocab from text

# # In[12]:


# text = text['text']


# # In[15]:


# vocab_set = set(new_vocab)


# # In[24]:


# temp = []
# start = time()
# for string in tqdm(text):
#     split = string.split()
#     filtered = [i for i in split if i in vocab_set]
# print(time()-start)

