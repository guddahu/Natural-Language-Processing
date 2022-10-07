import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('IMDB Dataset.csv')

df.loc[df['sentiment'] == 'positive', 'sentiment'] = int(1)
df.loc[df['sentiment'] == 'negative', 'sentiment'] = int(0)
df = df[:1000]
df['review']=df['review'].str.lower()
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
X =[]
maxi = df["review"].str.len().max()

for i in df['review']:
    j = tokenizer.encode(i)
    j = j + [0] *(maxi - len(j))
    X.append(j)
#print(X)
X_train,X_test,Y_train, Y_test = train_test_split(X, df['sentiment'], test_size=0.25, random_state=30)    

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0)
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')
clf.fit(X_train,Y_train)

y_test_pred=clf.predict(X_test)
from sklearn.metrics import classification_report
report=classification_report(Y_test, y_test_pred,output_dict=True)
print(report)