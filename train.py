import pandas as pd
from sklearn.naive_bayes import GaussianNB

import pickle

# Reading data from file
df = pd.read_csv('./data/data.csv')

features = df[['test_participation',
               'chats_forum_interactions', 'introspection']]

predictions = df['active']

# Import Gaussian Naive Bayes model

# Create a Gaussian Classifier
model = GaussianNB()
print('Training the model ...')
# Train the model using the training sets
model.fit(features, predictions)
print('Training complete ....')

print('Saving model to ./model/ActiveVsReflective.model ....')
pickle.dump(model, open('./model/ActiveVsReflective.model', 'wb'))
print('done ...')
