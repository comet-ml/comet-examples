from __future__ import print_function

from comet_ml import Experiment
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Setting the API key (saved as environment variable)
experiment = Experiment(
    #api_key="YOUR API KEY",
    # or
    api_key=os.environ.get("COMET_API_KEY"),
    project_name='comet-examples')

# Get dataset and put into train,test lists
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(
    subset='test', categories=categories, shuffle=True, random_state=42)

# log hash of your dataset to Comet.ml
experiment.log_dataset_hash(twenty_train)

# Build training pipeline

text_clf = Pipeline([('vect', CountVectorizer()),  # Counts occurrences of each word
                     # Normalize the counts based on document length
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',  # Call classifier with vector
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
                     ])

text_clf.fit(twenty_train.data, twenty_train.target)
#
# Predict unseen test data based on fitted classifer
predicted = text_clf.predict(twenty_test.data)

# Compute accuracy
acc = accuracy_score(twenty_test.target, predicted)
print(acc)
experiment.log_metric(name="accuracy_score", value=acc)
