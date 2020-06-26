from hw07_entity_types import utils

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def train_evaluate_type_prediction(embeddings_file, train_file, test_file):
    # Reading word vectors
    word_vectors, word_to_id = utils.read_word2vec_file(embeddings_file)
    # Reading train data
    x_train, y_train, type_to_id = utils.read_entity_types_file(train_file, word_vectors, word_to_id)
    # Reading test data
    x_test, y_test, _ = utils.read_entity_types_file(test_file, word_vectors, word_to_id, type_to_id)

    # Training
    classifier = None # TODO: Exercise 3

    # Prediction
    prediction_test = None # TODO

    return 0, 0, 0 # TODO: Remove after initializing 'classifier' and 'prediction_test'

    # Total number of positives
    relevant = y_test.sum()

    # True Positives and false positives
    predicted = prediction_test.sum()

    # True Positive.
    relevant_predicted = 0 # TODO

    precision = 0 # TODO
    recall = 0 # TODO
    f_score = 0 # TODO
    return precision, recall, f_score


'''
Tip to practice F1-measure (used everywhere, important!) use 'todense()'
and calculate precision, recall and F1 score by hand.

gold labels:
------------

[[1 0 0]
 [1 1 0]
 [0 0 1]
 [0 0 1]
 [0 0 1]]

relevant: ?

predicted labels:
-----------------

[[1 0 0]
 [0 0 1]
 [0 0 1]
 [0 0 1]
 [1 0 0]]

predicted: ?


gold & prediction comparison (gold multiply predicted):
-----------------------------

[[1 0 0]
 [0 0 0]
 [0 0 1]
 [0 0 1]
 [0 0 0]]

true positives (relevant+predicted, match?): ?

'''
