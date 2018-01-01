import numpy as np
import tensorflow as tf
from sklearn import model_selection


# load data and split data
raw = np.loadtxt('diabete.data', delimiter=',')
X, y = raw[:, :8], raw[:, -1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=11)

# input 8 features to a DNN with 3 hidden layers
feature_columns = [tf.feature_column.numeric_column('x', shape=[8])]
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[16, 32, 16], n_classes=2,
                                        model_dir='tmp/diabete_dnn')

# training and evaluation
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': X_train}, y=y_train, num_epochs=None, shuffle=True)
classifier.train(input_fn=train_input_fn, steps=2000)
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': X_test}, y=y_test, num_epochs=1, shuffle=False)
accuracy_score = classifier.evaluate(input_fn=test_input_fn)['accuracy']

print('\nTest Accuracy: %.2f%%' % accuracy_score)