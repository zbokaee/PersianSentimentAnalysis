import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split
import nltk
from keras.models import model_from_json
import numpy as np
import logging
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import multiprocessing
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import gensim
from gensim.models.word2vec import Word2Vec
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import pandas as pd

#----------initialing----------#

max_tweet_length = 30
np.random.seed(1000)
avg_length = 0.0
max_length = 0
nb_epochs = 100
batch_size = 32
nb_epoch = 12
array = []
corpus= []
labels= []
tokenized_corpus = []
use_gpu = True
#----------Config----------#

config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(),
                        inter_op_parallelism_threads=multiprocessing.cpu_count(),
                        allow_soft_placement=True,
                        device_count={'CPU': 1,'GPU': 1 if use_gpu else 0})

session = tf.Session(config=config)
K.set_session(session)
#----------Reading----------#

model_location = 'C:/PycharmProjects/untitled6'
with open('d1.csv', 'r', encoding='utf-8') as df:

    for i, line in enumerate(df):

        if i == 0:

            continue

        parts = line.strip().split(',')

        labels.append(int(parts[1].strip()))

        tweet = parts[3].strip()

        if tweet.startswith('"'):

            tweet = tweet[1:]

        if tweet.endswith('"'):

            tweet = tweet[::-1]

        corpus.append(tweet.strip().lower())

corpus = str(corpus)
print('Corpus size: {}'.format(len(corpus)))
#----------Tokenize----------#

totkenized_sentences = nltk.sent_tokenize(corpus)

for each_senteces in totkenized_sentences:

       words = nltk.tokenize.word_tokenize(each_senteces)

       tokenized_corpus.append(words)
#----------Word2Vector----------#

vector_size = 512
window_size = 10

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
word2vec = Word2Vec(sentences=tokenized_corpus,size=vector_size,window=window_size,min_count=1,negative=20,iter=50,seed=1000,
                    workers=multiprocessing.cpu_count())

word2vec.save('texta.model')
word2vec.wv.save_word2vec_format('text.model.bin', binary=True)
model= gensim.models.KeyedVectors.load_word2vec_format('text.model.bin', binary=True)
X_vecs = word2vec.wv
print('Shape of embedding matrix: ', X_vecs)
#----------Split----------#

x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)

x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

X1 = np.array(x_train)

print("X_train shape: " + str(X1.shape))

X1 = np.array(x_test)

print("X_test shape: " + str(X1.shape))

X1 = np.array(X_val)

print("X_val shape: " + str(X1.shape))

X1 = np.array(y_train)

print("y_train shape: " + str(X1.shape))

X1 = np.array(y_test)

print("y_test shape: " + str(X1.shape))

X1 = np.array(y_val)

print("y_val shape: " + str(X1.shape))

train_size = len(x_train)

test_size = len(x_test)

indexes = set(np.random.choice(len(tokenized_corpus), train_size + test_size,replace=False))

#----------Seperation----------#

X_train = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())

Y_train = np.zeros((train_size, 2), dtype=np.int32)

X_test = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())

Y_test = np.zeros((test_size, 2), dtype=np.int32)

for i, index in enumerate(indexes):

     for t, token in enumerate(tokenized_corpus[index]):
         if t >= max_tweet_length:
             break

         if token not in X_vecs:
             continue

         if i < train_size:
             X_train[i, t, :] = X_vecs[token]


         else:
             X_test[i - train_size, t, :] = X_vecs[token]

     if i < train_size:
         if labels[index] == 0 :
             Y_train[i, :] = [1.0, 0.0]

         elif labels[index] == 1 :
             Y_train[i, :] = [0.0, 1.0]

         else:
             Y_train[i, :] = [1.0, 1.0]
     else:

         if labels[index] == 0 :
             Y_test[i - train_size, :] = [1.0, 0.0]

         elif labels[index] == 1 :
             Y_test[i - train_size, :] = [0.0, 1.0]

         else:
             Y_test[i - train_size, :] = [1.0, 1.0]

#----------Keras convolutional model----------#

model = Sequential()

model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same', input_shape=(max_tweet_length,vector_size)))

model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))

model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))

model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))

model.add(Dropout(0.25))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='tanh'))

model.add(Dense(256, activation='tanh'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

rnn = model.fit(np.array(X_train),np.array(Y_train),
           batch_size=batch_size,
           shuffle=True,
           epochs=nb_epochs,
           validation_data=(np.array(X_test),np.array(Y_test)),
           callbacks=[EarlyStopping(min_delta=0.00025, patience=2)])

print('Save model...')
model.save_weights('model2.h5')

#----------Acc----------#

score = model.evaluate(X_val, y_val)

print("Test Accuracy: %.2f%%" % (score[1]*100))
#---------Saving---------#

model_json = model.to_json()

with open('model.json', 'w') as json_file:

    json_file.write(model_json)

model.save_weights('model3.h5')

print('saved model!')

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

model = model_from_json(loaded_model_json)
#------Precision-Recall------#

y_pred = model.predict(x_test)

yy_true = [np.argmax(i) for i in y_test]

yy_scores = [np.argmax(i) for i in y_pred]

print("Recall: " + str(recall_score(yy_true, yy_scores, average='weighted')))

print("Precision: " + str(precision_score(yy_true, yy_scores, average='weighted')))

print("F1 Score: " + str(f1_score(yy_true, yy_scores, average='weighted')))
#----------confusion_matrix----------#

Y_pred = model.predict(x_test, verbose=2)

y_pred = np.argmax(Y_pred, axis=1)

for ix in range(3):

    print(ix, confusion_matrix(np.argmax(y_test, axis=1), y_pred)[ix].sum())

cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

print(cm)

df_cm = pd.DataFrame(cm, range(3), range(3))

plt.figure(figsize=(10,7))

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=False)

sn.set_context("poster")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.savefig('Plots/confusionMatrix.png')

plt.show()
#------------- ROC Curve------------#

n_classes = 3
lw = 2
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(yy_true))[:, i], np.array(pd.get_dummies(yy_scores))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(np.array(pd.get_dummies(yy_true))[:, i], np.array(pd.get_dummies(yy_scores))[:, i])
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(8,5))
plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]),
color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),
         color='green', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.savefig('Plots/ROCcurve.png')

plt.show()
#----------Graph----------#
plt.figure(0)

plt.plot(rnn.history['acc'],'r')

plt.plot(rnn.history['val_acc'],'g')

plt.xticks(np.arange(0, nb_epoch+1, nb_epoch/5))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy CNN l=10, epochs=20") # for max length = 10 and 20 epochs

plt.legend(['train', 'validation'])

plt.figure(1)

plt.plot(rnn.history['loss'],'r')

plt.plot(rnn.history['val_loss'],'g')

plt.xticks(np.arange(0, nb_epoch+1, nb_epoch/5))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Training vs Validation Loss CNN l=10, epochs=20") # for max length = 10 and 20 epochs

plt.legend(['train', 'validation'])

plt.show()
#----------Predict----------#

labels = ['negative', 'positive','neutral']
final = []

evalSentence = input('Input a sentence to be evaluated, or Enter to quit: ')
words = nltk.word_tokenize(evalSentence)

for word in words:
    if word in X_vecs:
        final.append(X_vecs[words])

X_test = pad_sequences(final,maxlen=max_tweet_length, padding='post')
y_pred=model.predict(X_test)

print("%s sentiment; %f%% confidence" % (labels[np.argmax(y_pred)], y_pred[0][np.argmax(y_pred)] * 100))
