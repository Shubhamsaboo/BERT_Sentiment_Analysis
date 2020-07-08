import tokenization
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 8
VALIDATION_SPLIT = 0.20

TRAINING_FILE = "data/IMDB.csv"

BERT_PATH = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

MODEL_JSON = "model/model_sentiment.json"
MODEL_WEIGHT = "model/model_sentiment.h5"

# setting up the "EarlyStopping" callback
EARLY_STOP = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=5,
                           verbose=True,
                           mode='auto',
                           baseline=None,
                           restore_best_weights=False)
CALLBACKS = [EARLY_STOP]

bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# Creating Tokenizer (do lower case = False)
TOKENIZER = tokenization.FullTokenizer(vocab_file, False)

# Sentiment Dictionary
mapping = {
    1: 'positive',
    0: 'negative',
}
