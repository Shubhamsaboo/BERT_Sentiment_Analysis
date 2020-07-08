
import pandas as pd
from tensorflow.keras.utils import to_categorical
from bert_encode import bert_encode
from model import build_model
from config import TRAINING_FILE, TOKENIZER, MAX_LEN, bert_layer, VALIDATION_SPLIT, TRAIN_BATCH_SIZE, EPOCHS,CALLBACKS,\
    MODEL_WEIGHT, MODEL_JSON
from sklearn import model_selection

# Loading the training data
dfx = pd.read_csv(TRAINING_FILE).fillna("none")
dfx.sentiment = dfx.sentiment.apply(lambda x: 1 if x == "positive" else 0)

# Train/Test Split
df_train, df_valid = model_selection.train_test_split(dfx, test_size=0.1, random_state=42, stratify=dfx.sentiment.values)

# Resetting the index
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

# Encoding the training data
train_input = bert_encode(df_train['review'].values, TOKENIZER, max_len=MAX_LEN)

train_labels = to_categorical(df_train['sentiment'])

model = build_model(bert_layer, max_len=MAX_LEN)

model.fit(x=train_input, y=train_labels,
          batch_size=TRAIN_BATCH_SIZE, epochs=EPOCHS,
          verbose=2, callbacks=CALLBACKS,
          validation_split=VALIDATION_SPLIT)

# Saving the trained model
# serialize model to JSON
model_json = model.to_json()
with open(MODEL_JSON, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(MODEL_WEIGHT)
