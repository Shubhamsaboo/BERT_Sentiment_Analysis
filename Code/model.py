from config import BERT_PATH, mapping, MODEL_JSON, MODEL_WEIGHT, MAX_LEN
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

# setting model optimization parameters:
optimizer = Adam(lr=2e-6)
loss = 'binary_crossentropy'
metrics = ['accuracy']


def build_model(bert_layer, max_len=MAX_LEN):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, pooled_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = pooled_output[:, 0, :]
    out = Dense(2, activation='softmax')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
