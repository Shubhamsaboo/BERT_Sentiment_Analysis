import numpy as np
from bert_encode import bert_encode
import tokenization
import json
from tensorflow.keras.models import model_from_json
import tensorflow_hub as hub
from config import BERT_PATH, mapping, MODEL_JSON, MODEL_WEIGHT, MAX_LEN, TOKENIZER

# Loading Model
# load json and create model
json_file = open(MODEL_JSON)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json, custom_objects={'KerasLayer': hub.KerasLayer})

# load weights into new model
model.load_weights(MODEL_WEIGHT)


class Categorize:

    def get(self, text):
        global model
        """
        helper function: check if a text provided would be classified under which technology
        argument: <string> with SMS text to be checked
        if no argument provided, read the user's input
        """

        # tokenize the SMS text and pad sequence to match training sequences length
        text = [text, ]
        sequence = bert_encode(text, TOKENIZER, max_len=MAX_LEN)

        # predict class and give feedback
        prediction = model.predict(sequence)
        pred_class = np.argmax(prediction, axis=None, out=None)

        is_tech = "The technology which it belong to is: " + mapping.get(str(pred_class))

        return is_tech, prediction
