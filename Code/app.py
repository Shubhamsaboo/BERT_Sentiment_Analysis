import flask
from flask import Flask
from flask import request
from categorize import Categorize


app = Flask(__name__)

MODEL = None
DEVICE = "cuda"


@app.route('/predict', methods=['GET'])
def get_category():
    text = request.args.get('text', '')
    response = {}
    # Returns a numpy array od size (1,2) [neg_pred_score, pos_pred_score]
    sentiment, negative_prediction = Categorize.get(text)
    negative_prediction = negative_prediction[0][0]
    positive_prediction = 1 - negative_prediction
    response["response"] = {
        'sentence': str(text),
        'positive': str(positive_prediction),
        'negative': str(negative_prediction),
        'sentiment': sentiment
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run()
