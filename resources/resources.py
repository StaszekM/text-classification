from flask_jsonpify import jsonify
from flask_restful import Resource
from flask import request

from modelAccess.model_loader import get_model

print('Retrieving model...')
model, get_most_influential_vars = get_model()
print('Model loaded.')


def get_description(prediction):
    if prediction < 0.45:
        return "The review is negative."
    if 0.45 <= prediction <= 0.55:
        return "The model gave an inconclusive answer - the review is neither positive nor negative"
    return "The review is positive."


class Status(Resource):
    def get(self):
        return {"version": 0.2, "running": True}


class Prediction(Resource):
    def get(self):
        if not request.is_json:
            return {"message": "No JSON body was provided in the request."}, 422

        json = request.get_json()
        review = json.get("review")
        if review is None or type(review) != str:
            return {"message": "Expected \"review\" field with a string value in body."}, 422

        result = model.predict([review])[0][0]
        most_influential_vars = get_most_influential_vars(review, result > 0.5)

        return jsonify(result=result.item(), mostInfulentialVars=most_influential_vars,
                       description=get_description(result))
