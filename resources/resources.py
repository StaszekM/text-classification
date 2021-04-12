from flask_jsonpify import jsonify
from flask_restful import Resource
from flask import request

from modelAccess.model_loader import get_model

model, get_most_influential_vars = get_model()


class Status(Resource):
    def get(self):
        return {"version": 0.1, "running": True}


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

        return jsonify(result=result.item(), mostInfulentialVars=most_influential_vars)
