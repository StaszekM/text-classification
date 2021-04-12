from flask import Flask
from flask_restful import Api

from resources.resources import Status, Prediction

app = Flask(__name__)
api = Api(app)

api.add_resource(Status, '/status')
api.add_resource(Prediction, '/predict')