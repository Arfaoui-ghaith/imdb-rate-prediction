from Preprocessing import Preprocessing
from Naive_bayes import Naive_bayes
from KNN import KNeighbors
from Tree import Tree
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import os

app = Flask(__name__)
api = Api(app)

app.static_folder = 'static'
@app.route('/')
def home():
    return render_template('index.html')

class ImdbPredictRate(Resource):
    
    def post(self,algorithm):
        data = request.json
        return jsonify(data)

api.add_resource(ImdbPredictRate, '/<string:algorithm>')

if __name__ == '__main__':
    app.run(debug=True)
"""
p = Preprocessing()
n = Tree(p.getData())
r = n.predict([2012,102,10,2000])
print(str(r[0]*100)[:5],"% ----- ","more than 7.0" if int(r[1])>0 else "equel or less then 7.0" )

"""
