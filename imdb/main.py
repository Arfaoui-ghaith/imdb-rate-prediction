from Preprocessing import Preprocessing
from Naive_bayes import Naive_bayes
from KNN import KNeighbors
from Tree import Tree
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import os
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

app.static_folder = 'static'
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/genrelist', methods = ['GET'])
def get():
    res = Preprocessing().getGenre()
    i=0
    dic = {}
    while(i<len(res[0])):
        dic[res[1][i]] = str(res[0][i])
        i=i+1
    print(dic)
    return json.dumps(dic)

class ImdbPredictRate(Resource):
    
    def post(self,algorithm,year,duration,genre,votes):
        dataset = Preprocessing()
        if(algorithm == 'KNN'):
            p = KNeighbors(dataset.getData()).predict([int(year),int(duration),int(genre),int(votes)])
            return jsonify(str(p[0]*100)[:5]+"% more than 7.0" if int(p[1])>0 else str(p[0]*100)[:5]+"% equel or less then 7.0")
        
        if(algorithm == 'Naive_bayes'):
            p = Naive_bayes(dataset.getData()).predict([int(year),int(duration),int(genre),int(votes)])
            return jsonify(str(p[0]*100)[:5]+"% more than 7.0" if int(p[1])>0 else str(p[0]*100)[:5]+"% equel or less then 7.0")
        
        if(algorithm == 'Tree'):
            p = Tree(dataset.getData()).predict([int(year),int(duration),int(genre),int(votes)])
            return jsonify(str(p[0]*100)[:5]+"% more than 7.0" if int(p[1])>0 else str(p[0]*100)[:5]+"% equel or less then 7.0")
        return jsonify({ 'message': 'Please Select the right algorithm.' })

api.add_resource(ImdbPredictRate, '/<string:algorithm>/<string:year>/<string:duration>/<string:genre>/<string:votes>')

if __name__ == '__main__':
    app.run(debug=True)
"""
p = Preprocessing()
n = Tree(p.getData())
r = n.predict([2012,102,10,2000])
print(str(r[0]*100)[:5],"% ----- ","more than 7.0" if int(r[1])>0 else "equel or less then 7.0" )

"""
