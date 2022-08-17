import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg

app = Flask("mpg_prediction")

@app.route('/', methods=['POST'])

def predict():
    vehicle_config = request.get_json()

    with open('./model_files/model.bin' ,'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    
    predictions = predict_mpg(vehicle_config,model)

    response = {
        'mpg_prediction' : list(predictions)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
