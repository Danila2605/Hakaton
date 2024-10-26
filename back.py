from flask import Flask, request, render_template, jsonify
from main import predict_all
from flask import send_from_directory

app = Flask(__name__)

@app.route('/api/data/<question>', methods=['GET'])
def get_data(question):
    predicted_service, predicted_solution, predicted_instruction = predict_all(question)
    return jsonify({'predicted_service': predicted_service,'predicted_solution': predicted_solution, 'predicted_instruction': predicted_instruction})

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    app.run(debug=True)