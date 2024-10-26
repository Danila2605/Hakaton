from flask import Flask, request, render_template, jsonify
from main import predict

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    question = request.args.get('question')
    return jsonify({'answer': predict(question)})

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)