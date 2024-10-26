from flask import Flask, request, render_template, jsonify
from main import predict
from flask import send_from_directory

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    question = request.args.get('question')
    return jsonify({'answer': predict(question)})

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    app.run(debug=True)