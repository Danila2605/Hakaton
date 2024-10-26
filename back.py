from flask import Flask, request, render_template, jsonify
from main import predict
from flask import send_from_directory

print(predict("I can't connect to the internet"))
print(predict("The software keeps crashing"))
print(predict("How do I use this feature?"))
print(predict("I haven't received any emails"))

app = Flask(__name__)

@app.route('/api/data/<question>', methods=['GET'])
def get_data(question):
    print(question)
    qq = predict(question)
    return jsonify({'answer': qq})

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    app.run(debug=True)