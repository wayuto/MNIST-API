from flask import Flask, request
from mnist_api import *
from utils import parseImgStr

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home() -> str:
    if matrix := request.form.get('matrix'):
        matrix = parseImgStr(matrix)
        return str(predict(matrix))
    else: 
        return "Request this URL with a POST method to get the response."

if __name__ == '__main__':
    app.run(
            host="0.0.0.0",
            port=5000,
            debug=True)
