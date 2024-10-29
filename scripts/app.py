# app.py
from flask import Flask, request
from mplsandbox import MPLSANDBOX
app = Flask(__name__)

@app.route('/run-app', methods=['POST'])
def run_app():
    data = request.json
    data["app"] = True
    app = MPLSANDBOX(data)
    return app.get_basic_info()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True) 