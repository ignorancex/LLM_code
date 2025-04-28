
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, render_template, request, send_file
from Server import analyzePhage
import random

app = Flask(__name__)

@app.route('/')
def render():
    return render_template("index.html")

@app.route("/success", methods=["POST"])
def rank():
    uploaded_file = request.files['file']
    global f
    f = uploaded_file.filename.split(".")[0]
    
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        try:
            analyzePhage(uploaded_file.filename, f + "_ranks")
        except:
            return render_template("error.html")
    return render_template("success.html", name=f)

@app.route("/download", methods=["GET"])
def download():
    filename = f + "_ranks.csv"
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    app.run(use_reloader=False, debug=True, port=port)