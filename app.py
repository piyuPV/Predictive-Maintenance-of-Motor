from flask import Flask,render_template,request
import numpy as np
from Motor_Maintainance import give_pred

app = Flask(__name__)

@app.route("/")
@app.route("/Hello")
def home():
    return render_template("Motor.html")

@app.route("/result",methods=['POST','GET'])
def result():
    outpt= request.form.to_dict()
    current=outpt["current"]
    voltage=outpt['voltage']
    temperature=outpt['temperature']
    humidity=outpt['humidity']
    vibration=outpt['vibration']
    test=np.array([[current,voltage,temperature,humidity,vibration]])
    results=give_pred(test)
    return render_template("Motor.html", name = give_pred(test))


if __name__=='__main__':
    app.run(debug=True,port=5000)