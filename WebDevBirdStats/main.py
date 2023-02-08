from flask import Flask, redirect, url_for, render_template, request
import pymysql
import json



app = Flask(__name__)



def mysqlconnect():
    conn = pymysql.connect(
        host='localhost',
        user='dylan',
        password='pw',
        db="nano_detections",
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor)
    cursor = conn.cursor()
    cursor.execute("Select SPECIES, COUNT(*) as COUNT FROM detections GROUP BY species")
    output = cursor.fetchall()

    conn.close()
    return output

@app.route("/", methods=['GET'])
def home():
    return render_template("base.html", json_file=json_file)

@app.route("/pie", methods=['GET'])
def pie():
    return render_template("pie.html", json_file=json_file)

@app.route("/bar", methods=['GET'])
def bar():
    return render_template("bar.html", json_file=json_file)

if __name__ == "__main__":
    output = mysqlconnect()
    print(f"json: {json.dumps(output)}")
    with open('static/json_data.json', 'w') as outfile:
        outfile.write(json.dumps(output, indent=4))
    outfile.close()
    json_file = json.dumps(output, indent=4)
    app.run(debug=True)
