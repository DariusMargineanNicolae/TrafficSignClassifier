from flask import Flask, render_template, request
import testing


app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    imagefile = request.files['imagefile']
    img_name= imagefile.filename
    image_path="C:\\Users\\Nicolae PC\\PyTorch\\TrafficSigns_Classification\\static\\" + imagefile.filename
    imagefile.save(image_path)

    pred_dict={}
    pred_dict[image_path]=testing.prediction(image_path,testing.transformer)
    text = testing.return_value_of_a_dict(pred_dict, image_path)

    return render_template('index.html', prediction=text, image_name = img_name)



if __name__ == "__main__":
    app.run(port=3000, debug=True)
