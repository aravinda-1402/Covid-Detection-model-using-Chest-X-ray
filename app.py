from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(__name__)
model = load_model('covid_detection_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']

        # Read the file stream into BytesIO object
        file_stream = BytesIO(file.read())

        # Load and preprocess the image
        img = image.load_img(file_stream, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Make prediction
        prediction = model.predict(img_array)

        # Get the class label
        if prediction[0][0] > 0.5:
            result = 'Normal'
        elif prediction[0][1] > 0.5:
            result = 'Pneumonia'
        else:
            result = 'COVID-19'

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
