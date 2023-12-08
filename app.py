from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import tensorflow as tf
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

# Load model TensorFlow
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')  # Buat file HTML sesuai kebutuhan

@socketio.on('predict_request')
def handle_predict(data):
    try:
        # Asumsikan data adalah array numerik yang akan digunakan untuk prediksi
        input_data = np.array(data['input'])

        # Lakukan prediksi dengan model
        prediction = model.predict(np.expand_dims(input_data, axis=0))

        # Format hasil prediksi sebagai JSON
        result = {'prediction': prediction.flatten().tolist()}

        # Kirim hasil prediksi ke klien
        emit('prediction_response', result)

    except Exception as e:
        emit('prediction_response', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
