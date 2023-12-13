from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from google.cloud import translate_v2, texttospeech

app = Flask(__name__)

# Inisialisasi klien untuk layanan Google Cloud Translation dan Text-to-Speech
translate_client = translate_v2.Client()
text_to_speech_client = texttospeech.TextToSpeechClient()

def preprocess_image(img_array):
    # Ubah ukuran gambar sesuai dengan kebutuhan model
    target_size = (224, 224)  # Ganti sesuai dengan ukuran input model
    img = Image.fromarray(img_array)
    img = img.resize(target_size, Image.ANTIALIAS)

    # Normalisasi nilai pixel ke rentang 0-1
    img_array = np.array(img) / 255.0

    # Sesuaikan dimensi gambar sesuai dengan kebutuhan model
    img_array = np.expand_dims(img_array, axis=0)

    # Ganti format data sesuai dengan kebutuhan model
    img_array = img_array.astype(np.float32)

    return img_array

def postprocess_output(output_data):
    translation_result = output_data[0]
    return translation_result

def text_to_speech(text):
    # Terjemahkan teks ke suara menggunakan Google Cloud Text-to-Speech
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        name='en-US-Wavenet-D',
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = text_to_speech_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content

@app.route('/api/detect_sign_language', methods=['POST'])
def detect_sign_language():
    # Ambil gambar dari permintaan POST
    image = request.files.get('image')

    # Proses gambar menggunakan model TensorFlow Lite
    input_data = preprocess_image(np.array(Image.open(image)))

    interpreter = tf.lite.Interpreter(model_path='path/to/your/model.tflite')
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    # Terjemahkan hasil deteksi ke dalam teks atau format lainnya
    translation_result = postprocess_output(output_data)

    # Kirim hasil terjemahan sebagai respons JSON
    return jsonify({"result": translation_result})

if __name__ == '__main__':
    app.run(debug=True)
