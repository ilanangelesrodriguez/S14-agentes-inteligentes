from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os
from io import BytesIO
import tensorflow as tf

app = Flask(__name__)

# Configuración de la carpeta para almacenar imágenes temporales
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Ruta para mostrar la página
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para el entrenamiento de la red neuronal (como la tenías)
@app.route('/train', methods=['POST'])
def train():
    # Código de entrenamiento (similar al que tenías)
    # Aquí puedes mantener la lógica de tu red neuronal
    return jsonify({'accuracy': '95%', 'accuracies': [90, 92, 93, 94, 95]})

# Ruta para la clasificación de imágenes
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Cargar la imagen
    image = Image.open(file)
    image = image.resize((32, 32))  # Ajustar tamaño para la red neuronal
    image_array = np.array(image) / 255.0  # Normalización de la imagen
    image_array = image_array.reshape((1, 32, 32, 3))  # Ajustar el formato de entrada

    # Cargar el modelo de red neuronal preentrenado (esto debe ser reemplazado por tu modelo real)
    # Aquí estamos utilizando un modelo ficticio solo para simular
    # Puedes cargar tu modelo con algo como `model = tf.keras.models.load_model('tu_modelo.h5')`

    # Simulando una predicción con valores aleatorios
    prediction = np.random.choice(['Setosa', 'Versicolor', 'Virginica'])
    
    # Simulando la precisión de la predicción (esto puede ser calculado por tu modelo)
    accuracy = np.random.uniform(85, 99)  # Precisión aleatoria entre 85% y 99%

    return jsonify({'prediction': prediction, 'accuracy': f'{accuracy:.2f}%'})

if __name__ == '__main__':
    app.run(debug=True , port=5002)
