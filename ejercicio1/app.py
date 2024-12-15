from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Cargar y procesar datos
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y).toarray()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Inicializar pesos y sesgos
np.random.seed(42)
weights_hidden1 = np.random.rand(4, 16)
bias_hidden1 = np.random.rand(1, 16)
weights_hidden2 = np.random.rand(16, 10)
bias_hidden2 = np.random.rand(1, 10)
weights_output = np.random.rand(10, 3)
bias_output = np.random.rand(1, 3)

# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Propagación hacia adelante
def forward_propagation(X, weights_hidden1, bias_hidden1, weights_hidden2, bias_hidden2, weights_output, bias_output, activation_func):
    if activation_func == 'sigmoid':
        activation = sigmoid
    elif activation_func == 'relu':
        activation = relu
    else:
        raise ValueError("Función de activación no soportada.")

    # Primera capa oculta
    z_hidden1 = np.dot(X, weights_hidden1) + bias_hidden1
    a_hidden1 = activation(z_hidden1)

    # Segunda capa oculta
    z_hidden2 = np.dot(a_hidden1, weights_hidden2) + bias_hidden2
    a_hidden2 = activation(z_hidden2)

    # Capa de salida
    z_output = np.dot(a_hidden2, weights_output) + bias_output
    a_output = sigmoid(z_output)

    return a_hidden1, a_hidden2, a_output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global weights_hidden1, bias_hidden1, weights_hidden2, bias_hidden2, weights_output, bias_output
    learning_rate = 0.005
    epochs = int(request.form.get('epochs', 10000))
    activation_func = request.form.get('activation', 'sigmoid')

    accuracies = []  # Lista para almacenar la precisión por época

    for epoch in range(epochs):
        # Propagación hacia adelante
        a_hidden1, a_hidden2, a_output = forward_propagation(
            X_train, weights_hidden1, bias_hidden1, weights_hidden2, bias_hidden2,
            weights_output, bias_output, activation_func
        )

        # Errores
        error_output = y_train - a_output
        if activation_func == 'sigmoid':
            error_hidden2 = np.dot(error_output, weights_output.T) * sigmoid_derivative(a_hidden2)
            error_hidden1 = np.dot(error_hidden2, weights_hidden2.T) * sigmoid_derivative(a_hidden1)
        elif activation_func == 'relu':
            error_hidden2 = np.dot(error_output, weights_output.T) * relu_derivative(a_hidden2)
            error_hidden1 = np.dot(error_hidden2, weights_hidden2.T) * relu_derivative(a_hidden1)

        # Actualizar pesos y sesgos
        weights_output += np.dot(a_hidden2.T, error_output) * learning_rate
        bias_output += np.sum(error_output, axis=0, keepdims=True) * learning_rate
        weights_hidden2 += np.dot(a_hidden1.T, error_hidden2) * learning_rate
        bias_hidden2 += np.sum(error_hidden2, axis=0, keepdims=True) * learning_rate
        weights_hidden1 += np.dot(X_train.T, error_hidden1) * learning_rate
        bias_hidden1 += np.sum(error_hidden1, axis=0, keepdims=True) * learning_rate

        # Calcular precisión
        _, _, y_pred = forward_propagation(
            X_test, weights_hidden1, bias_hidden1, weights_hidden2, bias_hidden2,
            weights_output, bias_output, activation_func
        )
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        accuracies.append(accuracy * 100)

    return jsonify({'accuracy': f'{accuracies[-1]:.2f}%', 'accuracies': accuracies})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
