import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from flask import Flask, request, jsonify

# Set up the Flask app
app = Flask(__name__)

# Build the generator model (it learns to generate anomaly scores)
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),  # First hidden layer
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),  # Deeper layers for obfuscation
        layers.Dense(1, activation='sigmoid')  # Output: single value (anomaly score)
    ])
    return model

# Build the discriminator model (it evaluates whether the data is real or fake)
def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(1024, activation='relu', input_dim=input_dim),  # First hidden layer
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output: real or fake classification
    ])
    return model

# Build the GAN model combining the generator and the discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze discriminator during generator training
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Hyperparameters
latent_dim = 100  # Latent space dimension
input_dim = 10    # Input data dimensions (e.g., 10 features of vehicle data)
epochs = 1000
batch_size = 64

# Create the generator, discriminator, and GAN models
generator = build_generator(latent_dim)
discriminator = build_discriminator(input_dim)
gan = build_gan(generator, discriminator)

# Compile the models
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Dummy data: Replace with actual vehicle data during deployment
# Here, we simulate vehicle data with random values for anomaly detection
def generate_vehicle_data(batch_size):
    return np.random.rand(batch_size, input_dim)  # Replace with real vehicle data when available

# Define the function to calculate anomaly score using GAN
def generate_anomaly_score(vehicle_data):
    noise = np.random.normal(0, 1, (1, latent_dim))  # Latent noise for generator
    fake_data = generator.predict(noise)
    
    # Use the discriminator to evaluate whether the generated data is anomalous
    anomaly_score = discriminator.predict(fake_data)
    return anomaly_score[0][0]  # Return the anomaly score

# API endpoint to get anomaly score
@app.route('/api/anomaly-score', methods=['POST'])
def get_anomaly_score():
    try:
        vehicle_data = request.json  # Get vehicle data from the POST request
        vehicle_features = np.array(vehicle_data['features']).reshape(1, input_dim)  # Format the input data

        # Get anomaly score based on the vehicle features
        anomaly_score = generate_anomaly_score(vehicle_features)

        # Return the anomaly score in the response
        return jsonify({"score": float(anomaly_score)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9050)
