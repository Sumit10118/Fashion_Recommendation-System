import os
from flask import Flask, request, render_template, send_from_directory
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

app = Flask(__name__)

# Load feature list and filenames
feature_list = np.array(pickle.load(open('pickle files/final_combined_features.pkl', 'rb')))
filenames = pickle.load(open('pickle files/filenames.pkl', 'rb'))



# Define the absolute path to your images folder
IMAGES_FOLDER_PATH = "C:/Users/amitk/OneDrive/Desktop/WEBSITE/Fashion_Recommendation-System/recommend_images"

# Load pre-trained model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        upload_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(upload_path)
        return upload_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

# Function to extract features from the uploaded image
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to get recommended images
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    
    return indices

    # Assuming 'filenames' is your list of filenames



# Route for the Home Page
@app.route('/')
def home_page():
    return render_template('index.html')

# Route for the Model Page
@app.route('/model', methods=['GET', 'POST'])
def model_page():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename != '':
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                try:
                    # Extract features and get recommendations
                    features = feature_extraction(file_path, model)
                    indices = recommend(features, feature_list)
                    
                    

                    # Construct absolute paths for recommended images
                    recommended_images = [
                        f'/recommend_images/{filenames[index]}'
                        for index in indices[0]
                    ]

                      # Debugging line

                    
                    return render_template(
                        'index.html',
                        uploaded_image=f'/uploads/{uploaded_file.filename}',
                        recommendations=recommended_images
                    )
                except Exception as e:
                    print(f"Error during feature extraction or recommendation: {e}")
                    return render_template('index.html', error="Error processing the image.")
        else:
            return render_template('index.html', error="No file uploaded.")
    return render_template('index.html')

# Route to serve uploaded images
@app.route('/uploads/<path:filename>')
def send_uploaded_image(filename):
    return send_from_directory('uploads', filename)

# Route to serve images from the images folder
@app.route('/recommend_images/<path:filename>')
def send_recommend_image(filename):
    return send_from_directory(IMAGES_FOLDER_PATH, filename)


if __name__ == '__main__':
    app.run(debug=True)
