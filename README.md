
# Fashion Recommendation System

This is a Fashion Recommendation System that suggests similar fashion items based on an input image. The system uses a pre-trained ResNet50 model to extract image features and leverages the Nearest Neighbors algorithm for image similarity recommendations.

## Project Overview
This project is designed to help users discover fashion items that are similar to a given input image. By leveraging the power of deep learning and image feature extraction, the system finds and recommends visually similar items from a pre-defined set of images.

## Datasets
- The dataset used in this project was obtained from Kaggle's **Fashion Product Images Dataset**. 
- You can find the dataset here: [Fashion Product Images Dataset on Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

## Project Structure
```
├── images/                         # Folder containing all the fashion item images
├── pickle files/                   # Folder containing precomputed features and filenames
│   ├── final_combined_features.pkl # Pickle file with extracted image features
│   ├── filenames.pkl               # Pickle file with image filenames
├── uploads/                        # Folder to store uploaded images
├── app.py                          # Flask app for serving the web interface
├── extract_features.py             # Script for feature extraction and similarity computation
├── static/                         # Static files for the web app (CSS, JS, images)
├── templates/
│   ├── index.html                  # HTML template for the web interface
├── README.md                       # Project documentation
├── requirements.txt                # List of dependencies
```

## Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fashion-recommendation-system.git
   cd fashion-recommendation-system
   ```

2. **Install dependencies**:
   Use the `requirements.txt` file to install the necessary Python packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   - Download the **Fashion Product Images Dataset** from Kaggle and place the images in the `images/` folder.

4. **Run the Flask App**:
   ```bash
   python app.py
   ```
   - The app will be available at `http://127.0.0.1:5000`.

## Usage
- Open your browser and go to `http://127.0.0.1:5000`.
- Upload an image of a fashion item to get similar recommendations.

## Feature Extraction
- The project uses a **ResNet50** model pre-trained on ImageNet for feature extraction.
- Extracted features are stored in a normalized form and used to find similar images using the **Nearest Neighbors** algorithm.

## Dependencies
See `requirements.txt` for a full list of packages required for this project.

## Acknowledgements
- **Dataset**: Thanks to [Param Aggarwal](https://www.kaggle.com/paramaggarwal) for the Fashion Product Images Dataset on Kaggle.
- **Model**: This project utilizes TensorFlow's pre-trained **ResNet50** model.

## License
This project is open source and available under the [MIT License](LICENSE).


