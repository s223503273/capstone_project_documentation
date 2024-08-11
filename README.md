# capstone_project_documentation
##Image Metadata Extraction and Feature Extraction
###Overview
This repository contains a Python script that extracts metadata and features from images in a specified folder. The script uses OpenCV for face detection and feature extraction, and exifread for metadata extraction.
###Features
Extracts metadata from images (Image Make, Image Model, EXIF DateTimeOriginal, EXIF ExifImageWidth, EXIF ExifImageLength)
Detects faces in images using OpenCV's Haar cascade classifier
Extracts features from detected faces (Age, Gender, Hair Color, Mask)
Saves extracted metadata and features to a CSV file
###Requirements
Python 3.8+
OpenCV 4.5+
exifread 2.3+
pandas 1.3+
numpy 1.20+
glob 0.7+
###Usage
Clone the repository to your local machine
Install the required packages using pip: pip install -r requirements.txt
Specify the folder containing the images in the folder_path variable
Run the script using Python: python image_metadata_extraction.py
The extracted metadata and features will be saved to a CSV file named combined_image_metadata.csv
###Functions
extract_features(image_path): Extracts features from an image using OpenCV
get_image_paths(folder): Returns a list of image file paths in the specified folder
process_image(img_path): Extracts metadata and features from an image and returns a dictionary
###Variables
ages, genders, hair_colors, mask_types: Lists of categories for feature extraction
desired_columns: List of metadata columns to extract
folder_path: Path to the folder containing the images
metadata_list: List of metadata dictionaries for each image
###Output
combined_image_metadata.csv: CSV file containing extracted metadata and features
