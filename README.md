import cv2
import numpy as np
import pandas as pd
import os
import exifread
import glob

# Define categories
ages = ['0-18', '19-30', '31-50', '51+']
genders = ['Male', 'Female']
hair_colors = ['Black', 'Brown', 'Blonde', 'Red', 'Gray']
mask_types = ['No Mask', 'Surgical Mask', 'N95 Mask']

# Create a pandas dataframe to store the dataset
df = pd.DataFrame(columns=['Image Path', 'Age', 'Gender', 'Hair Color', 'Mask', 'Image Make', 'Image Model', 'EXIF DateTimeOriginal', 'EXIF ExifImageWidth', 'EXIF ExifImageLength', 'Location'])

# Function to extract features from an image
def extract_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml'))
        if face_cascade.empty():
            print(f"Error loading Haar cascade file for image: {image_path}")
            return None
        
        faces = face_cascade.detectMultiScale(gray)
        if len(faces) == 0:
            print(f"No faces detected in: {image_path}")
            return None
        
        # Age
        age = ages[np.random.randint(0, len(ages))]  # Random age for demonstration
        
        # Gender
        gender = genders[np.random.randint(0, len(genders))]  # Random gender for demonstration
        
        # Hair Color
        hair_color = hair_colors[np.random.randint(0, len(hair_colors))]  # Random hair color for demonstration
        
        # Mask
        mask = mask_types[np.random.randint(0, len(mask_types))]  # Random mask type for demonstration
        
        return {
            'Image Path': image_path,
            'Age': age,
            'Gender': gender,
            'Hair Color': hair_color,
            'Mask': mask
        }
    
    except cv2.error as e:
        print(f"OpenCV error processing {image_path}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to get all image file paths in the given folder
def get_image_paths(folder):
    image_paths = glob.glob(folder + "/*.jpg") + glob.glob(folder + "/*.png")
    print(f"Found {len(image_paths)} images")  # Debugging statement
    return image_paths

# List to store metadata dictionaries for each image
metadata_list = []

# Specify the columns you want to extract
desired_columns = [
    'Image Make', 'Image Model', 'EXIF DateTimeOriginal', 'EXIF ExifImageWidth', 'EXIF ExifImageLength'
]

# Specify the folder containing the images
folder_path = r'C:/Users/vaibh/OneDrive/Desktop/metadata_extraction_dataset'

# Iterate through each image file in the folder
for img_path in get_image_paths(folder_path):
    print(f"Processing image: {img_path}")  # Debugging statement
    features = extract_features(img_path)
    if features is not None:
        with open(img_path, 'rb') as image_file:
            # Return Exif tags
            tags = exifread.process_file(image_file)
        
        # Create a dictionary to store the selected metadata
        info_dict = features
        
        for tag in desired_columns:
            if tag in tags:
                info_dict[tag] = str(tags[tag])
                print(f"Found {tag}: {tags[tag]}")  # Debugging statement
            else:
                info_dict[tag] = None  # Fill with None if the tag is not available
                print(f"{tag} not found")  # Debugging statement
        
        # Add location
        info_dict['Location'] = folder_path
        
        # Append the metadata dictionary to the list
        metadata_list.append(info_dict)

# Create a DataFrame from the list of metadata dictionaries
metadata_df = pd.DataFrame(metadata_list)

# Display the DataFrame to the user
print(metadata_df)

# Save the DataFrame to a CSV file
csv_filename = 'combined_image_metadata.csv'
metadata_df.to_csv(csv_filename, index=False)
