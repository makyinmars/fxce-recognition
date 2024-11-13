from fastai.vision.all import *
import face_recognition
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
import os

class FaceMatcher:
    def __init__(self, model_path: str = 'face_model.pkl'):
        self.model_path = model_path
        self.learn = None
        self.face_encodings_cache = {}

    def train_model(self, training_path: str):
        """Train the FastAI model on a directory of images"""
        path = Path(training_path)
        dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, 
                                         item_tfms=Resize(224))
        self.learn = vision_learner(dls, resnet34, metrics=error_rate)
        self.learn.fine_tune(3)
        self.learn.export(self.model_path)

    def get_face_encoding(self, image_path: str) -> Optional[np.ndarray]:
        """Get face encoding from an image"""
        if image_path in self.face_encodings_cache:
            return self.face_encodings_cache[image_path]

        # Load image
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = face_recognition.load_image_file(BytesIO(response.content))
        else:
            img = face_recognition.load_image_file(image_path)

        # Get face encoding
        face_encodings = face_recognition.face_encodings(img)
        if not face_encodings:
            return None
        
        encoding = face_encodings[0]
        self.face_encodings_cache[image_path] = encoding
        return encoding

    def find_similar_faces(self, 
                          target_image_path: str,
                          candidate_images: List[str],
                          num_results: int = 50,
                          similarity_threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find similar faces in candidate images compared to target image
        Returns: List of tuples (image_path, similarity_score)
        """
        target_encoding = self.get_face_encoding(target_image_path)
        if target_encoding is None:
            raise ValueError("No face found in target image")

        results = []
        for img_path in candidate_images:
            encoding = self.get_face_encoding(img_path)
            if encoding is not None:
                # Calculate face distance (lower = more similar)
                face_distance = face_recognition.face_distance([target_encoding], encoding)[0]
                # Convert distance to similarity score (higher = more similar)
                similarity = 1 - face_distance
                if similarity >= similarity_threshold:
                    results.append((img_path, similarity))

        # Sort by similarity score and return top matches
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_results]

# Example usage:
def main():
    matcher = FaceMatcher()

    
    # Get all PNG files from the facx directory
    facx_dir = Path('facx')
    candidate_images = list(facx_dir.glob('*.jpg'))
    candidate_images = [str(path) for path in candidate_images]
    
    # Define target images (you can modify this list as needed)
    target_images = [
        'facx/4XFCWCM9FC.jpg',
        'facx/Y8XX9VIORW.jpg',
        # Add more target images as needed
    ]
    
    try:
        # Process each target image
        for target_image in target_images:
            print(f"\nFinding matches for target image: {target_image}")
            try:
                similar_faces = matcher.find_similar_faces(target_image, candidate_images)
                print(f"Found {len(similar_faces)} matches:")
                for image_path, similarity in similar_faces:
                    print(f"Image: {image_path}, Similarity: {similarity:.2f}")
            except Exception as e:
                print(f"Error processing {target_image}: {str(e)}")
                continue
    except Exception as e:
        print(f"Error: {str(e)}")
    # except Exception as e:
    #     print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
