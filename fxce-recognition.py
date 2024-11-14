from fastai.vision.all import *
import torch.nn.functional as F
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
        self.image_embeddings = {}  # Cache for image embeddings

    def _get_embedding(self, image_path: str) -> torch.Tensor:
        """Get embedding vector for an image using ResNet34"""
        if image_path in self.image_embeddings:
            return self.image_embeddings[image_path]
        
        # Load and preprocess image
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = PILImage.create(BytesIO(response.content))
        else:
            img = PILImage.create(image_path)
        
        # Get embedding using the model
        self.learn.model.eval()
        with torch.no_grad():
            embedding = self.learn.model[:-1](self.learn.preprocess(img).unsqueeze(0))
            embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
            
        embedding = embedding.squeeze().cpu()
        self.image_embeddings[image_path] = embedding
        return embedding

    def train_model(self, training_path: str, force_retrain: bool = False):
        """Train the FastAI model on a directory of images"""
        # Check if model already exists
        if os.path.exists(self.model_path) and not force_retrain:
            print("Loading existing model...")
            try:
                self.learn = load_learner(self.model_path)
                return
            except Exception as e:
                print(f"Error loading existing model: {e}")
                print("Will retrain model...")

        print("Training new model...")
        path = Path(training_path)
        dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, 
                                         item_tfms=Resize(224))
        self.learn = vision_learner(dls, resnet34, metrics=error_rate)
        self.learn.fine_tune(3)
        self.learn.export(self.model_path)

    

    def find_similar_faces(self, 
                          target_image_path: str,
                          candidate_images: List[str],
                          num_results: int = 50,
                          similarity_threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find similar faces using ResNet34 embeddings
        Returns: List of tuples (image_path, similarity_score)
        """
        if self.learn is None:
            raise ValueError("Model not trained. Call train_model first.")

        target_embedding = self._get_embedding(target_image_path)
        
        results = []
        for img_path in candidate_images:
            try:
                embedding = self._get_embedding(img_path)
                # Calculate cosine similarity
                similarity = F.cosine_similarity(target_embedding.unsqueeze(0), 
                                              embedding.unsqueeze(0)).item()
                if similarity >= similarity_threshold:
                    results.append((img_path, similarity))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

        # Sort by similarity score and return top matches
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_results]

# Example usage:
def main():
    matcher = FaceMatcher()
    
    # Train the model using images from facx directory
    print("Training model on facx directory images...")
    matcher.train_model('facx')
    print("Training completed!")

    # Get all JPG files from the facx directory
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
