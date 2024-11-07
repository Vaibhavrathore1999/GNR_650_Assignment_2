import torch, warnings
from torch import nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import os
from tqdm import tqdm
# import pandas as pd
import pickle
# import pillow_avif
import ast
from scipy import io
from scipy.io import savemat
import numpy as np

warnings.filterwarnings("ignore", message="xFormers is not available")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained ViT-B/16 model with DINO weights from torch hub
def load_vit_dino():
    # model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14',pretrained=True)
    model.head = nn.Identity()  # Remove the classification head
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# Function to process the image and get embeddings
def get_image_embedding(model, image_path):
    # Define the transformation for the image (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to the input size of ViT-L/14
        transforms.ToTensor(),          # Convert the image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )   # Normalize with ImageNet mean and std
    ])

    try:
        # Load image from the path
        img = Image.open(image_path)
        # If the image is palette-based (like GIF/PNG) and has transparency, convert it to RGBA
        if img.mode == "P" or img.mode == "LA" or (img.mode == "RGBA" and "transparency" in img.info):
            img = img.convert("RGBA")
        img = img.convert('RGB')  # Ensure 3-channel image (RGB)
        img_t = transform(img)  # Apply transformations
        img_t = img_t.unsqueeze(0)  # Add batch dimension

        # Get the embedding from the ViT model
        with torch.no_grad():
            embedding = model(img_t.to(device))

        return embedding

    except UnidentifiedImageError:
        print(f"Unsupported or invalid image format: {image_path}")
        return torch.zeros(1, 1536).to(device)

# Save the embeddings to a pickle file
def save_embeddings(embeddings, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)

# Example usage
if __name__ == "__main__":
    # csv_file = "question_image_dict.csv"  # Path to your CSV file
    model = load_vit_dino()
    # output_file = "/home/ninad/vaibhav_r/GNR_650/Assignment_2/IAB-GZSL/data/AWA2/full_image_embeddings.pkl"  # Path to save the image embeddings
    # Load the .mat file
    root_path="/home/ninad/vaibhav_r/GNR_650/Assignment_2/IAB-GZSL/AwA2"
    data = io.loadmat('/home/ninad/vaibhav_r/GNR_650/Assignment_2/IAB-GZSL/data/AWA2/res101.mat')
    dummy_feature_array=np.zeros((1536,37322))
    feature_dict = {k:v for k,v in data.items()}
    for i,image_path in tqdm(enumerate(feature_dict["image_files"][:,0]), total=feature_dict["image_files"].shape[0]):
        feature=get_image_embedding(model,os.path.join(root_path,image_path[0].split("//")[-1])).to('cpu').numpy()
        dummy_feature_array[:,i]=(feature)
    feature_dict["features"]=dummy_feature_array
    savemat("IAB-GZSL/data/AWA2/vitG14.mat", feature_dict)
