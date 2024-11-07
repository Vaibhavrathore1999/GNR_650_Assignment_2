# import pickle
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import torch.hub
# from tqdm import tqdm
# import torch.optim as optim
# import wandb

# # Initialize wandb
# wandb.init(project="GNR_650_Assignment_2")  # Replace 'your_username' with your wandb username

# # Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_embeddings(embedding_path):
#     with open(embedding_path, 'rb') as file:
#         embeddings = pickle.load(file)
#     embeddings = {k: torch.tensor(v, dtype=torch.float32) for k, v in embeddings.items()}
#     return embeddings

# def load_pretrained_vit(trainable_blocks=1):
#     model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
#     model.head = nn.Identity()  # Remove the head

#     # Freeze all parameters initially
#     for param in model.parameters():
#         param.requires_grad = False

#     # Count the number of blocks in the ViT model
#     num_blocks = len(model.blocks)

#     # Unfreeze the last 'trainable_blocks' blocks
#     for i in range(num_blocks - trainable_blocks, num_blocks):
#         for param in model.blocks[i].parameters():
#             param.requires_grad = True

#     model.to(device)
#     return model

# def get_trainable_params(model):
#     return [param for param in model.parameters() if param.requires_grad]

# def load_class_names(filename):
#     with open(filename, 'r') as file:
#         class_names = file.read().splitlines()
#     return class_names

# def get_train_transforms():
#     return transforms.Compose([
#         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random cropping and resizing
#         transforms.RandomHorizontalFlip(),  # Horizontal flipping
#         transforms.RandomRotation(15),  # Random rotations
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

# def get_test_transforms():
#     return transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

# class CustomDataset(Dataset):
#     def __init__(self, root_dir, class_names, embeddings, transform=None, is_train=False):
#         self.root_dir = root_dir
#         self.class_names = class_names
#         self.embeddings = embeddings
#         self.transform = transform
#         self.is_train = is_train
#         self.files = [os.path.join(root, f) for root, _, files in os.walk(root_dir) for f in files if f.split('_')[0] in class_names]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         img_path = self.files[idx]
#         image = Image.open(img_path).convert('RGB')
#         label = os.path.basename(os.path.dirname(img_path))
#         if self.is_train:
#             transform = get_train_transforms()
#         else:
#             transform = get_test_transforms()
#         if transform:
#             image = transform(image)
#         return image, label


# class VisualProjectionHead(nn.Module):
#     def __init__(self, input_dim, output_dim=512, hidden_dims=None):
#         super(VisualProjectionHead, self).__init__()
#         if hidden_dims is None:
#             hidden_dims = [1024, 768]  # Example dimensions
#         layers = []
#         last_dim = input_dim
#         for dim in hidden_dims:
#             layers.append(nn.Linear(last_dim, dim))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.BatchNorm1d(dim))
#             last_dim = dim
#         layers.append(nn.Linear(last_dim, output_dim))
#         layers.append(nn.ReLU(inplace=True))  # Optionally add ReLU or other activation to the output
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)

# class SemanticProjectionHead(nn.Module):
#     def __init__(self, input_dim, output_dim=512, hidden_dims=None):
#         super(SemanticProjectionHead, self).__init__()
#         if hidden_dims is None:
#             hidden_dims = [1024, 768]  # Example dimensions, should be tailored
#         layers = []
#         last_dim = input_dim
#         for dim in hidden_dims:
#             layers.append(nn.Linear(last_dim, dim))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.BatchNorm1d(dim))
#             last_dim = dim
#         layers.append(nn.Linear(last_dim, output_dim))
#         layers.append(nn.ReLU(inplace=True))  # Optionally add ReLU or other activation to the output
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)


# def evaluate_model(data_loader, vit_model, visual_projection, semantic_projection, class_embeddings_tensor, device, class_names_list):
#     vit_model.eval()
#     visual_projection.eval()
#     semantic_projection.eval()
#     total_correct = 0
#     total_images = 0
#     with torch.no_grad():
#         with tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") as tepoch:
#             for images, labels in tepoch:
#                 images = images.to(device)
#                 features = vit_model(images)
#                 visual_latent = visual_projection(features)
#                 semantic_indices = [class_names_list.index(label) for label in labels]
#                 semantic_features = class_embeddings_tensor[semantic_indices].to(device)
#                 semantic_latent = semantic_projection(semantic_features)
#                 similarities = F.cosine_similarity(visual_latent.unsqueeze(1), semantic_latent.unsqueeze(0), dim=2)
#                 predicted_indices = similarities.max(dim=1).indices
#                 correct_predictions = (predicted_indices == torch.tensor(semantic_indices, device=device)).sum().item()
#                 total_correct += correct_predictions
#                 total_images += len(labels)
#                 tepoch.set_postfix(accuracy=f"{100 * total_correct / total_images:.2f}%")
#     accuracy = total_correct / total_images
#     return accuracy

# def main():
#     embeddings = load_embeddings('Baseline1/class_embeddings.pkl')
#     vit_model = load_pretrained_vit(trainable_blocks=1)
#     train_classes = load_class_names('Baseline1/AwA2/trainclasses.txt')
#     test_classes = load_class_names('Baseline1/AwA2/testclasses.txt')
#     visual_projection = VisualProjectionHead(768, 512)
#     semantic_projection = SemanticProjectionHead(300, 512)
#     visual_projection.to(device)
#     semantic_projection.to(device)

    
#     # Set up datasets with respective transformations
#     train_dataset = CustomDataset(
#         root_dir='Baseline1/AwA2/JPEGImages',
#         class_names=train_classes,
#         embeddings=embeddings,
#         is_train=True  # Enable augmentation
#     )
#     train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#     test_dataset = CustomDataset(
#         root_dir='Baseline1/AwA2/JPEGImages',
#         class_names=test_classes,
#         embeddings=embeddings,
#         is_train=False  # Disable augmentation for testing
#     )
#     test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#     class_embeddings_tensor = torch.stack([torch.tensor(v, dtype=torch.float32) for v in embeddings.values()]).to(device)
#     class_names_list = list(embeddings.keys())
#     optimizer = optim.SGD(
#     get_trainable_params(vit_model) +
#     list(visual_projection.parameters()) +
#     list(semantic_projection.parameters()),
#     lr=0.0001
#     )
#     vit_model.train()
#     visual_projection.train()
#     semantic_projection.train()

#     for epoch in range(200):
#         with tqdm(train_data_loader, desc=f'Epoch {epoch + 1}', unit='batch') as tepoch:
#             for images, labels in tepoch:
#                 images = images.to(device)
#                 features = vit_model(images)
#                 visual_latent = visual_projection(features)
#                 semantic_indices = [class_names_list.index(label) for label in labels]
#                 semantic_features = class_embeddings_tensor[semantic_indices]
#                 semantic_latent = semantic_projection(semantic_features)
#                 loss = 1.0 - F.cosine_similarity(visual_latent, semantic_latent).mean()
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 tepoch.set_postfix(loss=loss.item())
#                 wandb.log({'epoch': epoch, 'loss': loss.item()})
        
#         test_accuracy = evaluate_model(test_data_loader, vit_model, visual_projection, semantic_projection, class_embeddings_tensor, device, class_names_list)
#         wandb.log({'epoch': epoch, 'test_accuracy': test_accuracy})
#         print(f'Test Accuracy after Epoch {epoch}: {test_accuracy:.4f}')

# if __name__ == "__main__":
#     main()



#--------Loss function: CE+MSE----------#
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.hub
from tqdm import tqdm
import torch.optim as optim
import wandb

# Initialize wandb
wandb.init(project="GNR_650_Assignment_2")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_embeddings(embedding_path):
    with open(embedding_path, 'rb') as file:
        embeddings = pickle.load(file)
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in embeddings.items()}

def load_pretrained_vit(trainable_blocks=1):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
    model.head = nn.Identity()
    for param in model.parameters():
        param.requires_grad = False
    num_blocks = len(model.blocks)
    for i in range(num_blocks - trainable_blocks, num_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True
    model.to(device)
    return model

def get_trainable_params(model):
    return [param for param in model.parameters() if param.requires_grad]

def load_class_names(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class CustomDataset(Dataset):
    def __init__(self, root_dir, class_names, embeddings, is_train=False):
        self.root_dir = root_dir
        self.class_names = class_names
        self.embeddings = embeddings
        self.transform = get_transforms(is_train)
        self.files = []
        self.labels = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                class_id = f.split('_')[0]
                if class_id in class_names:
                    self.files.append(os.path.join(root, f))
                    self.labels.append(class_names.index(class_id))  # Ensuring labels are indexed from 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label_index = self.labels[idx]  # Directly use preprocessed label
        embedding = self.embeddings[self.class_names[label_index]]
        return image, embedding, label_index


class VisualProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=512, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]  # Adjust dimensions as needed
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(dim))
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SemanticProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=512, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]  # Adjust dimensions as needed
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(dim))
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_epoch(loader, model, visual_head, semantic_head, optimizer, device):
    model.train()
    visual_head.train()
    semantic_head.train()
    total_loss = 0
    for images, semantic_embeddings, labels in tqdm(loader):
        images = images.to(device)
        semantic_embeddings = semantic_embeddings.to(device)  # Pre-transformed class embeddings
        labels = labels.to(device)

        optimizer.zero_grad()
        image_features = model(images)
        visual_embeddings = visual_head(image_features)
        transformed_semantic_embeddings = semantic_head(semantic_embeddings)

        # Compute similarities; visual_embeddings against each transformed_semantic_embedding
        # Reshape as needed to create a [batch_size x num_classes] tensor for F.cross_entropy
        cosine_sim = visual_embeddings @ transformed_semantic_embeddings.T  # Assuming dimensions allow for matrix multiplication
        mse_loss = F.mse_loss(visual_embeddings, transformed_semantic_embeddings)
        ce_loss = F.cross_entropy(cosine_sim, labels)  # labels should be class indices

        loss = mse_loss + ce_loss
        loss.backward()
        optimizer.step()
        wandb.log({'batch loss': loss})
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(data_loader, model, device):
    model.eval()
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        with tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") as tepoch:
            for images, _, labels in tepoch:
                images = images.to(device)
                labels=labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_images += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                tepoch.set_postfix(accuracy=f"{100 * total_correct / total_images:.2f}%")
    accuracy = total_correct / total_images
    return accuracy

def main():
    embeddings = load_embeddings('Baseline1/class_embeddings.pkl')
    vit_model = load_pretrained_vit(trainable_blocks=1)
    visual_projection = VisualProjectionHead(768, 512)
    semantic_projection = SemanticProjectionHead(300, 512)

    vit_model.to(device)
    visual_projection.to(device)
    semantic_projection.to(device)

    train_classes = load_class_names('Baseline1/AwA2/trainclasses.txt')
    test_classes = load_class_names('Baseline1/AwA2/testclasses.txt')

    train_dataset = CustomDataset('Baseline1/AwA2/JPEGImages', train_classes, embeddings, is_train=True)
    test_dataset = CustomDataset('Baseline1/AwA2/JPEGImages', test_classes, embeddings, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    optimizer = optim.SGD(get_trainable_params(vit_model) + list(visual_projection.parameters()) + list(semantic_projection.parameters()), lr=0.001)

    for epoch in range(10):
        loss = train_epoch(train_loader, vit_model, visual_projection, semantic_projection, optimizer, device)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
        wandb.log({'epoch': epoch, 'loss': loss})

        test_accuracy = evaluate_model(test_loader, vit_model, visual_projection, semantic_projection, device)
        wandb.log({"test_accuracy": test_accuracy})
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
