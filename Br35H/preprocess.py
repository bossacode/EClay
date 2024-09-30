import torch
from torchvision.transforms import Compose, Lambda, Resize, ToTensor, Grayscale
from PIL import Image
from sklearn.model_selection import train_test_split
import os


root_dir = "./dataset/"


def center_crop_to_square(img: Image.Image):
    """Crop the image to a square based on the smaller dimension."""
    width, height = img.size
    
    # Find the smaller dimension
    new_size = min(width, height)
    
    # Calculate the cropping box (center crop)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2
    
    # Crop the image and return
    return img.crop((left, top, right, bottom))


transform = Compose([
    Lambda(center_crop_to_square),  # Apply the custom center crop
    Resize(size=(224, 224)),
    ToTensor()
])

grayscale = Grayscale(num_output_channels=1)


def preprocess(val_size=0.3, test_size=0.3):
    data_list = []
    label_list = []

    # label 0 data
    for img_name in os.listdir(root_dir + "no/"):
        img = Image.open(root_dir + f"no/{img_name}")
        data = transform(img)
        if data.shape[0] <= 3:  # filter images with 4 channels
            data = grayscale(data)
        else:
            continue
        data_list.append(data)
        label_list.append(0)

    # label 0 data
    for img_name in os.listdir(root_dir + "yes/"):
        img = Image.open(root_dir + f"yes/{img_name}")
        data = transform(img)
        if data.shape[0] <= 3:  # filter images with 4 channels
            data = grayscale(data)
        else:
            continue
        data_list.append(data)
        label_list.append(1)
    
    X = torch.stack(data_list)
    y = torch.tensor(label_list)

    # split test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123, shuffle=True, stratify=y)
    # split val data
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=123, shuffle=True, stratify=y_train)

    # save data
    processed_dir = root_dir + "/processed/"
    os.makedirs(processed_dir, exist_ok=True)
    torch.save((X_tr, y_tr), f=processed_dir + f"train.pt")
    torch.save((X_val, y_val), f=processed_dir + f"val.pt")
    torch.save((X_test, y_test), f=processed_dir + f"test.pt")


if __name__ == "__main__":
    preprocess(val_size=0.3, test_size=0.7)