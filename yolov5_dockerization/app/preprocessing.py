from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path, target_size):
    # Load the image and resize to the target size
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)

    return image
