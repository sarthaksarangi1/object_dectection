import torch
from preprocessing import preprocess_image
from newyolo.models.experimental import attempt_load
from newyolo import Detector

class YOLOv5Detector:
    def __init__(self, model_weights_path, target_size, device='cpu'):
        # Load the YOLOv5 model
        self.model = attempt_load(model_weights_path, map_location=device)
        self.model.to(device)
        self.model.eval()

        # Create the detector object
        self.detector = Detector(self.model)

        # Save the target size
        self.target_size = target_size

    def detect(self, image_path):
        # Preprocess the image
        image = preprocess_image(image_path, self.target_size)

        # Run object detection on the image using the YOLOv5 model
        results = self.detector.detect(image)

        return results

