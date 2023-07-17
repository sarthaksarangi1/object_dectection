from flask import Flask, jsonify, request
from prediction import YOLOv5Detector

app = Flask(__name__)

# Define the endpoint for object detection
@app.route('/detect', methods=['POST'])
def detect_objects():
    # Get the uploaded image file
    image_file = request.files['image']

    # Perform object detection on the image
    results = detector.detect(image_file)

    # Convert the results to JSON format and return them
    return jsonify(results.pandas().to_dict(orient='records'))

if __name__ == '__main__':
    # Initialize the detector object
    model_weights_path = 'weights/best.pt'
    target_size = (640, 640)
    device = 'cpu'
    detector = YOLOv5Detector(model_weights_path, target_size, device)

    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
