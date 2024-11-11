from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pytesseract

# Load model with weights
def load_model(model_path, device):
    model = fasterrcnn_resnet50_fpn(weights=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_inference(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)

    detections = outputs[0]
    boxes = detections['boxes']
    scores = detections['scores']
    labels = detections['labels']
    
    print("All Detections (without filtering):")
    filtered_boxes = []
    for box, score, label in zip(boxes, scores, labels):
        print(f"Box: {box.cpu().numpy()}, Score: {score.item()}, Label: {label.item()}")
        filtered_boxes.append((box.cpu().numpy(), score.item(), label.item()))
    
    return filtered_boxes, image


def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, score, label in detections:
        # Convert coordinates to integers
        x_min, y_min, x_max, y_max = map(int, box)
        
        # Filter boxes with very low confidence
        if score < 0.3:  # Change this threshold as needed
            continue
        
        print(f"Drawing box: ({x_min}, {y_min}, {x_max}, {y_max}), Score: {score}, Label: {label}")
        
        # Draw rectangle and label
        draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)
        draw.text((x_min, y_min), f"{label}: {score:.2f}", fill="red", font=font)

    image.show()
    return image





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "resume_detection_model.pth"
    image_path = "resume-data/doc_000051.png"  # Use a training image for testing

    model = load_model(model_path, device)
    detections, image = run_inference(model, image_path, device)

    # Draw bounding boxes on the image
    image_with_boxes = draw_boxes(image, detections)

    # Optionally, save the image with bounding boxes
    image_with_boxes.save("output_with_boxes.jpg")

if __name__ == "__main__":
    main()


