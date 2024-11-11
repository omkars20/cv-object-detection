# train.py
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_model():
    # Initialize a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.train()
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0

    for images, targets in data_loader:
        # Move images to the correct device
        images = [img.to(device) for img in images]
        
        # Convert each target to a dictionary if not already
        formatted_targets = []
        for t in targets:
            valid_boxes = []
            valid_labels = []

            for ann in t:
                x, y, w, h = ann["bbox"]
                # Filter out invalid boxes with non-positive width or height
                if w > 0 and h > 0:
                    valid_boxes.append([x, y, x + w, y + h])
                    valid_labels.append(ann["category_id"])

            # Skip if no valid boxes found for this image
            if valid_boxes:
                t_dict = {
                    "boxes": torch.tensor(valid_boxes, dtype=torch.float32, device=device),
                    "labels": torch.tensor(valid_labels, dtype=torch.int64, device=device)
                }
                formatted_targets.append(t_dict)
            else:
                print("Warning: Skipping image with no valid bounding boxes")

        # Skip iteration if no valid targets
        if not formatted_targets:
            continue

        # Forward pass
        loss_dict = model(images, formatted_targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return total_loss / len(data_loader)



def train_model(model, train_loader, valid_loader, device, num_epochs=5, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training step
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation step (forward pass only, no gradients, with training mode for loss)
        model.train()  # Temporarily use training mode to get loss during validation
        total_val_loss = 0
        with torch.no_grad():  # Disable gradient calculations for validation
            num_valid_samples = 0  # Track valid samples to calculate avg loss correctly
            for images, targets in valid_loader:
                images = [img.to(device) for img in images]

                # Convert targets to the required dictionary format
                formatted_targets = []
                for t in targets:
                    valid_boxes = []
                    valid_labels = []

                    for ann in t:
                        x, y, w, h = ann["bbox"]
                        # Filter out invalid boxes with non-positive width or height
                        if w > 0 and h > 0:
                            valid_boxes.append([x, y, x + w, y + h])
                            valid_labels.append(ann["category_id"])

                    # Only add if there are valid boxes
                    if valid_boxes:
                        t_dict = {
                            "boxes": torch.tensor(valid_boxes, dtype=torch.float32, device=device),
                            "labels": torch.tensor(valid_labels, dtype=torch.int64, device=device)
                        }
                        formatted_targets.append(t_dict)

                # Skip iteration if no valid targets
                if not formatted_targets:
                    continue

                # Forward pass with training mode to calculate loss
                loss_dict = model(images, formatted_targets)
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()
                num_valid_samples += 1

        # Calculate average validation loss if there are valid samples
        if num_valid_samples > 0:
            avg_val_loss = total_val_loss / num_valid_samples
        else:
            avg_val_loss = 0.0
            print("Warning: No valid samples found during validation.")

        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        model.eval()  # Set model back to evaluation mode after validation

    # Save the trained model
    torch.save(model.state_dict(), "resume_detection_model.pth")
    print("Model saved as resume_detection_model.pth")
