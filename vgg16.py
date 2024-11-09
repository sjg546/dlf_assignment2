import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score


# Use GPU if available, way faster
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10  # 10 classes of images in the CIFAR-10 dataset
num_epochs = 20
batch_size = 32
learning_rate = 0.001

filename = f"vgg16_cifar10_{num_epochs}_{learning_rate}_{batch_size}.pth"
# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(device)

# Load pretrained vgg16 model
def get_pretrained_vgg16(num_classes=10):
    model = models.vgg16(pretrained=True)
    
    # Modify the classifier to match the number of classes
    model.classifier[6] = nn.Linear(4096, num_classes)  # Replace the last layer
    
    return model

# Main training function
def train_pretrained_vgg16():        
    # Load CIFAR-10 dataset
    training_data = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    
    model = get_pretrained_vgg16(num_classes=num_classes).to(device)
        
    criteria = nn.CrossEntropyLoss()

    for param in model.features.parameters():
        param.requires_grad = False

    # Only the classifier parameters will be updated
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Set to training mode
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for i, (images, labels) in enumerate(train_loader):
            print(f"I: {i}")
            images, labels= images.to(device), labels.to(device)

            optimizer.zero_grad() 
            outputs = model(images)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch:{epoch+1}/{num_epochs} , Step:{i}/{len(train_loader)}')

    torch.save(model.state_dict(), filename)
    print(f'Model saved to {filename}')

    print('Finished Training')
# Function to test the model on a test dataset
def test_model():

    # Load the saved model
    model = get_pretrained_vgg16(num_classes=10)
    model.load_state_dict(torch.load(filename))  
    model = model.to(device)  

    # Switch to evaluation mode
    model.eval()

    # Load CIFAR-10 test dataset
    test_data = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Store predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels,all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1score = f1_score(all_labels, all_predictions, average='weighted') 

    with open("vgg16_results.txt", "a") as myfile:
        myfile.write(f"{filename} : Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1Score:{f1score}\n")


if __name__ == "__main__":
    # train_pretrained_vgg16()
    test_model()
