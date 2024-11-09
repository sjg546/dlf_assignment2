import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score

# Hyperparameters
num_classes = 10
num_epochs = 10
batch_size = 32
learning_rate = 0.001    

# Use GPU if available, way faster
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms to match the data used in the pretrained model
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pretrained resnet34 model
def get_pretrained_resnet34(num_classes=10):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Main training function
def train_pretrained_resnet_34():    
    # Load CIFAR-10 dataset
    training_data = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    
    # Initialize the pretrained model
    model = get_pretrained_resnet34(num_classes=num_classes).to(device)

    criteria = nn.CrossEntropyLoss()
    
    for param in model.parameters():
        param.requires_grad = False

    # ADAM optimiser, only optimise final fully connected layer
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
    # Set to training mode
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for i, (images, labels) in enumerate(train_loader):
            print(f"I: {i}")
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch:{epoch+1}/{num_epochs} , Step:{i}/{len(train_loader)}')

    torch.save(model.state_dict(), 'resnet34_cifar10.pth')
    print('Model saved to resnet34_cifar10.pth')

    print('Finished Training')

def test_model():
    # Load the saved model state
    model = get_pretrained_resnet34(num_classes=10)
    model.load_state_dict(torch.load('resnet34_cifar10.pth'))
    model = model.to(device) 

    # Switch to eval mode
    model.eval()
    
    # Load test dataset
    test_data = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) 
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1score = f1_score(all_labels, all_predictions, average='weighted')

    with open("resnet34_results.txt", "a") as myfile:
        myfile.write(f"Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1Score:{f1score}")

if __name__ == "__main__":
    # train_pretrained_resnet_34()
    test_model()
