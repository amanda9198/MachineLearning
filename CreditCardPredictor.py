import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming 'data' and 'labels' are your existing dataset

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network for Credit Risk Detection
class CreditRiskNN(nn.Module):
    def __init__(self, input_size):
        super(CreditRiskNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)

# Create DataLoader
dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = CreditRiskNN(input_size=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test_scaled))
    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == torch.FloatTensor(y_test.values).reshape(-1, 1)).float().mean()
    print(f'Test Accuracy: {accuracy.item():.4f}')

# Compare with Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_accuracy = rf_model.score(X_test_scaled, y_test)
print(f'Random Forest Accuracy: {rf_accuracy:.4f}')

# Visualize feature importance
feature_importance = rf_model.feature_importances_
plt.bar(range(len(feature_importance)), feature_importance)
plt.title('Feature Importance in Credit Risk Prediction')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()