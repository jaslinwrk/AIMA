import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class InventoryPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(InventoryPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data_from_df(df):
    print(f"Loaded data columns: {df.columns.tolist()}")
    weather_columns = ['temperature_2m_max', 'temperature_2m_min', 'daylight_duration']
    product_columns = [col for col in df.columns if col not in ['date'] + weather_columns]
    
    X = df[weather_columns].values
    y = df[product_columns].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, input_size, output_size):
    model = InventoryPredictor(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    epochs_without_improvement = 0
    best_loss = float('inf')

    for epoch in range(400):  # Increased max epochs, will stop early if needed
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 10:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model