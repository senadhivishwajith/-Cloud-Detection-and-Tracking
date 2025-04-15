from dataset import CloudMaskSequenceDataset
from model import ConvLSTMModel
from train import train_model
import torch
from torch.utils.data import DataLoader

csv_path = "/Users/vishwajithjayathissa/Documents/Final Year Project/imageCapturing2.0/Cloud Tracking/cloudTracking/data/cloud_sun_tracking_seg.csv"
image_folder = "/Users/vishwajithjayathissa/Documents/Final Year Project/imageCapturing2.0/Cloud Tracking/cloudTracking/outputs/annotated"

batch_size = 32
learning_rate = 1e-3
epochs = 15
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

dataset = CloudMaskSequenceDataset(csv_path, image_folder)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = ConvLSTMModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, val_loader, optimizer, device, epochs=epochs)
print("✅ Training complete. Model saved as convlstm_model.pt.")