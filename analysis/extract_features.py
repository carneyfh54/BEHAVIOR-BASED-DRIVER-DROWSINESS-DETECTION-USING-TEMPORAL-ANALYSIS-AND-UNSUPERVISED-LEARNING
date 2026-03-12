import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        # Simple repeat last hidden state for decoding
        h_repeated = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(h_repeated)
        return decoded

# Load data
data = np.load("alert_sequences.npy")
dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMAutoEncoder(input_dim=data.shape[-1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    total_loss = 0
    for batch in loader:
        seq = batch[0]
        recon = model(seq)
        loss = criterion(recon, seq)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/100 - Loss: {total_loss / len(loader):.6f}")

torch.save(model.state_dict(), "lstm_autoencoder.pth")