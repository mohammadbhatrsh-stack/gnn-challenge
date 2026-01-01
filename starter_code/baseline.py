import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os

from dataset import TopologicalDataset
from model import GINModel

# ----------------------------
# Paths (root of repo)
# ----------------------------
# Get the absolute path of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(REPO_ROOT, "data")
SUBMISSIONS_DIR = os.path.join(REPO_ROOT, "submissions")

# Make sure submissions folder exists
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# ----------------------------
# Load data splits
# ----------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

dataset = TopologicalDataset("MUTAG", topo_config="degree")

train_graphs = [dataset[i] for i in train_df.graph_index]
test_graphs = [dataset[i] for i in test_df.graph_index]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32)

# ----------------------------
# Model
# ----------------------------
model = GINModel(
    input_dim=dataset.num_features,
    output_dim=dataset.num_classes
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# Training
# ----------------------------
for epoch in range(50):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} completed")

# ----------------------------
# Prediction
# ----------------------------
model.eval()
predictions = []

with torch.no_grad():
    for data in test_loader:
        out = model(data)
        predictions.extend(out.argmax(dim=1).tolist())

# ----------------------------
# Save submission
# ----------------------------
submission_path = os.path.join(SUBMISSIONS_DIR, "sample_submission.csv")

submission = pd.DataFrame({
    "graph_index": test_df.graph_index,
    "target": predictions
})

submission.to_csv(submission_path, index=False)
print(f"Saved submission to: {submission_path}")
