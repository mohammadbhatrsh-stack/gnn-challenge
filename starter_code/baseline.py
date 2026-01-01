import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from dataset import TopologicalDataset
from model import GINModel

# ----------------------------
# Load data splits
# ----------------------------
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

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

submission = pd.DataFrame({
    "graph_index": test_df.graph_index,
    "target": predictions
})

submission.to_csv("../submissions/sample_submission.csv", index=False)
print("Saved submissions/sample_submission.csv")
