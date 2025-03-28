import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from torchinfo import summary

import matplotlib.pyplot as plt
from torchviz import make_dot

########################################################################
# 1) Custom Transformer Encoder Layers to Extract Attention Weights
########################################################################

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """
    A slight modification of PyTorch's TransformerEncoderLayer that returns
    the self-attention weights along with the usual output.
    """
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # The default forward pass in nn.TransformerEncoderLayer:
        # self_attn(src, src, src, ...)
        # We'll capture attn_weights by setting need_weights=True.
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,         # <--- we want attention weights
            average_attn_weights=False # <--- do not average across heads
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # feedforward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src, attn_weights


class TransformerEncoderWithAttn(nn.Module):
    """
    Stacks multiple TransformerEncoderLayerWithAttn layers and
    collects their attention weights.
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Returns:
          - output: final layer output of shape (batch_size, seq_len, d_model)
          - attn_maps_list: a list of attention matrices [layer_1, layer_2, ...],
            where each element is shape (batch_size, nhead, seq_len, seq_len).
        """
        attn_maps_list = []
        output = src

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask,
                                       src_key_padding_mask=src_key_padding_mask)
            attn_maps_list.append(attn_weights)

        return output, attn_maps_list


########################################################################
# 2) PositionalEncoding (same as before)
########################################################################

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to embeddings to provide sequence (frame) order information.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer so it’s not trainable but still moves to GPU if needed
        self.register_buffer('pe', pe.unsqueeze(0)) # shape = (1, max_len, d_model)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        We add the encoding up to seq_len to x.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


########################################################################
# 3) MotionTransformer model that returns logits + attention maps
########################################################################

class MotionTransformer(nn.Module):
    """
    Transformer-based model for classification of videos (real vs. fake) 
    using motion-delta features.
    
    Input shape: (batch_size, seq_len=40, feature_dim=136)
    Output: Probability distribution over 2 classes: [Real, Fake].

    We will return both:
      - logits
      - attn_maps_list (list of attention weights for each layer)
    """
    def __init__(
        self,
        feature_dim=136,     # dimension of motion-delta features per frame
        d_model=128,         # internal embedding dimension used by the Transformer
        nhead=4,             # number of attention heads
        num_layers=2,        # number of Transformer encoder layers
        dim_feedforward=256, 
        dropout=0.1,
        num_classes=2
    ):
        super(MotionTransformer, self).__init__()

        self.feature_dim = feature_dim
        self.d_model = d_model

        # 1) Linear projection of input features to d_model
        self.input_linear = nn.Linear(feature_dim, d_model)

        # 2) Positional encoding module
        self.pos_encoder = PositionalEncoding(d_model)

        # 3) Custom Transformer Encoder that returns attention
        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoderWithAttn(encoder_layer, num_layers=num_layers)

        # 4) Classification head: we pool over the sequence dimension (e.g., average pooling)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, feature_dim)
        Returns:
          - logits of shape (batch_size, num_classes)
          - attn_maps_list: list of shape (num_layers, batch_size, n_heads, seq_len, seq_len)
        """
        # 1) Project input to d_model
        x = self.input_linear(x)  # (batch_size, seq_len, d_model)

        # 2) Add positional encoding
        x = self.pos_encoder(x)   # (batch_size, seq_len, d_model)

        # 3) Pass through Transformer encoder (collect attention weights)
        x, attn_maps_list = self.transformer_encoder(x) # (batch_size, seq_len, d_model), [layer_1, ...]

        # 4) Pool over the time dimension (seq_len). Here we do mean pooling.
        x = torch.mean(x, dim=1) # (batch_size, d_model)

        # 5) Classification head
        logits = self.classifier(x) # (batch_size, num_classes)

        return logits, attn_maps_list


########################################################################
# 4) Dataset & Training
########################################################################

class MotionDataset(Dataset):
    """
    Dataset wrapping for motion-delta features. The data is typically loaded from
    motion_features.pkl, which is assumed to store (all_motion, all_labels).

    all_motion shape might be: (num_samples, seq_len=40, feature_dim=136)
    all_labels shape: (num_samples,)
    """
    def __init__(self, motion_data, labels):
        super(MotionDataset, self).__init__()
        self.motion_data = motion_data  # shape: (N, 40, 136)
        self.labels = labels            # shape: (N,)

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        x = self.motion_data[idx] # shape: (40, 136)
        y = self.labels[idx]      # 0 or 1
        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long) # For classification
        return x, y


def train_transformer_model(
    motion_features_path,
    batch_size=10,
    num_epochs=20,
    learning_rate=1e-4,
    device='cuda' # set to 'cuda' if GPU is available
):
    """
    Example training routine:
      1. Loads the motion feature data from the pickle file.
      2. Splits into train/val or train/test (as needed).
      3. Creates a DataLoader.
      4. Initializes and trains the transformer model.
      5. Returns the trained model.
    """

    # 1) Load data
    with open(motion_features_path, 'rb') as f:
        (all_motion, all_labels) = pickle.load(f)  
        # all_motion.shape = (num_samples, seq_len=40, feature_dim=136)
        # all_labels.shape = (num_samples,)

    # For simplicity, let's do a quick train/val split (80% train, 20% val)
    num_samples = len(all_motion)
    split_idx = int(0.8 * num_samples)
    train_motion = all_motion[:split_idx]
    train_labels = all_labels[:split_idx]
    val_motion = all_motion[split_idx:]
    val_labels = all_labels[split_idx:]

    # 2) Create datasets and loaders
    train_dataset = MotionDataset(train_motion, train_labels)
    val_dataset = MotionDataset(val_motion, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3) Initialize model, loss, optimizer
    model = MotionTransformer(feature_dim=all_motion.shape[2])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4) Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device) # shape: (B, 40, 136)
            batch_y = batch_y.to(device) # shape: (B,)

            optimizer.zero_grad()
            logits, _ = model(batch_x) # shape: (B, 2); we can ignore attention here
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item()

                # Compute accuracy
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / len(val_dataset)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")

    print("Training complete.")
    return model


########################################################################
# 5) Visualization Code
########################################################################

def visualize_architecture(model, save_path="motion_transformer_graph"):
    """
    Creates a high-level architecture graph of the model using torchviz.
    """
    # Create a small dummy input: batch=1, seq_len=40, feature_dim=136
    dummy_input = torch.randn(1, 40, model.feature_dim)
    logits, attn_maps_list = model(dummy_input)

    # Use torchviz to create a graph
    graph = make_dot(logits, params=dict(model.named_parameters()))
    graph.render(save_path, format="png")
    print(f"Architecture graph saved to {save_path}.png")


def visualize_attention(model, device="cpu", num_frames=40):
    """
    Runs a forward pass on dummy data and plots the attention matrix
    for each layer and head (on a single sample).
    """
    model.eval()
    # We'll create a dummy batch of size=1
    dummy_input = torch.randn(1, num_frames, model.feature_dim).to(device)
    with torch.no_grad():
        _, attn_maps_list = model(dummy_input)

    # attn_maps_list is a list of shape [num_layers], each element:
    # (batch_size=1, n_heads, seq_len, seq_len)
    for layer_idx, attn_map in enumerate(attn_maps_list):
        # attn_map shape = (1, n_heads, seq_len, seq_len)
        n_heads = attn_map.shape[1]
        for head_idx in range(n_heads):
            # We'll plot the attention for the single sample
            attention_matrix = attn_map[0, head_idx].detach().cpu().numpy() # shape (seq_len, seq_len)
            plt.figure(figsize=(5, 4))
            plt.imshow(attention_matrix, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f"Layer {layer_idx+1}, Head {head_idx+1} Attention")
            plt.xlabel("Key Frames")
            plt.ylabel("Query Frames")
            plt.show()


########################################################################
# 6) Main script usage
########################################################################

if __name__ == "__main__":
    # Example usage:
    trained_model = train_transformer_model(
        motion_features_path='features/motion_features.pkl',
        batch_size=10,
        num_epochs=20,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # Summarize the layer-by-layer shapes and parameter counts
    summary(trained_model, input_size=(10, 40, 136))

    # Generate a high-level architecture graph
    #visualize_architecture(trained_model, save_path="motion_transformer_graph")

    # Visualize attention weights from a single dummy sample
    visualize_attention(trained_model, device='cuda' if torch.cuda.is_available() else 'cpu')
