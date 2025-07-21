import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from eval import eval_subbox_to_outside
import json

def convert_string_patterns_to_int(array, mask_value, mask_symbol="*"):
    vectorized = np.vectorize(lambda x: mask_value if x == mask_symbol else int(x))
    return vectorized(array).astype(int)

class TransformerModel(nn.Module):
    """
    Simple Transformer model for 2D-to-2D symbolic pattern prediction.

    Architecture:
        - Embedding of input tokens
        - Learned 2D positional encoding
        - Transformer encoder layers
        - Linear head mapping to output vocabulary
    """
    def __init__(self, vocab_size, emb_dim=64, nhead=4, num_layers=2, H=19, W=19):
        """
        Initializes the Transformer model.

        Parameters:
            vocab_size (int): Number of possible input/output symbols (e.g., 2 or 3).
            emb_dim (int): Dimension of internal embeddings.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            H (int): Height of the 2D grid.
            W (int): Width of the 2D grid.
        """
        super().__init__()
        self.H = H
        self.W = W
        self.seq_len = H * W

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.seq_len, emb_dim))  # learned 2D positional embedding

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
            x (Tensor): shape (B, H, W), integer-encoded input patterns.

        Returns:
            logits (Tensor): shape (B, H, W, vocab_size), predicted logits at each position.
        """
        B, H, W = x.shape
        assert H == self.H and W == self.W

        x_flat = x.view(B, -1)                              # (B, L)
        x_emb = self.token_embedding(x_flat)                # (B, L, D)
        x_pos = x_emb + self.position_embedding             # (B, L, D)

        z = self.transformer(x_pos)                         # (B, L, D)
        logits = self.head(z)                               # (B, L, vocab_size)

        return logits.view(B, H, W, -1)                      # (B, H, W, vocab_size)
    

def train_transformer_model(model, input_stack, target_stack, model_path, 
                            mask_token=None, epochs=20, batch_size=32, lr=1e-3):
    """
    Trains a transformer model to predict target patterns from input patterns.

    Parameters:
        model (nn.Module): TransformerModel instance.
        input_stack (np.ndarray): shape (N, H, W), integer-encoded input patterns.
        target_stack (np.ndarray): shape (N, H, W), integer-encoded target patterns.
        model_path (str): Path to save the trained model.
        mask_token (int or None): If given, ignored in loss computation.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        lr (float): Learning rate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_tensor = torch.tensor(input_stack, dtype=torch.long).to(device)
    target_tensor = torch.tensor(target_stack, dtype=torch.long).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=mask_token) if mask_token is not None else nn.CrossEntropyLoss()

    N = input_tensor.size(0)
    H, W = input_tensor.size(1), input_tensor.size(2)


    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = input_tensor[idx]
            y_batch = target_tensor[idx]

            optimizer.zero_grad()
            logits = model(x_batch)  # (B, H, W, vocab_size)
            logits = logits.permute(0, 3, 1, 2)  # (B, vocab_size, H, W)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # === Step 1: Load patterns ===
    NAME = "subshift_0"
    ALPHABET_SIZE = 3
    LOAD_MODEL = True

    masked_patterns = np.load(f"outside_subbox_masked_patterns/{NAME}/all_patterns.npy")
    masked_patterns = convert_string_patterns_to_int(masked_patterns,mask_value=ALPHABET_SIZE)
    complement_patterns = np.load(f"subbox_masked_patterns/{NAME}/all_patterns.npy")
    complement_patterns = convert_string_patterns_to_int(masked_patterns,mask_value=ALPHABET_SIZE)

    from sklearn.model_selection import train_test_split

    # === Step 1.5: Split into training and test sets ===
    x_train, x_test, y_train, y_test = train_test_split(
        masked_patterns, complement_patterns, test_size=0.1, random_state=42
    )

    # === Step 2: Create and train model ===
    H, W = masked_patterns.shape[1:]
    model = TransformerModel(
        vocab_size=ALPHABET_SIZE+1,
        emb_dim=64,
        nhead=4,
        num_layers=2,
        H=H,
        W=W
    )

    if LOAD_MODEL is False: 
        train_transformer_model(
            model=model,
            input_stack=x_train,
            target_stack=y_train,
            model_path="models/transformer.pt",
            mask_token=ALPHABET_SIZE,
            epochs=20,
            batch_size=32,
            lr=1e-3
        )
    else: 
        model.load_state_dict(torch.load("models/transformer.pt", map_location=torch.device("cpu")))

    model.eval()
    x_test_tensor = torch.tensor(x_test, dtype=torch.long)

    with torch.no_grad():
        logits = model(x_test_tensor)  # shape: (B, H, W, vocab_size)
        pred = torch.argmax(logits, dim=-1)  # shape: (B, H, W)
        pred = pred.cpu().numpy()
        pred = pred.reshape((312,3,3))

    print(pred[0,:,:])
    print(x_test[0,:,:])
    print(y_test[0,:,:])
    # with open("samples.json", "r") as f:
    #     samples = json.load(f)
    # eval = eval_subbox_to_outside(x_test,pred,box_size=1,forbidden_patterns=samples[NAME]["forbidden_pairs"])
    # print(eval)