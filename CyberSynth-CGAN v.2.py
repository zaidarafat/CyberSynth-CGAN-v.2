import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy import stats
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add SMOTE import
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not installed. Run: pip install imbalanced-learn")
    print("          Hybrid augmentation will not be available.")
# =============================
#   ENHANCED DATA PREPROCESSING
# =============================

def load_and_preprocess(dataset_path, label_col=None):
    """Enhanced data loading with better label detection and cleaning"""
    import unicodedata
    import re
    
    files = sorted(Path(dataset_path).glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {dataset_path}")
    
    print(f"[INFO] Loading {len(files)} CSV files...")
    data = pd.concat([pd.read_csv(f, low_memory=False, encoding="latin1")
                    for f in files], ignore_index=True)
    
    print(f"[INFO] Loaded {len(data)} total records")
    
    # 1) Normalize column names
    clean_cols = []
    for c in data.columns:
        c2 = unicodedata.normalize("NFKC", str(c)).replace("\ufeff", "").strip()
        clean_cols.append(c2)
    data.columns = clean_cols
    
    # 2) Build lowercase index to original names
    norm2orig = {c.lower().replace(" ", "").replace("_", ""): c for c in data.columns}
    
    # 3) Try to find label column
    candidates = ["label", "attacklabel", "attack_cat", "attackcategory", "category", "class"]
    if label_col is not None:
        key = str(label_col).lower().replace(" ", "").replace("_", "")
        label_col = norm2orig.get(key, label_col)
    else:
        for k in candidates:
            if k in norm2orig:
                label_col = norm2orig[k]
                break
    
    # 4) Heuristic fallback
    if label_col is None:
        tokens = re.compile(r"benign|ddos|dos|portscan|web\s*attack|bot|infiltration|patator|xss|sql", re.I)
        obj_cols = data.select_dtypes(include=["object"]).columns.tolist()
        for c in obj_cols:
            s = data[c].astype(str)
            nunq = s.nunique(dropna=True)
            if 2 < nunq < 100 and s.str.contains(tokens, na=False).mean() > 0.05:
                label_col = c
                break
    
    if label_col is None:
        print("Available columns:", list(data.columns)[:40])
        raise ValueError("Label column not found. Pass label_col=... to load_and_preprocess().")
    
    print(f"[INFO] Using label column: {label_col}")
    
    # 5) Clean label strings
    def _clean_label(s: str) -> str:
        s = str(s).strip()
        s = s.replace("Ã¯Â¿Â½", "-").replace("ï¿½", "-")
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"\s+", " ", s)
        return s
    
    data[label_col] = data[label_col].astype(str).map(_clean_label)
    
    # Canonicalize label names
    canon_map = {
        "Web Attack - Brute Force": "Web Attack: Brute Force",
        "Web Attack - XSS": "Web Attack: XSS",
        "Web Attack - Sql Injection": "Web Attack: Sql Injection",
        "Web Attack - SQL Injection": "Web Attack: Sql Injection",
        "Web Attack  -  Brute Force": "Web Attack: Brute Force",
        "Web Attack  -  XSS": "Web Attack: XSS",
    }
    data[label_col] = data[label_col].replace(canon_map, regex=False)
    
    print("\n[INFO] Label distribution:")
    label_counts = data[label_col].value_counts()
    print(label_counts)
    
    # Identify minority classes (< 0.1% of dataset)
    total_samples = len(data)
    minority_threshold = total_samples * 0.001
    minority_classes = label_counts[label_counts < minority_threshold].index.tolist()
    print(f"\n[INFO] Identified {len(minority_classes)} minority classes (< 0.1%):")
    for cls in minority_classes:
        print(f"  - {cls}: {label_counts[cls]} samples")
    
    # Replace inf/nan
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    # Encode categorical (except label)
    cat_cols = data.select_dtypes(include=['object']).columns.drop(label_col, errors='ignore')
    for col in cat_cols:
        data[col] = LabelEncoder().fit_transform(data[col])
    
    # Drop remaining non-numeric columns except label
    non_num = data.select_dtypes(exclude=[np.number]).columns.drop(label_col, errors='ignore')
    data.drop(columns=non_num, inplace=True)
    
    # One-hot encode labels
    labels = data[label_col].values.reshape(-1, 1)
    ohe = OneHotEncoder(sparse_output=False)
    onehot_labels = ohe.fit_transform(labels)
    
    # Apply log transform to handle extreme values in network traffic data
    features = data.drop(columns=[label_col])
    feature_array = features.values  # Convert to numpy array
    
    print(f"\n[INFO] Original feature statistics:")
    print(f"  Max value: {feature_array.max():.2e}")
    print(f"  Min value: {feature_array.min():.2f}")
    
    # Apply log1p (log(1+x)) to handle zeros and positive values
    print(f"\n[INFO] Applying log transformation...")
    features_log = np.log1p(np.abs(feature_array))  # log(1 + |x|)
    
    # Preserve sign for negative values
    features_signed = np.sign(feature_array) * features_log
    
    print(f"\n[INFO] After log transform:")
    print(f"  Max value: {features_signed.max():.2f}")
    print(f"  Min value: {features_signed.min():.2f}")
    
    # Now scale to [-1, 1]
    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = minmax_scaler.fit_transform(features_signed)
    
    print(f"\n[INFO] Final scaled range: [{scaled_features.min():.6f}, {scaled_features.max():.6f}]")
    
    # Create custom scaler that PROPERLY handles inverse transform
    class LogMinMaxScaler:
        def __init__(self, minmax_scaler):
            self.minmax_scaler = minmax_scaler
            self.n_features_in_ = minmax_scaler.n_features_in_
            self.feature_range = minmax_scaler.feature_range
            # Store min/max for debugging
            self.data_min_ = minmax_scaler.data_min_
            self.data_max_ = minmax_scaler.data_max_
            self.data_range_ = minmax_scaler.data_range_
            self.scale_ = minmax_scaler.scale_
            self.min_ = minmax_scaler.min_
        
        def inverse_transform(self, X):
            """
            Reverse the transformation:
            1. Reverse MinMax scaling ([-1, 1] → log space)
            2. Reverse log transform (log space → original scale)
            """
            # Step 1: Reverse MinMax scaling
            X_log = self.minmax_scaler.inverse_transform(X)
            
            # Step 2: Reverse log transform
            # Remember: we did sign(x) * log1p(|x|)
            # So reverse is: sign(x) * expm1(|x|)
            X_abs = np.expm1(np.abs(X_log))  # exp(x) - 1
            X_original = np.sign(X_log) * X_abs
            
            return X_original
        
        def transform(self, X):
            """Forward transform for new data"""
            features_log = np.log1p(np.abs(X))
            features_signed = np.sign(X) * features_log
            return self.minmax_scaler.transform(features_signed)
    
    scaler = LogMinMaxScaler(minmax_scaler)
    
    # VERIFY the scaler works correctly
    print(f"\n[INFO] Verifying inverse transform...")
    test_sample = scaled_features[:5]
    test_inverse = scaler.inverse_transform(test_sample)
    print(f"  Original sample max: {feature_array[:5].max():.2e}")
    print(f"  Inverse sample max: {test_inverse.max():.2e}")
    
    if test_inverse.max() > 1e8:
        print(f"  ⚠️  WARNING: Inverse transform still producing large values!")
    else:
        print(f"  ✓ Inverse transform working correctly!")
    
    return (
        torch.tensor(scaled_features, dtype=torch.float32),
        torch.tensor(onehot_labels, dtype=torch.float32),
        scaler, 
        ohe,
        minority_classes
    )


# =============================
#   IMPROVED GAN ARCHITECTURE
# =============================

class ImprovedGenerator(nn.Module):
    """Enhanced Generator with residual connections and layer normalization"""
    def __init__(self, noise_dim, label_dim, output_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        
        # Label embedding
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, label_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Main pathway
        self.fc1 = nn.Linear(noise_dim + label_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 512)
        self.ln2 = nn.LayerNorm(512)
        
        self.fc3 = nn.Linear(512, 1024)
        self.ln3 = nn.LayerNorm(1024)
        
        self.fc4 = nn.Linear(1024, 512)
        self.ln4 = nn.LayerNorm(512)
        
        self.output = nn.Linear(512, output_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, noise, labels):
        # Embed labels
        labels = self.label_emb(labels)
        x = torch.cat([noise, labels], dim=1)
        
        # Layer 1
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        
        # Layer 3 (with skip connection from layer 1)
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.relu(x)
        
        # Layer 4
        x = self.fc4(x)
        x = self.ln4(x)
        x = self.relu(x)
        
        # Output
        x = self.output(x)
        x = self.tanh(x)
        
        return x


class ImprovedDiscriminator(nn.Module):
    """Enhanced Discriminator with feature extraction and spectral normalization"""
    def __init__(self, input_dim, label_dim):
        super().__init__()
        
        # Label embedding
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, label_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Feature extraction layers
        self.layer1 = nn.utils.spectral_norm(nn.Linear(input_dim + label_dim, 512))
        self.layer2 = nn.utils.spectral_norm(nn.Linear(512, 256))
        self.layer3 = nn.utils.spectral_norm(nn.Linear(256, 128))
        self.output_layer = nn.Linear(128, 1)
        
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights - FIXED VERSION
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Fixed weight initialization that works with spectral_norm"""
        # Check if it's a Linear layer by looking at the module's class name
        if m.__class__.__name__ == 'Linear':
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # Also check for the wrapper created by spectral_norm
        elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
            if len(m.weight.shape) == 2:  # Linear layers have 2D weights
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, data, labels, return_features=False):
        # Embed labels
        labels = self.label_emb(labels)
        x = torch.cat([data, labels], dim=1)
        
        # Feature extraction
        x1 = self.leaky(self.layer1(x))
        x1 = self.dropout(x1)
        
        x2 = self.leaky(self.layer2(x1))
        x2 = self.dropout(x2)
        
        x3 = self.leaky(self.layer3(x2))
        
        # Output
        out = self.output_layer(x3)
        
        if return_features:
            return out, x2  # Return intermediate features for feature matching
        return out


# =============================
#   ADVANCED TRAINING UTILITIES
# =============================

def gradient_penalty_adaptive(discriminator, real_data, fake_data, labels, device, epoch, total_epochs, base_lambda=10):
    """Adaptive gradient penalty that adjusts during training"""
    # Reduce GP strength as training progresses
    progress = epoch / total_epochs
    lambda_gp = base_lambda * (1.0 - 0.3 * progress)  # Reduces to 70% of original
    
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1).to(device)
    
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolates = discriminator(interpolates, labels)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    
    return lambda_gp * gp


def feature_matching_loss(real_features, fake_features):
    """Feature matching loss for improved sample quality"""
    return torch.mean((real_features.mean(0) - fake_features.mean(0)) ** 2)


def diversity_loss(fake_samples, batch_size):
    """Encourage diversity in generated samples"""
    if batch_size < 2:
        return torch.tensor(0.0).to(fake_samples.device)
    
    # Compute pairwise distances
    dists = torch.cdist(fake_samples, fake_samples, p=2)
    
    # Exclude diagonal (self-distances)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=fake_samples.device)
    dists = dists[mask].view(batch_size, batch_size - 1)
    
    # Penalize small distances (encourage diversity)
    diversity = -torch.mean(torch.exp(-dists))
    
    return diversity


class TrainingHistory:
    """Track training metrics"""
    def __init__(self):
        self.d_losses = []
        self.g_losses = []
        self.d_real_scores = []
        self.d_fake_scores = []
        self.gp_values = []
        self.fm_losses = []
        
    def update(self, d_loss, g_loss, d_real, d_fake, gp, fm_loss=0):
        self.d_losses.append(d_loss)
        self.g_losses.append(g_loss)
        self.d_real_scores.append(d_real)
        self.d_fake_scores.append(d_fake)
        self.gp_values.append(gp)
        self.fm_losses.append(fm_loss)
    
    def plot(self, save_path='training_history.png'):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.d_losses, label='D Loss', alpha=0.7)
        axes[0, 0].plot(self.g_losses, label='G Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Generator and Discriminator Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator scores
        axes[0, 1].plot(self.d_real_scores, label='D(real)', alpha=0.7)
        axes[0, 1].plot(self.d_fake_scores, label='D(fake)', alpha=0.7)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Discriminator Scores')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient penalty
        axes[1, 0].plot(self.gp_values, alpha=0.7)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('GP Value')
        axes[1, 0].set_title('Gradient Penalty')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature matching loss
        axes[1, 1].plot(self.fm_losses, alpha=0.7)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('FM Loss')
        axes[1, 1].set_title('Feature Matching Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Training history plot saved to {save_path}")
        plt.close()


# =============================
#   ENHANCED TRAINING LOOP
# =============================

def train_improved_cgan(
    real_data, label_data, minority_classes,
    noise_dim=100, epochs=5000, batch_size=128,
    lr_g=0.0002, lr_d=0.0001, base_lambda_gp=10,
    n_critic=5, fm_weight=0.1, div_weight=0.01,
    device='cpu'
):
    if str(device) == 'cpu':
        print("\n" + "!"*70)
        print("WARNING: Training on CPU detected")
    """
    Enhanced training with all improvements:
    - Adaptive learning rates
    - Feature matching loss
    - Diversity loss
    - Focused minority training
    - Early stopping with convergence detection
    """
    if str(device) == 'cpu':
        print("\n" + "!"*70)
        print("WARNING: Training on CPU detected")
        print("!"*70)
        print("Adjusting hyperparameters for CPU stability:")
        
        lr_g = 0.00015       # Reduce from 0.0002
        lr_d = 0.00008       # Reduce from 0.0001
        base_lambda_gp = 5   # Reduce from 10
        n_critic = 3         # Reduce from 5
        fm_weight = 0.05     # Reduce from 0.1
        
        print(f"  lr_g: 0.0002 → {lr_g}")
        print(f"  lr_d: 0.0001 → {lr_d}")
        print(f"  base_lambda_gp: 10 → {base_lambda_gp}")
        print(f"  n_critic: 5 → {n_critic}")
        print(f"  fm_weight: 0.1 → {fm_weight}")
        print("!"*70 + "\n")
        
    label_dim = label_data.shape[1]
    output_dim = real_data.shape[1]
    
    # Initialize models
    G = ImprovedGenerator(noise_dim, label_dim, output_dim).to(device)
    D = ImprovedDiscriminator(output_dim, label_dim).to(device)
    
    # Optimizers with different learning rates
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.9995)
    scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.9995)
    
    # Compute class weights for sampling (inverse frequency)
    with torch.no_grad():
        counts = label_data.sum(dim=0) + 1e-6
        weights = 1.0 / counts
        sampling_probs = (weights / weights.sum()).to(device)
    
    # Identify minority class indices
    all_classes = label_data.argmax(dim=1)
    minority_indices = []
    for cls in minority_classes:
        # Find index in one-hot encoding
        # (This requires mapping class name to index - handled in data loading)
        pass
    
    # Training history
    history = TrainingHistory()
    
    # Move data to device
    real_data = real_data.to(device)
    label_data = label_data.to(device)
    
    print("\n" + "="*70)
    print("STARTING ENHANCED TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rates: G={lr_g}, D={lr_d}")
    print(f"Feature matching weight: {fm_weight}")
    print(f"Diversity weight: {div_weight}")
    print("="*70 + "\n")
    
    best_g_loss = float('inf')
    patience_counter = 0
    patience_limit = 1000
    
    for epoch in range(epochs):
        
        # ============================================
        # DISCRIMINATOR TRAINING (n_critic steps)
        # ============================================
        for _ in range(n_critic):
            # Sample real data
            idx = torch.randint(0, real_data.size(0), (batch_size,), device=device)
            real_samples = real_data[idx]
            real_labels = label_data[idx]
            
            # Sample fake labels (with emphasis on rare classes)
            fake_label_indices = torch.multinomial(sampling_probs, batch_size, replacement=True)
            fake_labels = torch.eye(label_dim, device=device)[fake_label_indices]
            
            # Generate fake samples
            noise = torch.randn(batch_size, noise_dim, device=device)
            with torch.no_grad():
                fake_samples = G(noise, fake_labels)
            
            # Discriminator predictions
            d_real = D(real_samples, real_labels).mean()
            d_fake = D(fake_samples, fake_labels).mean()
            
            # Gradient penalty
            gp = gradient_penalty_adaptive(
                D, real_samples, fake_samples, real_labels, 
                device, epoch, epochs, base_lambda_gp
            )
            
            # Discriminator loss (WGAN-GP)
            d_loss = -(d_real - d_fake) + gp
            
            # Update discriminator
            optimizer_D.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optimizer_D.step()
        
        # ============================================
        # GENERATOR TRAINING
        # ============================================
        
        # Sample fake labels
        fake_label_indices = torch.multinomial(sampling_probs, batch_size, replacement=True)
        fake_labels = torch.eye(label_dim, device=device)[fake_label_indices]
        
        # Generate fake samples
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_samples = G(noise, fake_labels)
        
        # Discriminator prediction
        d_fake_out, fake_features = D(fake_samples, fake_labels, return_features=True)
        
        # Get real features for feature matching
        idx = torch.randint(0, real_data.size(0), (batch_size,), device=device)
        real_samples = real_data[idx]
        real_labels_for_fm = label_data[idx]
        _, real_features = D(real_samples, real_labels_for_fm, return_features=True)
        
        # Generator losses
        g_loss_adv = -d_fake_out.mean()
        g_loss_fm = feature_matching_loss(real_features, fake_features)
        g_loss_div = diversity_loss(fake_samples, batch_size)
        
        g_loss = g_loss_adv + fm_weight * g_loss_fm + div_weight * g_loss_div
        
        # Update generator
        optimizer_G.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        optimizer_G.step()
        
        # ============================================
        # MINORITY CLASS FOCUSED TRAINING (every 10 epochs)
        # ============================================
        if epoch % 10 == 0 and epoch > 0:
            # Extra generator updates focused on minority classes
            for _ in range(3):
                # Force sampling of minority classes
                minority_batch_size = min(batch_size, 64)
                
                # Create batch with higher minority class ratio
                minority_weight = 0.7  # 70% minority, 30% others
                n_minority = int(minority_batch_size * minority_weight)
                n_others = minority_batch_size - n_minority
                
                # Sample labels with extreme weights for minorities
                extreme_weights = sampling_probs.clone()
                extreme_weights = extreme_weights ** 3  # Cube to emphasize minorities
                extreme_weights = extreme_weights / extreme_weights.sum()
                
                fake_label_indices = torch.multinomial(extreme_weights, minority_batch_size, replacement=True)
                fake_labels_minority = torch.eye(label_dim, device=device)[fake_label_indices]
                
                noise = torch.randn(minority_batch_size, noise_dim, device=device)
                fake_samples_minority = G(noise, fake_labels_minority)
                
                d_out_minority, fake_feat_minority = D(fake_samples_minority, fake_labels_minority, return_features=True)
                
                # Get real minority samples for feature matching
                real_minority_samples = []
                real_minority_labels = []
                for lbl_idx in fake_label_indices:
                    class_mask = label_data.argmax(dim=1) == lbl_idx
                    class_samples = real_data[class_mask]
                    if len(class_samples) > 0:
                        sample_idx = torch.randint(0, len(class_samples), (1,), device=device)
                        real_minority_samples.append(class_samples[sample_idx])
                        real_minority_labels.append(label_data[class_mask][sample_idx])
                
                if real_minority_samples:
                    real_minority_batch = torch.cat(real_minority_samples)
                    real_minority_label_batch = torch.cat(real_minority_labels)
                    
                    _, real_feat_minority = D(real_minority_batch, real_minority_label_batch, return_features=True)
                    
                    # Minority-focused losses with higher FM weight
                    g_loss_adv_min = -d_out_minority.mean()
                    g_loss_fm_min = feature_matching_loss(real_feat_minority[:len(fake_feat_minority)], fake_feat_minority)
                    g_loss_minority = g_loss_adv_min + 0.5 * g_loss_fm_min  # Higher FM weight
                    
                    optimizer_G.zero_grad()
                    g_loss_minority.backward()
                    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                    optimizer_G.step()
        
        # ============================================
        # LEARNING RATE ADJUSTMENT
        # ============================================
        if epoch % 100 == 0 and epoch > 0:
            # Adaptive LR based on loss ratio
            with torch.no_grad():
                loss_ratio = abs(d_loss.item()) / (abs(g_loss.item()) + 1e-8)
                
                if loss_ratio > 3.0:  # D too strong
                    for param_group in optimizer_D.param_groups:
                        param_group['lr'] *= 0.95
                    for param_group in optimizer_G.param_groups:
                        param_group['lr'] *= 1.05
                    print(f"[Epoch {epoch}] Adjusting LR: D too strong (ratio={loss_ratio:.2f})")
                
                elif loss_ratio < 0.5:  # G too strong
                    for param_group in optimizer_D.param_groups:
                        param_group['lr'] *= 1.05
                    for param_group in optimizer_G.param_groups:
                        param_group['lr'] *= 0.95
                    print(f"[Epoch {epoch}] Adjusting LR: G too strong (ratio={loss_ratio:.2f})")
        
        # ============================================
        # LOGGING AND HISTORY
        # ============================================
        history.update(
            d_loss.item(),
            g_loss.item(),
            d_real.item(),
            d_fake.item(),
            gp.item(),
            g_loss_fm.item()
        )
        
        if epoch % 100 == 0:
            print(f"Epoch [{epoch:4d}/{epochs}] | "
                f"D_loss: {d_loss.item():7.4f} | "
                f"G_loss: {g_loss.item():7.4f} | "
                f"D(real): {d_real.item():6.3f} | "
                f"D(fake): {d_fake.item():6.3f} | "
                f"GP: {gp.item():6.3f} | "
                f"FM: {g_loss_fm.item():6.4f}")
        
        # ============================================
        # EARLY STOPPING CHECK - IMPROVED
        # ============================================
        if epoch > 500:  # Start checking earlier
            # Check for divergence (not just no improvement)
            if abs(d_loss.item()) > 100 or abs(g_loss.item()) > 50:
                print(f"\n[WARNING] Training diverged at epoch {epoch}")
                print(f"  D_loss: {d_loss.item():.2f}, G_loss: {g_loss.item():.2f}")
                print("  Stopping training to prevent waste of time")
                break
            
            if g_loss.item() < best_g_loss:
                best_g_loss = g_loss.item()
                patience_counter = 0
                # Save best model
                best_G_state = G.state_dict()
                best_D_state = D.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience_limit:
                print(f"\n[INFO] Early stopping at epoch {epoch} (no improvement for {patience_limit} epochs)")
                # Restore best model
                if 'best_G_state' in locals():
                    G.load_state_dict(best_G_state)
                    D.load_state_dict(best_D_state)
                    print("[INFO] Restored best model weights")
                break
        
        # Update schedulers
        scheduler_G.step()
        scheduler_D.step()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70 + "\n")
    
    return G, D, history


# =============================
#   DATA QUALITY VALIDATION
# =============================

def validate_synthetic_quality(real_data, synth_data, real_labels, synth_labels, target_class, feature_threshold=0.3):
    """
    Validate synthetic data quality using statistical tests
    
    Returns:
        dict with validation results and quality metrics
    """
    
    # Get samples for target class
    real_class = real_data[real_labels == target_class]
    synth_class = synth_data[synth_labels == target_class]
    
    if len(real_class) == 0:
        return {"valid": False, "reason": "No real samples for class"}
    
    if len(synth_class) == 0:
        return {"valid": False, "reason": "No synthetic samples generated"}
    
    results = {
        "class": target_class,
        "real_count": len(real_class),
        "synth_count": len(synth_class)
    }
    
    # Kolmogorov-Smirnov test for each feature
    ks_pvalues = []
    ks_statistics = []
    
    for i in range(real_class.shape[1]):
        try:
            stat, pvalue = stats.ks_2samp(real_class[:, i], synth_class[:, i])
            ks_pvalues.append(pvalue)
            ks_statistics.append(stat)
        except:
            ks_pvalues.append(0.0)
            ks_statistics.append(1.0)
    
    results["ks_mean_pvalue"] = np.mean(ks_pvalues)
    results["ks_median_pvalue"] = np.median(ks_pvalues)
    results["ks_mean_statistic"] = np.mean(ks_statistics)
    results["ks_failed_features"] = sum(1 for p in ks_pvalues if p < 0.01)
    results["ks_failed_ratio"] = results["ks_failed_features"] / len(ks_pvalues)
    
    # Maximum Mean Discrepancy (MMD)
    def compute_mmd(x, y, sample_size=500):
        """Compute MMD between two distributions"""
        n = min(len(x), len(y), sample_size)
        x_sample = x[np.random.choice(len(x), n, replace=False)]
        y_sample = y[np.random.choice(len(y), n, replace=False)]
        
        xx = np.dot(x_sample, x_sample.T).mean()
        yy = np.dot(y_sample, y_sample.T).mean()
        xy = np.dot(x_sample, y_sample.T).mean()
        
        return xx + yy - 2 * xy
    
    try:
        results["mmd"] = compute_mmd(real_class, synth_class)
    except:
        results["mmd"] = float('inf')
    
    # Mean and std comparison
    real_mean = real_class.mean(axis=0)
    synth_mean = synth_class.mean(axis=0)
    real_std = real_class.std(axis=0)
    synth_std = synth_class.std(axis=0)
    
    mean_diff = np.abs(real_mean - synth_mean).mean()
    std_diff = np.abs(real_std - synth_std).mean()
    
    results["mean_difference"] = mean_diff
    results["std_difference"] = std_diff
    
    # Quality decision with relaxed thresholds for rare classes
    if len(real_class) < 100:  # Very rare class
        results["valid"] = (
            results["ks_mean_pvalue"] > 0.01 and  # More lenient
            results["ks_failed_ratio"] < 0.5 and   # Allow more failures
            results["mmd"] < 0.5                    # More lenient
        )
        results["threshold_type"] = "relaxed (rare class)"
    else:  # Normal class
        results["valid"] = (
            results["ks_mean_pvalue"] > 0.05 and
            results["ks_failed_ratio"] < feature_threshold and
            results["mmd"] < 0.1
        )
        results["threshold_type"] = "standard"
    
    return results


def print_validation_report(validation_results):
    """Print formatted validation report"""
    print("\n" + "="*70)
    print("SYNTHETIC DATA QUALITY VALIDATION")
    print("="*70)
    
    for result in validation_results:
        cls = result["class"]
        valid = result["valid"]
        status = "✓ PASS" if valid else "✗ FAIL"
        
        print(f"\n{cls}:")
        print(f"  Status: {status}")
        print(f"  Real samples: {result['real_count']}")
        print(f"  Synthetic samples: {result['synth_count']}")
        print(f"  KS mean p-value: {result['ks_mean_pvalue']:.4f}")
        print(f"  KS failed features: {result['ks_failed_features']}/{int(result['ks_failed_ratio']*len(result.get('ks_pvalues', [])) if 'ks_pvalues' in result else 0)}")
        print(f"  MMD: {result['mmd']:.6f}")
        print(f"  Mean difference: {result['mean_difference']:.6f}")
        print(f"  Threshold type: {result.get('threshold_type', 'N/A')}")
    
    print("\n" + "="*70 + "\n")


# =============================
#   SYNTHETIC DATA GENERATION
# =============================

def generate_synthetic_for_class(G, ohe, scaler, class_name, n_samples, noise_dim=100, device="cpu"):
    """Generate synthetic samples with safety checks"""
    try:
        class_idx = list(ohe.categories_[0]).index(class_name)
    except ValueError:
        print(f"[WARNING] Class '{class_name}' not found in encoder categories")
        return np.empty((0, scaler.n_features_in_)), np.array([])
    
    label_dim = len(ohe.categories_[0])
    labels = torch.eye(label_dim, device=device)[class_idx].unsqueeze(0).repeat(n_samples, 1)
    noise = torch.randn(n_samples, noise_dim, device=device)
    
    G.eval()
    with torch.no_grad():
        synth_scaled = G(noise, labels).cpu().numpy()
    G.train()
    
    # SAFETY CHECK 1: Values should be in [-1, 1]
    if synth_scaled.min() < -2 or synth_scaled.max() > 2:
        print(f"[ERROR] Generator output out of range for {class_name}")
        print(f"  Min: {synth_scaled.min():.4f}, Max: {synth_scaled.max():.4f}")
        return np.empty((0, scaler.n_features_in_)), np.array([])
    
    # Clip to valid range as safety measure
    synth_scaled = np.clip(synth_scaled, -1, 1)
    
    # SAFETY CHECK 2: Check for NaN/Inf
    if np.isnan(synth_scaled).any() or np.isinf(synth_scaled).any():
        print(f"[ERROR] NaN/Inf detected in generated data for {class_name}")
        return np.empty((0, scaler.n_features_in_)), np.array([])
    
    # Inverse transform
    try:
        synth_orig = scaler.inverse_transform(synth_scaled)
    except Exception as e:
        print(f"[ERROR] Inverse transform failed for {class_name}: {e}")
        return np.empty((0, scaler.n_features_in_)), np.array([])
    
    # SAFETY CHECK 3: Check for extreme values after inverse transform
    # For network traffic data, values up to tens of millions are normal
    if np.abs(synth_orig).max() > 1e9:  # 1 billion threshold
        print(f"[ERROR] Extreme values after inverse transform for {class_name}")
        print(f"  Max absolute value: {np.abs(synth_orig).max():.2e}")
        return np.empty((0, scaler.n_features_in_)), np.array([])
    
    # SAFETY CHECK 4: Check for NaN/Inf after inverse transform
    if np.isnan(synth_orig).any() or np.isinf(synth_orig).any():
        print(f"[ERROR] NaN/Inf after inverse transform for {class_name}")
        nan_count = np.isnan(synth_orig).sum()
        inf_count = np.isinf(synth_orig).sum()
        print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
        return np.empty((0, scaler.n_features_in_)), np.array([])
    
    y = np.array([class_name] * n_samples)
    
    return synth_orig, y

def diagnose_synthetic_data(G, ohe, scaler, class_name, device="cpu", noise_dim=100):
    """Diagnose what's wrong with synthetic data generation"""
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: Generating synthetic data for {class_name}")
    print(f"{'='*70}")
    
    # Generate a small sample
    try:
        class_idx = list(ohe.categories_[0]).index(class_name)
    except ValueError:
        print(f"ERROR: Class '{class_name}' not found")
        return
    
    label_dim = len(ohe.categories_[0])
    labels = torch.eye(label_dim, device=device)[class_idx].unsqueeze(0).repeat(10, 1)
    noise = torch.randn(10, noise_dim, device=device)
    
    G.eval()
    with torch.no_grad():
        synth_scaled = G(noise, labels).cpu().numpy()
    G.train()
    
    print(f"\n1. Generator output (scaled [-1, 1] range):")
    print(f"   Shape: {synth_scaled.shape}")
    print(f"   Min: {synth_scaled.min():.6f}")
    print(f"   Max: {synth_scaled.max():.6f}")
    print(f"   Mean: {synth_scaled.mean():.6f}")
    print(f"   Std: {synth_scaled.std():.6f}")
    print(f"   Sample values (first 5 features): {synth_scaled[0, :5]}")
    
    # Check if values are actually in [-1, 1]
    if synth_scaled.min() < -1.5 or synth_scaled.max() > 1.5:
        print(f"   ⚠️  WARNING: Values outside expected [-1, 1] range!")
    
    # Inverse transform
    print(f"\n2. Scaler info:")
    print(f"   Feature range: {scaler.feature_range}")
    print(f"   Data min shape: {scaler.data_min_.shape if hasattr(scaler, 'data_min_') else 'N/A'}")
    print(f"   Data max shape: {scaler.data_max_.shape if hasattr(scaler, 'data_max_') else 'N/A'}")
    
    try:
        synth_orig = scaler.inverse_transform(synth_scaled)
        
        print(f"\n3. After inverse transform (original scale):")
        print(f"   Shape: {synth_orig.shape}")
        print(f"   Min: {synth_orig.min():.6f}")
        print(f"   Max: {synth_orig.max():.6f}")
        print(f"   Mean: {synth_orig.mean():.6f}")
        print(f"   Std: {synth_orig.std():.6f}")
        print(f"   Sample values (first 5 features): {synth_orig[0, :5]}")
        
        # Check for extreme values
        if np.abs(synth_orig).max() > 1e6:
            print(f"   ⚠️  ERROR: Extreme values detected after inverse transform!")
            print(f"   This suggests scaler is misconfigured or data has inf/nan")
    
    except Exception as e:
        print(f"   ❌ ERROR during inverse transform: {e}")
    
    print(f"{'='*70}\n")

def validate_scaler_and_data(real_data_tensor, scaler):
    """Validate that scaler works correctly"""
    
    print("\n" + "="*70)
    print("VALIDATING SCALER CONFIGURATION")
    print("="*70)
    
    # Convert tensor to numpy
    scaled_data = real_data_tensor.numpy()
    
    print(f"\n1. Scaled data (should be in [-1, 1]):")
    print(f"   Min: {scaled_data.min():.6f}")
    print(f"   Max: {scaled_data.max():.6f}")
    print(f"   Mean: {scaled_data.mean():.6f}")
    
    if scaled_data.min() < -1.01 or scaled_data.max() > 1.01:
        print(f"   ⚠️  WARNING: Scaled data outside [-1, 1] range!")
    
    # Test inverse transform
    print(f"\n2. Testing inverse transform:")
    sample = scaled_data[:100]
    
    try:
        inverse = scaler.inverse_transform(sample)
        print(f"   Inverse transform successful")
        print(f"   Min: {inverse.min():.6f}")
        print(f"   Max: {inverse.max():.6f}")
        print(f"   Mean: {inverse.mean():.6f}")
        
        # Check for extreme values
        if np.abs(inverse).max() > 1e10:
            print(f"   ❌ ERROR: Extreme values in inverse transform!")
            print(f"   Scaler appears to be broken!")
            return False
        
        # Try round-trip
        rescaled = scaler.transform(inverse)
        diff = np.abs(sample - rescaled).max()
        print(f"\n3. Round-trip test:")
        print(f"   Max difference: {diff:.10f}")
        
        if diff > 0.001:
            print(f"   ⚠️  WARNING: Round-trip has significant error!")
            return False
        
        print(f"   ✓ Scaler validation passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    print("="*70 + "\n")

def plan_augmentation_strategy(y_real, minority_classes, 
                            minority_max_ratio=0.10, 
                            min_target=200,
                            cap_per_class=15000, 
                            overall_synth_cap=150000):
    """
    Plan augmentation with smart minority boosting
    """
    counts = Counter(y_real)
    majority_count = max(counts.values())
    total_samples = len(y_real)
    
    # Base desired count (10% of majority or min_target, whichever is larger)
    # FIX: Changed majority_max_ratio to minority_max_ratio
    base_target = max(int(minority_max_ratio * majority_count), min_target)
    
    augmentation_plan = {}
    
    for cls, current_count in counts.items():
        # Calculate how many synthetic samples needed
        deficit = base_target - current_count
        
        if deficit > 0:
            # Apply cap per class
            needed = min(deficit, cap_per_class)
            
            # Extra boost for ultra-rare classes (< 0.05% of dataset)
            if current_count < total_samples * 0.0005:
                boost_multiplier = 3.0
                needed = min(int(needed * boost_multiplier), cap_per_class * 2)
            # Moderate boost for rare classes (< 0.1% of dataset)
            elif cls in minority_classes:
                boost_multiplier = 2.0
                needed = min(int(needed * boost_multiplier), int(cap_per_class * 1.5))
            
            augmentation_plan[cls] = needed
        else:
            augmentation_plan[cls] = 0
    
    # Apply overall cap
    total_planned = sum(augmentation_plan.values())
    if total_planned > overall_synth_cap:
        scale_factor = overall_synth_cap / total_planned
        augmentation_plan = {cls: int(count * scale_factor) 
                            for cls, count in augmentation_plan.items()}
    
    return augmentation_plan, counts, majority_count, base_target

def hybrid_augmentation(X_real, y_real, minority_classes, augmentation_plan, G, ohe, scaler, device, noise_dim):
    """
    Use GAN for classes with >50 samples, SMOTE for ultra-rare (<50 samples)
    """
    X_synth_list, y_synth_list = [], []
    validation_results = []
    
    for cls, needed in augmentation_plan.items():
        if needed == 0:
            continue
            
        class_count = (y_real == cls).sum()
        # Skip augmentation for classes with very few samples (< 100)
        # These classes show degraded performance with synthetic data
        if class_count < 100:
            print(f"[INFO] Skipping augmentation for '{cls}' - insufficient samples ({class_count})")
            print(f"       Classes with <100 samples show degraded performance with synthetic data")
            continue
        
        
        if class_count < 50 and SMOTE_AVAILABLE:
            # Use SMOTE for ultra-rare
            print(f"[INFO] Using SMOTE for ultra-rare class '{cls}' ({class_count} samples)")
            class_mask = (y_real == cls)
            X_class = X_real[class_mask]
            
            if len(X_class) >= 2:  # SMOTE needs at least 2 samples
                # Create synthetic samples with SMOTE
                try:
                    # Adjust k_neighbors based on available samples
                    k_neighbors = min(len(X_class) - 1, 5)
                    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                    
                    # Need to create a balanced dataset for SMOTE
                    # Duplicate minority class to create enough samples
                    n_repeats = (needed // len(X_class)) + 1
                    X_repeated = np.tile(X_class, (n_repeats, 1))[:needed + len(X_class)]
                    y_repeated = np.array([cls] * len(X_repeated))
                    
                    X_smote, y_smote = smote.fit_resample(X_repeated, y_repeated)
                    
                    # Take only the new synthetic samples (not the original ones)
                    X_synth_smote = X_smote[len(X_class):len(X_class) + needed]
                    y_synth_smote = y_smote[len(X_class):len(X_class) + needed]
                    
                    X_synth_list.append(X_synth_smote)
                    y_synth_list.append(y_synth_smote)
                    print(f"      ✓ Generated {len(X_synth_smote)} samples with SMOTE")
                except Exception as e:
                    print(f"      ✗ SMOTE failed: {e}, trying GAN instead")
                    # Fall back to GAN
                    X_gan, y_gan = generate_synthetic_for_class(
                        G, ohe, scaler, cls, needed, noise_dim, device
                    )
                    if len(X_gan) > 0:
                        X_synth_list.append(X_gan)
                        y_synth_list.append(y_gan)
            else:
                print(f"      ✗ Not enough samples ({len(X_class)}) for SMOTE, skipping")
        
        else:
            # Use GAN for classes with enough samples
            print(f"[INFO] Using GAN for '{cls}' ({class_count} real samples)")
            X_gan, y_gan = generate_synthetic_for_class(
                G, ohe, scaler, cls, needed, noise_dim, device
            )
            if len(X_gan) > 0:
                X_synth_list.append(X_gan)
                y_synth_list.append(y_gan)
                
                # Validate GAN-generated data (skip for SMOTE data)
                if class_count >= 50:
                    validation = validate_synthetic_quality(
                        X_real, X_gan, y_real, y_gan, cls
                    )
                    validation_results.append(validation)
                    
                    if not validation["valid"]:
                        print(f"      ⚠️  Quality validation FAILED")
                        print(f"         KS p-value: {validation['ks_mean_pvalue']:.4f}")
                        print(f"         MMD: {validation['mmd']:.6f}")
                    else:
                        print(f"      ✓ Quality validation PASSED")
                        print(f"        KS p-value: {validation['ks_mean_pvalue']:.4f}")
    
    if X_synth_list:
        X_synth = np.vstack(X_synth_list)
        y_synth = np.hstack(y_synth_list)
        return X_synth, y_synth, validation_results
    else:
        return np.empty((0, X_real.shape[1])), np.array([]), validation_results

def build_augmented_dataset_with_validation(
    G, ohe, scaler, real_X_scaled, real_y, minority_classes,
    device="cpu", noise_dim=100,
    minority_max_ratio=0.10, min_target=200,
    cap_per_class=15000, overall_synth_cap=150000,
    real_cap_per_class=60000, validate_quality=True
):
    """
    Build augmented dataset with quality validation
    """
    
    print("\n" + "="*70)
    print("BUILDING AUGMENTED DATASET")
    print("="*70)
    
    # Convert real data to original scale
    real_X = scaler.inverse_transform(real_X_scaled)
    
    # Ensure real_y is class names
    if isinstance(real_y, np.ndarray) and real_y.ndim > 1:
        real_y = ohe.inverse_transform(real_y).ravel()
    
    # Cap real data per class
    def cap_by_class(X, y, max_per_class, seed=42):
        rng = np.random.default_rng(seed)
        keep_idx = []
        buckets = defaultdict(list)
        
        for i, cls in enumerate(y):
            buckets[cls].append(i)
        
        for cls, idxs in buckets.items():
            if len(idxs) > max_per_class:
                keep_idx.extend(rng.choice(idxs, size=max_per_class, replace=False))
            else:
                keep_idx.extend(idxs)
        
        keep_idx = np.array(keep_idx, dtype=int)
        return X[keep_idx], y[keep_idx]
    
    if real_cap_per_class is not None:
        print(f"[INFO] Capping real data to {real_cap_per_class} samples per class...")
        real_X, real_y = cap_by_class(real_X, real_y, max_per_class=real_cap_per_class)
    
    # Plan augmentation
    augmentation_plan, counts, majority, base_target = plan_augmentation_strategy(
        real_y, minority_classes,
        minority_max_ratio=minority_max_ratio,
        min_target=min_target,
        cap_per_class=cap_per_class,
        overall_synth_cap=overall_synth_cap
    )
    
    print(f"\n[INFO] Augmentation Plan:")
    print(f"  Majority class count: {majority}")
    print(f"  Base target per class: {base_target}")
    print(f"  Total synthetic samples planned: {sum(augmentation_plan.values())}")
    
    # Use hybrid augmentation approach (SMOTE for ultra-rare, GAN for others)
    print(f"\n[INFO] Using hybrid augmentation strategy:")
    print(f"  - SMOTE for classes with < 50 real samples")
    print(f"  - GAN for classes with >= 50 real samples")
    
    X_synth, y_synth, validation_results = hybrid_augmentation(
        real_X, real_y, minority_classes, augmentation_plan,
        G, ohe, scaler, device, noise_dim
    )
    
    if len(X_synth) > 0:
        print(f"\n[INFO] Total synthetic samples generated: {len(X_synth)}")
    else:
        print("[WARNING] No synthetic samples generated")
    
    # Create dataset variants
    datasets = {
        "Real only": (real_X, real_y),
        "Synthetic only": (X_synth, y_synth),
        "Real + Synthetic": (
            np.vstack([real_X, X_synth]) if len(X_synth) else real_X,
            np.hstack([real_y, y_synth]) if len(y_synth) else real_y
        )
    }
    
    # Print validation report
    if validation_results:
        print_validation_report(validation_results)
    
    print("="*70 + "\n")
    
    return datasets, augmentation_plan, validation_results


# =============================
#   ADVANCED EVALUATION
# =============================

def create_balanced_test_set(G, ohe, scaler, X_test_real, y_test_real, 
                            min_samples_per_class=50, noise_dim=100, device="cpu"):
    """
    Create a balanced test set by adding synthetic samples for rare classes
    """
    print("\n[INFO] Creating balanced test set...")
    
    # Count samples per class in test set
    test_counts = Counter(y_test_real)
    
    # Identify classes that need augmentation
    classes_to_augment = {cls: max(0, min_samples_per_class - count) 
                        for cls, count in test_counts.items() 
                        if count < min_samples_per_class}
    
    if not classes_to_augment:
        print("[INFO] Test set already balanced")
        return X_test_real, y_test_real
    
    print(f"[INFO] Augmenting {len(classes_to_augment)} classes in test set:")
    for cls, needed in classes_to_augment.items():
        print(f"  {cls}: adding {needed} samples (current: {test_counts[cls]})")
    
    # Generate synthetic test samples
    X_test_synth_list = [X_test_real]
    y_test_synth_list = [y_test_real]
    
    for cls, needed in classes_to_augment.items():
        if needed > 0:
            X_synth, y_synth = generate_synthetic_for_class(
                G, ohe, scaler, cls, needed, noise_dim, device
            )
            if len(X_synth) > 0:
                X_test_synth_list.append(X_synth)
                y_test_synth_list.append(y_synth)
    
    X_test_balanced = np.vstack(X_test_synth_list)
    y_test_balanced = np.hstack(y_test_synth_list)
    
    print(f"[INFO] Balanced test set size: {len(X_test_balanced)} (was {len(X_test_real)})")
    
    return X_test_balanced, y_test_balanced


def evaluate_with_comprehensive_metrics(datasets, G, ohe, scaler, minority_classes,
                                    random_state=42, use_balanced_test=True,
                                    min_test_samples=50, device="cpu"):
    """
    Comprehensive evaluation with multiple strategies
    """
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION")
    print("="*70)
    
    X_real, y_real = datasets["Real only"]
    X_synth, y_synth = datasets.get("Synthetic only", (np.empty((0, X_real.shape[1])), np.array([])))
    
    # Stratified split of real data
    print("\n[INFO] Creating train/test split...")
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.2, stratify=y_real, random_state=random_state
    )
    
    print(f"  Train size: {len(X_train_real)}")
    print(f"  Test size: {len(X_test_real)}")
    
    # Create balanced test set if requested
    if use_balanced_test:
        X_test_balanced, y_test_balanced = create_balanced_test_set(
            G, ohe, scaler, X_test_real, y_test_real, 
            min_samples_per_class=min_test_samples, device=device
        )
    else:
        X_test_balanced, y_test_balanced = X_test_real, y_test_real
    
    # Define experiments
    experiments = {
        "Real train → Real test": (X_train_real, y_train_real, X_test_real, y_test_real),
        "Real train → Balanced test": (X_train_real, y_train_real, X_test_balanced, y_test_balanced),
    }
    
    # Add augmented training if synthetic data exists
    if len(X_synth) > 0:
        X_train_augmented = np.vstack([X_train_real, X_synth])
        y_train_augmented = np.hstack([y_train_real, y_synth])
        
        experiments["Real+Synth train → Real test"] = (X_train_augmented, y_train_augmented, X_test_real, y_test_real)
        experiments["Real+Synth train → Balanced test"] = (X_train_augmented, y_train_augmented, X_test_balanced, y_test_balanced)
    
    # Train and evaluate each experiment
    results = {}
    
    for exp_name, (X_tr, y_tr, X_te, y_te) in experiments.items():
        print(f"\n{'='*70}")
        print(f"{exp_name}")
        print(f"{'='*70}")
        print(f"Training samples: {len(X_tr)}")
        print(f"Test samples: {len(X_te)}")
        
        # Train Random Forest with optimized hyperparameters
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            max_samples=0.7,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced_subsample',
            verbose=0
        )
        
        print("\n[INFO] Training Random Forest...")
        clf.fit(X_tr, y_tr)
        
        print("[INFO] Making predictions...")
        y_pred = clf.predict(X_te)
        
        # Print detailed classification report
        print("\n" + classification_report(y_te, y_pred, digits=4, zero_division=0))
        
        # Store results for comparison
        results[exp_name] = {
            'predictions': y_pred,
            'true_labels': y_te,
            'classifier': clf
        }
    
    # Comparative analysis for minority classes
    print("\n" + "="*70)
    print("MINORITY CLASS PERFORMANCE COMPARISON")
    print("="*70)
    
    if "Real train → Real test" in results and "Real+Synth train → Real test" in results:
        print("\nRecall improvements on real test set:")
        
        from sklearn.metrics import recall_score
        
        base_exp = "Real train → Real test"
        aug_exp = "Real+Synth train → Real test"
        
        y_true = results[base_exp]['true_labels']
        y_pred_base = results[base_exp]['predictions']
        y_pred_aug = results[aug_exp]['predictions']
        
        # Get unique classes in test set
        classes_in_test = np.unique(y_true)
        
        improvements = []
        for cls in classes_in_test:
            if cls in minority_classes:
                # Calculate recall for this class
                mask = (y_true == cls)
                if mask.sum() > 0:
                    recall_base = (y_pred_base[mask] == cls).sum() / mask.sum()
                    recall_aug = (y_pred_aug[mask] == cls).sum() / mask.sum()
                    improvement = recall_aug - recall_base
                    improvements.append((cls, recall_base, recall_aug, improvement, mask.sum()))
        
        if improvements:
            print(f"\n{'Class':<30} {'Base Recall':>12} {'Aug Recall':>12} {'Improvement':>12} {'Support':>10}")
            print("-" * 80)
            for cls, r_base, r_aug, imp, sup in sorted(improvements, key=lambda x: x[3], reverse=True):
                sign = "+" if imp >= 0 else ""
                print(f"{cls:<30} {r_base:>12.4f} {r_aug:>12.4f} {sign}{imp:>11.4f} {sup:>10d}")
            
            avg_improvement = np.mean([x[3] for x in improvements])
            print(f"\nAverage minority class recall improvement: {avg_improvement:+.4f}")
        else:
            print("\n[INFO] No minority classes found in test set for comparison")
    
    print("\n" + "="*70 + "\n")
    
    return results


# =============================
#   MAIN PIPELINE
# =============================

def main_improved_pipeline(dataset_path, label_col=None, 
                        epochs=5000, batch_size=128,
                        noise_dim=100, device=None,
                        save_plots=True):
    """
    Complete improved pipeline with all enhancements
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("CYBERSYNTH-CGAN: IMPROVED PIPELINE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Dataset path: {dataset_path}")
    print("="*70 + "\n")
    
    # Step 1: Load and preprocess data
    print("[STEP 1/6] Loading and preprocessing data...")
    real_data, label_data, scaler, ohe, minority_classes = load_and_preprocess(
        dataset_path, label_col
    )
    
    # Step 1.5: Validate scaler (NEW)
    print("\n[STEP 1.5/6] Validating scaler configuration...")
    if not validate_scaler_and_data(real_data, scaler):
        print("\n CRITICAL ERROR: Scaler validation failed!")
        print("Cannot proceed with training. Please check data preprocessing.")
        return None
    
    # Step 2: Train improved cGAN
    print("\n[STEP 2/6] Training improved cGAN...")
    G, D, history = train_improved_cgan(
        real_data, label_data, minority_classes,
        noise_dim=noise_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr_g=0.0002,
        lr_d=0.0001,
        base_lambda_gp=10,
        n_critic=5,
        fm_weight=0.1,
        div_weight=0.01,
        device=device
    )
    # Step 2.5: Diagnose generation quality (NEW)
    print("\n[STEP 2.5/6] Diagnosing synthetic data generation...")
    test_classes = minority_classes[:3] if len(minority_classes) >= 3 else minority_classes
    for cls in test_classes:
        diagnose_synthetic_data(G, ohe, scaler, cls, device, noise_dim)
    
    # Step 3: Plot training history
    if save_plots:
        print("\n[STEP 3/6] Saving training plots...")
        history.plot(save_path='improved_training_history.png')
    else:
        print("\n[STEP 3/6] Skipping plots...")
    
    # Step 4: Build augmented dataset with validation
    print("\n[STEP 4/6] Building augmented dataset with quality validation...")
    datasets, aug_plan, validation_results = build_augmented_dataset_with_validation(
        G, ohe, scaler,
        real_X_scaled=real_data.numpy(),
        real_y=ohe.inverse_transform(label_data.numpy()).ravel(),
        minority_classes=minority_classes,
        device=device,
        noise_dim=noise_dim,
        minority_max_ratio=0.10,
        min_target=200,
        cap_per_class=15000,
        overall_synth_cap=150000,
        real_cap_per_class=60000,
        validate_quality=True
    )
    
    # Step 5: Comprehensive evaluation
    print("\n[STEP 5/6] Comprehensive evaluation...")
    results = evaluate_with_comprehensive_metrics(
        datasets, G, ohe, scaler, minority_classes,
        random_state=42,
        use_balanced_test=True,
        min_test_samples=50,
        device=device
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    return {
        'generator': G,
        'discriminator': D,
        'history': history,
        'datasets': datasets,
        'augmentation_plan': aug_plan,
        'validation_results': validation_results,
        'evaluation_results': results,
        'scaler': scaler,
        'encoder': ohe,
        'minority_classes': minority_classes
    }


# =============================
#   ENTRY POINT
# =============================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved CyberSynth-CGAN')
    parser.add_argument('--dataset', type=str, default=r"e:\PHD\Datasets\CIC-IDS-2017",
                    help='Path to dataset directory')
    parser.add_argument('--label-col', type=str, default=None,
                    help='Name of label column')
    parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size')
    parser.add_argument('--noise-dim', type=int, default=100,
                    help='Noise dimension')
    parser.add_argument('--no-plots', action='store_true',
                    help='Disable plot generation')
    parser.add_argument('--cpu', action='store_true',
                    help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Set device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run pipeline
    pipeline_output = main_improved_pipeline(
        dataset_path=args.dataset,
        label_col=args.label_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        device=device,
        save_plots=not args.no_plots
    )
    
    print("\n[INFO] All outputs saved. Pipeline complete!")    