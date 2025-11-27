import os
import pickle
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# Try importing joblib for scaler loading
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. Model definitions
# =========================
class Classifier(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.noise_std = 0.05
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(64, classes)

    def add_noise(self, x):
        return x + self.noise_std * torch.randn_like(x)

    def forward(self, x, return_embed=False):
        x = self.add_noise(x)
        emb = self.net(x)
        logits = self.fc(emb)
        return (logits, emb) if return_embed else logits

class TripletNet(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )
    def forward(self, x):
        return self.fc(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return x + self.fc2(self.relu(self.fc1(x)))

class MetaDetector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 128)
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.res1(x)
        x = self.res2(x)
        return self.fc2(x)

# =========================
# 2. Utility: load artifacts
# =========================
def safe_load_pickle(filepath):
    """Try loading with pickle first, then joblib if available"""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        if HAS_JOBLIB:
            try:
                return joblib.load(filepath)
            except Exception as je:
                raise RuntimeError(f"Failed to load {filepath} with both pickle and joblib: {e}, {je}")
        else:
            raise RuntimeError(f"Failed to load {filepath}: {e}")

@st.cache_resource
def load_artifacts(models_dir="saved_models"):
    # required numpy artifacts
    X_test = np.load(os.path.join(models_dir, "X_test.npy"))
    y_test = np.load(os.path.join(models_dir, "y_test.npy"))
    
    scaler = safe_load_pickle(os.path.join(models_dir, "scaler.pkl"))
    
    le_path_pkl = os.path.join(models_dir, "label_encoder.pkl")
    le_path_pt = os.path.join(models_dir, "label_encoder.pt")
    
    if os.path.exists(le_path_pkl):
        label_encoder = safe_load_pickle(le_path_pkl)
    elif os.path.exists(le_path_pt):
        label_encoder = torch.load(le_path_pt, map_location=DEVICE)
    else:
        raise FileNotFoundError("label_encoder not found (.pkl or .pt)")

    threshold_path = os.path.join(models_dir, "optimal_threshold.npy")
    if os.path.exists(threshold_path):
        optimal_threshold = float(np.load(threshold_path))
    else:
        optimal_threshold = 0.0
        # in streamlit context we can't call st.warning here (cache context),
        # will show a message later if needed.

    dim = X_test.shape[1]
    n_classes = len(label_encoder.classes_)

    classifier = Classifier(dim, n_classes).to(DEVICE)
    classifier.load_state_dict(torch.load(os.path.join(models_dir, "classifier.pt"), map_location=DEVICE))
    classifier.eval()

    meta_detector = MetaDetector(64).to(DEVICE)
    meta_detector.load_state_dict(torch.load(os.path.join(models_dir, "meta_detector.pt"), map_location=DEVICE))
    meta_detector.eval()

    triplet_state = torch.load(os.path.join(models_dir, "triplet_model.pt"), map_location=DEVICE)
    in_dim = triplet_state["fc.0.weight"].shape[1]
    triplet_model = TripletNet(in_dim).to(DEVICE)
    triplet_model.load_state_dict(triplet_state)
    triplet_model.eval()

    return X_test, y_test, scaler, label_encoder, classifier, triplet_model, meta_detector, optimal_threshold

# =========================
# 3. Feature extraction & PGD
# =========================
criterion = nn.CrossEntropyLoss()

def extract_features(model, x, y):
    """
    x: torch tensor (batch, dim)
    y: torch tensor (batch,)
    returns: features tensor detached on cpu
    """
    x = x.to(DEVICE)
    x.requires_grad_(True)

    logits, emb = model(x, return_embed=True)
    probs = torch.softmax(logits, dim=1)

    top2 = torch.topk(logits, 2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).unsqueeze(1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1, keepdim=True)

    loss = criterion(logits, y.to(DEVICE))
    grad = torch.autograd.grad(loss, x)[0]
    grad_flat = grad.view(grad.size(0), -1)
    grad_norm = grad_flat.norm(dim=1, keepdim=True)
    grad_mean = grad_flat.abs().mean(dim=1, keepdim=True)
    grad_var = grad_flat.abs().var(dim=1, keepdim=True)

    feat = torch.cat([
        emb, logits, probs, margin, entropy,
        grad_mean, grad_var, grad_norm
    ], dim=1)

    return feat.detach().cpu()

def pgd_multi_eps(model, x, y, eps_list=(0.05, 0.1, 0.15), alpha=0.02, iters=10):
    """
    returns concatenated adversarial examples for each epsilon in eps_list along dim=0
    If input x has shape (B, D) then output shape is (B * len(eps_list), D) ordered by eps.
    """
    all_adv = []
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    for eps in eps_list:
        x0 = x.detach()
        x_adv = x0.clone()

        for _ in range(iters):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]

            with torch.no_grad():
                x_adv = x_adv + alpha * grad.sign()
                x_adv = x0 + torch.clamp(x_adv - x0, -eps, eps)

        all_adv.append(x_adv.detach().cpu())

    return torch.cat(all_adv, dim=0)

def meta_predict(triplet_model, meta_detector, features, threshold=0.0):
    """
    features: torch tensor on cpu (N, feat_dim)
    returns scores (numpy) and binary predictions (numpy)
    """
    with torch.no_grad():
        features_t = features.to(DEVICE)
        trip = triplet_model(features_t)
        logits = meta_detector(trip)
        scores = logits.view(-1).cpu().numpy()

    predictions = (scores > threshold).astype(int)
    return scores, predictions

# =========================
# 4. Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="PBMC3K Adversarial Detection", layout="wide")

    st.title("🧬 PBMC3K Adversarial Example Detection Demo")

    st.markdown("""
This interactive demo shows how adversarial attacks affect a gene-expression classifier,
and how a **meta-detector** identifies manipulated inputs using internal classifier signals.
""")

    st.info("""
### What this system does:
- The **main classifier** predicts the cell cluster from PBMC gene-expression data.
- A **PGD adversarial attack** perturbs the input slightly to fool the classifier.
- The **meta-detector** examines the classifier’s internal behavior to detect adversarial inputs.
""")

    try:
        (X_test, y_test, scaler, label_encoder,
         classifier, triplet_model, meta_detector, optimal_threshold) = load_artifacts()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    st.sidebar.markdown("**Detection Threshold**")
    st.sidebar.metric("Optimal Threshold", f"{optimal_threshold:.4f}")
    st.sidebar.caption("If the meta-detector score > threshold → sample is flagged as adversarial.")
    st.sidebar.markdown("---")

    idx = st.sidebar.slider("Select a test cell index", 0, len(X_test)-1, 0)
    generate_adv = st.sidebar.checkbox("Generate PGD adversarial example", value=True)
    show_raw_vector = st.sidebar.checkbox("Show gene vector (first 50 dims)", value=False)

    col1, col2 = st.columns(2)

    # CLEAN INPUT PANEL
    with col1:
        st.subheader("✅ Clean Input")
        st.caption("The unmodified gene expression vector is fed to the classifier.")

        x_raw = X_test[idx:idx+1]
        y_true = int(y_test[idx])
        x_scaled = scaler.transform(x_raw).astype(np.float32)
        x_tensor = torch.from_numpy(x_scaled)

        with torch.no_grad():
            logits, emb = classifier(x_tensor.to(DEVICE), return_embed=True)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(probs.argmax())

        st.metric("True Cluster", f"{y_true} ({label_encoder.classes_[y_true]})")
        st.metric("Predicted Cluster", f"{pred_class} ({label_encoder.classes_[pred_class]})")

        st.write("**Prediction Confidence:**")
        st.caption("How confident the classifier is about each possible cell cluster.")
        st.bar_chart(probs)

        y_tensor = torch.tensor([y_true], dtype=torch.long)
        features_clean = extract_features(classifier, x_tensor, y_tensor)

        scores_clean, meta_pred_clean = meta_predict(
            triplet_model, meta_detector, features_clean, threshold=optimal_threshold)

        status = "🚨 Adversarial" if meta_pred_clean[0] else "✅ Non-adversarial"
        st.metric("Meta-detector Result", status, delta=f"score: {scores_clean[0]:.3f}")
        st.caption("""
The meta-detector examines internal classifier signals (embeddings, logits, gradients)
to check if the input behaves unusually — a sign of adversarial manipulation.
""")

        if show_raw_vector:
            with st.expander("View Scaled Gene Expression Vector"):
                st.write(x_scaled[0][:50])

    # ADVERSARIAL INPUT PANEL
    with col2:
        st.subheader("⚠️ Adversarial Input")
        st.caption("""
A PGD attack slightly perturbs the gene vector to fool the classifier.
""")

        if generate_adv:
            y_tensor = torch.tensor([y_true], dtype=torch.long)
            adv_tensor = pgd_multi_eps(classifier, x_tensor, y_tensor)
            adv_single = adv_tensor[-1:].detach()

            with torch.no_grad():
                logits_adv, _ = classifier(adv_single.to(DEVICE), return_embed=True)
                probs_adv = torch.softmax(logits_adv, dim=1).cpu().numpy()[0]
                pred_adv = int(probs_adv.argmax())

            st.metric("Adversarial Prediction", f"{pred_adv} ({label_encoder.classes_[pred_adv]})")

            if pred_adv != y_true:
                st.warning(f"⚠️ Attack successful! Prediction changed from {y_true} → {pred_adv}")
            else:
                st.success("Attack failed — classifier prediction unchanged.")

            st.write("**Adversarial Confidence:**")
            st.bar_chart(probs_adv)

            y_adv_rep = torch.tensor([y_true], dtype=torch.long)
            # features for the adversarial single example (last epsilon)
            features_adv = extract_features(classifier, adv_single.cpu(), y_adv_rep)

            scores_adv, meta_pred_adv = meta_predict(
                triplet_model, meta_detector, features_adv, threshold=optimal_threshold)

            status_adv = "🚨 Adversarial" if meta_pred_adv[0] else "✅ Non-adversarial"
            st.metric("Meta-detector Result", status_adv, delta=f"score: {scores_adv[0]:.3f}")
        else:
            st.info("Enable PGD generation to see the adversarial attack results.")

    # GLOBAL METRICS
    st.markdown("---")
    st.subheader("📊 Global Performance Metrics")
    st.caption("""
These metrics summarize how well the classifier and meta-detector perform across the full test set.
""")

    if st.button("🔄 Compute Full Test Set Evaluation"):
        with st.spinner("Running evaluation..."):
            # --- classifier predictions (clean)
            X_scaled = scaler.transform(X_test).astype(np.float32)
            X_tensor = torch.from_numpy(X_scaled)
            y_tensor = torch.from_numpy(y_test.astype(np.int64))

            with torch.no_grad():
                logits_all = classifier(X_tensor.to(DEVICE))
                preds_all = logits_all.argmax(1).cpu().numpy()

            clf_acc = accuracy_score(y_test, preds_all)

            # --- meta-detector on full clean set (features computed from full set)
            feats_clean = extract_features(classifier, X_tensor, y_tensor)  # (N, feat_dim)
            scores_c, meta_c = meta_predict(triplet_model, meta_detector, feats_clean, threshold=optimal_threshold)
            # meta_c == 0 indicates clean predicted
            meta_clean_acc = (meta_c == 0).mean()

            # --- adversarial: pick a subset to keep computation reasonable (200 or all if small)
            subset = np.random.choice(len(X_test), size=min(200, len(X_test)), replace=False)
            xs = torch.from_numpy(X_scaled[subset])
            ys = torch.from_numpy(y_test[subset].astype(np.int64))

            # generate adversarial examples (3 epsilons stacked)
            adv_batch = pgd_multi_eps(classifier, xs, ys)  # shape (3*B, D)
            ys_rep = ys.repeat(3)

            feats_adv = extract_features(classifier, adv_batch, ys_rep)
            scores_a, meta_a = meta_predict(triplet_model, meta_detector, feats_adv, threshold=optimal_threshold)
            meta_adv_acc = (meta_a == 1).mean()

            # --- confusion matrix for meta-detector on evaluated data
            y_true_cm = np.concatenate([np.zeros_like(meta_c), np.ones_like(meta_a)])
            y_pred_cm = np.concatenate([meta_c, meta_a])
            cm = confusion_matrix(y_true_cm, y_pred_cm)

            # --- prepare triplet embeddings for UMAP (use full clean feat and adv subset feats)
            try:
                clean_trip = triplet_model(feats_clean.to(DEVICE)).detach().cpu().numpy()
                adv_trip = triplet_model(feats_adv.to(DEVICE)).detach().cpu().numpy()

            except Exception:
                # fallback small pieces to avoid memory errors
                clean_trip = triplet_model(feats_clean[:1000].to(DEVICE)).detach().cpu().numpy()
                adv_trip = triplet_model(feats_adv.to(DEVICE)).detach().cpu().numpy()


            # --- per-class clean accuracy
            n_classes = len(label_encoder.classes_)
            class_acc = []
            for cls in range(n_classes):
                mask = (y_test == cls)
                if mask.sum() == 0:
                    class_acc.append(0.0)
                else:
                    class_acc.append((preds_all[mask] == cls).mean())

            class_acc = np.array(class_acc)

            # --- per-class flip rate using our subset adversarial results
            # compute adversarial predictions for each epsilon block
            with torch.no_grad():
                logits_adv_all = classifier(adv_batch.to(DEVICE))
                preds_adv_all = logits_adv_all.argmax(1).cpu().numpy()  # length 3*B

            # reshape to (3, B)
            try:
                preds_adv_reshaped = preds_adv_all.reshape(3, -1)
            except Exception:
                # safety
                B = len(subset)
                preds_adv_reshaped = preds_adv_all[:3*B].reshape(3, -1)

            # clean preds for subset
            clean_preds_subset = preds_all[subset]  # (B,)
            flipped_any = (preds_adv_reshaped != clean_preds_subset).any(axis=0)  # (B,)
            # compute flip rate per class (only over samples present in subset)
            flip_rates = np.zeros(n_classes)
            for cls in range(n_classes):
                idxs = np.where(ys.cpu().numpy() == cls)[0]
                if len(idxs) == 0:
                    flip_rates[cls] = np.nan  # no samples in subset
                else:
                    flip_rates[cls] = flipped_any[idxs].mean()

            # --- ROC curve and optimal threshold
            scores_all = np.concatenate([scores_c, scores_a])
            labels_all = np.concatenate([np.zeros_like(scores_c), np.ones_like(scores_a)])
            fpr, tpr, thresholds = roc_curve(labels_all, scores_all)
            roc_auc = auc(fpr, tpr)
            youden = tpr - fpr
            idx = np.nanargmax(youden)
            optimal_thresh_val = thresholds[idx]

        # summary metrics
        colA, colB, colC = st.columns(3)
        colA.metric("Classifier Accuracy", f"{clf_acc:.1%}")
        colA.caption("How accurately the classifier predicts true cell clusters.")

        colB.metric("Meta Clean Detection", f"{meta_clean_acc:.1%}")
        colB.caption("Percentage of normal samples correctly recognized as clean.")

        colC.metric("Meta Adv Detection", f"{meta_adv_acc:.1%}")
        colC.caption("Percentage of adversarial inputs correctly detected.")

        # ---------------------------------------
        # VISUALIZATIONS (each inside an expander)
        # ---------------------------------------
        st.markdown("---")
        st.subheader("📈 Detailed Visual Diagnostics")

        # 1. Confusion Matrix
        with st.expander("1️⃣ Confusion Matrix (Meta-Detector Performance)"):
            st.write("""
            **What this shows:**  
            The confusion matrix compares the meta-detector's predicted labels vs. the true labels  
            (0 = clean, 1 = adversarial).

            **How to read:**  
            - **Top-left (True Negative):** Clean samples correctly recognized  
            - **Top-right (False Positive):** Clean samples incorrectly flagged as adversarial  
            - **Bottom-left (False Negative):** Adversarial samples missed  
            - **Bottom-right (True Positive):** Adversarial samples correctly detected  
            """)
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Meta-Model Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # 2. Meta-Model Score Distribution
        with st.expander("2️⃣ Meta-Model Score Distribution"):
            st.write("""
            **Purpose:**  
            Visualizes how clean vs adversarial samples distribute across meta-model scores.

            **Interpretation:**  
            - Clean inputs cluster at **lower scores**  
            - Adversarial inputs shift to the **right (higher scores)**  
            - The dashed line is the **decision threshold**:  
              - Score > threshold → flagged as adversarial  
              - Score < threshold → considered clean
            """)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(scores_c, bins=50, alpha=0.6, color="green", label="Clean")
            ax.hist(scores_a, bins=50, alpha=0.6, color="red", label="Adversarial")
            ax.axvline(x=optimal_threshold, linestyle="--", color="black", label="Decision Boundary")
            ax.set_title("Meta-Model Score Distribution")
            ax.set_xlabel("Meta-Model Score")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)

        # 3. UMAP of Triplet Embeddings
        with st.expander("3️⃣ UMAP of Triplet Embeddings"):
            st.write("""
            **What this shows:**  
            A 2D visualization of the meta-feature embeddings learned by the triplet network.

            **What to infer:**  
            - Clean samples should form more compact pockets  
            - Adversarial samples often spread and overlap — indicating embedding distortion  
            """)
            try:
                all_trip = np.vstack([clean_trip, adv_trip])
                labels_trip = np.array([0]*len(clean_trip) + [1]*len(adv_trip))
                um = umap.UMAP(random_state=42)
                embed_2d = um.fit_transform(all_trip)

                fig, ax = plt.subplots(figsize=(7, 7))
                ax.scatter(embed_2d[labels_trip==0, 0], embed_2d[labels_trip==0, 1],
                            s=10, alpha=0.7, label="Clean", color="green")
                ax.scatter(embed_2d[labels_trip==1, 0], embed_2d[labels_trip==1, 1],
                            s=10, alpha=0.7, label="Adversarial", color="red")
                ax.set_title("UMAP of Triplet Embeddings (Clean vs Adversarial)")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.write("UMAP failed (possibly too large). Error:", e)
                st.info("Try reducing test set size or precomputing UMAP embeddings offline.")

        # 4. Per-Class Clean Accuracy
        with st.expander("4️⃣ Per-Class Clean Accuracy"):
            st.write("""
            **What this shows:**  
            How accurately the classifier predicts each PBMC cell cluster **without attack**.

            **Interpretation:**  
            - Higher bars → class is easy to classify  
            - Lower bars → class is difficult / noisy
            """)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(range(len(class_acc)), class_acc)
            ax.set_title("Per-Class Clean Accuracy")
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1.05)
            st.pyplot(fig)

        # 5. Per-Class Flip Rate
        with st.expander("5️⃣ Per-Class Flip Rate (Adversarial Vulnerability)"):
            st.write("""
            **Purpose:**  
            Indicates how easily each cluster gets fooled by adversarial perturbations (computed on the evaluation subset).

            **Interpretation:**  
            - Flip rate = 1.0 → almost always fooled  
            - Flip rate = 0.0 → robust to attack  
            """)
            fig, ax = plt.subplots(figsize=(7, 4))
            # replace nan with 0 for plotting but indicate missing with hatch
            plot_rates = np.nan_to_num(flip_rates, nan=0.0)
            colors = plt.cm.Reds(np.linspace(0.3, 1, len(plot_rates)))
            ax.bar(range(len(plot_rates)), plot_rates, color=colors)
            ax.set_title("Per-Class Flip Rate")
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("Flip Rate")
            ax.set_ylim(0, 1.05)
            st.pyplot(fig)
            st.caption("Note: Flip rates computed on the random subset used for adversarial evaluation; classes missing in the subset show as 0.0 here.")

        # 6. ROC Curve + Threshold Calibration
        with st.expander("6️⃣ ROC Curve & Threshold Calibration"):
            st.write("""
            **Purpose:**  
            Shows how well the meta-detector separates clean vs. adversarial samples  
            at all possible thresholds.

            **How to read:**  
            - **AUC** closer to 1 → excellent detector  
            - **Diagonal line** → random guessing  
            - Red dot → threshold where (TPR - FPR) is maximized (Youden's J)
            """)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, linewidth=3, label=f"ROC Curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")
            ax.scatter(fpr[idx], tpr[idx], s=80, color="red",
                       label=f"Optimal Threshold = {optimal_thresh_val:.3f}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve - Meta-Detector Threshold Calibration")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

    else:
        st.info("Click the button to evaluate the full test set.")

if __name__ == "__main__":
    main()
