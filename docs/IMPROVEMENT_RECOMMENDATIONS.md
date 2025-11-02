# Comprehensive Improvement Plan for Ocular Disease Recognition

**Current Performance:** 89.22% label accuracy (but only 48.75% sample accuracy)

**Key Problems Identified:**
1. Systematic under-prediction (high false negatives)
2. Poor multi-label performance (48.75% sample accuracy)
3. AMD and Hypertension completely fail (F1 < 0.50)
4. High miss rates for Diabetes (54%), Cataract (49%), AMD (61%), Hypertension (84%)

---

## Root Cause Analysis

### Why Is Performance Disappointing?

1. **Class Imbalance Not Properly Addressed**
   - Hypertension: Only 45 samples (3.2%)
   - Myopia: Only 50 samples (3.6%)
   - Model learns to predict "absent" for rare diseases

2. **Multi-Label Complexity**
   - 88% of samples are single-label
   - Model doesn't learn multi-disease patterns well
   - No explicit multi-label loss weighting

3. **Single Threshold (0.5) for All Diseases**
   - Optimal threshold varies by disease
   - Rare diseases need lower thresholds
   - Current approach is suboptimal

4. **Limited Data Augmentation During Training**
   - Only basic augmentations used
   - No disease-specific augmentations
   - TTA helps but should be in training too

5. **Metadata Integration Failed**
   - Enhanced metadata (18 features) provided no gain
   - Tiny refinement network (4K params) can't utilize features
   - Age/gender alone insufficient

---

## Improvement Recommendations

## 1. DATA PREPROCESSING IMPROVEMENTS

### A. Advanced Image Preprocessing

```python
# Current: Basic resize and normalize
# Recommended: Disease-specific preprocessing

class DiseaseSpecificPreprocessing:
    """Apply preprocessing tailored to each disease type."""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    def enhance_blood_vessels(self, image):
        """Enhance vasculature for diabetic retinopathy, hypertension."""
        green_channel = image[:, :, 1]  # Green channel best for vessels
        enhanced = self.clahe.apply(green_channel)
        return enhanced
    
    def enhance_macula(self, image):
        """Enhance central region for AMD detection."""
        # Crop and enhance center 60% of image
        h, w = image.shape[:2]
        center_crop = image[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        enhanced = cv2.detailEnhance(center_crop, sigma_s=10, sigma_r=0.15)
        return enhanced
    
    def remove_noise(self, image):
        """Denoise while preserving edges."""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def detect_optic_disc(self, image):
        """Locate and crop around optic disc for glaucoma."""
        # Use circular Hough transform to find optic disc
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=10, maxRadius=50)
        return circles
```

**Expected Gain:** +2-3% overall, +5-10% for specific diseases

### B. Proper Class Balancing

```python
# Current: No balancing
# Recommended: Multi-strategy approach

# Strategy 1: Oversample minority classes
from imblearn.over_sampling import SMOTE, ADASYN

# For each rare disease, oversample to at least 150 samples
def balance_dataset(X, y, label_columns):
    """Balance each disease label independently."""
    balanced_data = []
    
    for i, disease in enumerate(label_columns):
        if y[:, i].sum() < 150:  # If fewer than 150 positive samples
            # Use SMOTE or duplicate samples
            positive_indices = np.where(y[:, i] == 1)[0]
            needed = 150 - len(positive_indices)
            
            # Duplicate with augmentation
            for _ in range(needed):
                idx = np.random.choice(positive_indices)
                augmented_image = augment_image(X[idx])
                balanced_data.append((augmented_image, y[idx]))
    
    return balanced_data

# Strategy 2: Class weights in loss function
class_weights = {}
for i, label in enumerate(LABEL_COLUMNS):
    pos_count = train_labels[:, i].sum()
    neg_count = len(train_labels) - pos_count
    class_weights[i] = neg_count / pos_count  # Weight inversely proportional to frequency

# Use in BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights[i] for i in range(8)]))
```

**Expected Gain:** +5-8% for rare diseases (AMD, Hypertension, Myopia)

### C. Multi-Label Aware Augmentation

```python
# Current: Same augmentation for all images
# Recommended: Preserve disease-relevant features

def disease_aware_augmentation(image, labels):
    """Apply augmentations that don't destroy disease markers."""
    
    augmented = image.copy()
    
    # Safe augmentations for all diseases
    if random.random() > 0.5:
        augmented = horizontal_flip(augmented)
    
    # Avoid brightness changes for diseases with subtle color changes
    if labels[LABEL_MAP['Diabetes']] == 0 and labels[LABEL_MAP['AMD']] == 0:
        # Only adjust brightness if NOT diabetes or AMD
        augmented = adjust_brightness(augmented, factor=random.uniform(0.9, 1.1))
    
    # Preserve center for AMD
    if labels[LABEL_MAP['AMD']] == 1:
        # Avoid aggressive crops that remove macula
        augmented = center_crop(augmented, crop_fraction=0.9)
    else:
        augmented = random_crop(augmented, crop_fraction=0.85)
    
    # Add subtle noise (helps with overfitting)
    augmented = add_gaussian_noise(augmented, std=0.01)
    
    return augmented
```

**Expected Gain:** +1-2% overall, preserve disease features better

---

## 2. MODEL ARCHITECTURE IMPROVEMENTS

### A. Multi-Task Learning with Auxiliary Tasks

```python
class MultiTaskModel(nn.Module):
    """Learn disease detection + auxiliary tasks simultaneously."""
    
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Main task: disease classification
        self.disease_head = nn.Linear(2048, num_classes)
        
        # Auxiliary task 1: Disease group classification
        # (vascular, degenerative, structural, normal)
        self.group_head = nn.Linear(2048, 4)
        
        # Auxiliary task 2: Image quality prediction
        self.quality_head = nn.Linear(2048, 1)
        
        # Auxiliary task 3: Anatomical landmark detection
        # (optic disc location, macula location)
        self.landmark_head = nn.Linear(2048, 4)
    
    def forward(self, x):
        features = self.features(x).flatten(1)
        
        # Multi-task outputs
        disease_logits = self.disease_head(features)
        group_logits = self.group_head(features)
        quality = self.quality_head(features)
        landmarks = self.landmark_head(features)
        
        return disease_logits, group_logits, quality, landmarks

# Multi-task loss
def multi_task_loss(disease_pred, group_pred, quality_pred, landmark_pred,
                    disease_true, group_true, quality_true, landmark_true):
    
    loss_disease = F.binary_cross_entropy_with_logits(disease_pred, disease_true)
    loss_group = F.cross_entropy(group_pred, group_true)
    loss_quality = F.mse_loss(quality_pred, quality_true)
    loss_landmark = F.mse_loss(landmark_pred, landmark_true)
    
    # Weighted combination
    total_loss = loss_disease + 0.3 * loss_group + 0.1 * loss_quality + 0.2 * loss_landmark
    return total_loss
```

**Expected Gain:** +3-5% by learning better representations

### B. Attention Mechanisms for Disease Localization

```python
class AttentionModel(nn.Module):
    """Focus on disease-relevant regions."""
    
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.feature_layers = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global average pooling after attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # Extract features
        features = self.feature_layers(x)  # [B, 2048, 7, 7]
        
        # Generate attention maps (one per disease)
        attention_maps = self.attention(features)  # [B, 8, 7, 7]
        
        # Apply attention
        attended_features = features.unsqueeze(2) * attention_maps.unsqueeze(1)
        attended_features = attended_features.sum(dim=2)  # Sum over diseases
        
        # Pool and classify
        pooled = self.gap(attended_features).flatten(1)
        logits = self.classifier(pooled)
        
        return logits, attention_maps
```

**Expected Gain:** +2-4% by focusing on relevant regions

### C. Disease-Specific Expert Models

```python
# Current: Single model for all diseases
# Recommended: Ensemble of disease-specific experts

class DiseaseExpertEnsemble(nn.Module):
    """Separate expert models for disease groups."""
    
    def __init__(self):
        super().__init__()
        
        # Expert 1: Vascular diseases (Diabetes, Hypertension)
        self.vascular_expert = create_expert_model(['Diabetes', 'Hypertension'])
        
        # Expert 2: Degenerative diseases (AMD, Cataract)
        self.degenerative_expert = create_expert_model(['AMD', 'Cataract'])
        
        # Expert 3: Structural diseases (Glaucoma, Myopia)
        self.structural_expert = create_expert_model(['Glaucoma', 'Myopia'])
        
        # Expert 4: Normal vs abnormal
        self.screening_expert = create_expert_model(['Normal'])
        
        # Gating network decides which expert to use
        self.gating = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Extract shared features
        shared_features = extract_features(x)
        
        # Get expert predictions
        vascular_pred = self.vascular_expert(shared_features)
        degenerative_pred = self.degenerative_expert(shared_features)
        structural_pred = self.structural_expert(shared_features)
        screening_pred = self.screening_expert(shared_features)
        
        # Gating weights
        gate_weights = self.gating(shared_features)
        
        # Combine predictions
        final_pred = combine_expert_predictions(
            [vascular_pred, degenerative_pred, structural_pred, screening_pred],
            gate_weights
        )
        
        return final_pred

def create_expert_model(disease_list):
    """Create a model specialized for specific diseases."""
    # Train only on samples with these diseases
    # Use disease-specific augmentations
    # Optimize for these diseases specifically
    pass
```

**Expected Gain:** +4-7% by specialization

---

## 3. TRAINING IMPROVEMENTS

### A. Focal Loss for Hard Negatives

```python
# Current: BCEWithLogitsLoss (treats all errors equally)
# Recommended: Focal Loss (focuses on hard examples)

class FocalLoss(nn.Module):
    """Focus on hard-to-classify examples."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of correct class
        
        # Focal term: (1-pt)^gamma
        # Harder examples (low pt) get more weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing term
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_weight * focal_weight * BCE_loss
        return loss.mean()

# Use in training
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**Expected Gain:** +2-3% by focusing on difficult cases

### B. Per-Disease Optimal Thresholds

```python
# Current: Fixed 0.5 threshold for all diseases
# Recommended: Optimize threshold per disease

def find_optimal_thresholds(model, val_loader, device):
    """Find best threshold for each disease to maximize F1."""
    
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    optimal_thresholds = []
    
    for i in range(8):
        best_f1 = 0
        best_threshold = 0.5
        
        # Try thresholds from 0.1 to 0.9
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs[:, i] > threshold).astype(float)
            f1 = f1_score(all_labels[:, i], preds)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        print(f"{LABEL_NAMES[LABEL_COLUMNS[i]]}: threshold={best_threshold:.2f}, F1={best_f1:.3f}")
    
    return optimal_thresholds

# Example output might be:
# Normal: threshold=0.45, F1=0.720
# Diabetes: threshold=0.35, F1=0.580
# Glaucoma: threshold=0.40, F1=0.650
# AMD: threshold=0.25, F1=0.550  (lower threshold for rare disease!)
# Hypertension: threshold=0.20, F1=0.400  (much lower!)
```

**Expected Gain:** +3-5% overall, +10-15% for rare diseases

### C. Curriculum Learning

```python
# Current: Train on all samples equally from start
# Recommended: Start with easy examples, gradually add harder ones

class CurriculumScheduler:
    """Gradually increase training difficulty."""
    
    def __init__(self, train_dataset, num_epochs):
        self.train_dataset = train_dataset
        self.num_epochs = num_epochs
        
        # Score each sample by difficulty
        self.difficulty_scores = self.compute_difficulty()
    
    def compute_difficulty(self):
        """Assign difficulty score to each sample."""
        difficulties = []
        
        for image, labels in self.train_dataset:
            # Difficulty based on:
            # 1. Number of diseases (more = harder)
            num_diseases = labels.sum()
            
            # 2. Rare diseases (rare = harder)
            rare_disease_penalty = 0
            if labels[LABEL_MAP['AMD']] == 1:
                rare_disease_penalty += 2
            if labels[LABEL_MAP['Hypertension']] == 1:
                rare_disease_penalty += 3
            
            # 3. Image quality (poor quality = harder)
            quality_score = assess_image_quality(image)
            
            difficulty = num_diseases + rare_disease_penalty + (1 - quality_score)
            difficulties.append(difficulty)
        
        return np.array(difficulties)
    
    def get_epoch_samples(self, epoch):
        """Return samples for this epoch based on curriculum."""
        # In early epochs, use easier samples
        # Gradually add harder samples
        
        progress = epoch / self.num_epochs  # 0 to 1
        
        # Start with easiest 50%, end with all samples
        percentile_threshold = 50 + 50 * progress
        threshold = np.percentile(self.difficulty_scores, percentile_threshold)
        
        # Include all samples below difficulty threshold
        indices = np.where(self.difficulty_scores <= threshold)[0]
        
        return Subset(self.train_dataset, indices)

# Use in training loop
curriculum = CurriculumScheduler(train_dataset, num_epochs=30)

for epoch in range(num_epochs):
    # Get subset of data for this epoch
    epoch_dataset = curriculum.get_epoch_samples(epoch)
    epoch_loader = DataLoader(epoch_dataset, batch_size=32, shuffle=True)
    
    # Train on this subset
    train_epoch(model, epoch_loader, optimizer, criterion)
```

**Expected Gain:** +1-3% by learning fundamentals first

### D. Multi-Label Loss Functions

```python
# Current: Treat each label independently
# Recommended: Model label dependencies

class MultiLabelContrastiveLoss(nn.Module):
    """Learn that certain disease combinations are more likely."""
    
    def __init__(self, num_classes=8):
        super().__init__()
        self.num_classes = num_classes
        
        # Learn disease co-occurrence patterns
        self.cooccurrence_weight = nn.Parameter(torch.randn(num_classes, num_classes))
    
    def forward(self, logits, targets):
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Co-occurrence penalty
        # Penalize unlikely disease combinations
        probs = torch.sigmoid(logits)
        
        # For each pair of diseases
        cooccurrence_loss = 0
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
                # Predicted co-occurrence
                pred_cooccur = probs[:, i] * probs[:, j]
                
                # True co-occurrence
                true_cooccur = targets[:, i] * targets[:, j]
                
                # Weighted by learned pattern
                weight = torch.sigmoid(self.cooccurrence_weight[i, j])
                cooccurrence_loss += weight * (pred_cooccur - true_cooccur) ** 2
        
        total_loss = bce_loss + 0.1 * cooccurrence_loss
        return total_loss
```

**Expected Gain:** +2-3% for multi-label cases

---

## 4. ADVANCED TECHNIQUES

### A. Self-Supervised Pre-Training

```python
# Current: Use ImageNet pre-trained models
# Recommended: Pre-train on unlabeled fundus images

class SimCLR:
    """Self-supervised contrastive learning on fundus images."""
    
    def __init__(self, backbone):
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    
    def contrastive_loss(self, z1, z2, temperature=0.5):
        """NT-Xent loss for contrastive learning."""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Similarity matrix
        similarity = torch.matmul(z1, z2.T) / temperature
        
        # Positive pairs are on diagonal
        labels = torch.arange(len(z1)).to(z1.device)
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def train(self, unlabeled_fundus_images):
        """Pre-train on large unlabeled fundus image dataset."""
        for images in unlabeled_fundus_images:
            # Create two augmented views
            view1 = augment(images)
            view2 = augment(images)
            
            # Get representations
            h1 = self.backbone(view1)
            h2 = self.backbone(view2)
            
            # Project to lower dimension
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)
            
            # Contrastive loss
            loss = self.contrastive_loss(z1, z2)
            loss.backward()
            optimizer.step()

# Then fine-tune on labeled data
```

**Expected Gain:** +5-8% with large unlabeled fundus dataset

### B. Knowledge Distillation from Specialist Models

```python
# Train large teacher models for each disease
# Distill knowledge into smaller student model

class KnowledgeDistillation:
    """Learn from specialist teacher models."""
    
    def __init__(self, student, teachers):
        self.student = student
        self.teachers = teachers  # List of disease-specific teachers
    
    def distillation_loss(self, student_logits, teacher_logits, targets, temperature=3.0):
        """Combine hard targets and soft targets from teacher."""
        
        # Hard loss (actual labels)
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, targets)
        
        # Soft loss (teacher predictions)
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Combine
        total_loss = 0.5 * hard_loss + 0.5 * (temperature ** 2) * soft_loss
        return total_loss
    
    def train_student(self, train_loader):
        """Train student to mimic teacher models."""
        for images, labels in train_loader:
            # Get student predictions
            student_logits = self.student(images)
            
            # Get teacher predictions (no gradient)
            teacher_logits_list = []
            for teacher in self.teachers:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                teacher_logits_list.append(teacher_logits)
            
            # Average teacher predictions
            teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
            
            # Distillation loss
            loss = self.distillation_loss(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()
```

**Expected Gain:** +3-5% by learning from specialists

### C. Semi-Supervised Learning with Pseudo-Labels

```python
# Use unlabeled fundus images with pseudo-labels

class SemiSupervisedTrainer:
    """Train with labeled + pseudo-labeled data."""
    
    def __init__(self, model):
        self.model = model
    
    def generate_pseudo_labels(self, unlabeled_images, confidence_threshold=0.8):
        """Generate high-confidence pseudo-labels."""
        pseudo_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for image in unlabeled_images:
                output = self.model(image)
                probs = torch.sigmoid(output)
                
                # Only use high-confidence predictions
                confident_labels = (probs > confidence_threshold) | (probs < 1 - confidence_threshold)
                
                if confident_labels.all():
                    # All labels are confident
                    pseudo_labels.append((image, (probs > 0.5).float()))
        
        return pseudo_labels
    
    def train_semi_supervised(self, labeled_data, unlabeled_data):
        """Train with both labeled and pseudo-labeled data."""
        
        for epoch in range(num_epochs):
            # Generate pseudo-labels for this epoch
            pseudo_labeled = self.generate_pseudo_labels(unlabeled_data)
            
            # Combine labeled and pseudo-labeled
            combined_data = labeled_data + pseudo_labeled
            
            # Train on combined data
            train_epoch(self.model, combined_data, optimizer, criterion)
```

**Expected Gain:** +2-4% with additional unlabeled data

---

## 5. ENSEMBLE IMPROVEMENTS

### A. Better Ensemble Strategy

```python
# Current: Equal weights, simple averaging
# Recommended: Learned optimal weights per disease

class LearnedEnsemble(nn.Module):
    """Learn optimal ensemble weights per disease."""
    
    def __init__(self, models):
        super().__init__()
        self.models = models
        
        # Learn separate weights for each disease
        # Shape: [num_models, num_diseases]
        self.weights = nn.Parameter(torch.ones(len(models), 8) / len(models))
    
    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_models, batch, num_diseases]
        
        # Normalize weights per disease
        normalized_weights = F.softmax(self.weights, dim=0)  # [num_models, num_diseases]
        
        # Weighted average per disease
        # Broadcasting: [num_models, batch, num_diseases] * [num_models, 1, num_diseases]
        weighted_preds = predictions * normalized_weights.unsqueeze(1)
        ensemble_pred = weighted_preds.sum(dim=0)
        
        return ensemble_pred

# Train ensemble weights on validation set
ensemble = LearnedEnsemble([resnet50, efficientnet, densenet])
optimize_ensemble_weights(ensemble, val_loader)
```

**Expected Gain:** +1-2% over equal weighting

### B. Diverse Model Selection

```python
# Current: ResNet50, EfficientNet-B3, DenseNet-121 (all CNN)
# Recommended: Mix CNN with Vision Transformers

ensemble_models = [
    models.resnet50(),           # CNN: Good at local features
    models.efficientnet_b4(),     # CNN: Efficient architecture
    VisionTransformer(),         # Transformer: Global context
    SwinTransformer(),           # Transformer: Hierarchical
    ConvNeXt()                   # Modern CNN: Best of both worlds
]

# More diverse architectures = better ensemble
```

**Expected Gain:** +2-3% from architecture diversity

---

## 6. DATA COLLECTION RECOMMENDATIONS

### A. Targeted Data Collection

**Collect more samples for failing diseases:**

| Disease | Current | Target | Needed |
|---------|---------|--------|--------|
| Hypertension | 45 | 300 | +255 |
| AMD | 105 | 300 | +195 |
| Myopia | 50 | 200 | +150 |
| Diabetes | 356 | 600 | +244 |

**Focus on multi-label cases:**
- Current: 12% multi-label
- Target: 25% multi-label
- Need: +180 multi-disease samples

**Expected Gain:** +10-15% for rare diseases

### B. External Dataset Integration

```python
# Combine with public datasets:
# - APTOS 2019 (Diabetes)
# - ODIR-5K (Multi-disease)
# - REFUGE (Glaucoma)
# - iChallenge-AMD (AMD)

def merge_datasets(odir_data, aptos_data, refuge_data):
    """Carefully merge multiple datasets."""
    
    # Standardize labels
    merged_labels = standardize_disease_labels(
        [odir_data.labels, aptos_data.labels, refuge_data.labels]
    )
    
    # Normalize images (different camera types)
    merged_images = normalize_across_datasets(
        [odir_data.images, aptos_data.images, refuge_data.images]
    )
    
    return merged_images, merged_labels
```

**Expected Gain:** +5-10% from more diverse training data

---

## PRIORITY ROADMAP

### Phase 1: Quick Wins (1-2 weeks)

1. **Optimal Thresholds per Disease** → +3-5% gain
   - Implement threshold optimization on validation set
   - Expected: Hypertension F1 from 0.25 to 0.35-0.40

2. **Class Imbalance Handling** → +5-8% for rare diseases
   - Add class weights to loss function
   - Oversample minority classes

3. **Focal Loss** → +2-3% overall
   - Replace BCE with Focal Loss
   - Focus on hard examples

**Expected Total: 87.32% → 92-93%**

### Phase 2: Architecture Improvements (2-4 weeks)

4. **Attention Mechanisms** → +2-4%
   - Add spatial attention to locate disease regions
   - Disease-specific attention maps

5. **Multi-Task Learning** → +3-5%
   - Add auxiliary tasks (disease groups, quality)
   - Better feature representations

6. **Better Ensemble** → +2-3%
   - Add Vision Transformer
   - Learn optimal weights per disease

**Expected Total: 92-93% → 96-97%**

### Phase 3: Data & Advanced Techniques (4-8 weeks)

7. **Targeted Data Collection** → +10-15% for rare diseases
   - Collect 255 Hypertension samples
   - Collect 195 AMD samples

8. **External Dataset Integration** → +5-10%
   - Merge with ODIR-5K, APTOS
   - More diverse training data

9. **Self-Supervised Pre-Training** → +5-8%
   - Pre-train on unlabeled fundus images
   - Better initialization

**Expected Total: 96-97% → 98-99%**

---

## REALISTIC EXPECTATIONS

### Conservative Estimates:
- Phase 1: 89.22% → 92-93% (achievable in 2 weeks)
- Phase 2: 92-93% → 95-96% (achievable in 4 weeks)
- Phase 3: 95-96% → 97-98% (requires more data, 8 weeks)

### Per-Disease Targets After All Improvements:

| Disease | Current F1 | Target F1 | Improvement Strategy |
|---------|-----------|-----------|---------------------|
| Normal | 0.688 | 0.75 | Better thresholds, focal loss |
| Diabetes | 0.505 | 0.70 | More data, class weights, attention |
| Glaucoma | 0.601 | 0.75 | Attention on optic disc |
| Cataract | 0.586 | 0.70 | Better preprocessing |
| AMD | 0.485 | 0.65 | More data, macula attention |
| Hypertension | 0.250 | 0.55 | More data, vascular expert |
| Myopia | 0.824 | 0.90 | Already good, minor tweaks |
| Other | 0.520 | 0.65 | Better multi-label loss |

---

## IMPLEMENTATION CHECKLIST

### Week 1-2: Quick Wins
- [ ] Implement per-disease optimal thresholds
- [ ] Add class weights to loss function
- [ ] Implement Focal Loss
- [ ] Add SMOTE/oversampling for minority classes
- [ ] Re-train and evaluate

### Week 3-4: Model Architecture
- [ ] Implement attention mechanisms
- [ ] Add multi-task learning heads
- [ ] Add Vision Transformer to ensemble
- [ ] Implement learned ensemble weights
- [ ] Re-train and evaluate

### Week 5-8: Data & Advanced
- [ ] Collect more Hypertension samples
- [ ] Collect more AMD samples
- [ ] Integrate external datasets
- [ ] Implement self-supervised pre-training
- [ ] Final training and evaluation

---

## CONCLUSION

The current 89.22% label accuracy (48.75% sample accuracy) is disappointing primarily due to:

1. **Class imbalance** (rare diseases have 3-5% prevalence)
2. **Fixed thresholds** (0.5 suboptimal for all diseases)
3. **Weak multi-label learning** (labels treated independently)
4. **Insufficient data** for rare diseases

**Most Impactful Changes:**
1. **Per-disease thresholds** (+3-5%, easiest to implement)
2. **Class balancing** (+5-8% for rare diseases)
3. **More data collection** (+10-15% for failing diseases)
4. **Attention mechanisms** (+2-4%, better localization)
5. **Multi-task learning** (+3-5%, better representations)

**Realistic Target After All Improvements: 95-98% label accuracy, 75-80% sample accuracy**

The current model is a good baseline, but significant improvements are possible with proper handling of class imbalance, better loss functions, and more targeted data collection.
