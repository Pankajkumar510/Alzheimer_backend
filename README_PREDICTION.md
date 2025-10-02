# Alzheimer's Disease Prediction System

This project implements a multimodal Alzheimer's disease prediction system using trained MRI and PET scan models.

## Trained Models

### MRI Model
- **Architecture**: ConvNext Tiny
- **Model Path**: `mri_20251002-101653/mri_best_model.pth`
- **Performance**: 84.5% F1 Score
- **Classes**: 4 classes
  - MildDemented (index 0)
  - ModerateDemented (index 1) 
  - NonDemented (index 2)
  - VeryMildDemented (index 3)

### PET Model
- **Architecture**: Vision Transformer Small (vit_small_patch16_224)
- **Model Path**: `pet_20251002-022237/pet_best_model.pth`
- **Performance**: 96.6% F1 Score
- **Classes**: 5 classes
  - AD (index 0)
  - CN (index 1)
  - EMCI (index 2)
  - LMCI (index 3) 
  - MCI (index 4)

## Usage

### Standalone Prediction Script

The main prediction script is `predict_with_trained_models.py` which can be used for:

1. **MRI-only prediction**:
```bash
python predict_with_trained_models.py --mri path/to/mri_image.jpg
```

2. **PET-only prediction**:
```bash
python predict_with_trained_models.py --pet path/to/pet_image.jpg
```

3. **Multimodal prediction** (both MRI and PET):
```bash
python predict_with_trained_models.py --mri path/to/mri.jpg --pet path/to/pet.jpg
```

4. **Save results to JSON file**:
```bash
python predict_with_trained_models.py --mri path/to/mri.jpg --pet path/to/pet.jpg --output results.json
```

5. **Verbose output** (shows all class probabilities):
```bash
python predict_with_trained_models.py --mri path/to/mri.jpg --verbose
```

### Example Output

```
============================================================
PREDICTION RESULTS
============================================================

MRI Prediction:
  Class: NonDemented
  Confidence: 0.892

PET Prediction:
  Class: CN
  Confidence: 0.934

FUSED Prediction:
  Class: CN
  Confidence: 0.913
  Method: weighted_average
============================================================
```

### API Integration

The existing FastAPI backend at `Alzheimer_backend/src/api/routes.py` has been updated to use your trained models. It will automatically detect and load:
- MRI model from `mri_20251002-101653/mri_best_model.pth`
- PET model from `pet_20251002-022237/pet_best_model.pth`

## Class Mapping and Fusion

### Fusion Strategy
The system maps different class taxonomies to a common fusion space:

**MRI Classes → Fusion Classes**:
- MildDemented → MCI
- ModerateDemented → AD
- NonDemented → CN
- VeryMildDemented → EMCI

**PET Classes** are already in the fusion space:
- AD, CN, EMCI, LMCI, MCI

### Fusion Classes
The final fusion predictions use 5 classes:
- **CN**: Cognitively Normal
- **EMCI**: Early Mild Cognitive Impairment  
- **LMCI**: Late Mild Cognitive Impairment
- **MCI**: Mild Cognitive Impairment
- **AD**: Alzheimer's Disease

## Files Overview

- `predict_with_trained_models.py` - Main standalone prediction script
- `test_models.py` - Test script to verify model loading
- `inspect_models.py` - Utility to inspect model architectures
- `mri_20251002-101653/mri_best_model.pth` - Trained MRI model
- `pet_20251002-022237/pet_best_model.pth` - Trained PET model
- `fusion_mapping.json` - Class mapping configuration
- `Alzheimer_backend/meta/mri_classes.json` - MRI class mappings
- `Alzheimer_backend/meta/pet_classes.json` - PET class mappings

## Model Architecture Details

### MRI Model (ConvNext Tiny)
- Input: 224x224 RGB images
- Preprocessing: ImageNet normalization
- Stem channels: 96
- Depth configuration: [3, 3, 9, 3]

### PET Model (ViT Small)
- Input: 224x224 RGB images  
- Preprocessing: ImageNet normalization
- Embedding dimension: 384
- Patch size: 16x16

## Requirements

Make sure you have the required dependencies installed:
```bash
pip install torch torchvision timm pillow numpy
```

## Testing

To verify that your models load correctly:
```bash
python test_models.py
```

Expected output:
```
✓ SUCCESS: Both models loaded successfully!
```

## Notes

- Models automatically detect CUDA availability and use GPU if available
- Image preprocessing is handled automatically (resize, normalize)
- Supports common image formats: JPG, JPEG, PNG
- The system is designed for research purposes
- Both individual and fused predictions are provided for multimodal inputs