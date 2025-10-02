#!/usr/bin/env python3
"""
Standalone prediction script using trained MRI and PET models.
This script specifically uses your trained models:
- MRI: mri_20251002-101653/mri_best_model.pth 
- PET: pet_20251002-022237/pet_best_model.pth

Usage:
    python predict_with_trained_models.py --mri path/to/mri_image.jpg
    python predict_with_trained_models.py --pet path/to/pet_image.jpg
    python predict_with_trained_models.py --mri path/to/mri.jpg --pet path/to/pet.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Add the backend src to path
sys.path.append(str(Path(__file__).parent / "Alzheimer_backend" / "src"))

from models.convnext_mri import create_mri_model
from models.vit_pet import create_vit_model

# Constants for preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AlzheimerPredictor:
    """
    Predictor class that uses your trained MRI and PET models.
    """
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent
        
        self.base_dir = Path(base_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model paths
        self.mri_model_path = self.base_dir / "mri_20251002-101653" / "mri_best_model.pth"
        self.pet_model_path = self.base_dir / "pet_20251002-022237" / "pet_best_model.pth"
        
        # Class mappings from fusion_mapping.json
        self.fusion_mapping_path = self.base_dir / "fusion_mapping.json"
        
        # Load class mappings from existing meta files
        mri_classes_path = self.base_dir / "Alzheimer_backend" / "meta" / "mri_classes.json"
        pet_classes_path = self.base_dir / "Alzheimer_backend" / "meta" / "pet_classes.json"
        
        # Load MRI classes
        if mri_classes_path.exists():
            with open(mri_classes_path, 'r') as f:
                mri_class_to_idx = json.load(f)
            # Sort by index to get correct order
            self.mri_classes = [k for k, v in sorted(mri_class_to_idx.items(), key=lambda x: x[1])]
        else:
            self.mri_classes = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
            
        # Load PET classes  
        if pet_classes_path.exists():
            with open(pet_classes_path, 'r') as f:
                pet_class_to_idx = json.load(f)
            # Sort by index to get correct order
            self.pet_classes = [k for k, v in sorted(pet_class_to_idx.items(), key=lambda x: x[1])]
        else:
            self.pet_classes = ["CN", "EMCI", "LMCI", "MCI", "AD"]
            
        self.fusion_classes = ["CN", "EMCI", "LMCI", "MCI", "AD"]
        
        # Load models
        self.mri_model = None
        self.pet_model = None
        self._load_models()
        
        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    def _load_models(self):
        """Load the trained MRI and PET models."""
        print(f"Device: {self.device}")
        
        # Load MRI model
        if self.mri_model_path.exists():
            print(f"Loading MRI model from: {self.mri_model_path}")
            checkpoint = torch.load(self.mri_model_path, map_location=self.device)
            
            # Create MRI model (ConvNext Tiny)
            num_mri_classes = len(self.mri_classes)
            self.mri_model = create_mri_model(
                num_classes=num_mri_classes,
                model_name="convnext_tiny",
                pretrained=False
            )
            
            # Load weights
            if "model_state" in checkpoint:
                self.mri_model.load_state_dict(checkpoint["model_state"])
            else:
                self.mri_model.load_state_dict(checkpoint)
            
            self.mri_model.to(self.device)
            self.mri_model.eval()
            print(f"✓ MRI model loaded successfully ({num_mri_classes} classes)")
        else:
            print(f"⚠ MRI model not found at: {self.mri_model_path}")
        
        # Load PET model  
        if self.pet_model_path.exists():
            print(f"Loading PET model from: {self.pet_model_path}")
            checkpoint = torch.load(self.pet_model_path, map_location=self.device)
            
            # Create PET model (Vision Transformer Small)
            num_pet_classes = len(self.pet_classes)
            self.pet_model = create_vit_model(
                num_classes=num_pet_classes,
                model_name="vit_small_patch16_224",
                pretrained=False
            )
            
            # Load weights
            if "model_state" in checkpoint:
                self.pet_model.load_state_dict(checkpoint["model_state"])
            else:
                self.pet_model.load_state_dict(checkpoint)
            
            self.pet_model.to(self.device)
            self.pet_model.eval()
            print(f"✓ PET model loaded successfully ({num_pet_classes} classes)")
        else:
            print(f"⚠ PET model not found at: {self.pet_model_path}")
    
    def predict_mri(self, image_path: str) -> Dict[str, Any]:
        """
        Predict using MRI model.
        
        Args:
            image_path: Path to MRI image
            
        Returns:
            Dictionary with prediction results
        """
        if self.mri_model is None:
            return {"error": "MRI model not loaded"}
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.mri_model(input_tensor)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_idx = np.argmax(probabilities)
                predicted_class = self.mri_classes[predicted_idx]
                confidence = float(probabilities[predicted_idx])
            
            return {
                "modality": "MRI",
                "predicted_class": predicted_class,
                "predicted_index": int(predicted_idx),
                "confidence": confidence,
                "probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.mri_classes, probabilities)
                },
                "model_path": str(self.mri_model_path)
            }
            
        except Exception as e:
            return {"error": f"Error processing MRI image: {str(e)}"}
    
    def predict_pet(self, image_path: str) -> Dict[str, Any]:
        """
        Predict using PET model.
        
        Args:
            image_path: Path to PET image
            
        Returns:
            Dictionary with prediction results
        """
        if self.pet_model is None:
            return {"error": "PET model not loaded"}
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.pet_model(input_tensor)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_idx = np.argmax(probabilities)
                predicted_class = self.pet_classes[predicted_idx]
                confidence = float(probabilities[predicted_idx])
            
            return {
                "modality": "PET",
                "predicted_class": predicted_class,
                "predicted_index": int(predicted_idx),
                "confidence": confidence,
                "probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.pet_classes, probabilities)
                },
                "model_path": str(self.pet_model_path)
            }
            
        except Exception as e:
            return {"error": f"Error processing PET image: {str(e)}"}
    
    def predict_multimodal(self, mri_path: Optional[str] = None, pet_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict using both MRI and PET models and fuse results.
        
        Args:
            mri_path: Path to MRI image (optional)
            pet_path: Path to PET image (optional)
            
        Returns:
            Dictionary with individual and fused predictions
        """
        results = {
            "multimodal_prediction": True,
            "individual_predictions": {},
            "fusion_result": None
        }
        
        # Get individual predictions
        predictions = []
        weights = []
        
        if mri_path:
            mri_result = self.predict_mri(mri_path)
            results["individual_predictions"]["mri"] = mri_result
            if "error" not in mri_result:
                # Map MRI classes to fusion space using the actual class ordering
                # MRI classes order from meta: [MildDemented, ModerateDemented, NonDemented, VeryMildDemented]
                # PET/Fusion classes order: [AD, CN, EMCI, LMCI, MCI]
                mri_probs = np.array(list(mri_result["probabilities"].values()))
                
                # Create mapping based on actual class order
                fusion_probs_mri = np.zeros(5)  # [AD, CN, EMCI, LMCI, MCI]
                
                # Map according to actual indices:
                # MildDemented (idx 0) -> MCI (idx 4)
                # ModerateDemented (idx 1) -> AD (idx 0)
                # NonDemented (idx 2) -> CN (idx 1)
                # VeryMildDemented (idx 3) -> EMCI (idx 2)
                fusion_probs_mri[4] = mri_probs[0]  # MildDemented -> MCI
                fusion_probs_mri[0] = mri_probs[1]  # ModerateDemented -> AD
                fusion_probs_mri[1] = mri_probs[2]  # NonDemented -> CN
                fusion_probs_mri[2] = mri_probs[3]  # VeryMildDemented -> EMCI
                # LMCI (idx 3) gets 0.0 as there's no direct mapping
                
                predictions.append(fusion_probs_mri)
                weights.append(1.0)
        
        if pet_path:
            pet_result = self.predict_pet(pet_path)
            results["individual_predictions"]["pet"] = pet_result
            if "error" not in pet_result:
                # PET is already in fusion space
                pet_probs = np.array(list(pet_result["probabilities"].values()))
                predictions.append(pet_probs)
                weights.append(1.0)
        
        # Fuse predictions if we have any
        if predictions:
            # Weighted average fusion
            weights = np.array(weights) / sum(weights)  # Normalize weights
            fused_probs = np.average(predictions, axis=0, weights=weights)
            fused_probs = fused_probs / fused_probs.sum()  # Ensure probabilities sum to 1
            
            predicted_idx = np.argmax(fused_probs)
            predicted_class = self.fusion_classes[predicted_idx]
            confidence = float(fused_probs[predicted_idx])
            
            results["fusion_result"] = {
                "predicted_class": predicted_class,
                "predicted_index": int(predicted_idx),
                "confidence": confidence,
                "probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.fusion_classes, fused_probs)
                },
                "fusion_method": "weighted_average",
                "weights_used": weights.tolist()
            }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Alzheimer's Detection using Trained MRI and PET Models")
    parser.add_argument("--mri", type=str, help="Path to MRI image")
    parser.add_argument("--pet", type=str, help="Path to PET image") 
    parser.add_argument("--output", type=str, help="Output JSON file path (optional)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.mri and not args.pet:
        print("Error: Please provide at least one image (--mri or --pet)")
        sys.exit(1)
    
    # Initialize predictor
    print("Initializing Alzheimer's Disease Predictor...")
    predictor = AlzheimerPredictor()
    
    # Make predictions
    if args.mri and args.pet:
        # Multimodal prediction
        print(f"\nRunning multimodal prediction:")
        print(f"  MRI: {args.mri}")
        print(f"  PET: {args.pet}")
        results = predictor.predict_multimodal(args.mri, args.pet)
    elif args.mri:
        # MRI only
        print(f"\nRunning MRI prediction: {args.mri}")
        results = predictor.predict_mri(args.mri)
    else:
        # PET only
        print(f"\nRunning PET prediction: {args.pet}")
        results = predictor.predict_pet(args.pet)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    if "multimodal_prediction" in results:
        # Multimodal results
        for modality, result in results["individual_predictions"].items():
            if "error" not in result:
                print(f"\n{modality.upper()} Prediction:")
                print(f"  Class: {result['predicted_class']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                if args.verbose:
                    print("  All probabilities:")
                    for cls, prob in result["probabilities"].items():
                        print(f"    {cls}: {prob:.3f}")
        
        if results["fusion_result"]:
            print(f"\nFUSED Prediction:")
            fusion = results["fusion_result"]
            print(f"  Class: {fusion['predicted_class']}")
            print(f"  Confidence: {fusion['confidence']:.3f}")
            print(f"  Method: {fusion['fusion_method']}")
            if args.verbose:
                print("  All probabilities:")
                for cls, prob in fusion["probabilities"].items():
                    print(f"    {cls}: {prob:.3f}")
    else:
        # Single modality result
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Modality: {results['modality']}")
            print(f"Predicted Class: {results['predicted_class']}")
            print(f"Confidence: {results['confidence']:.3f}")
            if args.verbose:
                print("All probabilities:")
                for cls, prob in results["probabilities"].items():
                    print(f"  {cls}: {prob:.3f}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()