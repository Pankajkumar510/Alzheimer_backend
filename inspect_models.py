#!/usr/bin/env python3
"""
Script to inspect the saved model architectures to determine the correct model names.
"""

import torch
from pathlib import Path


def inspect_model(model_path, model_name):
    """Inspect a saved model to determine its architecture."""
    print(f"\n=== Inspecting {model_name} ===")
    print(f"Path: {model_path}")
    
    if not Path(model_path).exists():
        print("Model file not found!")
        return
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
            
        print(f"Model state dict keys (first 10): {list(state_dict.keys())[:10]}")
        
        # For ConvNext models, check stem dimensions to determine variant
        if "stem.0.weight" in state_dict:
            stem_channels = state_dict["stem.0.weight"].shape[0]
            print(f"Stem output channels: {stem_channels}")
            
            # ConvNext variants by stem channels:
            # convnext_tiny: 96
            # convnext_small: 96  
            # convnext_base: 128
            # convnext_large: 192
            if stem_channels == 96:
                print("Detected ConvNext architecture: convnext_tiny or convnext_small")
                
                # Check depth configuration to differentiate variants
                # ConvNext depths: tiny=[3,3,9,3], small=[3,3,27,3], base=[3,3,27,3], large=[3,3,27,3]
                stage2_blocks = []
                for i in range(30):  # Check up to 30 blocks
                    if f"stages.2.blocks.{i}.gamma" in state_dict:
                        stage2_blocks.append(i)
                    else:
                        break
                        
                stage2_depth = len(stage2_blocks)
                print(f"Stage 2 depth (number of blocks): {stage2_depth}")
                
                if stage2_depth == 9:
                    print("Final determination: convnext_tiny")
                elif stage2_depth == 27:
                    if "stages.3.blocks.0.gamma" in state_dict:
                        stage3_dim = state_dict["stages.3.blocks.0.gamma"].shape[0]
                        print(f"Stage 3 dimensions: {stage3_dim}")
                        if stage3_dim == 768:
                            print("Final determination: convnext_small")
                        elif stage3_dim == 1024:
                            print("Final determination: convnext_base or convnext_large")
                        else:
                            print("Final determination: Unknown ConvNext variant")
                    else:
                        print("Final determination: convnext_small (assumed)")
                else:
                    print(f"Final determination: Unknown ConvNext variant (stage2_depth={stage2_depth})")
                    
            elif stem_channels == 128:
                print("Detected ConvNext architecture: convnext_base")
            elif stem_channels == 192:
                print("Detected ConvNext architecture: convnext_large")
            else:
                print(f"Unknown ConvNext variant with {stem_channels} stem channels")
        
        # For ViT models, check embedding dimensions
        if "pos_embed" in state_dict:
            embed_dim = state_dict["pos_embed"].shape[-1]
            print(f"Embedding dimension: {embed_dim}")
            
            # ViT variants by embedding dimension:
            if embed_dim == 192:
                print("Detected ViT architecture: vit_tiny_patch16_224")
            elif embed_dim == 384:
                print("Detected ViT architecture: vit_small_patch16_224")
            elif embed_dim == 768:
                print("Detected ViT architecture: vit_base_patch16_224")
            elif embed_dim == 1024:
                print("Detected ViT architecture: vit_large_patch16_224")
            else:
                print(f"Unknown ViT variant with {embed_dim} embedding dimension")
        
        # Check number of classes
        for key in ["head.fc.weight", "head.weight", "classifier.weight", "fc.weight"]:
            if key in state_dict:
                num_classes = state_dict[key].shape[0]
                print(f"Number of classes: {num_classes}")
                break
        
        # Check if there's class information
        if "classes" in checkpoint:
            print(f"Stored classes: {checkpoint['classes']}")
            
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")


def main():
    base_dir = Path("C:/Users/panka/OneDrive/Desktop/alzheimer")
    
    mri_path = base_dir / "mri_20251002-101653" / "mri_best_model.pth"
    pet_path = base_dir / "pet_20251002-022237" / "pet_best_model.pth"
    
    inspect_model(mri_path, "MRI Model")
    inspect_model(pet_path, "PET Model")


if __name__ == "__main__":
    main()