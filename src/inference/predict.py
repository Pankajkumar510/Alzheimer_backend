import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.models.convnext_mri import create_mri_model
from src.models.vit_pet import create_vit_model
from src.models.mlp_cognitive import CognitiveMLP, load_scaler
from src.models.fusion import average_fusion, weighted_fusion
from src.inference.explainability import GradCAM, attention_rollout_vit


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InferenceEngine:
    def __init__(
        self,
        mri_classes_json: Optional[str] = None,
        pet_classes_json: Optional[str] = None,
        classes_json: Optional[str] = None,
        mri_weights: Optional[str] = None,
        pet_weights: Optional[str] = None,
        cognitive_dir: Optional[str] = None,
        fusion_mapping_json: Optional[str] = None,
    ):
        # Load class mappings; allow separate taxonomies for MRI and PET
        if mri_classes_json and Path(mri_classes_json).exists():
            with open(mri_classes_json, "r", encoding="utf-8") as f:
                self.mri_class_to_idx = json.load(f)
        elif classes_json and Path(classes_json).exists():
            with open(classes_json, "r", encoding="utf-8") as f:
                self.mri_class_to_idx = json.load(f)
        else:
            self.mri_class_to_idx = {}
        if pet_classes_json and Path(pet_classes_json).exists():
            with open(pet_classes_json, "r", encoding="utf-8") as f:
                self.pet_class_to_idx = json.load(f)
        elif classes_json and Path(classes_json).exists():
            with open(classes_json, "r", encoding="utf-8") as f:
                self.pet_class_to_idx = json.load(f)
        else:
            self.pet_class_to_idx = {}

        self.mri_idx_to_class = {v: k for k, v in self.mri_class_to_idx.items()}
        self.pet_idx_to_class = {v: k for k, v in self.pet_class_to_idx.items()}

        # Fusion mapping (optional). If provided, we can fuse across different taxonomies by mapping
        # probabilities into a shared fusion class space.
        self.fusion = None
        if fusion_mapping_json and Path(fusion_mapping_json).exists():
            with open(fusion_mapping_json, "r", encoding="utf-8") as f:
                fmap = json.load(f)
            self.fusion = {
                "fusion_classes": fmap["fusion_classes"],
                "mri_classes": fmap["mri_classes"],
                "pet_classes": fmap["pet_classes"],
                "mri_M": fmap["mri_to_fusion"],
                "pet_M": fmap["pet_to_fusion"],
            }
            # Reorder mapping matrices to match current class_to_idx ordering
            import numpy as _np
            def build_matrix(src_classes_list, src_class_to_idx, M):
                # M is list of rows aligned to src_classes_list order
                src_len = len(src_classes_list)
                fus_len = len(M[0]) if M else 0
                R = _np.zeros((len(src_class_to_idx), fus_len), dtype=_np.float32)
                for i, cname in enumerate(src_classes_list):
                    if cname in src_class_to_idx:
                        R[src_class_to_idx[cname], :] = _np.array(M[i], dtype=_np.float32)
                return R
            self.fusion["mri_R"] = build_matrix(self.fusion["mri_classes"], self.mri_class_to_idx, self.fusion["mri_M"]) if self.mri_class_to_idx else None
            self.fusion["pet_R"] = build_matrix(self.fusion["pet_classes"], self.pet_class_to_idx, self.fusion["pet_M"]) if self.pet_class_to_idx else None
            self.fusion["idx_to_fusion"] = {i: c for i, c in enumerate(self.fusion["fusion_classes"])}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mri_model = None
        self.pet_model = None
        self.cog_model = None
        self.cog_scaler = None
        self.cog_features = []
        self.debug = {
            "cwd": str(Path.cwd()),
            "mri": {"weights": mri_weights, "classes": len(self.mri_class_to_idx), "loaded": False},
            "pet": {"weights": pet_weights, "classes": len(self.pet_class_to_idx), "loaded": False},
            "cognitive": {"loaded": False, "mode": None},
        }

        def _infer_num_classes_from_state_dict(sd):
            # Try common classifier keys first
            for key in [
                "head.fc.weight",
                "head.weight",
                "classifier.weight",
                "fc.weight",
            ]:
                if key in sd and hasattr(sd[key], "shape"):
                    shape = sd[key].shape
                    if len(shape) == 2:
                        return int(shape[0])
            for key in [
                "head.fc.bias",
                "head.bias",
                "classifier.bias",
                "fc.bias",
            ]:
                if key in sd and hasattr(sd[key], "shape") and len(sd[key].shape) == 1:
                    return int(sd[key].shape[0])
            # Fallback: any 2D weight tensor
            for k, v in sd.items():
                if hasattr(v, "ndim") and v.ndim == 2 and 2 <= v.shape[0] <= 1000:
                    return int(v.shape[0])
            return None

        # Initialize models with their respective class counts if available
        if mri_weights and Path(mri_weights).exists():
            ckpt = torch.load(mri_weights, map_location=self.device)
            sd = ckpt["model_state"]
            nclasses = len(self.mri_class_to_idx) if self.mri_class_to_idx else _infer_num_classes_from_state_dict(sd) or 2
            
            # Detect ConvNext architecture from state dict
            model_name = "convnext_base"  # default
            if "stem.0.weight" in sd:
                stem_channels = sd["stem.0.weight"].shape[0]
                if stem_channels == 96:
                    # Check stage 2 depth to distinguish tiny vs small
                    stage2_depth = 0
                    for i in range(30):
                        if f"stages.2.blocks.{i}.gamma" in sd:
                            stage2_depth += 1
                        else:
                            break
                    if stage2_depth == 9:
                        model_name = "convnext_tiny"
                    else:
                        model_name = "convnext_small"
                elif stem_channels == 128:
                    model_name = "convnext_base"
                elif stem_channels == 192:
                    model_name = "convnext_large"
            
            self.mri_model = create_mri_model(nclasses, model_name=model_name, pretrained=False)
            self.mri_model.load_state_dict(sd)  # type: ignore
            self.mri_model.to(self.device).eval()
            self.debug["mri"]["loaded"] = True
            self.debug["mri"]["nclasses"] = nclasses
            self.debug["mri"]["model_name"] = model_name
        elif self.mri_class_to_idx:
            # Fallback demo: untrained head produces non-informative predictions but enables UI flow
            nclasses = len(self.mri_class_to_idx)
            self.mri_model = create_mri_model(nclasses, model_name="convnext_tiny", pretrained=False)
            self.mri_model.to(self.device).eval()
            self.debug["mri"]["loaded"] = True
            self.debug["mri"]["nclasses"] = nclasses
            self.debug["mri"]["mode"] = "untrained"
        if pet_weights and Path(pet_weights).exists():
            ckpt = torch.load(pet_weights, map_location=self.device)
            sd = ckpt["model_state"]
            embed_dim = None
            # Try to infer ViT variant from checkpoint shapes
            for key in ("pos_embed", "cls_token"):
                if key in sd:
                    shape = sd[key].shape
                    embed_dim = shape[-1]
                    break
            model_name = "vit_base_patch16_224"
            if embed_dim is not None:
                if embed_dim <= 192:
                    model_name = "vit_tiny_patch16_224"
                elif embed_dim <= 384:
                    model_name = "vit_small_patch16_224"
                elif embed_dim <= 768:
                    model_name = "vit_base_patch16_224"
            nclasses = len(self.pet_class_to_idx) if self.pet_class_to_idx else _infer_num_classes_from_state_dict(sd) or 2
            self.pet_model = create_vit_model(nclasses, model_name=model_name, pretrained=False)
            self.pet_model.load_state_dict(sd)  # type: ignore
            self.pet_model.to(self.device).eval()
            self.debug["pet"]["loaded"] = True
            self.debug["pet"]["nclasses"] = nclasses
        elif self.pet_class_to_idx:
            # Fallback demo: untrained head enables UI flow without weights
            model_name = "vit_small_patch16_224"
            nclasses = len(self.pet_class_to_idx)
            self.pet_model = create_vit_model(nclasses, model_name=model_name, pretrained=False)
            self.pet_model.to(self.device).eval()
            self.debug["pet"]["loaded"] = True
            self.debug["pet"]["nclasses"] = nclasses
            self.debug["pet"]["mode"] = "untrained"
        if cognitive_dir:
            cog_w = Path(cognitive_dir) / "best_model.pth"
            scaler_p = Path(cognitive_dir) / "scaler.joblib"
            if cog_w.exists() and scaler_p.exists():
                ckpt = torch.load(str(cog_w), map_location=self.device)
                self.cog_model = CognitiveMLP(input_dim=len(ckpt["features"]), num_classes=len(self.mri_class_to_idx) or len(self.pet_class_to_idx) or 2)
                self.cog_model.load_state_dict(ckpt["model_state"])  # type: ignore
                self.cog_model.to(self.device).eval()
                self.cog_features = ckpt.get("features", [])
                self.cog_scaler = load_scaler(str(scaler_p))
                self.debug["cognitive"]["loaded"] = True
                self.debug["cognitive"]["mode"] = "mlp"

        # Determine if direct fusion is possible (identical class sets)
        self.fusion_enabled_direct = self.mri_class_to_idx and self.pet_class_to_idx and (
            self.mri_class_to_idx == self.pet_class_to_idx
        )
        self.fusion_idx_to_class = {v: k for k, v in self.mri_class_to_idx.items()} if self.fusion_enabled_direct else None

        self.tx = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def _predict_image(self, model, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        x = self.tx(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def _explain_mri(self, image_path: str) -> Optional[np.ndarray]:
        if self.mri_model is None:
            return None
        img = Image.open(image_path).convert("RGB")
        x = self.tx(img).unsqueeze(0).to(self.device)
        self.mri_model.zero_grad()
        logits = self.mri_model(x)
        cam = GradCAM(self.mri_model)(logits)
        return cam

    def _explain_pet(self, image_path: str) -> Optional[np.ndarray]:
        if self.pet_model is None:
            return None
        # Using attention rollout as a coarse explanation
        heat = attention_rollout_vit(self.pet_model)
        return heat

    def predict(
        self,
        patient_id: str,
        mri_path: Optional[str],
        pet_path: Optional[str],
        cognitive: Optional[Dict[str, Any]],
        save_explain_dir: Optional[str] = None,
        fusion: str = "average",
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {
            "patient_id": patient_id,
            "inputs": {"mri_path": mri_path, "pet_path": pet_path, "cognitive": cognitive},
            "predictions": {},
            "explainability": {},
            "debug": self.debug,
        }
        probs_list = []
        weights = []
        if mri_path and self.mri_model:
            mri_probs = self._predict_image(self.mri_model, mri_path)
            mri_label = self.mri_idx_to_class[int(np.argmax(mri_probs))] if self.mri_idx_to_class else int(np.argmax(mri_probs))
            outputs["predictions"]["mri"] = {
                "label": mri_label,
                "probabilities": mri_probs.tolist(),
            }
            probs_list.append(mri_probs)
            weights.append(1.0)
            if save_explain_dir:
                import cv2
                cam = self._explain_mri(mri_path)
                if cam is not None:
                    cam_u8 = (cam * 255).astype(np.uint8)
                    out_p = Path(save_explain_dir) / f"mri_{patient_id}.png"
                    Path(save_explain_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_p), cam_u8)
                    outputs["explainability"]["mri_heatmap"] = str(out_p).replace("\\", "/")
        if pet_path and self.pet_model:
            pet_probs = self._predict_image(self.pet_model, pet_path)
            pet_label = self.pet_idx_to_class[int(np.argmax(pet_probs))] if self.pet_idx_to_class else int(np.argmax(pet_probs))
            outputs["predictions"]["pet"] = {
                "label": pet_label,
                "probabilities": pet_probs.tolist(),
            }
            probs_list.append(pet_probs)
            weights.append(1.0)
            if save_explain_dir:
                import cv2
                heat = self._explain_pet(pet_path)
                if heat is not None:
                    heat_u8 = (heat * 255).astype(np.uint8)
                    out_p = Path(save_explain_dir) / f"pet_{patient_id}.png"
                    Path(save_explain_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_p), heat_u8)
                    outputs["explainability"]["pet_heatmap"] = str(out_p).replace("\\", "/")
        if cognitive is not None:
            if self.cog_model and self.cog_scaler is not None and self.cog_features:
                X = np.array([cognitive.get(k, 0.0) for k in self.cog_features], dtype=np.float32) if isinstance(cognitive, dict) else np.zeros((len(self.cog_features),), dtype=np.float32)
                X = (X - self.cog_scaler.mean_) / np.sqrt(self.cog_scaler.var_ + 1e-8)
                xt = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.cog_model(xt)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                # Treat cognitive probs as 5-class fusion space if lengths match, else leave as-is
                outputs["predictions"]["cognitive"] = {
                    "label": int(np.argmax(probs)),
                    "probabilities": probs.tolist(),
                }
                probs_list.append(probs)
                weights.append(0.5)
                self.debug["cognitive"]["mode"] = self.debug["cognitive"].get("mode") or "mlp"
            else:
                # Heuristic scoring: convert answers to an impairment score -> 5-class distribution [CN, EMCI, LMCI, MCI, AD]
                def _norm(v, lo, hi):
                    try:
                        v = float(v)
                        return max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-6)))
                    except Exception:
                        return 0.0
                score = 0.0
                total = 0.0
                # Orientation correctness
                if isinstance(cognitive, dict):
                    ori_year = 1.0 if str(cognitive.get("orientation_1", "")).strip() == "2025" else 0.0
                    ori_season = 1.0 if str(cognitive.get("orientation_2", "")).lower() in {"winter"} else 0.0
                    ori_month = 1.0 if str(cognitive.get("orientation_3", "")).lower() in {"december"} else 0.0
                    score += (1.0 - (ori_year + ori_season + ori_month) / 3.0) * 2.0; total += 2.0
                    # Memory related MCQ: worse -> higher impairment
                    memory_map = {"much better":0, "somewhat better":0.25, "about the same":0.5, "somewhat worse":0.75, "much worse":1.0}
                    mem1 = memory_map.get(str(cognitive.get("memory_1", "")).lower(), 0.5)
                    mem2_map = {"never":0.0, "rarely":0.25, "sometimes":0.5, "often":0.75, "always":1.0}
                    mem2 = mem2_map.get(str(cognitive.get("memory_2", "")).lower(), 0.5)
                    score += (mem1 + mem2) * 1.5; total += 1.5
                    # Attention numeric: 65 expected
                    att = cognitive.get("attention_1")
                    att_err = abs((float(att) if isinstance(att,(int,float,str)) and str(att).replace('.','',1).isdigit() else 0) - 65.0)
                    att_score = min(1.0, att_err / 20.0)
                    score += att_score * 1.5; total += 1.5
                    # Language frequency questions
                    lang1 = mem2_map.get(str(cognitive.get("language_1", "")).lower(), 0.5)
                    lang2 = mem2_map.get(str(cognitive.get("language_2", "")).lower(), 0.5)
                    score += (lang1 + lang2) * 1.0; total += 1.0
                    # Visuospatial
                    vis = mem2_map.get(str(cognitive.get("visuospatial_1", "")).lower(), 0.5)
                    score += vis * 0.8; total += 0.8
                    # Lifestyle
                    edu = _norm(cognitive.get("lifestyle_1", 12), 0, 25)
                    act = _norm(cognitive.get("lifestyle_2", 3), 1, 5)
                    cvd = {"no":0.0, "yes - mild":0.33, "yes - moderate":0.66, "yes - severe":1.0}.get(str(cognitive.get("lifestyle_3", "no")).lower(), 0.33)
                    # Lower education and activity increase risk; cvd increases risk
                    score += (1 - min(1.0, (edu*0.5 + (1-act)*0.5))) * 1.0; total += 1.0
                    score += cvd * 1.0; total += 1.0
                score = score / (total + 1e-6)
                # Map score to 5 classes with soft distribution
                centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
                probs = np.exp(-((score - centers) ** 2) / (2 * (0.12 ** 2)))
                probs = probs / (probs.sum() + 1e-8)
                idx = int(np.argmax(probs))
                fusion_classes = self.fusion["fusion_classes"] if self.fusion else ["CN","EMCI","LMCI","MCI","AD"]
                outputs["predictions"]["cognitive"] = {
                    "label": fusion_classes[idx],
                    "probabilities": probs.tolist(),
                }
                probs_list.append(probs)
                weights.append(0.5)
                self.debug["cognitive"]["mode"] = self.debug["cognitive"].get("mode") or "heuristic"

        # Fusion strategy:
        # 1) If a fusion mapping is provided, map each modality's probabilities into the fusion space and fuse there.
        # 2) Else, if direct fusion is possible (identical class sets), fuse directly.
        if probs_list:
            if self.fusion and (self.fusion.get("mri_R") is not None or self.fusion.get("pet_R") is not None):
                parts = []
                if "mri" in outputs["predictions"] and self.fusion.get("mri_R") is not None:
                    mri_p = np.array(outputs["predictions"]["mri"]["probabilities"], dtype=np.float32)
                    fm = mri_p @ self.fusion["mri_R"]  # to fusion space
                    parts.append(fm)
                if "pet" in outputs["predictions"] and self.fusion.get("pet_R") is not None:
                    pet_p = np.array(outputs["predictions"]["pet"]["probabilities"], dtype=np.float32)
                    fp = pet_p @ self.fusion["pet_R"]
                    parts.append(fp)
                if "cognitive" in outputs["predictions"]:
                    # If cognitive probs are already in fusion space, include them directly
                    cog_p = np.array(outputs["predictions"]["cognitive"]["probabilities"], dtype=np.float32)
                    if cog_p.ndim == 1 and (not self.fusion or len(cog_p) == len(self.fusion.get("fusion_classes", [])) or len(self.fusion["fusion_classes"]) == len(cog_p)):
                        parts.append(cog_p)
                if parts:
                    stack = np.stack(parts, axis=0)
                    fused = stack.mean(axis=0) if fusion != "weighted" else stack.mean(axis=0)
                    fused = fused / (fused.sum() + 1e-8)
                    fid = int(np.argmax(fused))
                    
                    # Calculate combined confidence as average of individual MRI and PET max confidences
                    individual_confidences = []
                    if "mri" in outputs["predictions"]:
                        mri_probs_raw = np.array(outputs["predictions"]["mri"]["probabilities"])
                        individual_confidences.append(float(mri_probs_raw.max()))
                    if "pet" in outputs["predictions"]:
                        pet_probs_raw = np.array(outputs["predictions"]["pet"]["probabilities"])
                        individual_confidences.append(float(pet_probs_raw.max()))
                    
                    # Use average of individual confidences instead of fused probability max
                    # If no MRI/PET available, fall back to original fused confidence
                    if len(individual_confidences) >= 2:
                        # Both MRI and PET available - use average
                        combined_confidence = np.mean(individual_confidences)
                    elif len(individual_confidences) == 1:
                        # Only one modality available - use its confidence directly
                        combined_confidence = individual_confidences[0]
                    else:
                        # No MRI/PET available - use original fused confidence
                        combined_confidence = float(fused.max())
                    
                    # Add debug info to output
                    outputs["debug"]["fusion_calculation"] = {
                        "individual_confidences": individual_confidences,
                        "combined_confidence": combined_confidence,
                        "original_fused_max": float(fused.max()),
                        "fusion_method": "average_of_individuals" if len(individual_confidences) >= 2 else "single_modality" if len(individual_confidences) == 1 else "fused_max",
                        "modalities_used": len(individual_confidences),
                        "mri_available": "mri" in outputs["predictions"],
                        "pet_available": "pet" in outputs["predictions"]
                    }
                    
                    # Adjust the fused probabilities to reflect the combined confidence
                    # Keep the predicted class the same but scale the max probability to match combined confidence
                    fused_adjusted = fused.copy()
                    if combined_confidence > 0:
                        # Scale the winning class probability to match the combined confidence
                        fused_adjusted[fid] = combined_confidence
                        # Redistribute the remaining probability among other classes
                        remaining_prob = 1.0 - combined_confidence
                        other_indices = [i for i in range(len(fused_adjusted)) if i != fid]
                        if other_indices and remaining_prob > 0:
                            # Distribute remaining probability proportionally among other classes
                            other_probs_sum = sum(fused[i] for i in other_indices)
                            if other_probs_sum > 0:
                                for i in other_indices:
                                    fused_adjusted[i] = (fused[i] / other_probs_sum) * remaining_prob
                            else:
                                # If all other probs are 0, distribute equally
                                for i in other_indices:
                                    fused_adjusted[i] = remaining_prob / len(other_indices)
                    
                    outputs["predictions"]["fusion"] = {
                        "label": self.fusion["idx_to_fusion"][fid],
                        "probabilities": fused_adjusted.tolist(),
                    }
                    outputs["debug"]["fusion_path_taken"] = "mapping_path"
            elif self.fusion_enabled_direct and len(set(map(len, probs_list))) == 1:
                if fusion == "weighted":
                    fused = weighted_fusion(probs_list, weights)
                else:
                    fused = average_fusion(probs_list)
                outputs["predictions"]["fusion"] = {
                    "label": self.fusion_idx_to_class[int(np.argmax(fused))] if self.fusion_idx_to_class else int(np.argmax(fused)),
                    "probabilities": fused.tolist(),
                }
            else:
                outputs["predictions"]["fusion_skipped_reason"] = "No fusion mapping and class taxonomies differ; fusion not applied."
        return outputs
