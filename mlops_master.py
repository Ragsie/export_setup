#!/usr/bin/env python3
"""
OmniMow - Complete MLOps Pipeline for TrueNAS Server (Qualcomm QCS6490 / Radxa Dragon Q6A Edition)
=============================================================================================
This script is your complete, production-ready, self-driving MLOps brain on your TrueNAS server.
It handles all the heavy lifting: Active Learning, auto-segmentation, review,
automatic training and export to your new 12 TOPS Qualcomm Snapdragon NPU.

Features:
1. Grounded-SAM (DINO Base + SAM ViT-Base) for auto-annotation at 23:00 (or manually).
2. Precise mask subtraction in NumPy: Automatically subtracts obstacles (poo, toy, dog, etc.)
   from your grass and pavement polygons, so your robot learns precise garden boundaries.
3. Torchvision Batched NMS: Removes all annoying duplicate polygons and bboxes.
4. Watchdog with Debounce: Watches your approved dataset and starts YOLOv26-seg training (800x800)
   automatically 10 seconds after you have moved images over from review.
5. Qualcomm QNN Export: Calls export.py, which compiles a Hexagon v68 HTP context binary (.bin)
   and bundles it with the QNN runtime from the same SDK into a deploy package (.tar.gz) for the Radxa.

Paths on your TrueNAS server:
- /mnt/Meta-pool/yolo_training/incoming_raw  (Uncertain images from the robot)
- /mnt/Meta-pool/yolo_training/review        (Contains classes.txt for AnyLabeling)
- /mnt/Meta-pool/yolo_training/dataset       (The official training dataset)
"""

import os
import time
import shutil
import urllib.request
import threading
import subprocess
import argparse
import numpy as np
import cv2
import torch
import schedule
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ultralytics import YOLO

# ==============================================================================
# 1. CONFIGURATION & PATHS (Matches your TrueNAS container)
# ==============================================================================
BASE_DIR = "/home/jovyan/work"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train/images")
TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, "train/labels")

INCOMING_DIR = os.path.join(BASE_DIR, "incoming_raw")
REVIEW_DIR = os.path.join(BASE_DIR, "review")
REVIEW_IMAGES_DIR = os.path.join(REVIEW_DIR, "images")
REVIEW_LABELS_DIR = os.path.join(REVIEW_DIR, "labels")

YAML_PATH = os.path.join(BASE_DIR, "data.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
EXPORT_DIR = os.path.join(BASE_DIR, "exported_models")
WEIGHTS_DIR = os.path.join(BASE_DIR, "model_weights")

# Qualcomm QNN configuration (None = auto-detect, see export.find_qnn_root)
QNN_SDK_ROOT = None
EXPORT_MODEL_NAME = "MowerAIn_seg"
# Calibration images for quantization (should resemble what the robot sees). Train + valid gives
# more images; calibration is not evaluation, so training images are fine.
CALIB_DIR = [TRAIN_IMAGES_DIR, os.path.join(DATASET_DIR, "valid/images")]
TRAIN_RUN_NAME = "worx_segmentation_v2"
# Model input on the NPU (height, width). Training is 800x800, but the camera is 16:9, so at
# 800x800, 44% of the input is grey letterbox border. 448x800 covers the image (2 px border) and saves
# ~44% NPU work without retraining. Set to 800 for square input.
EXPORT_IMGSZ = (448, 800)
# QNN runtime version on the SBC (from device/qnn_info.py). The export stops if the SDK is newer,
# so the model can always run on the Radxa's own runtime. None = allow newer SDK (requires the deploy package's runtime).
DEVICE_QNN_VERSION = "2.42.0.251225"

# Models & Weights
YOLO_SEG_MODEL = "yolo26n-seg.pt"
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam_vit_b_01ec64.pth")
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# ==============================================================================
# 2. CLASSES & SEARCH TERMS (CLASS_MAP) - Matches data.yaml 1:1
# ==============================================================================
CLASS_MAP = {
    0: {"name": "Animal", "prompts": ["dog", "puppy", "canine", "cat", "kitten", "feline", "pet"]},
    1: {"name": "Grass", "prompts": ["lawn", "green grass", "turf", "mowed grass"]},
    2: {"name": "Pavement", "prompts": ["concrete slab", "paving stones", "patio tiles", "sidewalk", "cobblestone"]},
    3: {"name": "Poop", "prompts": ["dog feces", "dog poop", "animal droppings", "dark feces on grass", "poop on lawn"]},
    4: {"name": "Toy", "prompts": ["dog toy", "cloth toy", "plush toy", "frisbee", "ball", "stuffed animal"]},
    5: {"name": "Tree", "prompts": ["tree trunk", "bush", "shrub", "plant pot"]},
    6: {"name": "Wall", "prompts": ["brick wall", "wooden fence", "black wooden fence", "building facade", "shed wall"]}
}

ALL_PROMPTS = []
for entry in CLASS_MAP.values():
    ALL_PROMPTS.extend(entry["prompts"])

TEXT_PROMPT = ". ".join(ALL_PROMPTS) + "."

# Thread locks and timers for controlling events
training_lock = threading.Lock()
pending_training_timer = None

# Create all required directories at startup
REQUIRED_DIRS = [
    INCOMING_DIR,
    REVIEW_IMAGES_DIR,
    REVIEW_LABELS_DIR,
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    os.path.join(DATASET_DIR, "valid/images"),
    os.path.join(DATASET_DIR, "valid/labels"),
    RUNS_DIR,
    EXPORT_DIR,
    WEIGHTS_DIR
]

for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)

# Ensure classes.txt for AnyLabeling or other review tools
def sync_classes_txt():
    classes_content = "\n".join([cdata["name"] for cdata in CLASS_MAP.values()]) + "\n"
    target_paths = [
        os.path.join(REVIEW_DIR, "classes.txt"),
        os.path.join(REVIEW_LABELS_DIR, "classes.txt"),
        os.path.join(DATASET_DIR, "classes.txt"),
        os.path.join(TRAIN_LABELS_DIR, "classes.txt")
    ]
    for p in target_paths:
        try:
            with open(p, "w") as f:
                f.write(classes_content)
        except Exception:
            pass

sync_classes_txt()

def ensure_sam_weights():
    """Automatically downloads SAM ViT-Base if the file is missing"""
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"📥 Downloading SAM ViT-Base to {SAM_CHECKPOINT}...", flush=True)
        urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)
        print("✅ SAM ViT-Base download complete!", flush=True)

# ==============================================================================
# 3. CONTOUR & POLYGON PROCESSING (Split for mask subtraction)
# ==============================================================================
def mask_to_yolo_polygons(mask, img_w, img_h, min_contour_points=6):
    """Converts a binary mask (incl. split regions) to normalized polygon strings."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    poly_strings = []
    for contour in contours:
        # Ignore microscopic noise islands (< 25 pixels)
        if cv2.contourArea(contour) < 25:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # YOLO requires at least 3 points (6 numbers)
        if len(approx) < (min_contour_points // 2):
            continue

        coords = []
        for point in approx:
            px, py = point[0]
            norm_x = max(0.0, min(1.0, px / img_w))
            norm_y = max(0.0, min(1.0, py / img_h))
            coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

        if coords:
            poly_strings.append(" ".join(coords))

    return poly_strings

# ==============================================================================
# 4. GROUNDED-SAM AUTO-ANNOTATION WITH MASK SUBTRACTION (Runs at 23:00)
# ==============================================================================
def run_auto_annotation():
    """Scans incoming_raw for uncertain images, runs Grounded-SAM with NMS and mask subtraction"""
    sync_classes_txt()
    valid_exts = ('.png', '.jpg', '.jpeg')
    raw_images = [f for f in os.listdir(INCOMING_DIR) if f.lower().endswith(valid_exts)]
    
    if not raw_images:
        print("ℹ️ [AUTO-ANNOTATE] No new uncertain images in incoming_raw.", flush=True)
        return

    print(f"\n🏷️ [AUTO-ANNOTATE] Starting Grounding-DINO + SAM Segmentation on {len(raw_images)} images...", flush=True)

    try:
        import torch
        from torchvision.ops import batched_nms
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from segment_anything import sam_model_registry, SamPredictor

        ensure_sam_weights()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        print(f"⚙️ Loading models on GPU: {device.upper()} (Mixed Precision / VRAM protected)...", flush=True)

        # 1. Load Grounding DINO Base
        dino_model_id = "IDEA-Research/grounding-dino-base"
        dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

        # 2. Load SAM ViT-Base
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(device)
        sam_predictor = SamPredictor(sam)

        for img_name in raw_images:
            img_path = os.path.join(INCOMING_DIR, img_name)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue

            h, w, _ = image_bgr.shape
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Detect bounding boxes with Grounding DINO under autocast (Mixed Precision)
            inputs = dino_processor(images=image_rgb, text=TEXT_PROMPT, return_tensors="pt").to(device)

            with torch.no_grad():
                if device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = dino_model(**inputs)
                else:
                    outputs = dino_model(**inputs)

            results = dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=0.22,       # Low threshold to catch small or hidden objects in the grass
                text_threshold=0.20,
                target_sizes=[(h, w)]
            )[0]

            # Match the detected bboxes back to our 7 classes
            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]

            raw_boxes = []
            raw_scores = []
            raw_classes = []

            for box, score, label in zip(boxes, scores, labels):
                label_clean = label.lower().strip()
                matched_id = None
                for cid, cdata in CLASS_MAP.items():
                    if any(p in label_clean for p in cdata["prompts"]):
                        matched_id = cid
                        break
                if matched_id is not None:
                    raw_boxes.append(box.float())
                    raw_scores.append(score.float())
                    raw_classes.append(matched_id)

            # Run Torchvision Batched NMS to remove duplicates
            valid_boxes = []
            matched_classes = []

            if raw_boxes:
                b_tensor = torch.stack(raw_boxes)
                s_tensor = torch.stack(raw_scores)
                c_tensor = torch.tensor(raw_classes, device=device)

                # iou_threshold=0.45 removes overlapping duplicates within the same class
                keep_indices = batched_nms(b_tensor, s_tensor, c_tensor, iou_threshold=0.45)

                for idx in keep_indices.tolist():
                    valid_boxes.append(raw_boxes[idx])
                    matched_classes.append(raw_classes[idx])

            base_name = os.path.splitext(img_name)[0]
            label_file = os.path.join(REVIEW_LABELS_DIR, f"{base_name}.txt")

            if valid_boxes:
                boxes_tensor = torch.stack(valid_boxes)
                sam_predictor.set_image(image_rgb)
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_tensor, image_rgb.shape[:2])
                
                # Generate all binary masks
                raw_item_masks = []
                with torch.no_grad():
                    for single_box in transformed_boxes.split(1):
                        if device == "cuda":
                            with torch.cuda.amp.autocast():
                                mask, _, _ = sam_predictor.predict_torch(
                                    point_coords=None,
                                    point_labels=None,
                                    boxes=single_box,
                                    multimask_output=False,
                                )
                        else:
                            mask, _, _ = sam_predictor.predict_torch(
                                point_coords=None,
                                point_labels=None,
                                boxes=single_box,
                                multimask_output=False,
                            )
                        raw_item_masks.append(mask[0][0].cpu().numpy().astype(np.uint8))

                # --- MASK SUBTRACTION ALGORITHM ---
                # Build a combined union of all obstacles (everything except grass and pavement)
                obstacle_union = np.zeros((h, w), dtype=np.uint8)
                for cid, m_np in zip(matched_classes, raw_item_masks):
                    if cid not in [1, 2]:  # dog (0), poo (3), toy (4), tree (5), wall (6)
                        obstacle_union = np.bitwise_or(obstacle_union, m_np)

                # Subtract obstacles from grass (1) and pavement (2) masks for millimeter-precise edges
                final_labels = []
                for cid, m_np in zip(matched_classes, raw_item_masks):
                    if cid in [1, 2]:  # Grass or Pavement
                        processed_mask = np.where(obstacle_union > 0, 0, m_np).astype(np.uint8)
                    else:
                        processed_mask = m_np

                    # Generate YOLO-seg polygon strings
                    polys = mask_to_yolo_polygons(processed_mask, w, h)
                    for poly in polys:
                        final_labels.append(f"{cid} {poly}\n")

                with open(label_file, "w") as lf:
                    lf.writelines(final_labels)
                polygons_written = len(final_labels)
            else:
                open(label_file, "w").close()
                polygons_written = 0

            # Move the finished pair to the review folder
            shutil.move(img_path, os.path.join(REVIEW_IMAGES_DIR, img_name))
            print(f"👁️ Ready for review ({polygons_written} polygons segmented): {img_name}", flush=True)

        # Free resources from the GPU
        del dino_model, sam, sam_predictor
        torch.cuda.empty_cache()
        print("🎉 [AUTO-ANNOTATE] Done! Images and labels are ready in 'review/'", flush=True)

    except Exception as e:
        print(f"❌ Error during Grounded-SAM annotation: {e}", flush=True)
# ==============================================================================
# 5. QUALCOMM QNN EXPORT (Hexagon v68 HTP Context Binary via export.py)
# ==============================================================================
def export_to_qualcomm_qnn(pt_model_path, imgsz=EXPORT_IMGSZ):
    """
    Runs the full export.py chain: .pt -> ONNX -> QNN (INT8, calibrated on the validation images)
    -> x86_64 .so -> Hexagon HTP context binary (.bin) -> deploy package (.tar.gz) for Radxa Dragon Q6A.
    Returns the path to the deploy package, or None on failure.
    """
    print(f"\n🔄 [QUALCOMM QNN EXPORT] Starting export of {pt_model_path}...", flush=True)

    if not os.path.exists(pt_model_path):
        print(f"❌ Error: Could not find PyTorch model at {pt_model_path}", flush=True)
        return None

    try:
        from export import export_pipeline
        return export_pipeline(
            pt_model_path,
            calib_dir=CALIB_DIR,
            out_dir=EXPORT_DIR,
            imgsz=imgsz,
            qnn_root=QNN_SDK_ROOT,
            base_name=EXPORT_MODEL_NAME,
            device_qnn_version=DEVICE_QNN_VERSION,
        )
    except Exception as e:
        print(f"❌ Error during QNN export: {e}", flush=True)
        return None

# ==============================================================================
# 6. YOLOv26-SEG TRAINING (Runs immediately when approved files are added)
# ==============================================================================
def run_training_job():
    """Trains your YOLOv26-seg model on your approved dataset"""
    if not training_lock.acquire(blocking=False):
        print("⚠️ Training already running. Ignoring new trigger.", flush=True)
        return

    try:
        # Free all previously allocated GPU memory
        torch.cuda.empty_cache()

        if not os.path.exists(YAML_PATH):
            print(f"❌ Error: Cannot find {YAML_PATH}. Aborting training.", flush=True)
            return

        valid_exts = ('.png', '.jpg', '.jpeg')
        train_images = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.lower().endswith(valid_exts)]
        
        if len(train_images) == 0:
            print(f"⚠️ No images found in {TRAIN_IMAGES_DIR}. Skipping training.", flush=True)
            return

        print("\n" + "=" * 50, flush=True)
        print(f"🚀 [AUTO-TRAIN] New files approved! Starting training of {YOLO_SEG_MODEL} (800x800)...", flush=True)
        print(f"📊 Training set size: {len(train_images)} images", flush=True)
        print("=" * 50, flush=True)

        model = YOLO(YOLO_SEG_MODEL)
        
        # Improved hyperparameters and augmentations specifically for lawn mowers (800x800 px)
        model.train(
            data=YAML_PATH,
            epochs=150,           # Full training
            patience=25,
            batch=8,              # Matches your GTX 1660 VRAM limit
            imgsz=800,            # Train at high resolution for small poops and toys
            device=0,             # NVIDIA GPU
            workers=4,
            project=os.path.join(RUNS_DIR, "train"),
            name=TRAIN_RUN_NAME,
            exist_ok=True,
            
            # Geometric variation for oblique camera angles
            mosaic=1.0,
            close_mosaic=15,
            copy_paste=0.3,
            scale=0.6,
            erasing=0.2,
            degrees=5.0,
            translate=0.1,
            fliplr=0.5,
            flipud=0.0,
            
            # Light & weather (sun, shadows, glare in dew)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4
        )
        print("✅ YOLO segmentation training complete!", flush=True)

        best_pt_path = os.path.join(RUNS_DIR, "train", TRAIN_RUN_NAME, "weights", "best.pt")
        if os.path.exists(best_pt_path):
            # Export and compile for the Qualcomm Snapdragon NPU!
            export_to_qualcomm_qnn(best_pt_path)
        else:
            print(f"❌ Could not find {best_pt_path} - skipping export.", flush=True)

    except Exception as e:
        print(f"❌ Error during YOLO training: {e}", flush=True)
    finally:
        training_lock.release()

def schedule_instant_training():
    """Debounce timer: Waits 10 seconds after the last file transfer before training starts"""
    global pending_training_timer
    if pending_training_timer is not None:
        pending_training_timer.cancel()
    
    print("⏳ Detected new files... Automatically starting YOLO training in 10 seconds...", flush=True)
    pending_training_timer = threading.Timer(10.0, run_training_job)
    pending_training_timer.start()

# ==============================================================================
# 7. WATCHDOG DATASET OBSERVER
# ==============================================================================
class DatasetChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            valid_exts = ('.png', '.jpg', '.jpeg', '.txt')
            if event.src_path.lower().endswith(valid_exts):
                schedule_instant_training()

    def on_moved(self, event):
        if not event.is_directory:
            valid_exts = ('.png', '.jpg', '.jpeg', '.txt')
            if event.dest_path.lower().endswith(valid_exts):
                schedule_instant_training()

# Grounded-SAM is scheduled to run on its own every night
schedule.every().day.at("23:00").do(run_auto_annotation)

def main():
    # Only watch your approved dataset
    event_handler = DatasetChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, TRAIN_IMAGES_DIR, recursive=False)
    observer.schedule(event_handler, TRAIN_LABELS_DIR, recursive=False)
    observer.start()

    print("🤖 OmniMow MLOps Pipeline Started!", flush=True)
    print(f"🌙 23:00 Schedule: Grounded-SAM scans '{INCOMING_DIR}' -> '{REVIEW_DIR}'")
    print(f"⚡ Training Trigger: Auto-starts YOLOv26-seg training when you move approved files to '{TRAIN_IMAGES_DIR}'")
    print("----------------------------------------------------------------", flush=True)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
