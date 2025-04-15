import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from datetime import datetime
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.spatial.distance import euclidean

# === CONFIG ===
IMAGE_FOLDER = "/Users/vishwajithjayathissa/Documents/Final Year Project/imageCapturing2.0/Cloud Tracking/cloudTracking/data/Inverter_Station_1"
MODEL_PATH = "/Users/vishwajithjayathissa/Documents/Final Year Project/imageCapturing2.0/Cloud Tracking/cloudTracking/models/segmentation_100epochs_final.pt"
OUTPUT_DIR = "outputs"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
NPZ_DIR = os.path.join(OUTPUT_DIR, "cloud_convlstm_npz_ukf")
CSV_PATH = os.path.join(OUTPUT_DIR, "cloud_tracking_ukf.csv")

os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(NPZ_DIR, exist_ok=True)

# === DEVICE ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = YOLO(MODEL_PATH)
model.to(device)

# === UKF SETUP ===
def fx(x, dt):
    x[0] += x[2] * dt
    x[1] += x[3] * dt
    return x

def hx(x):
    return x[:2]

def create_ukf():
    points = MerweScaledSigmaPoints(5, alpha=0.1, beta=2., kappa=0.)
    ukf = UKF(dim_x=5, dim_z=2, fx=fx, hx=hx, dt=1.0, points=points)
    ukf.x = np.zeros(5)
    ukf.P *= 10
    ukf.R *= 1.0
    ukf.Q *= 0.01
    return ukf

# === MAIN ===
image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".jpg")])
ukf_trackers = {}  # cloud_id -> {'ukf': UKF, 'sequence': list, 'last_seen': int}
tracking_data = []
next_cloud_id = 0
MAX_DISTANCE = 50

for frame_idx, fname in enumerate(tqdm(image_files, desc="🔍 Processing frames")):
    frame_path = os.path.join(IMAGE_FOLDER, fname)
    frame = cv2.imread(frame_path)
    results = model.predict(source=frame, device=device, verbose=False)[0]
    annotated = frame.copy()
    timestamp = fname.replace(".jpg", "")

    if results.masks is None:
        continue

    masks = results.masks.data.cpu().numpy()
    assigned_ids = set()
    current_centroids = []

    for i, segmentation in enumerate(masks):
        xyxy = results.boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(results.boxes.conf[i])
        if conf < 0.7:
            continue
        label = results.names[int(results.boxes.cls[i])]
        area = segmentation.sum()
        x1, y1, x2, y2 = xyxy
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        assigned_id = None

        for cloud_id, tracker_info in ukf_trackers.items():
            if cloud_id in assigned_ids:
                continue
            prev_cx, prev_cy = tracker_info['ukf'].x[:2]
            dist = euclidean((cx, cy), (prev_cx, prev_cy))
            if dist < MAX_DISTANCE:
                assigned_id = cloud_id
                break

        if assigned_id is None:
            assigned_id = next_cloud_id
            ukf_trackers[assigned_id] = {
                'ukf': create_ukf(),
                'sequence': [],
                'last_seen': frame_idx
            }
            next_cloud_id += 1

        tracker = ukf_trackers[assigned_id]['ukf']
        tracker.predict()
        tracker.update(np.array([cx, cy]))
        ukf_trackers[assigned_id]['last_seen'] = frame_idx
        assigned_ids.add(assigned_id)

        mask_resized = cv2.resize((segmentation * 255).astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
        ukf_trackers[assigned_id]['sequence'].append(mask_resized)

        tracking_data.append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "object_id": assigned_id,
            "centroid_x": cx,
            "centroid_y": cy,
            "vx": tracker.x[2],
            "vy": tracker.x[3],
            "area": area,
            "confidence": conf,
            "filename": fname
        })

        mask_resized_full = cv2.resize(segmentation.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros_like(annotated, dtype=np.uint8)
        color_mask[mask_resized_full.astype(bool)] = (180, 100, 255)
        annotated = cv2.addWeighted(annotated, 1.0, color_mask, 0.5, 0)
        cv2.rectangle(annotated, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(annotated, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    out_img_path = os.path.join(ANNOTATED_DIR, f"tracked_{fname}")
    cv2.imwrite(out_img_path, annotated)

# === SAVE TRACKING CSV ===
pd.DataFrame(tracking_data).to_csv(CSV_PATH, index=False)

# === SAVE .NPZ FILES ===
saved_npz = 0
for cloud_id, tracker_info in ukf_trackers.items():
    seq = tracker_info['sequence']
    if len(seq) >= 30:
        sequence = np.stack(seq[:30]).astype(np.float32) / 255.0
        np.savez_compressed(
            os.path.join(NPZ_DIR, f"cloud_{cloud_id}_start_0.npz"),
            mask_sequence=sequence,
            cloud_id=cloud_id,
            start_frame=0,
            image_filenames=image_files[:30]
        )
        saved_npz += 1

print(f"✅ Saved tracking CSV: {CSV_PATH}")
print(f"✅ Saved {saved_npz} cloud .npz sequences to: {NPZ_DIR}")
