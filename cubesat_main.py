import time
import os
import cv2
import numpy as np
import socket
import threading
import queue
from picamera2 import Picamera2

WIDTH, HEIGHT         = 1024, 768
DISTANCE_M            = 2.0
FOCAL_LENGTH_PX       = 700
CAPTURE_INTERVAL_S    = 12
STORAGE_LIMIT_MB      = 14000
PHOTO_DIR             = "Photos"
LAPTOP_BT_MAC         = "70:D8:23:96:BA:DA"
BLUETOOTH_CHANNEL     = 12

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (WIDTH, HEIGHT)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
os.makedirs(PHOTO_DIR, exist_ok=True)
orb = cv2.ORB_create()

# Two queues — PSR files go in priority queue and get sent first
normal_queue   = queue.Queue()
priority_queue = queue.Queue()

def send_file_now(filepath):
    """Blocking send — used for PSR files to guarantee delivery."""
    if not os.path.exists(filepath):
        print(f"[BT] PSR file missing: {filepath}")
        return False
    for attempt in range(5):
        try:
            sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
            sock.settimeout(30)
            sock.connect((LAPTOP_BT_MAC, BLUETOOTH_CHANNEL))
            with open(filepath, "rb") as f:
                data = f.read()
            filename = os.path.basename(filepath)
            sock.send(f"{filename}:{len(data)}\n".encode())
            sock.sendall(data)
            sock.close()
            print(f"[BT] PSR SENT: {filename}")
            os.remove(filepath)
            print(f"[BT] PSR Deleted: {filename}")
            return True
        except Exception as e:
            print(f"[BT] PSR attempt {attempt+1} failed: {e}")
            time.sleep(3)
    print(f"[BT] PSR send FAILED after 5 attempts — file kept on Pi")
    return False

def bluetooth_sender():
    """Background sender for normal files — checks priority queue first."""
    while True:
        try:
            filepath = priority_queue.get_nowait()
        except queue.Empty:
            try:
                filepath = normal_queue.get(timeout=1)
            except queue.Empty:
                continue

        if not os.path.exists(filepath):
            print(f"[BT] File gone before send: {filepath}")
            continue

        success = False
        for attempt in range(5):
            try:
                sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
                sock.settimeout(30)
                sock.connect((LAPTOP_BT_MAC, BLUETOOTH_CHANNEL))
                with open(filepath, "rb") as f:
                    data = f.read()
                filename = os.path.basename(filepath)
                sock.send(f"{filename}:{len(data)}\n".encode())
                sock.sendall(data)
                sock.close()
                print(f"[BT] Sent {filename}")
                os.remove(filepath)
                print(f"[BT] Deleted {filename}")
                success = True
                break
            except Exception as e:
                print(f"[BT] Attempt {attempt+1} failed: {e}")
                time.sleep(3)
        if not success:
            print(f"[BT] Gave up on {filepath} — keeping file")

sender_thread = threading.Thread(target=bluetooth_sender, daemon=True)
sender_thread.start()

def queue_file(filepath):
    if filepath and os.path.exists(filepath):
        normal_queue.put(filepath)
    else:
        print(f"[QUEUE] Skipped missing file: {filepath}")

def queue_psr_file(filepath):
    """PSR files go to priority queue — sent before any normal files."""
    if filepath and os.path.exists(filepath):
        priority_queue.put(filepath)
        print(f"[QUEUE] PSR file queued with priority: {os.path.basename(filepath)}")
    else:
        print(f"[QUEUE] PSR file missing at queue time: {filepath}")

def get_used_mb():
    total_bytes = sum(
        os.path.getsize(os.path.join(PHOTO_DIR, f))
        for f in os.listdir(PHOTO_DIR)
        if os.path.isfile(os.path.join(PHOTO_DIR, f))
    )
    return total_bytes / (1024 * 1024)

def check_and_manage_storage():
    used = get_used_mb()
    if used >= STORAGE_LIMIT_MB:
        print("[STORAGE] Full — skipping capture")
        return False
    return True

# ── SR-1: Imaging Capability ─────────────────────────────────
def capture_gray():
    frame    = picam2.capture_array()
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filename = os.path.join(PHOTO_DIR, f"photo_{int(time.time())}.png")
    cv2.imwrite(filename, gray)
    print(f"[CAPTURE] Saved {filename} ({os.path.getsize(filename)/1024:.1f} KB)")
    return gray, filename

# ── SR-2: PSR shadow segmentation ────────────────────────────
def segment_light_shadow(gray):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t, gray > t, gray <= t

def estimate_height(pixel_height, focal_length_px, distance_m):
    return (pixel_height * distance_m) / focal_length_px

def compare_images(img1, img2):
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return 0.0, 0.0, [], k1, k2
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 4:
        return 0.0, 0.0, matches, k1, k2
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    H, _ = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC)
    if H is None:
        return 0.0, 0.0, matches, k1, k2
    return H[0, 2], H[1, 2], matches, k1, k2

def overlay_light_shadow(gray, light, shadow, filename):
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[light]  = [0, 255, 0]
    overlay[shadow] = [0, 0, 255]
    out = filename.replace(".png", "_overlay.png")
    cv2.imwrite(out, overlay)
    return out

def overlay_height(gray, light, pixel_height, height_m, filename):
    overlay  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    y_coords = np.where(light)[0]
    if len(y_coords) > 0:
        top, bottom = int(y_coords.min()), int(y_coords.max())
        cv2.rectangle(overlay, (0, top), (gray.shape[1]-1, bottom), (255, 0, 0), 2)
        cv2.putText(overlay, f"{height_m:.2f}m", (10, max(top-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    out = filename.replace(".png", "_height.png")
    cv2.imwrite(out, overlay)
    return out

def overlay_matches(img1, img2, matches, k1, k2, filename):
    if not matches:
        return None
    match_img = cv2.drawMatches(img1, k1, img2, k2, matches[:20], None, flags=2)
    out = filename.replace(".png", "_matches.png")
    cv2.imwrite(out, match_img)
    return out

# ── MR-1 + MR-2: PSR Detection & Binary Map ──────────────────
def detect_psr(shadow_masks):
    if len(shadow_masks) < 3:
        print("[PSR] Need at least 3 images for PSR detection")
        return None
    psr_map = shadow_masks[0].copy()
    for mask in shadow_masks[1:]:
        psr_map = np.logical_and(psr_map, mask)
    return psr_map

def save_psr_overlay(gray, psr_map, cycle_number):
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[psr_map] = [0, 0, 255]
    psr_percent = (np.sum(psr_map) / psr_map.size) * 100
    cv2.putText(overlay, f"PSR: {psr_percent:.1f}% of frame",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    psr_filename = os.path.join(PHOTO_DIR, f"psr_cycle_{cycle_number}_{int(time.time())}.png")
    cv2.imwrite(psr_filename, overlay)
    print(f"[PSR] {psr_percent:.1f}% of frame is PSR — saved as {os.path.basename(psr_filename)}")

    # Also save a binary map as a numpy file — fulfills MR-2
    binary_filename = psr_filename.replace(".png", "_binary.npy")
    np.save(binary_filename, psr_map.astype(np.uint8))
    print(f"[PSR] Binary map saved: {os.path.basename(binary_filename)}")

    return psr_filename, binary_filename

# ── MR-3: Autonomous Operation ────────────────────────────────
def main():
    print("[MAIN] Starting — clearing old photos and taking first picture now")
    for f in os.listdir(PHOTO_DIR):
        os.remove(os.path.join(PHOTO_DIR, f))
    print("[MAIN] Photos folder cleared")

    prev_gray    = None
    shadow_masks = []
    image_count  = 0
    psr_cycle    = 0

    while True:
        # Storage check
        if not check_and_manage_storage():
            time.sleep(5)
            continue

        # SR-1: Capture image
        gray, filename = capture_gray()
        image_count += 1

        # SR-2: Segment and store shadow mask for PSR stack
        threshold, light, shadow = segment_light_shadow(gray)
        y_coords     = np.where(light)[0]
        pixel_height = int(y_coords.max() - y_coords.min()) if len(y_coords) > 0 else 0
        height_m     = estimate_height(pixel_height, FOCAL_LENGTH_PX, DISTANCE_M) if pixel_height > 0 else 0.0
        print(f"[ANALYSIS] threshold={threshold} px_height={pixel_height} height={height_m:.3f}m")

        shadow_masks.append(shadow)

        # Generate overlays
        overlay_file = overlay_light_shadow(gray, light, shadow, filename)
        height_file  = overlay_height(gray, light, pixel_height, height_m, filename)

        # MR-4 / COM: Queue normal files for downlink
        queue_file(filename)
        queue_file(overlay_file)
        queue_file(height_file)

        # Motion comparison
        if prev_gray is not None:
            dx, dy, matches, k1, k2 = compare_images(prev_gray, gray)
            print(f"[MOTION] dx={dx:.2f} dy={dy:.2f} matches={len(matches)}")
            matches_file = overlay_matches(prev_gray, gray, matches, k1, k2, filename)
            if matches_file:
                queue_file(matches_file)

        # MR-1 + MR-2: Every 3 images run PSR detection
        if image_count % 3 == 0:
            psr_cycle += 1
            print(f"[PSR] Running PSR detection — cycle {psr_cycle}...")
            psr_map = detect_psr(shadow_masks)
            if psr_map is not None:
                psr_file, binary_file = save_psr_overlay(gray, psr_map, psr_cycle)
                time.sleep(0.5)
                # PSR files go to priority queue — sent before normal images
                queue_psr_file(psr_file)
                queue_psr_file(binary_file)
            shadow_masks = []

        prev_gray = gray
        print(f"[MAIN] Waiting {CAPTURE_INTERVAL_S}s until next capture...")
        time.sleep(CAPTURE_INTERVAL_S)

if __name__ == "__main__":
    main()