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

send_queue = queue.Queue()

def bluetooth_sender():
    while True:
        filepath = send_queue.get()
        if not os.path.exists(filepath):
            send_queue.task_done()
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
        send_queue.task_done()

sender_thread = threading.Thread(target=bluetooth_sender, daemon=True)
sender_thread.start()

def queue_file(filepath):
    send_queue.put(filepath)

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
        print("[STORAGE] Full and nothing to purge — skipping capture")
        return False
    return True

def capture_gray():
    frame    = picam2.capture_array()
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filename = os.path.join(PHOTO_DIR, f"photo_{int(time.time())}.png")
    cv2.imwrite(filename, gray)
    print(f"[CAPTURE] Saved {filename} ({os.path.getsize(filename)/1024:.1f} KB)")
    return gray, filename

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

def main():
    print("[MAIN] Starting — clearing old photos and taking first picture now")
    for f in os.listdir(PHOTO_DIR):
        os.remove(os.path.join(PHOTO_DIR, f))
    print("[MAIN] Photos folder cleared")

    prev_gray = None

    while True:
        if not check_and_manage_storage():
            time.sleep(5)
            continue

        gray, filename = capture_gray()

        threshold, light, shadow = segment_light_shadow(gray)
        y_coords     = np.where(light)[0]
        pixel_height = int(y_coords.max() - y_coords.min()) if len(y_coords) > 0 else 0
        height_m     = estimate_height(pixel_height, FOCAL_LENGTH_PX, DISTANCE_M) if pixel_height > 0 else 0.0
        print(f"[ANALYSIS] threshold={threshold} px_height={pixel_height} height={height_m:.3f}m")

        overlay_file = overlay_light_shadow(gray, light, shadow, filename)
        height_file  = overlay_height(gray, light, pixel_height, height_m, filename)

        queue_file(filename)
        queue_file(overlay_file)
        queue_file(height_file)

        if prev_gray is not None:
            dx, dy, matches, k1, k2 = compare_images(prev_gray, gray)
            print(f"[MOTION] dx={dx:.2f} dy={dy:.2f} matches={len(matches)}")
            matches_file = overlay_matches(prev_gray, gray, matches, k1, k2, filename)
            if matches_file:
                queue_file(matches_file)

        prev_gray = gray
        print(f"[MAIN] Waiting {CAPTURE_INTERVAL_S}s until next capture...")
        time.sleep(CAPTURE_INTERVAL_S)

if __name__ == "__main__":
    main()