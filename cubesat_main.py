import time
import os
import cv2
import numpy as np
import bluetooth
from picamera2 import Picamera2

WIDTH, HEIGHT         = 1024, 768
DISTANCE_M            = 2.0
FOCAL_LENGTH_PX       = 700
CAPTURE_INTERVAL_S    = 120
STORAGE_LIMIT_MB      = 14000
PHOTO_DIR             = "Photos"
DOWNLINKED_LOG        = "downlinked.txt"
LAPTOP_BT_MAC         = "70:D8:23:96:BA:DA"
BLUETOOTH_CHANNEL     = 12

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    {'format': 'YUV420', 'size': (WIDTH, HEIGHT)}
)
picam2.configure(config)
picam2.start()
os.makedirs(PHOTO_DIR, exist_ok=True)
orb = cv2.ORB_create()

def send_via_bluetooth(filepath):
    try:
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((LAPTOP_BT_MAC, BLUETOOTH_CHANNEL))
        with open(filepath, "rb") as f:
            data = f.read()
        filename = os.path.basename(filepath)
        sock.send(f"{filename}:{len(data)}\n".encode())
        sock.sendall(data)
        sock.close()
        mark_downlinked(filename)
        print(f"[BT] Sent {filename}")
        return True
    except Exception as e:
        print(f"[BT] Failed: {e}")
        return False

def send_all_pending():
    downlinked = load_downlinked()
    pending = [
        f for f in sorted(os.listdir(PHOTO_DIR))
        if f.endswith(".png")
        and "_overlay" not in f
        and "_height"  not in f
        and "_matches" not in f
        and f not in downlinked
    ]
    for fname in pending:
        send_via_bluetooth(os.path.join(PHOTO_DIR, fname))

def get_used_mb():
    total_bytes = sum(
        os.path.getsize(os.path.join(PHOTO_DIR, f))
        for f in os.listdir(PHOTO_DIR)
        if os.path.isfile(os.path.join(PHOTO_DIR, f))
    )
    return total_bytes / (1024 * 1024)

def load_downlinked():
    if not os.path.exists(DOWNLINKED_LOG):
        return set()
    with open(DOWNLINKED_LOG) as fh:
        return set(line.strip() for line in fh)

def mark_downlinked(filename):
    with open(DOWNLINKED_LOG, "a") as fh:
        fh.write(filename + "\n")

def purge_downlinked_images():
    downlinked = load_downlinked()
    candidates = sorted(
        [f for f in os.listdir(PHOTO_DIR) if f in downlinked],
        key=lambda f: os.path.getmtime(os.path.join(PHOTO_DIR, f))
    )
    for fname in candidates:
        if get_used_mb() < STORAGE_LIMIT_MB:
            break
        os.remove(os.path.join(PHOTO_DIR, fname))

def check_and_manage_storage():
    used = get_used_mb()
    if used >= STORAGE_LIMIT_MB:
        purge_downlinked_images()
        if get_used_mb() >= STORAGE_LIMIT_MB:
            return False
    return True

def capture_gray():
    time.sleep(CAPTURE_INTERVAL_S)
    yuv  = picam2.capture_array()
    gray = yuv[:HEIGHT, :WIDTH].copy()
    filename = os.path.join(PHOTO_DIR, f"photo_{int(time.time())}.png")
    cv2.imwrite(filename, gray)
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
    cv2.imwrite(filename.replace(".png", "_overlay.png"), overlay)

def overlay_height(gray, light, pixel_height, height_m, filename):
    overlay  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    y_coords = np.where(light)[0]
    if len(y_coords) > 0:
        top, bottom = int(y_coords.min()), int(y_coords.max())
        cv2.rectangle(overlay, (0, top), (gray.shape[1]-1, bottom), (255, 0, 0), 2)
        cv2.putText(overlay, f"{height_m:.2f}m", (10, max(top-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imwrite(filename.replace(".png", "_height.png"), overlay)

def overlay_matches(img1, img2, matches, k1, k2, filename):
    if not matches:
        return
    match_img = cv2.drawMatches(img1, k1, img2, k2, matches[:20], None, flags=2)
    cv2.imwrite(filename.replace(".png", "_matches.png"), match_img)

def main():
    send_all_pending()
    prev_gray = None

    while True:
        if not check_and_manage_storage():
            time.sleep(60)
            continue

        gray, filename = capture_gray()

        threshold, light, shadow = segment_light_shadow(gray)
        y_coords     = np.where(light)[0]
        pixel_height = int(y_coords.max() - y_coords.min()) if len(y_coords) > 0 else 0
        height_m     = estimate_height(pixel_height, FOCAL_LENGTH_PX, DISTANCE_M) if pixel_height > 0 else 0.0

        overlay_light_shadow(gray, light, shadow, filename)
        overlay_height(gray, light, pixel_height, height_m, filename)

        if prev_gray is not None:
            dx, dy, matches, k1, k2 = compare_images(prev_gray, gray)
            overlay_matches(prev_gray, gray, matches, k1, k2, filename)

        send_via_bluetooth(filename)
        prev_gray = gray

if __name__ == "__main__":
    main()