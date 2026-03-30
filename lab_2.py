import os
import numpy as np
from tensorflow import keras
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.transform import rotate as sk_rotate
from scipy import ndimage
from scipy.ndimage import gaussian_filter

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model.keras")
PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
DEBUG_DIR  = os.path.join(BASE_DIR, "debug_out")

ANGLES     = [-30, -20, -10, 0, 10, 20, 30]
ROF_WEIGHT = 0.015        
MNIST_BOX  = 20            
MNIST_SIZE = 28            


def save_debug(arr, folder, filename):
    out_dir = os.path.join(DEBUG_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    img.save(os.path.join(out_dir, filename))


model = keras.models.load_model(MODEL_PATH)


def preprocess(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]

    img = Image.open(image_path).convert("L")
    gray = np.array(img).astype(np.float32) / 255.0
    save_debug(gray, stem, "01_grayscale.png")

    gmin, gmax = gray.min(), gray.max()
    if gmax - gmin > 1e-6:
        stretched = (gray - gmin) / (gmax - gmin)
    else:
        stretched = gray.copy()
    save_debug(stretched, stem, "02_stretched.png")

    blurred = gaussian_filter(stretched, sigma=1.5)
    save_debug(blurred, stem, "03_blurred.png")

    thresh = threshold_otsu(blurred)
    binary = blurred < thresh
    save_debug(binary.astype(np.float32), stem, "04_binary.png")

    labeled = label(binary)
    props   = regionprops(labeled)
    if props:
        largest = max(props, key=lambda p: p.area)
        digit_mask = (labeled == largest.label)
    else:
        digit_mask = binary
    save_debug(digit_mask.astype(np.float32), stem, "05_digit_mask.png")

    coords = np.argwhere(digit_mask)
    if coords.size > 0:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1
    else:
        r0, c0, r1, c1 = 0, 0, gray.shape[0], gray.shape[1]
    margin = max(4, int(0.1 * max(r1 - r0, c1 - c0)))
    r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
    r1, c1 = min(gray.shape[0], r1 + margin), min(gray.shape[1], c1 + margin)

    crop_gray = stretched[r0:r1, c0:c1]
    crop_mask = digit_mask[r0:r1, c0:c1]
    cleaned = np.where(crop_mask, crop_gray, 1.0)
    save_debug(cleaned, stem, "06_cleaned_crop.png")

    h, w = cleaned.shape
    scale = MNIST_BOX / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pil_crop = Image.fromarray(np.clip(cleaned * 255, 0, 255).astype(np.uint8))
    resized = np.array(
        pil_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    ).astype(np.float32) / 255.0
    save_debug(resized, stem, "07_resized_20.png")

    inv = 1.0 - resized
    save_debug(inv, stem, "08_inverted.png")

    if inv.max() > 0:
        inv = inv / inv.max()

    inv[inv < 0.15] = 0.0
    save_debug(inv, stem, "09_normalised.png")

    canvas = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.float32)
    if inv.sum() > 0:
        cy, cx = ndimage.center_of_mass(inv)
    else:
        cy, cx = new_h / 2.0, new_w / 2.0
    off_y = int(round(MNIST_SIZE / 2 - cy))
    off_x = int(round(MNIST_SIZE / 2 - cx))

    sy0 = max(0, -off_y);  dy0 = max(0, off_y)
    sx0 = max(0, -off_x);  dx0 = max(0, off_x)
    ch  = min(new_h - sy0,  MNIST_SIZE - dy0)
    cw  = min(new_w - sx0,  MNIST_SIZE - dx0)
    if ch > 0 and cw > 0:
        canvas[dy0:dy0+ch, dx0:dx0+cw] = inv[sy0:sy0+ch, sx0:sx0+cw]
    save_debug(canvas, stem, "10_centred_28x28.png")

    result = denoise_tv_chambolle(canvas, weight=ROF_WEIGHT, channel_axis=None)
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    save_debug(result, stem, "11_rof_final.png")

    return result


def recognize(img28):
    best_digit, best_prob, best_angle = None, -1, 0

    for angle in ANGLES:
        rotated = sk_rotate(img28, angle, resize=False, mode="constant",
                            cval=0.0, preserve_range=True)
        probs = model.predict(np.clip(rotated, 0, 1)[np.newaxis, ...], verbose=0)[0]
        digit, prob = int(np.argmax(probs)), float(np.max(probs))

        print(f"    кут={angle:>4}°  цифра={digit}  впевненість={prob:.4f}")

        if prob > best_prob:
            best_digit, best_prob, best_angle = digit, prob, angle

    return best_digit, best_prob, best_angle


exts = (".png", ".jpg", ".jpeg", ".bmp")
photos = sorted(
    os.path.join(PHOTOS_DIR, f)
    for f in os.listdir(PHOTOS_DIR)
    if f.lower().endswith(exts)
)

for path in photos:
    name = os.path.basename(path)
    print(f"[{name}]")

    img28 = preprocess(path)
    digit, conf, angle = recognize(img28)

    print(f"  >>> Цифра: {digit}  (впевненість: {conf*100:.1f}%, кут: {angle}°)\n")
