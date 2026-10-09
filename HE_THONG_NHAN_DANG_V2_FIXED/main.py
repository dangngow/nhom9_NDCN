import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ============================================================
# HE THONG NHAN DIEN DO DUNG HOC TAP - V2 FIXED
# OpenCV 5 + HOG tu viet + SVM
# Khong dung cv2.HOGDescriptor
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLASSES = {
    "but": "BÚT BI",
    "butchi": "BÚT CHÌ",
    "tay": "TẨY",
    "thuoc": "THƯỚC"
}

IMAGE_SIZE = (128, 128)
model = None

# ------------------------------------------------------------
# TIEN XU LY / TIM VAT THE
# ------------------------------------------------------------

def make_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, a = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def best_area(mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        h, w = mask.shape
        best = 0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            ratio = area / float(h * w)
            if 0.005 < ratio < 0.90:
                best = max(best, area)
        return best

    mask = a if best_area(a) >= best_area(b) else b
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = mask.shape
    valid = [c for c in contours if cv2.contourArea(c) >= h * w * 0.005]
    return max(valid, key=cv2.contourArea) if valid else max(contours, key=cv2.contourArea)


def normalize_object(img):
    mask = make_mask(img)
    contour = largest_contour(mask)
    if contour is None:
        return None

    # Xac dinh goc va chuan hoa huong bang minAreaRect
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    if rw <= 1 or rh <= 1:
        return None
    if rw < rh:
        angle += 90

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    pts = contour.reshape(-1, 2).astype(np.float32)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts2 = np.hstack([pts, ones]) @ M.T
    x, y, bw, bh = cv2.boundingRect(pts2.astype(np.int32))

    pad = max(8, int(0.08 * max(bw, bh)))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Dua vat the vao canvas vuong, khong lam meo ti le
    ch, cw = crop.shape[:2]
    side = max(ch, cw) + 20
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    yy = (side - ch) // 2
    xx = (side - cw) // 2
    canvas[yy:yy+ch, xx:xx+cw] = crop
    return cv2.resize(canvas, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

# ------------------------------------------------------------
# HOG TU VIET - KHONG DUNG cv2.HOGDescriptor
# ------------------------------------------------------------

def hog_feature(img, cell=8, bins=9):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0

    h, w = gray.shape
    features = []
    bin_width = 180.0 / bins

    # histogram theo cell
    cells = []
    for y in range(0, h, cell):
        row = []
        for x in range(0, w, cell):
            hist = np.zeros(bins, dtype=np.float32)
            m = mag[y:y+cell, x:x+cell]
            a = ang[y:y+cell, x:x+cell]
            for mv, av in zip(m.ravel(), a.ravel()):
                pos = av / bin_width
                b0 = int(np.floor(pos)) % bins
                b1 = (b0 + 1) % bins
                frac = pos - np.floor(pos)
                hist[b0] += mv * (1.0 - frac)
                hist[b1] += mv * frac
            row.append(hist)
        cells.append(row)

    # block 2x2 + L2 normalization
    for y in range(len(cells) - 1):
        for x in range(len(cells[0]) - 1):
            block = np.concatenate([
                cells[y][x], cells[y][x+1],
                cells[y+1][x], cells[y+1][x+1]
            ])
            block = block / np.sqrt(np.sum(block * block) + 1e-6)
            features.extend(block)

    return np.asarray(features, dtype=np.float32)


def extract_feature(img):
    obj = normalize_object(img)
    if obj is None:
        return None
    return hog_feature(obj)

# ------------------------------------------------------------
# TANG CUONG GOC
# ------------------------------------------------------------

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def augment_images(img):
    result = [img]
    for angle in (-45, -30, -15, 15, 30, 45):
        result.append(rotate_image(img, angle))
    result.append(cv2.convertScaleAbs(img, alpha=0.85, beta=-10))
    result.append(cv2.convertScaleAbs(img, alpha=1.10, beta=10))
    return result

# ------------------------------------------------------------
# DATASET / SVM
# ------------------------------------------------------------

def image_files(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]


def train_model():
    X, y = [], []
    counts = {}

    for class_id, class_name in CLASSES.items():
        folder = os.path.join(DATA_DIR, class_id)
        # Cho phep ca ten but_chi neu nguoi dung da tao ten nay
        if class_id == "butchi" and not os.path.isdir(folder):
            alt = os.path.join(DATA_DIR, "but_chi")
            if os.path.isdir(alt):
                folder = alt

        files = image_files(folder)
        counts[class_name] = len(files)

        for path in files:
            img = cv2.imread(path)
            if img is None:
                continue
            for aug in augment_images(img):
                feat = extract_feature(aug)
                if feat is not None:
                    X.append(feat)
                    y.append(class_id)

    if len(X) < 8:
        raise ValueError(
            "Chua du du lieu. Hay cho anh vao 4 thu muc:\n"
            "data/but\n"
            "data/butchi\n"
            "data/tay\n"
            "data/thuoc"
        )

    if len(set(y)) < 2:
        raise ValueError("Can anh cua it nhat 2 loai de huan luyen.")

    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", probability=True,
            class_weight="balanced")
    )
    clf.fit(np.asarray(X), np.asarray(y))
    return clf, counts, len(X)


def classify(img):
    global model
    if model is None:
        model, _, _ = train_model()

    feat = extract_feature(img)
    if feat is None:
        return None, 0.0

    probs = model.predict_proba([feat])[0]
    i = int(np.argmax(probs))
    return model.classes_[i], float(probs[i] * 100)

# ------------------------------------------------------------
# VE KET QUA
# ------------------------------------------------------------

def draw_result(img, class_id, confidence):
    result = img.copy()
    contour = largest_contour(make_mask(img))
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
        text = f"{CLASSES[class_id]} - {confidence:.1f}%"
        cv2.putText(result, text, (x, max(35, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 120, 0), 3,
                    cv2.LINE_AA)
    return result

# ------------------------------------------------------------
# GIAO DIEN
# ------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("HE THONG NHAN DIEN DO DUNG HOC TAP - V2")
        self.root.geometry("1100x750")
        self.image = None
        self.photo = None
        self.counts = {v: 0 for v in CLASSES.values()}

        tk.Label(root, text="HE THONG NHAN DIEN VA PHAN LOAI DO DUNG HOC TAP",
                 font=("Arial", 20, "bold")).pack(pady=12)
        tk.Label(root, text="OpenCV + chuan hoa goc + HOG + SVM",
                 font=("Arial", 11)).pack()

        main = tk.Frame(root)
        main.pack(fill="both", expand=True, padx=15, pady=15)
        left = tk.Frame(main, bd=2, relief="groove")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = tk.Frame(main, width=300, bd=2, relief="groove")
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.image_label = tk.Label(left, text="CHUA CO ANH\n\nBam CHON ANH",
                                    font=("Arial", 16))
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(right, text="KET QUA", font=("Arial", 17, "bold")).pack(pady=15)
        self.result_label = tk.Label(right, text="Chua nhan dang",
                                     font=("Arial", 16, "bold"), wraplength=260)
        self.result_label.pack(pady=10)
        self.conf_label = tk.Label(right, text="Do tin cay: --", font=("Arial", 12))
        self.conf_label.pack(pady=5)
        tk.Label(right, text="THONG KE", font=("Arial", 14, "bold")).pack(pady=(30, 10))
        self.stats_label = tk.Label(right, text=self.stats_text(),
                                    font=("Arial", 12), justify="left")
        self.stats_label.pack()

        for text, cmd in [
            ("CHON ANH", self.choose_image),
            ("NHAN DANG", self.recognize),
            ("HUAN LUYEN LAI", self.retrain),
            ("RESET", self.reset),
            ("THOAT", root.destroy)
        ]:
            tk.Button(right, text=text, command=cmd, font=("Arial", 12, "bold"),
                      width=20, height=2).pack(pady=5)

        self.status = tk.Label(root, text="San sang", anchor="w", relief="sunken")
        self.status.pack(fill="x", side="bottom")

    def stats_text(self):
        return (f"But bi:   {self.counts['BÚT BI']}\n"
                f"But chi:  {self.counts['BÚT CHÌ']}\n"
                f"Tay:      {self.counts['TẨY']}\n"
                f"Thuoc:    {self.counts['THƯỚC']}")

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Chon anh",
            filetypes=[("Anh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tat ca", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Loi", "Khong doc duoc anh.")
            return
        self.image = img
        self.show_image(img)
        self.result_label.config(text="Chua nhan dang")
        self.conf_label.config(text="Do tin cay: --")
        self.status.config(text=f"Da chon: {os.path.basename(path)}")

    def show_image(self, img):
        h, w = img.shape[:2]
        scale = min(700 / w, 580 / h, 1.0)
        display = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))))
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(display))
        self.image_label.config(image=self.photo, text="")

    def recognize(self):
        if self.image is None:
            messagebox.showwarning("Thong bao", "Hay chon anh truoc.")
            return
        try:
            self.status.config(text="Dang nhan dang...")
            self.root.update()
            class_id, confidence = classify(self.image)
            if class_id is None:
                raise ValueError("Khong phat hien duoc vat the.")
            self.show_image(draw_result(self.image, class_id, confidence))
            name = CLASSES[class_id]
            if confidence < 55:
                self.result_label.config(text="KHONG CHAC CHAN")
            else:
                self.result_label.config(text=name)
                self.counts[name] += 1
            self.conf_label.config(text=f"Do tin cay: {confidence:.1f}%")
            self.stats_label.config(text=self.stats_text())
            self.status.config(text="Nhan dang hoan tat.")
        except Exception as e:
            messagebox.showerror("Loi nhan dang", str(e))
            self.status.config(text="Nhan dang that bai.")

    def retrain(self):
        global model
        try:
            self.status.config(text="Dang huan luyen...")
            self.root.update()
            model, counts, total = train_model()
            messagebox.showinfo(
                "Thanh cong",
                f"Da huan luyen lai!\n\nTong mau: {total}\n"
                f"But bi: {counts['BÚT BI']}\n"
                f"But chi: {counts['BÚT CHÌ']}\n"
                f"Tay: {counts['TẨY']}\n"
                f"Thuoc: {counts['THƯỚC']}"
            )
            self.status.config(text="Mo hinh da duoc huan luyen.")
        except Exception as e:
            messagebox.showerror("Loi huan luyen", str(e))
            self.status.config(text="Huan luyen that bai.")

    def reset(self):
        self.image = None
        self.photo = None
        self.image_label.config(image="", text="CHUA CO ANH\n\nBam CHON ANH")
        self.result_label.config(text="Chua nhan dang")
        self.conf_label.config(text="Do tin cay: --")
        self.counts = {v: 0 for v in CLASSES.values()}
        self.stats_label.config(text=self.stats_text())
        self.status.config(text="Da reset.")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
