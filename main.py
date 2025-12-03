"""
Fruit Ninja clone
- Uses your PNG assets in folder (ASSET_DIR)
- Requirements: opencv-python, mediapipe, numpy
- No sound (per request)
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import os
import re

# ---------------- CONFIG ----------------
ASSET_DIR = "fruit-ninja-assets"  # change if needed
CAM_SRC = "http://10.9.20.154:8080/video"  # camera input URL
FRAME_RATE = 30
SPAWN_INTERVAL = 0.9  # seconds between spawns
COMBO_WINDOW = 1.4  # seconds to chain slices for combo
# ----------------------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ----------------- HELPERS -----------------
def safe_imread(path):
    """Read PNG (with alpha) or return None."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

def draw_png(frame, png, cx, cy, angle=0):
    """Draw PNG centered at (cx,cy). png is BGR(A). Handles rotation if angle != 0."""
    if png is None:
        return
    h, w = png.shape[:2]
    # rotated image
    if angle != 0:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        png = cv2.warpAffine(png, M, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
    x1, y1 = int(cx - w//2), int(cy - h//2)
    x2, y2 = x1 + w, y1 + h

    # cropping if outside
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if fx1 >= fx2 or fy1 >= fy2:
        return

    px1, py1 = fx1 - x1, fy1 - y1
    px2, py2 = px1 + (fx2 - fx1), py1 + (fy2 - fy1)

    alpha = None
    if png.shape[2] == 4:
        alpha = png[py1:py2, px1:px2, 3] / 255.0
        for c in range(3):
            fg = png[py1:py2, px1:px2, c].astype(float)
            bg = frame[fy1:fy2, fx1:fx2, c].astype(float)
            frame[fy1:fy2, fx1:fx2, c] = (alpha * fg + (1 - alpha) * bg).astype(np.uint8)
    else:
        frame[fy1:fy2, fx1:fx2] = png[py1:py2, px1:px2]


def load_and_resize(path, size):
    img = safe_imread(path)
    if img is None:
        return None
    # preserve alpha if present
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

# ----------------- ASSET LOADER -----------------
# Looks for:
#   <name>.png
#   <name>_half_1.png
#   <name>_half_2.png
# Ignores files with "small" in filename (will fallback later to small if needed).

asset_files = os.listdir(ASSET_DIR) if os.path.isdir(ASSET_DIR) else []
asset_map = {}  # name -> {"whole": path, "half1": path, "half2": path}

for fn in asset_files:
    if not fn.lower().endswith(".png"):
        continue
    if "small" in fn.lower():
        continue
    m = re.match(r"^([a-zA-Z0-9]+)(?:_half_([12]))?\.png$", fn)
    if not m:
        continue
    name = m.group(1).lower()
    half = m.group(2)
    if name not in asset_map:
        asset_map[name] = {"whole": None, "half1": None, "half2": None}
    full = os.path.join(ASSET_DIR, fn)
    if half == "1":
        asset_map[name]["half1"] = full
    elif half == "2":
        asset_map[name]["half2"] = full
    else:
        asset_map[name]["whole"] = full

# If nothing found because only small variants exist, try fallback to small files
if not any(v["whole"] and v["half1"] and v["half2"] for v in asset_map.values()):
    for fn in asset_files:
        if not fn.lower().endswith(".png"):
            continue
        m = re.match(r"^([a-zA-Z0-9]+)(?:_half_([12]))?_small\.png$", fn)
        if not m:
            continue
        name = m.group(1).lower()
        half = m.group(2)
        if name not in asset_map:
            asset_map[name] = {"whole": None, "half1": None, "half2": None}
        full = os.path.join(ASSET_DIR, fn)
        if half == "1":
            asset_map[name]["half1"] = full
        elif half == "2":
            asset_map[name]["half2"] = full
        else:
            asset_map[name]["whole"] = full

# Build list of usable fruit types (have whole + halves)
fruit_types = []
for name, paths in asset_map.items():
    if paths["whole"] and paths["half1"] and paths["half2"]:
        fruit_types.append({"name": name, **paths})

# locate splash and bomb/explosion assets
def find_first_candidate(cands):
    for c in cands:
        p = os.path.join(ASSET_DIR, c)
        if os.path.exists(p):
            return p
    return None

splash_png = find_first_candidate([
    "splash_transparent.png", "splash_red.png", "splash_yellow.png",
    "splash_orange.png", "splash_transparent_small.png", "splash_red_small.png"
])
explosion_png = find_first_candidate(["explosion.png", "explosion_small.png"])
bomb_png = find_first_candidate(["bomb.png", "bomb_small.png"])
background_png = find_first_candidate(["background.png", "background_small.png"])

print("Detected fruit types:", [f["name"] for f in fruit_types])
print("splash:", splash_png, "explosion:", explosion_png, "bomb:", bomb_png, "background:", background_png)

if not fruit_types:
    raise SystemExit("No fruit types found in ASSET_DIR. Put <fruit>.png + <fruit>_half_1.png + <fruit>_half_2.png in the folder.")

# ----------------- GAME OBJECTS -----------------
class Fruit:
    def __init__(self, kind=None):
        if kind is None:
            self.kind = random.choice(fruit_types)
        else:
            self.kind = kind
        self.size = random.randint(80, 130)
        self.whole = load_and_resize(self.kind["whole"], self.size)
        self.half1 = load_and_resize(self.kind["half1"], self.size)
        self.half2 = load_and_resize(self.kind["half2"], self.size)
        # spawn near bottom with random x
        self.x = random.randint(120, 520)
        self.y = 520
        self.vx = random.uniform(-6, 6)
        self.vy = random.uniform(-20, -26)
        self.radius = self.size // 2
        self.alive = True
        self.is_bomb = False

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 1.0
        if self.y - self.radius > H + 50:
            self.alive = False

    def draw(self, frame):
        draw_png(frame, self.whole, int(self.x), int(self.y))

class Bomb(Fruit):
    def __init__(self):
        # Use bomb PNGs, size slightly smaller or similar
        self.size = random.randint(70, 110)
        self.whole = load_and_resize(bomb_png, self.size)
        # halves not meaningful for bombs; we'll use explosion instead
        self.half1 = None
        self.half2 = None
        self.x = random.randint(120, 520)
        self.y = 520
        self.vx = random.uniform(-6, 6)
        self.vy = random.uniform(-20, -26)
        self.radius = self.size // 2
        self.alive = True
        self.is_bomb = True

    def draw(self, frame):
        draw_png(frame, self.whole, int(self.x), int(self.y))

class Piece:
    def __init__(self, png, x, y, vx, vy, ang=0, rot_speed=0):
        self.png = png
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = ang
        self.rot_speed = rot_speed
        self.life = 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 1.6  # gravity for pieces
        self.angle += self.rot_speed
        self.life += 1

    def draw(self, frame):
        draw_png(frame, self.png, int(self.x), int(self.y), angle=self.angle)

    def dead(self):
        return self.y > H + 200 or self.life > 200

class Splash:
    def __init__(self, png, x, y):
        self.png = load_and_resize(png, random.randint(40, 90)) if png else None
        self.x = x
        self.y = y
        self.life = 0
        self.vx = random.uniform(-4, 4)
        self.vy = random.uniform(-6, -2)
        self.alpha = 1.0
        self.fade_speed = 0.04 + random.random() * 0.02

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.6
        self.life += 1
        self.alpha -= self.fade_speed
        if self.alpha < 0:
            self.alpha = 0

    def draw(self, frame):
        if self.png is None:
            return
        # draw with current alpha by blending image into a copy
        hpng, wpng = self.png.shape[:2]
        temp = self.png.copy().astype(np.float32)
        if temp.shape[2] == 4:
            temp[:, :, 3] = (temp[:, :, 3].astype(float) * self.alpha).astype(temp.dtype)
        draw_png(frame, temp.astype(np.uint8), int(self.x), int(self.y))

    def dead(self):
        return self.alpha <= 0 or self.y > H + 200 or self.life > 150

# ----------------- GAME STATE -----------------
cap = cv2.VideoCapture(CAM_SRC)
fruits = []
pieces = []
splashes = []
score = 0
last_spawn = time.time()
prev_point = None
last_cut_time = 0.0
combo_count = 0
combo_display_time = 0.0
game_over_flash = 0  # frames to flash red when bomb exploded

# load background image (fit to camera frame size later)
bg_img = safe_imread(background_png) if background_png else None

# ----------------- UTILS -----------------
def spawn_fruit_or_bomb():
    # small chance to spawn bomb
    if bomb_png and random.random() < 0.12:
        fruits.append(Bomb())
    else:
        fruits.append(Fruit())

def cut_fruit_obj(obj, dir_vec):
    global score, last_cut_time, combo_count, combo_display_time, game_over_flash
    if obj.is_bomb:
        # bomb exploded
        score = max(0, score - 5)
        game_over_flash = 12  # flash screen
        # spawn explosion pieces
        if explosion_png:
            expl = load_and_resize(explosion_png, max(80, obj.radius*2))
            pieces.append(Piece(expl, obj.x, obj.y, 0, -2, ang=0, rot_speed=0))
        # splat shock particles
        for _ in range(8):
            px = obj.x + random.uniform(-obj.radius/2, obj.radius/2)
            py = obj.y + random.uniform(-obj.radius/2, obj.radius/2)
            vxx = random.uniform(-8, 8)
            vyy = random.uniform(-6, -1)
            small = load_and_resize(explosion_png, random.randint(30, 60)) if explosion_png else None
            pieces.append(Piece(small, px, py, vxx, vyy, ang=random.uniform(0, 360), rot_speed=random.uniform(-10, 10)))
        # reset combo
        combo_count = 0
        combo_display_time = time.time()
        last_cut_time = 0
    else:
        # spawn the two halves and splashes
        dx, dy = dir_vec
        speed = 10 + random.random() * 3
        perp = (-dy, dx)
        L = math.hypot(perp[0], perp[1]) or 1
        perp = (perp[0]/L, perp[1]/L)
        vx1 = obj.vx + perp[0] * speed + random.uniform(-1.5, 1.5)
        vy1 = obj.vy + perp[1] * speed + random.uniform(-2, 2) - 2
        vx2 = obj.vx - perp[0] * speed + random.uniform(-1.5, 1.5)
        vy2 = obj.vy - perp[1] * speed + random.uniform(-2, 2) - 2

        rot1 = random.uniform(-18, 18)
        rot2 = random.uniform(-18, 18)

        pieces.append(Piece(obj.half1, obj.x, obj.y, vx1, vy1, ang=random.uniform(0, 360), rot_speed=rot1/2))
        pieces.append(Piece(obj.half2, obj.x, obj.y, vx2, vy2, ang=random.uniform(0, 360), rot_speed=rot2/2))

        # spawn splashes
        if splash_png:
            for _ in range(3 + random.randint(0, 3)):
                spl = Splash(splash_png, obj.x + random.uniform(-10, 10), obj.y + random.uniform(-10, 10))
                splashes.append(spl)

        # scoring + combo
        now = time.time()
        if now - last_cut_time <= COMBO_WINDOW:
            combo_count += 1
        else:
            combo_count = 1
        last_cut_time = now
        combo_display_time = now
        # score ramp: base 1 point, extra for combo
        gained = 1 + (combo_count - 1)
        score += gained

# ----------------- MAIN LOOP -----------------
ret, tmp = cap.read()
if not ret:
    # attempt to open default camera index 0
    cap = cv2.VideoCapture(0)
    ret, tmp = cap.read()
    if not ret:
        raise SystemExit("Cannot open camera (network or local).")

H, W = tmp.shape[:2]

# If background exists, resize once to frame size
if bg_img is not None:
    bg_img = cv2.resize(bg_img, (W, H), interpolation=cv2.INTER_AREA)

spawn_fruit_or_bomb()  # initial

while True:
    t0 = time.time()
    ret, cam_frame = cap.read()
    if not ret:
        break

    cam_frame = cv2.flip(cam_frame, 1)
    # create visible frame from background (not camera)
    if bg_img is not None:
        frame = bg_img.copy()
    else:
        frame = np.ones((H, W, 3), dtype=np.uint8) * 255

    # hand detection on camera frame but only draw fingertip dot on our white/bg frame
    rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingertip = None
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark[8]  # index fingertip
        fx, fy = int(lm.x * W), int(lm.y * H)
        fingertip = (fx, fy)
        # draw fingertip as small circle
        cv2.circle(frame, fingertip, 11, (0, 0, 255), -1)

    # spawn logic
    if time.time() - last_spawn > SPAWN_INTERVAL:
        spawn_fruit_or_bomb()
        last_spawn = time.time()

    # update / draw fruits
    for f in fruits:
        f.update()
        f.draw(frame)

    # update / draw pieces
    for p in pieces:
        p.update()
        p.draw(frame)

    # update / draw splashes
    for s in splashes:
        s.update()
        s.draw(frame)

    # cut detection (touch-based)
    if fingertip:
        for f in list(fruits):  # copy because we may remove
            if f.alive:
                d = math.dist(fingertip, (f.x, f.y))
                if d < f.radius:
                    # compute direction using previous fingertip (fallback to right)
                    if prev_point is not None:
                        dx = fingertip[0] - prev_point[0]
                        dy = fingertip[1] - prev_point[1]
                        L = math.hypot(dx, dy) or 1.0
                        dir_vec = (dx / L, dy / L)
                    else:
                        dir_vec = (1.0, 0.0)
                    cut_fruit_obj(f, dir_vec)
                    f.alive = False
                    # tiny highlight circle
                    cv2.circle(frame, (int(f.x), int(f.y)), int(f.radius + 6), (0, 180, 255), 3)

    prev_point = fingertip

    # cleanup
    fruits = [f for f in fruits if f.alive]
    pieces = [p for p in pieces if not p.dead()]
    splashes = [s for s in splashes if not s.dead()]

    # HUD: score
    cv2.putText(frame, f"Score: {score}", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10, 10, 10), 4)

    # HUD: combo
    if combo_count > 1 and (time.time() - combo_display_time) < 1.6:
        txt = f"COMBO x{combo_count}"
        cv2.putText(frame, txt, (W//2 - 140, 60), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 120, 255), 4)

    # flash screen red on bomb
    if game_over_flash > 0:
        overlay = frame.copy()
        overlay[:] = (0, 0, 255)
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        game_over_flash -= 1

    # show window
    cv2.imshow("Fruit Ninja - Enhanced (no sound)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # maintain approx frame rate
    dt = time.time() - t0
    wait = max(0, 1.0 / FRAME_RATE - dt)
    time.sleep(wait)

cap.release()
cv2.destroyAllWindows()

