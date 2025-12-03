import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import os
import re

# ---------------------------------------
# CONFIG
# ---------------------------------------
ASSET_DIR = "fruit-ninja-assets"
CAM_SRC = "http://172.20.3.208:8080/video"
FRAME_RATE = 30
SPAWN_INTERVAL = 0.9
COMBO_WINDOW = 1.4


# ---------------------------------------
# MEDIAPIPE HAND TRACKING (Stable settings)
# ---------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.50,
    min_tracking_confidence=0.30,
    model_complexity=1
)

# ------------- PNG DRAWING HELPERS ---------------
def safe_imread(path):
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def draw_png(frame, png, cx, cy, angle=0):
    if png is None:
        return

    h, w = png.shape[:2]

    # rotate if needed
    if angle != 0:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        png = cv2.warpAffine(
            png, M, (w, h),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_TRANSPARENT
        )

    # position
    x1, y1 = int(cx - w//2), int(cy - h//2)
    x2, y2 = x1 + w, y1 + h

    # bounds check
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    if fx1 >= fx2 or fy1 >= fy2:
        return

    px1, py1 = fx1 - x1, fy1 - y1
    px2, py2 = px1 + (fx2 - fx1), py1 + (fy2 - fy1)

    # alpha blend
    if png.shape[2] == 4:
        alpha = png[py1:py2, px1:px2, 3] / 255.0
        for c in range(3):
            fg = png[py1:py2, px1:px2, c].astype(float)
            bg = frame[fy1:fy2, fx1:fx2, c].astype(float)
            frame[fy1:fy2, fx1:fx2, c] = (alpha * fg + (1 - alpha) * bg).astype(np.uint8)
    else:
        frame[fy1:fy2, fx1:fx2] = png[py1:py2, px1:px2]


def load_resized(path, size):
    img = safe_imread(path)
    if img is None:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------
# ASSET LOADING (Auto detects fruit sets)
# ---------------------------------------
asset_files = os.listdir(ASSET_DIR)
asset_map = {}

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

fruit_types = []
for name, paths in asset_map.items():
    if paths["whole"] and paths["half1"] and paths["half2"]:
        fruit_types.append(paths)

# splash + bomb + explosion + background
def find_any(names):
    for n in names:
        p = os.path.join(ASSET_DIR, n)
        if os.path.exists(p):
            return p
    return None

splash_png = find_any([
    "splash_transparent.png",
    "splash_yellow.png",
    "splash_red.png",
    "splash_orange.png",
])
explosion_png = find_any(["explosion.png"])
bomb_png = find_any(["bomb.png"])
background_png = find_any(["background.png"])

print("Loaded fruit types:", len(fruit_types))

if not fruit_types:
    raise SystemExit("❌ No valid fruit sets found (fruit.png + fruit_half_1.png + fruit_half_2.png required)")


# ---------------------------------------
# GAME OBJECT CLASSES
# ---------------------------------------
class Fruit:
    def __init__(self):
        self.kind = random.choice(fruit_types)
        self.size = random.randint(80, 130)

        self.whole = load_resized(self.kind["whole"], self.size)
        self.half1 = load_resized(self.kind["half1"], self.size)
        self.half2 = load_resized(self.kind["half2"], self.size)

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
        if self.y > H + 80:
            self.alive = False

    def draw(self, frame):
        draw_png(frame, self.whole, int(self.x), int(self.y))


class Bomb(Fruit):
    def __init__(self):
        self.size = random.randint(70, 110)
        self.whole = load_resized(bomb_png, self.size)
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
    def __init__(self, png, x, y, vx, vy, ang=0, rot=0):
        self.png = png
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = ang
        self.rot_vel = rot
        self.life = 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 1.4
        self.angle += self.rot_vel
        self.life += 1

    def draw(self, frame):
        draw_png(frame, self.png, int(self.x), int(self.y), self.angle)

    def dead(self):
        return self.y > H + 150 or self.life > 200


class Splash:
    def __init__(self, png, x, y):
        self.png = load_resized(png, random.randint(40, 90))
        self.x = x
        self.y = y
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-4, -1)
        self.life = 0
        self.alpha = 1.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.5
        self.life += 1
        self.alpha -= 0.03

    def draw(self, frame):
        if self.png is None or self.alpha <= 0:
            return
        tmp = self.png.copy().astype(float)
        if tmp.shape[2] == 4:
            tmp[:, :, 3] *= self.alpha
        draw_png(frame, tmp.astype(np.uint8), int(self.x), int(self.y))

    def dead(self):
        return self.alpha <= 0 or self.y > H + 100


# ---------------------------------------
# CUT LOGIC
# ---------------------------------------
def spawn_fruit():
    if bomb_png and random.random() < 0.12:
        fruits.append(Bomb())
    else:
        fruits.append(Fruit())


def cut_target(target, dir_vec):
    global score, last_cut_time, combo, combo_time, flash_frames

    if target.is_bomb:
        # bomb explosion
        flash_frames = 12
        score = max(0, score - 5)

        # explosion visual
        if explosion_png:
            expl = load_resized(explosion_png, int(target.size * 1.3))
            pieces.append(Piece(expl, target.x, target.y, 0, -2))
        return

    # fruit sliced
    dx, dy = dir_vec
    speed = 10
    perp = (-dy, dx)
    L = math.hypot(perp[0], perp[1]) or 1
    perp = (perp[0]/L, perp[1]/L)

    vx1 = target.vx + perp[0] * speed
    vy1 = target.vy + perp[1] * speed - 2
    vx2 = target.vx - perp[0] * speed
    vy2 = target.vy - perp[1] * speed - 2

    pieces.append(Piece(target.half1, target.x, target.y, vx1, vy1, ang=0, rot=5))
    pieces.append(Piece(target.half2, target.x, target.y, vx2, vy2, ang=0, rot=-5))

    # splash
    if splash_png:
        for _ in range(random.randint(2, 5)):
            splashes.append(Splash(splash_png, target.x, target.y))

    # combo scoring
    now = time.time()
    if now - last_cut_time <= COMBO_WINDOW:
        combo += 1
    else:
        combo = 1

    last_cut_time = now
    combo_time = now
    score += combo


# ---------------------------------------
# MAIN LOOP SETUP
# ---------------------------------------
cap = cv2.VideoCapture(CAM_SRC)
ret, tmp = cap.read()

if not ret:
    print("Camera failed. Trying webcam index 0...")
    cap = cv2.VideoCapture(0)
    ret, tmp = cap.read()

if not ret:
    raise SystemExit("❌ No camera available.")

H, W = tmp.shape[:2]

# background load
bg = safe_imread(background_png)
if bg is not None:
    bg = cv2.resize(bg, (W, H))
    if bg.shape[2] == 4:  # convert RGBA → BGR
        bg = cv2.cvtColor(bg, cv2.COLOR_BGRA2BGR)

fruits = []
pieces = []
splashes = []
score = 0

prev_point = None
last_spawn = time.time()
last_cut_time = 0
combo = 0
combo_time = 0
flash_frames = 0

spawn_fruit()

# ---------------------------------------
# GAME LOOP
# ---------------------------------------
while True:
    t0 = time.time()
    ret, cam = cap.read()
    if not ret:
        break

    cam = cv2.flip(cam, 1)

    # background frame
    frame = bg.copy() if bg is not None else np.ones((H, W, 3), np.uint8) * 255

    # hand track
    rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fingertip = None
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark[8]
        fx, fy = int(lm.x * W), int(lm.y * H)
        fingertip = (fx, fy)
        cv2.circle(frame, fingertip, 12, (0, 0, 255), -1)

    # spawn
    if time.time() - last_spawn > SPAWN_INTERVAL:
        spawn_fruit()
        last_spawn = time.time()

    # update fruit
    for f in fruits:
        f.update()
        f.draw(frame)

    # update pieces
    for p in pieces:
        p.update()
        p.draw(frame)

    # update splash
    for s in splashes:
        s.update()
        s.draw(frame)

    # cutting
    if fingertip:
        for f in list(fruits):
            if f.alive and math.dist(fingertip, (f.x, f.y)) < f.radius:
                if prev_point:
                    dx = fingertip[0] - prev_point[0]
                    dy = fingertip[1] - prev_point[1]
                    L = math.hypot(dx, dy) or 1
                    dir_vec = (dx / L, dy / L)
                else:
                    dir_vec = (1, 0)

                cut_target(f, dir_vec)
                f.alive = False

                cv2.circle(frame, (int(f.x), int(f.y)), f.radius + 5, (0, 150, 255), 3)

    prev_point = fingertip

    # cleanup
    fruits = [f for f in fruits if f.alive]
    pieces = [p for p in pieces if not p.dead()]
    splashes = [s for s in splashes if not s.dead()]

    # HUD
    cv2.putText(frame, f"Score: {score}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

    if combo > 1 and time.time() - combo_time < 1.5:
        cv2.putText(frame, f"COMBO x{combo}", (W//2 - 120, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 140, 255), 4)

    # flash red on bomb
    if flash_frames > 0:
        overlay = frame.copy()
        overlay[:] = (0, 0, 255)
        cv2.addWeighted(overlay, 0.25, frame, 1 - 0.25, 0, frame)
        flash_frames -= 1

    cv2.imshow("Fruit Ninja - Enhanced", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # stabilize FPS
    dt = time.time() - t0
    wait = max(0, 1.0 / FRAME_RATE - dt)
    time.sleep(wait)

cap.release()
cv2.destroyAllWindows()

