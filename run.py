#!/usr/bin/env python3
"""
Fruit Ninja — merged version
- Uses camera feed as live background
- Loads PNG assets (whole + halves + splash + bomb + explosion) from ASSET_DIR
- Threaded camera capture, resized hand detection for performance
- Cut-on-touch and swipe trail, combo scoring, bomb handling
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import os
import re
import threading
from collections import deque

# -------------------------
# CONFIG
# -------------------------
ASSET_DIR = "fruit-ninja-assets"
CAM_SRC = "http://10.140.217.200:8080/video"   # change to your feed/0
FRAME_RATE = 30
SPAWN_INTERVAL = 0.9
COMBO_WINDOW = 1.4

# Performance tuning
PROCESS_EVERY = 1         # process every Nth frame for hand detection (1 = every frame)
DETECT_RESIZE = 0.5       # factor to resize frame for detection (0.5 = half-size)
MAX_FRUITS = 8

# Mediapipe settings (lower complexity for speed)
MP_MAX_HANDS = 1
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.3
MP_MODEL_COMPLEXITY = 0   # 0 or 1 (0 is faster)

# -------------------------
# UTIL: threaded camera reader
# -------------------------
class VideoGet:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera source: {src}")
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.stopped = True
        self.cap.release()

# -------------------------
# PNG helpers (from your code)
# -------------------------
def safe_imread(path):
    if not path or not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def draw_png(frame, png, cx, cy, angle=0):
    if png is None:
        return
    h, w = png.shape[:2]
    # rotate png around center if needed
    if angle != 0:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        png = cv2.warpAffine(png, M, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

    x1, y1 = int(cx - w//2), int(cy - h//2)
    x2, y2 = x1 + w, y1 + h

    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if fx1 >= fx2 or fy1 >= fy2:
        return
    px1, py1 = fx1 - x1, fy1 - y1
    px2, py2 = px1 + (fx2 - fx1), py1 + (fy2 - fy1)

    # alpha blend if 4 channels
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

# -------------------------
# Load assets (auto-detect fruit sets)
# -------------------------
asset_files = []
if os.path.isdir(ASSET_DIR):
    asset_files = os.listdir(ASSET_DIR)
else:
    print("Warning: ASSET_DIR does not exist. PNG assets will be skipped.")

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
    # prefer fruits that have whole image; halves optional (we fallback to ellipse halves)
    if paths["whole"]:
        fruit_types.append(paths)

splash_png = next((os.path.join(ASSET_DIR, n) for n in [
    "splash_transparent.png", "splash_yellow.png", "splash_red.png", "splash_orange.png"
] if os.path.exists(os.path.join(ASSET_DIR, n))), None)
explosion_png = next((os.path.join(ASSET_DIR, n) for n in ["explosion.png"] if os.path.exists(os.path.join(ASSET_DIR, n))), None)
bomb_png = next((os.path.join(ASSET_DIR, n) for n in ["bomb.png"] if os.path.exists(os.path.join(ASSET_DIR, n))), None)

print(f"Loaded fruit types: {len(fruit_types)}  splash:{bool(splash_png)} bomb:{bool(bomb_png)}")

# -------------------------
# Game classes (integrated)
# -------------------------
class FruitObj:
    """Fruit represented by PNG assets if available, else simple circle fallback."""
    def __init__(self):
        self.kind = None
        if fruit_types:
            self.kind = random.choice(fruit_types)
        self.size = random.randint(80, 120)
        self.whole = load_resized(self.kind["whole"], self.size) if self.kind and self.kind.get("whole") else None
        self.half1 = load_resized(self.kind.get("half1"), self.size) if self.kind and self.kind.get("half1") else None
        self.half2 = load_resized(self.kind.get("half2"), self.size) if self.kind and self.kind.get("half2") else None

        self.x = random.randint(120, 520)
        self.y = 520
        self.vx = random.uniform(-6, 6)
        self.vy = random.uniform(-20, -26)
        self.radius = self.size // 2 if self.size else random.randint(25, 35)
        self.alive = True
        self.is_bomb = False

        # fallback color
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 1.0
        if self.y > H + 80:
            self.alive = False

    def draw(self, frame):
        if self.whole is not None:
            draw_png(frame, self.whole, int(self.x), int(self.y))
        else:
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)

class BombObj(FruitObj):
    def __init__(self):
        super().__init__()
        self.size = random.randint(70, 110)
        self.whole = load_resized(bomb_png, self.size) if bomb_png else None
        self.half1 = None
        self.half2 = None
        self.radius = self.size // 2
        self.is_bomb = True

    def draw(self, frame):
        if self.whole is not None:
            draw_png(frame, self.whole, int(self.x), int(self.y))
        else:
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (0,0,255), -1)

class PiecePNG:
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

class PieceEllipse:
    def __init__(self, x, y, vx, vy, radius, color, start_angle, end_angle, angle):
        self.x = x; self.y = y; self.vx = vx; self.vy = vy
        self.radius = radius; self.color = color
        self.start_angle = start_angle; self.end_angle = end_angle; self.angle = angle
        self.rotate_speed = random.uniform(-10, 10)
        self.life = 0; self.max_life = 120
    def update(self):
        self.x += self.vx; self.y += self.vy
        self.vy += 0.8
        self.angle += self.rotate_speed
        self.life += 1
    def draw(self, frame):
        center = (int(self.x), int(self.y))
        axes = (int(self.radius), int(self.radius))
        cv2.ellipse(frame, center, axes, int(self.angle),
                    int(self.start_angle), int(self.end_angle), self.color, -1)
        cv2.ellipse(frame, center, axes, int(self.angle),
                    int(self.start_angle), int(self.end_angle), (20,20,20), 1)
    def dead(self, h=720):
        return self.life > self.max_life or self.y - self.radius > h

class Splash:
    def __init__(self, png, x, y):
        self.png = load_resized(png, random.randint(40,90)) if png else None
        self.x = x; self.y = y
        self.vx = random.uniform(-3,3); self.vy = random.uniform(-4,-1)
        self.life = 0; self.alpha = 1.0
    def update(self):
        self.x += self.vx; self.y += self.vy
        self.vy += 0.5; self.life += 1; self.alpha -= 0.03
    def draw(self, frame):
        if self.png is None or self.alpha <= 0:
            return
        tmp = self.png.copy().astype(float)
        if tmp.shape[2] == 4:
            tmp[:,:,3] = (tmp[:,:,3] * self.alpha).astype(tmp.dtype)
        draw_png(frame, tmp.astype(np.uint8), int(self.x), int(self.y))
    def dead(self):
        return self.alpha <= 0 or self.y > H + 100

# -------------------------
# Cut logic (spawn pieces + splash)
# -------------------------
def spawn_fruit():
    if bomb_png and random.random() < 0.12:
        fruits.append(BombObj())
    else:
        if len(fruits) < MAX_FRUITS:
            fruits.append(FruitObj())

def cut_target(target, dir_vec):
    global score, last_cut_time, combo, combo_time, flash_frames, pieces, splashes
    if target.is_bomb:
        # bomb explosion
        flash_frames = 12
        score = max(0, score - 5)
        if explosion_png:
            expl = load_resized(explosion_png, int(target.size * 1.3))
            pieces.append(PiecePNG(expl, target.x, target.y, 0, -2))
        return
    # sliced
    dx, dy = dir_vec
    speed = 10
    perp = (-dy, dx)
    L = math.hypot(perp[0], perp[1]) or 1
    perp = (perp[0]/L, perp[1]/L)
    vx1 = target.vx + perp[0] * speed
    vy1 = target.vy + perp[1] * speed - 2
    vx2 = target.vx - perp[0] * speed
    vy2 = target.vy - perp[1] * speed - 2

    # if halves available use PNG halves, else ellipse pieces
    if getattr(target, "half1", None) is not None and getattr(target, "half2", None) is not None:
        pieces.append(PiecePNG(target.half1, target.x, target.y, vx1, vy1, ang=0, rot=5))
        pieces.append(PiecePNG(target.half2, target.x, target.y, vx2, vy2, ang=0, rot=-5))
    else:
        # fallback ellipse halves
        r = getattr(target, "radius", 30)
        color = getattr(target, "color", (0,200,200))
        angle_deg = 0
        pieces.append(PieceEllipse(target.x, target.y, vx1, vy1, r, color, 0, 180, angle_deg))
        pieces.append(PieceEllipse(target.x, target.y, vx2, vy2, r, color, 180, 360, angle_deg))

    # splash
    if splash_png:
        for _ in range(random.randint(2,5)):
            splashes.append(Splash(splash_png, target.x, target.y))

    # combo scoring
    now = time.time()
    global combo
    if now - last_cut_time <= COMBO_WINDOW:
        combo += 1
    else:
        combo = 1
    last_cut_time = now
    combo_time = now
    score += combo

# -------------------------
# Swipe tracker (simple)
# -------------------------
class SwipeTracker:
    def __init__(self, max_len=12):
        self.points = deque(maxlen=max_len)
    def add(self, p):
        if p is not None:
            self.points.append(tuple(p))
    def clear(self):
        self.points.clear()
    def draw_trail(self, frame):
        pts = list(self.points)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (255,255,255), 3)
    def is_swipe(self):
        pts = list(self.points)
        if len(pts) < 3:
            return False
        p1 = np.array(pts[0]); p2 = np.array(pts[-1])
        return np.linalg.norm(p2 - p1) > 120

# -------------------------
# Setup: camera, mediapipe, game state
# -------------------------
vg = VideoGet(CAM_SRC)
# read one frame to get size
ret, tmp = vg.read()
if not ret or tmp is None:
    print("Camera failed. Trying webcam index 0...")
    vg.release()
    vg = VideoGet(0)
    ret, tmp = vg.read()
if not ret or tmp is None:
    raise SystemExit("❌ No camera available.")

H, W = tmp.shape[:2]
print("Frame size:", W, H)

# mediapipe hands (keep complexity low for speed)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=MP_MAX_HANDS,
    min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    model_complexity=MP_MODEL_COMPLEXITY
)

# game state
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
swipe = SwipeTracker()

spawn_fruit()

# -------------------------
# Main loop
# -------------------------
frame_count = 0
try:
    while True:
        t0 = time.time()
        ret, cam = vg.read()
        if not ret or cam is None:
            break
        frame_count += 1

        # flip mirror
        cam = cv2.flip(cam, 1)

        # use camera feed as dynamic background (we'll draw fruits/pieces on top)
        frame = cam.copy()

        # small resized frame for mediapipe to speed up detection
        do_process = (frame_count % PROCESS_EVERY) == 0
        fingertip = None
        if do_process:
            small = cv2.resize(frame, (0,0), fx=DETECT_RESIZE, fy=DETECT_RESIZE,
                               interpolation=cv2.INTER_LINEAR)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_small)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark[8]  # index fingertip
                # scale landmark coordinates back to full frame size
                fx = int(lm.x * small.shape[1] / DETECT_RESIZE)
                fy = int(lm.y * small.shape[0] / DETECT_RESIZE)
                fingertip = (fx, fy)
                swipe.add(fingertip)
                # draw small landmarks on full frame (use scaled normalized coords)
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0,200,0), thickness=1))
            else:
                swipe.clear()
        else:
            # on skipped frames, keep last fingertip for trail smoothing
            if swipe.points:
                fingertip = tuple(swipe.points[-1])

        # spawn
        if time.time() - last_spawn > SPAWN_INTERVAL:
            spawn_fruit()
            last_spawn = time.time()

        # update fruits
        for f in fruits:
            f.update()
            f.draw(frame)

        # update pieces
        for p in list(pieces):
            p.update()
            p.draw(frame)

        # update splashes
        for s in list(splashes):
            s.update()
            s.draw(frame)

        # cutting: cut on touch (closest fingertip distance)
        if fingertip:
            for f in list(fruits):
                if f.alive and math.dist(fingertip, (f.x, f.y)) < f.radius + 12:
                    # compute cut dir from previous swipe point
                    prev_pt = swipe.points[-2] if len(swipe.points) >= 2 else None
                    if prev_pt:
                        dx = fingertip[0] - prev_pt[0]; dy = fingertip[1] - prev_pt[1]
                        L = math.hypot(dx, dy) or 1.0
                        dir_vec = (dx / L, dy / L)
                    else:
                        dir_vec = (1.0, 0.0)
                    cut_target(f, dir_vec)
                    f.alive = False
                    # highlight splash
                    cv2.circle(frame, (int(f.x), int(f.y)), f.radius + 6, (0, 150, 255), -1)
                    swipe.clear()
                    break

        # optional: swipe detection cuts (if you want higher-speed gestures)
        # if swipe.is_swipe() and fingertip:
        #     for f in fruits:
        #         if f.alive and math.dist(fingertip, (f.x, f.y)) < f.radius + 15:
        #             cut_target(f, (1, 0))
        #             f.alive = False
        #     swipe.clear()

        # cleanup
        fruits = [f for f in fruits if f.alive]
        pieces = [p for p in pieces if not getattr(p, "dead", lambda *a: False)()]
        splashes = [s for s in splashes if not s.dead()]

        # draw swipe trail last (so it's on top)
        swipe.draw_trail(frame)

        # HUD
        cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
        if combo > 1 and time.time() - combo_time < 1.5:
            cv2.putText(frame, f"COMBO x{combo}", (W//2 - 120, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,140,255), 4)

        # flash on bomb
        if flash_frames > 0:
            overlay = frame.copy()
            overlay[:] = (0,0,255)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            flash_frames -= 1

        cv2.imshow("Fruit Ninja - Merged (camera bg)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # maintain framerate
        dt = time.time() - t0
        wait = max(0, 1.0 / FRAME_RATE - dt)
        time.sleep(wait)

except KeyboardInterrupt:
    pass
finally:
    vg.release()
    cv2.destroyAllWindows()

