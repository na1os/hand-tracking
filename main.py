import argparse
import math
import os
import sys
import time
import urllib.request
from typing import Optional, Tuple

import cv2
import numpy as np  # noqa: F401  (required by mediapipe at runtime)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pynput.mouse import Controller, Button

try:
    from pynput import keyboard as pynput_keyboard
    _HAVE_KB = True
except Exception:
    _HAVE_KB = False


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.expanduser("~/.cache/handmouse/hand_landmarker.task")

# Hand skeleton connections (pairs of landmark indices).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # palm base
]

# Hysteresis thresholds for pinch detection. The gap between enter and exit
# prevents rapid flicker around the threshold, which was causing the cursor
# to jitter and clicks to feel unreliable.
PINCH_ENTER = 0.35   # pinch_ratio below this -> pinch starts
PINCH_EXIT  = 0.55   # pinch_ratio above this -> pinch ends


def ensure_model() -> str:
    """Download the hand landmarker model on first run, cache it for later."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return MODEL_PATH
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading hand landmarker model (~8 MB, one time only)...")
    tmp = MODEL_PATH + ".tmp"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_PATH)
        return MODEL_PATH
    except Exception as e:
        print(f"Failed to download model: {e}", file=sys.stderr)
        print(f"Download it manually from {MODEL_URL} and place at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)


class OneEuroFilter:
    """Adaptive low-pass filter.

    Smooth when the hand is still (low jitter), responsive when the hand
    moves fast (no lag). Reference: Casiez et al., CHI 2012.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: Optional[float] = None
        self._dx = 0.0
        self._last_t: Optional[float] = None
        self._freq = 60.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        t = 1.0 / max(freq, 1e-6)
        return 1.0 / (1.0 + tau / t)

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()
        if self._last_t is not None and t > self._last_t:
            self._freq = 1.0 / (t - self._last_t)
        self._last_t = t

        a_d = self._alpha(self.d_cutoff, self._freq)
        if self._x is not None:
            dx = (x - self._x) * self._freq
        else:
            dx = 0.0
        self._dx = a_d * dx + (1 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a_x = self._alpha(cutoff, self._freq)
        if self._x is None:
            self._x = x
        else:
            self._x = a_x * x + (1 - a_x) * self._x
        return self._x

    def prime(self, x: float, t: Optional[float] = None):
        """Warm-start the filter at position x without resetting dx.

        Used after a pinch ends: the filter's last position is stale (from
        before the pinch), so we set it to the current hand position to
        avoid a jump, while keeping the velocity estimate.
        """
        if t is None:
            t = time.time()
        self._x = x
        self._last_t = t

    def reset(self):
        self._x = None
        self._dx = 0.0
        self._last_t = None


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_extended(lm, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    """A finger is extended when its tip is above its PIP joint (image y
    grows downward) and the tip-to-MCP distance is large enough."""
    return lm[tip_idx].y < lm[pip_idx].y and _dist(lm[tip_idx], lm[mcp_idx]) > 0.10


def pinch_ratio(lm) -> float:
    """Thumb-tip to index-tip distance, normalized by palm size."""
    palm = _dist(lm[0], lm[9])
    return _dist(lm[4], lm[8]) / max(palm, 1e-3)


class GestureClassifier:
    """Classifies hand poses with hysteresis on pinch to avoid flicker.

    The pinch state is sticky: once you enter pinch, you stay in pinch
    until the thumb/index separation grows past a more relaxed threshold.
    This eliminates the rapid on/off toggling that was making clicks
    unreliable and the cursor jittery.
    """

    def __init__(self):
        self._is_pinching = False

    def reset(self):
        self._is_pinching = False

    def classify(self, lm) -> str:
        pr = pinch_ratio(lm)

        # Hysteresis: enter pinch below PINCH_ENTER, exit above PINCH_EXIT.
        if self._is_pinching:
            if pr > PINCH_EXIT:
                self._is_pinching = False
        else:
            if pr < PINCH_ENTER:
                self._is_pinching = True

        if self._is_pinching:
            return "pinch"

        index_ext  = finger_extended(lm, 8, 6, 5)
        middle_ext = finger_extended(lm, 12, 10, 9)
        ring_ext   = finger_extended(lm, 16, 14, 13)
        pinky_ext  = finger_extended(lm, 20, 18, 17)
        ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        if ext_count == 0:
            return "fist"
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "peace"
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "point"
        if ext_count >= 3:
            return "open"
        return "none"


class HandMouse:
    """Glue between MediaPipe hand tracking and the system mouse."""

    def __init__(self,
                 camera_index: int = 0,
                 width: int = 640,
                 height: int = 480,
                 margin: float = 0.12,
                 min_cutoff: float = 0.4,
                 beta: float = 0.007,
                 dead_zone: int = 2,
                 mirror: bool = True,
                 gui: bool = False):
        self.camera_index = camera_index
        self.cam_w = width
        self.cam_h = height
        self.margin = margin
        self.mirror = mirror
        self.gui = gui
        self.dead_zone = dead_zone

        self.screen_w, self.screen_h = self._detect_screen_size()

        self.mouse = Controller()
        self.fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.fy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.classifier = GestureClassifier()

        # Click / gesture state.
        self.left_pressed = False
        self.cursor_locked = False
        self.last_gesture = "none"
        self.last_right_click = 0.0

        # Last position actually sent to the OS mouse. Used for the dead
        # zone: if the filtered target is within dead_zone pixels of this,
        # we don't move at all, which kills residual jitter when the hand
        # is held still.
        self._last_moved: Optional[Tuple[float, float]] = None

        # Pinch lock: when pinch starts, we capture the current mouse
        # position and the current knuckle position. While pinching,
        # the cursor follows only the knuckle DELTA from that captured
        # position, applied to the captured mouse position. The filter
        # is bypassed entirely during pinch so there is zero lag and
        # zero drift from the filter's residual settling motion.
        self._pinch_mouse_start: Optional[Tuple[int, int]] = None
        self._pinch_knuckle_start: Optional[Tuple[float, float]] = None

        # Load MediaPipe model.
        model_path = ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # Quit hotkeys (Q / Esc).
        self.should_quit = False
        self._kb_listener = None
        if _HAVE_KB:
            def _on_key(key):
                if key == pynput_keyboard.Key.esc:
                    self.should_quit = True
                    return False
                if hasattr(key, "char") and key.char in ("q", "Q"):
                    self.should_quit = True
                    return False
                return None
            try:
                self._kb_listener = pynput_keyboard.Listener(on_press=_on_key)
                self._kb_listener.daemon = True
                self._kb_listener.start()
            except Exception:
                self._kb_listener = None

    @staticmethod
    def _detect_screen_size() -> Tuple[int, int]:
        try:
            from screeninfo import get_monitors
            m = get_monitors()[0]
            return int(m.width), int(m.height)
        except Exception:
            pass
        try:
            import tkinter
            root = tkinter.Tk()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return w, h
        except Exception:
            pass
        return 1920, 1080

    def _map_to_screen(self, nx: float, ny: float) -> Tuple[int, int]:
        """Map normalized [0,1] hand coords to screen pixels.

        An active-area margin is applied so that the edges of the camera
        frame still reach the edges of the screen.
        """
        m = self.margin
        sx = (nx - m) / (1 - 2 * m)
        sy = (ny - m) / (1 - 2 * m)
        sx = max(0.0, min(1.0, sx))
        sy = max(0.0, min(1.0, sy))
        if self.mirror:
            sx = 1.0 - sx
        return int(sx * (self.screen_w - 1)), int(sy * (self.screen_h - 1))

    def _delta_to_screen_pixels(self, dx_n: float, dy_n: float) -> Tuple[int, int]:
        """Convert a normalized delta (relative to camera frame) to screen
        pixels. Same scale as _map_to_screen but for relative motion."""
        # Use the full screen range so 1.0 in normalized = full screen.
        px = int(dx_n * (self.screen_w - 1))
        py = int(dy_n * (self.screen_h - 1))
        return px, py

    def _update_cursor(self, lm, gesture: str, now: float):
        """Move the system mouse based on hand position.

        Behavior:
          - Pointing/open/peace/fist: cursor follows index fingertip,
            filtered through OneEuro for smoothness, with a small dead
            zone to kill residual jitter when the hand is held still.
          - Pinch: cursor is FROZEN at the position it had when the pinch
            started. While pinching, only the knuckle's delta is applied
            (so you can still drag), and the filter is bypassed. This
            means: zero cursor drift while clicking, zero lag during drag.
          - Fist: cursor locked entirely (no movement at all).
        """
        if gesture == "pinch":
            knuckle = lm[5]  # index MCP - doesn't move during pinch
            if self._pinch_mouse_start is None or self._pinch_knuckle_start is None:
                # First pinch frame: capture mouse pos and knuckle pos.
                self._pinch_mouse_start = self.mouse.position
                self._pinch_knuckle_start = (knuckle.x, knuckle.y)
            else:
                # Follow knuckle delta only. Stable, no filter, no drift.
                dx = knuckle.x - self._pinch_knuckle_start[0]
                dy = knuckle.y - self._pinch_knuckle_start[1]
                if self.mirror:
                    dx = -dx
                px, py = self._delta_to_screen_pixels(dx, dy)
                new_x = self._pinch_mouse_start[0] + px
                new_y = self._pinch_mouse_start[1] + py
                new_x = max(0, min(self.screen_w - 1, new_x))
                new_y = max(0, min(self.screen_h - 1, new_y))
                self.mouse.position = (new_x, new_y)
                self._last_moved = (new_x, new_y)
            return

        # Pinch just ended: prime the filter with the current hand position
        # so the cursor doesn't jump (the filter's last sample is stale).
        if self.last_gesture == "pinch":
            nx, ny = lm[8].x, lm[8].y
            sx, sy = self._map_to_screen(nx, ny)
            self.fx.prime(sx, now)
            self.fy.prime(sy, now)
            self._pinch_mouse_start = None
            self._pinch_knuckle_start = None
            self._last_moved = None  # force a real move on next iteration
            return

        if self.cursor_locked:
            return  # Fist: don't move.

        # Default: follow fingertip, filtered.
        nx, ny = lm[8].x, lm[8].y
        sx, sy = self._map_to_screen(nx, ny)
        sx_s = self.fx(sx, now)
        sy_s = self.fy(sy, now)

        # Dead zone: if filtered target is within dead_zone pixels of the
        # last actually-moved position, do nothing. This eliminates the
        # residual 1-2px tremor that OneEuro can't fully kill, without
        # affecting intentional movements (which are always larger).
        if self._last_moved is not None:
            ddx = sx_s - self._last_moved[0]
            ddy = sy_s - self._last_moved[1]
            if abs(ddx) < self.dead_zone and abs(ddy) < self.dead_zone:
                return

        ix, iy = int(sx_s), int(sy_s)
        self.mouse.position = (ix, iy)
        self._last_moved = (sx_s, sy_s)

    def _handle_gesture_edges(self, gesture: str, now: float):
        # Right-click on peace sign rising edge (with cooldown).
        if gesture == "peace" and self.last_gesture != "peace":
            if now - self.last_right_click > 0.3:
                self.mouse.click(Button.right, 1)
                self.last_right_click = now

        # Left press on pinch rising edge.
        if gesture == "pinch" and self.last_gesture != "pinch":
            self.mouse.press(Button.left)
            self.left_pressed = True
        elif gesture != "pinch" and self.last_gesture == "pinch" and self.left_pressed:
            self.mouse.release(Button.left)
            self.left_pressed = False

        self.cursor_locked = (gesture == "fist")
        self.last_gesture = gesture

    # ----- GUI drawing -----
    def _draw_hand(self, frame, lm, gesture: str):
        """Draw a clean hand skeleton outline on the camera frame."""
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm]

        # Color depends on gesture: pinch = green, fist = red, else cyan.
        if gesture == "pinch":
            color = (80, 230, 120)
        elif gesture == "fist":
            color = (90, 90, 230)
        elif gesture == "peace":
            color = (180, 120, 255)
        else:
            color = (90, 200, 255)

        # Skeleton lines.
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

        # Joints.
        for i, p in enumerate(pts):
            r = 4 if i in (4, 8, 12, 16, 20, 0) else 3
            cv2.circle(frame, p, r, (250, 250, 250), -1, cv2.LINE_AA)
            cv2.circle(frame, p, r, (0, 0, 0), 1, cv2.LINE_AA)

        # Highlight the active anchor: index fingertip normally, knuckle during pinch.
        anchor_idx = 5 if gesture == "pinch" else 8
        ax, ay = pts[anchor_idx]
        cv2.circle(frame, (ax, ay), 10, color, 2, cv2.LINE_AA)

        # Tiny gesture label in the corner (non-intrusive).
        label = gesture if gesture != "none" else "tracking"
        cv2.rectangle(frame, (8, 8), (8 + 14 * len(label) + 16, 34), (18, 18, 24), -1)
        cv2.putText(frame, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60)

        if not cap.isOpened():
            print(f"Could not open camera {self.camera_index}. "
                  f"Try --camera 1 or --camera 2.", file=sys.stderr)
            sys.exit(1)

        print(f"HandMouse running. Camera {self.camera_index}, "
              f"screen {self.screen_w}x{self.screen_h}.")
        if self.gui:
            print("GUI: on (camera window with hand outline).")
        else:
            print("GUI: off. Use --gui to show the camera window.")
        print("Gestures: pinch = left click (cursor freezes), "
              "peace = right-click, fist = lock cursor.")
        print("Press Q or Esc to quit.\n")

        win_name = "HandMouse"
        if self.gui:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, self.cam_w, self.cam_h)

        last_status = None

        try:
            while not self.should_quit:
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    print("Empty frame from camera.", file=sys.stderr)
                    break

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = self.landmarker.detect_for_video(mp_image, int(t0 * 1000))

                now = time.time()

                if res.hand_landmarks:
                    lm = res.hand_landmarks[0]
                    gesture = self.classifier.classify(lm)

                    self._update_cursor(lm, gesture, now)
                    self._handle_gesture_edges(gesture, now)

                    if self.gui:
                        self._draw_hand(frame, lm, gesture)

                    if gesture != last_status:
                        label = gesture if gesture != "none" else "tracking"
                        print(f"[{label}]")
                        last_status = gesture
                else:
                    # Hand lost: release any held click so we don't drag forever.
                    if self.left_pressed:
                        self.mouse.release(Button.left)
                        self.left_pressed = False
                    self._pinch_mouse_start = None
                    self._pinch_knuckle_start = None
                    self.classifier.reset()
                    self.last_gesture = "none"
                    if last_status not in (None, "lost"):
                        print("[hand lost]")
                        last_status = "lost"

                if self.gui:
                    cv2.imshow(win_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        self.should_quit = True
        except KeyboardInterrupt:
            pass
        finally:
            if self.left_pressed:
                self.mouse.release(Button.left)
            cap.release()
            if self.gui:
                cv2.destroyAllWindows()
            try:
                self.landmarker.close()
            except Exception:
                pass
            if self._kb_listener is not None:
                self._kb_listener.stop()
            print("\nHandMouse stopped.")


def ask_gui() -> bool:
    """Ask the user interactively whether to show the camera window.

    Returns True for GUI on, False for headless. Reads from stdin; if
    input is closed or interrupted, defaults to headless (False).
    """
    try:
        ans = input("Show camera window with hand outline?  (1 = yes, 0 = no)  [0]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("1", "y", "Y", "yes", "YES", "Yes")


def main():
    p = argparse.ArgumentParser(description="Hand tracking mouse control.")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default 0). Try 1 or 2 if 0 doesn't work.")
    p.add_argument("--width", type=int, default=640, help="Capture width (default 640)")
    p.add_argument("--height", type=int, default=480, help="Capture height (default 480)")
    p.add_argument("--margin", type=float, default=0.12,
                   help="Active-area margin, 0..0.45 (default 0.12). "
                        "Smaller = more of the camera frame reaches the screen edges.")
    p.add_argument("--beta", type=float, default=0.007,
                   help="One Euro beta. Lower = smoother but slower, "
                        "higher = faster but jitterier (default 0.007).")
    p.add_argument("--min-cutoff", type=float, default=0.4,
                   help="One Euro min cutoff (default 0.4). Lower = smoother when still.")
    p.add_argument("--dead-zone", type=int, default=2,
                   help="Dead zone in pixels: movements smaller than this are "
                        "ignored to kill residual jitter (default 2).")
    p.add_argument("--no-mirror", action="store_true",
                   help="Disable mirrored image (default is mirrored like a selfie cam).")
    p.add_argument("--gui", action="store_true",
                   help="Show camera window with hand outline. "
                        "Skips the interactive prompt.")
    p.add_argument("--no-gui", action="store_true",
                   help="Run headless (no window). Skips the interactive prompt.")
    args = p.parse_args()

    if not 0.0 <= args.margin < 0.45:
        print("--margin must be between 0 and 0.45", file=sys.stderr)
        sys.exit(1)

    # Decide GUI mode: explicit flags take priority, otherwise ask.
    if args.gui:
        gui = True
    elif args.no_gui:
        gui = False
    else:
        gui = ask_gui()

    app = HandMouse(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        margin=args.margin,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        dead_zone=args.dead_zone,
        mirror=not args.no_mirror,
        gui=gui,
    )
    app.run()


if __name__ == "__main__":
    main()
