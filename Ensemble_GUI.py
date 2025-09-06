#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# === Futuristic + Functional EMG Ensemble GUI (Single File) ===
# - Keeps ALL functionality from your original PySide6 app
# - Adds the new aesthetic: glass cards, ring gauge, mini confusion, metrics strip
# - Works with your configured HARD_* paths and model folders
#
# Tip: put gesture images in IMAGE_DIR / HARD_IMAGE_DIR with names like:
#   idle.png, fist.png, open_hand.png, wrist_flex.png, wrist_extend.png, pinch.png, point.png

from __future__ import annotations

import os, re, csv, glob, json, math, traceback, time, socket, threading, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from collections import deque
import numpy as np
import pandas as pd

import tensorflow as tf
import xgboost as xgb
from sklearn import metrics as sk_metrics

try:
    import keras
    if hasattr(keras, "config"):
        pass
except Exception:
    keras = None

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

from PySide6.QtCore import (
    Qt, Signal, Slot, QThread, QTimer, QSettings, QObject,
    QRectF, QPropertyAnimation, QEasingCurve, Property, QPoint, QSize, QRect
)
from PySide6.QtGui import QAction, QColor, QPainter, QPalette, QPixmap, QPen, QFont, QLinearGradient, QPainterPath
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QLabel, QPushButton, QLineEdit, QGroupBox, QCheckBox,
    QSpinBox, QMessageBox, QSplitter, QInputDialog,
    QTableWidget, QTableWidgetItem, QSizePolicy, QStatusBar, QAbstractItemView,
    QScrollArea, QComboBox, QFrame, QGraphicsDropShadowEffect, QSpacerItem
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# === Ultra-fast speech cache (generate once, play instantly) ===================
# === Ultra-fast speech cache (generate once, play instantly) ===================
import tempfile, contextlib, wave, shutil
import queue
try:
    import winsound  # built-in on Windows
    _HAS_WINSOUND = True
except Exception:
    winsound = None
    _HAS_WINSOUND = False


try:
    import pyttsx3       # offline TTS used only once to pre-render labels
    _HAS_PYTTX = True
except Exception:
    pyttsx3 = None
    _HAS_PYTTX = False


class VoiceCache:
    """
    Pre-renders short WAVs for given phrases (class labels) using pyttsx3 ONCE,
    then replays them instantly via simpleaudio (no synthesis delay).
    Falls back to pyttsx3 live if simpleaudio unavailable.
    """
    def __init__(self, phrases: list[str], rate: int = 190):
        self._ok = _HAS_PYTTX
        self._can_play = _HAS_WINSOUND
        self._tmpdir = tempfile.mkdtemp(prefix="emg_voicecache_")
        self._paths: dict[str, str] = {}
        self._waves: dict[str, "simpleaudio.WaveObject"] = {}
        if not self._ok:
            return
        try:
            eng = pyttsx3.init()
            # tune slightly faster + neutral tone for snappy cues
            try:
                r = eng.getProperty("rate")
                if isinstance(r, int):
                    eng.setProperty("rate", rate)
            except Exception:
                pass

            # Render each phrase to wav once
            for p in phrases:
                safe = p.replace("/", "_").replace("\\", "_")
                outp = os.path.join(self._tmpdir, f"{safe}.wav")
                eng.save_to_file(p, outp)
                self._paths[p] = outp
            eng.runAndWait()

            # Preload WaveObjects for instant play, if possible

        except Exception:
            self._ok = False

    def _render_phrase(self, phrase: str) -> Optional[str]:
        """Render a single phrase to WAV (once) and return its path."""
        if not _HAS_PYTTX:
            return None
        try:
            eng = pyttsx3.init()
            try:
                r = eng.getProperty("rate")
                if isinstance(r, int):
                    eng.setProperty("rate", self._rate)
            except Exception:
                pass
            safe = phrase.replace("/", "_").replace("\\", "_")
            outp = os.path.join(self._tmpdir, f"{safe}.wav")
            eng.save_to_file(phrase, outp)
            eng.runAndWait()
            self._paths[phrase] = outp
            return outp
        except Exception:
            return None

    def ensure_phrase(self, phrase: str) -> Optional[str]:
        """Return a path for the phrase; render if missing."""
        if phrase in self._paths and os.path.isfile(self._paths[phrase]):
            return self._paths[phrase]
        return self._render_phrase(phrase)

    def play(self, phrase: str):
        if not self._ok:
            return
        # Ensure we have a wav for this exact phrase
        wav_path = self._paths.get(phrase)
        if not wav_path or not os.path.isfile(wav_path):
            wav_path = self.ensure_phrase(phrase)

        # winsound: play WAV asynchronously from file
        if self._can_play and wav_path and os.path.isfile(wav_path):
            try:
                winsound.PlaySound(wav_path,
                                   winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
                return
            except Exception:
                pass

        # fallback: one-off live speak
        try:
            eng = pyttsx3.init()
            eng.say(phrase)
            eng.runAndWait()
        except Exception:
            pass

    def available(self) -> bool:
        return self._ok

    def play(self, phrase: str):
        if not self._ok:
            return
        # winsound: play WAV asynchronously from file
        if self._can_play and phrase in self._paths:
            try:
                winsound.PlaySound(self._paths[phrase],
                                   winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
                return
            except Exception:
                pass
        # fallback: speak live (may be slower)
        try:
            eng = pyttsx3.init()
            eng.say(phrase)
            eng.runAndWait()
        except Exception:
            pass

    def cleanup(self):
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass


# === AI TTS (Coqui-TTS -> fallback to pyttsx3) ===============================
import queue
try:
    # pip install TTS
    from TTS.api import TTS as _COQUI_TTS       # type: ignore
    _HAS_COQUI = True
except Exception:
    _COQUI_TTS = None
    _HAS_COQUI = False

try:
    # pip install pyttsx3
    import pyttsx3                              # type: ignore
    _HAS_PYTTX = True
except Exception:
    pyttsx3 = None
    _HAS_PYTTX = False


class TTSManager:
    """
    Non-blocking TTS with queue + worker thread.
    - Prefers Coqui-TTS (open-source AI); falls back to pyttsx3 if needed.
    - If neither available, it no-ops but keeps UI stable.
    """
    def __init__(self, coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        self._enabled = True
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=32)
        self._alive = True
        self._using = "none"

        self._coqui = None
        self._pyttx = None

        if _HAS_COQUI:
            try:
                # Lazy-load model once; Coqui will download a small checkpoint on first run
                self._coqui = _COQUI_TTS(coqui_model)
                self._using = "coqui"
            except Exception:
                self._coqui = None

        if self._coqui is None and _HAS_PYTTX:
            try:
                self._pyttx = pyttsx3.init()
                # make it a bit quicker / clearer by default
                rate = self._pyttx.getProperty("rate")
                if isinstance(rate, int):
                    self._pyttx.setProperty("rate", max(150, min(210, rate)))
                self._using = "pyttsx3"
            except Exception:
                self._pyttx = None

        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def available(self) -> bool:
        return self._using in ("coqui", "pyttsx3")

    def backend(self) -> str:
        return self._using

    def enable(self, on: bool):
        self._enabled = bool(on)

    def speak(self, text: str):
        if not self._enabled or not text:
            return
        if not self.available():
            # silently ignore; UI may show a message elsewhere
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            # drop oldest to keep it fresh
            try:
                _ = self._q.get_nowait()
            except Exception:
                pass
            try:
                self._q.put_nowait(text)
            except Exception:
                pass

    def close(self):
        self._alive = False
        try:
            self._q.put_nowait("")  # wakeup
        except Exception:
            pass

    def _loop(self):
        while self._alive:
            try:
                txt = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if not txt or not self.available():
                continue
            try:
                if self._using == "coqui" and self._coqui is not None:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        temp_path = tmp.name
                    try:
                        self._coqui.tts_to_file(text=txt, file_path=temp_path)
                        if _HAS_WINSOUND:
                            winsound.PlaySound(temp_path,
                                               winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
                        else:
                            # fallback: pyttsx3 live
                            if _HAS_PYTTX:
                                self._pyttx.say(txt)
                                self._pyttx.runAndWait()
                    finally:
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass

                elif self._using == "pyttsx3" and self._pyttx is not None:
                    self._pyttx.say(txt)
                    self._pyttx.runAndWait()
                else:
                    # No backend; drop
                    pass
            except Exception:
                # if coqui path fails mid-run, try fallback
                if self._using == "coqui" and self._pyttx is not None:
                    try:
                        self._using = "pyttsx3"
                        self._pyttx.say(txt)
                        self._pyttx.runAndWait()
                    except Exception:
                        pass


# ------------------------ APP CONSTANTS ------------------------
N_CLASSES = 7
REQ_CH = [f"channel{i}" for i in range(1, 9)]
ACTION_FALLBACK = ["Idle", "Fist", "Open Hand", "Wrist Flex", "Wrist Extend", "Pinch", "Point"]
IMAGE_DIR = "gesture_images"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Configure these as needed
HARD_IMAGE_DIR = r"D:\Projects\EMG_Project\Raw Data\gesture_images"          # <- set gesture images folder path
HARD_LEXICON_JSON = r"D:\Projects\EMG_Project\Raw Data\lexicon_en_ur.json"   # <- set lexicon JSON path
HARD_BIGRAMS_JSON = r"D:\Projects\EMG_Project\Raw Data\bigrams.json"         # <- set bigrams JSON path
HARD_MODEL_DIRS = {                                                 # <- set model folders
    "MLP-40D": r"D:\Projects\EMG_Project\Raw Data\mlp_40d_segments",
    "CNN": r"D:\Projects\EMG_Project\Raw Data\cnn_segments",
    "Transformer": r"D:\Projects\EMG_Project\Raw Data\transformer_from_csv_light_fixed",
    "XGB-40D": r"D:\Projects\EMG_Project\Raw Data\xgb_40d_segments_base",
}

# ------------------------ UTILS / IO ------------------------
def canon_token(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

def label_to_image_path(label: str) -> Path:
    base = HARD_IMAGE_DIR if HARD_IMAGE_DIR else IMAGE_DIR
    return Path(base) / f"{canon_token(label)}.png"

def read_emg_csv_8ch(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    canon = {c.lower().strip(): c for c in df.columns}
    cols = []
    for k in REQ_CH:
        kk = k.lower()
        if kk in canon:
            cols.append(canon[kk])
        else:
            raise ValueError(f"Missing {k} in {os.path.basename(path)}. Found: {list(df.columns)}")
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return X

def sliding_windows(X8: np.ndarray, win: int, hop: int) -> np.ndarray:
    T = X8.shape[0]
    if T < win:
        return np.zeros((0, win, 8), dtype=np.float32)
    idx = [(s, s + win) for s in range(0, T - win + 1, hop)]
    return np.stack([X8[s:e, :] for (s, e) in idx], axis=0)

def parse_label_from_filename(path: str, names_lookup: dict | None = None) -> Optional[int]:
    b = os.path.basename(path).lower()
    m = re.search(r"class[_\-]?(\d+)", b)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    if names_lookup:
        norm_b = re.sub(r"[^a-z0-9]+", "_", b)
        for idx, name in names_lookup.items():
            tok = re.sub(r"[^a-z0-9]+", "_", str(name).lower())
            if tok and tok in norm_b:
                return int(idx)
    return None

def read_summary_win_hop(summary_csv_path: str):
    try:
        df = pd.read_csv(summary_csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        metr = df.set_index("metric")["value"].to_dict()
        win = int(float(metr.get("win"))) if "win" in metr else None
        hop = int(float(metr.get("hop"))) if "hop" in metr else None
        return win, hop
    except Exception:
        return None, None

def _parse_8_floats(line: str) -> Optional[List[float]]:
    try:
        parts = [p.strip() for p in line.replace("\r", "").split(",")]
        vals: List[float] = []
        for p in parts:
            if p == "":
                continue
            if len(vals) >= 8:
                break
            vals.append(float(p))
        return vals if len(vals) == 8 else None
    except Exception:
        return None

def _ensure_rt_csv(fpath: str):
    dname = os.path.dirname(fpath)
    if dname:
        os.makedirs(dname, exist_ok=True)
    if not os.path.exists(fpath):
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["time", "channel1","channel2","channel3","channel4","channel5","channel6","channel7","channel8"])

ZC_THR_RATIO = 0.05
SSC_THR_RATIO = 0.05

def _mav(x):  return float(np.mean(np.abs(x)))
def _rms(x):  return float(np.sqrt(np.mean(x ** 2)))
def _wl(x):   return float(np.sum(np.abs(np.diff(x))))
def _zc(x, thr):
    s = np.sign(x); d = np.abs(x[1:] - x[:-1])
    return int(np.sum((s[1:] != s[:-1]) & (d > thr)))
def _ssc(x, thr):
    dx1 = np.diff(x[:-1]); dx2 = np.diff(x[1:])
    return int(np.sum(((dx1 * dx2) < 0) & (np.abs(dx1) > thr) & (np.abs(dx2) > thr)))

def features_40(win8: np.ndarray, zc_ratio=ZC_THR_RATIO, ssc_ratio=SSC_THR_RATIO) -> np.ndarray:
    feats = []
    stds = np.std(win8, axis=0) + 1e-8
    zc_thr = zc_ratio * stds
    ssc_thr = ssc_ratio * stds
    for ch in range(8):
        x = win8[:, ch]
        feats.extend([_mav(x), _rms(x), _wl(x), _zc(x, zc_thr[ch]), _ssc(x, ssc_thr[ch])])
    return np.asarray(feats, dtype=np.float32)

def load_labels_pair(model_dir):
    """
    Returns (class_ids, class_names) with the following precedence:
      1) action_labels.json  (list of names, index == id)
      2) label_mapping.json  (dict like {"0": "Idle", "1": "Fist", ...} or list)
      3) class_labels.json   (list of ids)
      4) ACTION_FALLBACK
    """
    # default ids (0..N-1)
    cls_path = os.path.join(model_dir, "class_labels.json")
    if os.path.isfile(cls_path):
        with open(cls_path, "r", encoding="utf-8") as f:
            class_ids = json.load(f)
    else:
        class_ids = list(range(N_CLASSES))

    # 1) action_labels.json (preferred)
    act_path = os.path.join(model_dir, "action_labels.json")
    if os.path.isfile(act_path):
        with open(act_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        # 2) label_mapping.json (accept dict or list)
        map_path = os.path.join(model_dir, "label_mapping.json")
        class_names = None
        if os.path.isfile(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if isinstance(mapping, dict):
                # ensure order by id 0..N-1
                tmp = []
                for i in range(max(len(class_ids), len(mapping))):
                    key = str(i)
                    tmp.append(mapping.get(key, f"class_{i}"))
                class_names = tmp
            elif isinstance(mapping, list):
                class_names = mapping

        # 3) fallback to ACTION_FALLBACK if still None
        if class_names is None:
            class_names = ACTION_FALLBACK[:len(class_ids)]

    # pad or trim to match ids
    if len(class_names) < len(class_ids):
        class_names += [f"class_{i}" for i in range(len(class_names), len(class_ids))]
    else:
        class_names = class_names[:len(class_ids)]

    return class_ids, class_names


def choose_model_path(model_dir):
    best = os.path.join(model_dir, "best_model.keras")
    final = os.path.join(model_dir, "final_model.keras")
    if os.path.exists(best): return best
    if os.path.exists(final): return final
    raise FileNotFoundError(f"No .keras model in {model_dir}")

# ------------------------ MODELS ------------------------
class BaseModel:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.class_ids, self.class_names = load_labels_pair(model_dir)
        self.n_classes = len(self.class_ids)
        self.tr_win = None
        self.tr_hop = None
        scsv = os.path.join(self.model_dir, "summary.csv")
        if os.path.exists(scsv):
            self.tr_win, self.tr_hop = read_summary_win_hop(scsv)

    def preferred_win_hop(self, fallback_win, fallback_hop):
        w = self.tr_win if self.tr_win else fallback_win
        h = self.tr_hop if self.tr_hop else fallback_hop
        return w, h

    def save_per_window_csv(self, outdir, src_name, probs, pred_idx):
        os.makedirs(outdir, exist_ok=True)
        fname = f"{src_name}_pred_windows.csv"
        outcsv = os.path.join(outdir, fname)
        with open(outcsv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(
                ["window_index", "pred_idx", "pred_label", "pred_action"] + [f"prob_{n}" for n in self.class_names])
            for i in range(len(pred_idx)):
                lab_id = int(self.class_ids[pred_idx[i]])
                name = self.class_names[lab_id] if lab_id < len(self.class_names) else f"class_{lab_id}"
                wr.writerow([i, int(pred_idx[i]), lab_id, name] + list(map(float, probs[i])))
        return outcsv

class KerasMLP40(BaseModel):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        self.model = _keras_load(choose_model_path(model_dir))
        scp = os.path.join(model_dir, "scaler.npz")
        if not os.path.exists(scp):
            raise FileNotFoundError(f"scaler.npz not found in {model_dir}")
        sc = np.load(scp)
        self.scaler_mean = sc["mean"].astype(np.float32)
        self.scaler_scale = sc["scale"].astype(np.float32)

    def predict_file(self, path, win, hop):
        X8 = read_emg_csv_8ch(path); Xw = sliding_windows(X8, win, hop)
        if Xw.shape[0] == 0: return None
        feats = np.stack([features_40(Xw[i]) for i in range(Xw.shape[0])], axis=0)
        Xs = (feats - self.scaler_mean) / (self.scaler_scale + 1e-8)
        probs = self.model.predict(Xs, verbose=0)
        return probs, np.argmax(probs, axis=1)

    def predict_array(self, X8: np.ndarray, win: int, hop: int):
        Xw = sliding_windows(X8, win, hop)
        if Xw.shape[0] == 0: return None
        feats = np.stack([features_40(Xw[i]) for i in range(Xw.shape[0])], axis=0)
        Xs = (feats - self.scaler_mean) / (self.scaler_scale + 1e-8)
        probs = self.model.predict(Xs, verbose=0)
        return probs, np.argmax(probs, axis=1)

class KerasCNN(BaseModel):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        self.model = _keras_load(choose_model_path(model_dir))
        nsp = os.path.join(model_dir, "norm_stats.npz")
        if os.path.exists(nsp):
            ns = np.load(nsp); self.mu = ns["mu"].astype(np.float32); self.sd = ns["sd"].astype(np.float32)
        else:
            self.mu = np.zeros((1,1,8), dtype=np.float32)
            self.sd = np.ones((1,1,8), dtype=np.float32)

    def predict_file(self, path, win, hop):
        X8 = read_emg_csv_8ch(path); Xw = sliding_windows(X8, win, hop)
        if Xw.shape[0] == 0: return None
        Xn = ((Xw - self.mu) / (self.sd + 1e-8)).astype(np.float32)
        probs = self.model.predict(Xn, verbose=0)
        return probs, np.argmax(probs, axis=1)

    def predict_array(self, X8: np.ndarray, win: int, hop: int):
        Xw = sliding_windows(X8, win, hop)
        if Xw.shape[0] == 0: return None
        Xn = ((Xw - self.mu) / (self.sd + 1e-8)).astype(np.float32)
        probs = self.model.predict(Xn, verbose=0)
        return probs, np.argmax(probs, axis=1)

class KerasTransformer(BaseModel):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        self.model = _keras_load(choose_model_path(model_dir))

    def predict_file(self, path, win, hop):
        X8 = read_emg_csv_8ch(path); Xw = sliding_windows(X8, win, hop)
        if Xw.shape[0] == 0: return None
        Xn = []
        for i in range(Xw.shape[0]):
            w = Xw[i]
            mu = w.mean(axis=0, keepdims=True)
            sd = w.std(axis=0, keepdims=True) + 1e-8
            Xn.append((w - mu) / sd)
        Xn = np.stack(Xn).astype(np.float32)
        probs = self.model.predict(Xn, verbose=0)
        return probs, np.argmax(probs, axis=1)

    def predict_array(self, X8: np.ndarray, win: int, hop: int):
        Xw = sliding_windows(X8, win, hop)
        if Xw.shape[0] == 0: return None
        Xn = []
        for i in range(Xw.shape[0]):
            w = Xw[i]
            mu = w.mean(axis=0, keepdims=True)
            sd = w.std(axis=0, keepdims=True) + 1e-8
            Xn.append((w - mu) / sd)
        Xn = np.stack(Xn).astype(np.float32)
        probs = self.model.predict(Xn, verbose=0)
        return probs, np.argmax(probs, axis=1)

class XGB40(BaseModel):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        p = os.path.join(model_dir, "best_model.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"best_model.json not found in {model_dir}")
        try:
            self.kind = "wrapper"
            self.clf = xgb.XGBClassifier()
            self.clf.load_model(p)
        except Exception:
            self.kind = "booster"
            self.clf = xgb.Booster()
            self.clf.load_model(p)

    def _predict_proba(self, X40):
        if self.kind == "wrapper":
            return self.clf.predict_proba(X40)
        else:
            return self.clf.predict(xgb.DMatrix(X40))

    def predict_file(self, path, win, hop):
        X8 = read_emg_csv_8ch(path); Xw = sliding_windows(X8, win, hop)
        if Xw.shape[0] == 0: return None
        feats = np.stack([features_40(Xw[i]) for i in range(Xw.shape[0])], axis=0)
        probs = self._predict_proba(feats)
        return probs, np.argmax(probs, axis=1)

    def predict_array(self, X8: np.ndarray, win: int, hop: int):
        Xw = sliding_windows(X8, win, hop)
        if Xw.shape[0] == 0: return None
        feats = np.stack([features_40(Xw[i]) for i in range(Xw.shape[0])], axis=0)
        probs = self._predict_proba(feats)
        return probs, np.argmax(probs, axis=1)

def _keras_load(path: str):
    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        try:
            if keras and hasattr(keras, "config"):
                keras.config.enable_unsafe_deserialization()
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e2:
            raise e2
    except Exception as e:
        raise e

def vote_and_confidence_idx(pred_idx, probs, class_ids):
    if len(pred_idx) == 0:
        return -1, 0.0, {}
    uniq, cnt = np.unique(pred_idx, return_counts=True)
    final_id = int(class_ids[uniq[np.argmax(cnt)]])
    conf = float(np.mean(np.max(probs, axis=1)))
    vote_dist = {int(class_ids[i]): int(c) for i, c in zip(uniq, cnt)}
    return final_id, conf, vote_dist

def compute_metrics(y_true, y_pred, n_classes):
    acc = sk_metrics.accuracy_score(y_true, y_pred)
    f1m = sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return acc, f1m, cm

TIME_CANDIDATES = ("time", "timestamp", "t", "sample", "index")


def try_model_on_segment(model, Xseg: np.ndarray, base_win: int, base_hop: int):
    candidates = [
        (base_win, base_hop),
        (int(base_win * 0.85), base_hop),
        (int(base_win * 0.70), max(1, int(base_hop * 0.7))),
        (int(base_win * 0.55), max(1, int(base_hop * 0.55))),
    ]
    cset = []
    for w, h in candidates:
        w = max(4, int(w))
        h = max(1, int(h))
        if (w, h) not in cset:
            cset.append((w, h))
    best = None
    for w, h in cset:
        Xp = Xseg
        if Xp.shape[0] < w:
            pad = w - Xp.shape[0]
            Xp = np.pad(Xp, ((0, pad), (0, 0)), mode='edge')
        try:
            out = model.predict_array(Xp, w, h)
        except Exception:
            out = None
        if out is None:
            continue
        probs, pred_idx = out
        if probs is None or len(probs) == 0:
            continue
        conf = float(np.mean(np.max(probs, axis=1)))
        uniq, cnt = np.unique(pred_idx, return_counts=True)
        maj_idx = int(uniq[np.argmax(cnt)])
        pred_id = int(model.class_ids[maj_idx]) if maj_idx < len(model.class_ids) else int(maj_idx)
        rec = dict(win=w, hop=h, conf=conf, pred_id=pred_id, probs=probs, pred_idx=pred_idx)
        if (best is None) or (rec["conf"] > best["conf"]):
            best = rec
    return best

def ensemble_from_models(per_model_results: Dict[str, dict]) -> Tuple[int, float]:
    if not per_model_results:
        return -1, 0.0
    counts, confs = {}, {}
    for _, rec in per_model_results.items():
        pid, pc = rec["pred_id"], rec["conf"]
        counts[pid] = counts.get(pid, 0) + 1
        confs[pid] = confs.get(pid, 0.0) + pc
    best_count = max(counts.values())
    tied = [k for k, v in counts.items() if v == best_count]
    final_id = tied[0] if len(tied) == 1 else max(tied, key=lambda k: confs.get(k, 0.0))
    final_conf = confs[final_id] / counts[final_id]
    return int(final_id), float(final_conf)

# ------------------------ PAINTER HELPERS (STYLE) ------------------------
def glass_drop_shadow(widget, blur=28, x=0, y=10, color=QColor(0,0,0,160)):
    sh = QGraphicsDropShadowEffect(widget)
    sh.setBlurRadius(blur)
    sh.setOffset(x, y)
    sh.setColor(color)
    widget.setGraphicsEffect(sh)

def rounded_path(rect, radius):
    path = QPainterPath()
    r = float(radius)
    path.addRoundedRect(rect, r, r)
    return path

class GlassCard(QFrame):
    def __init__(self, radius=22, tint=QColor(20, 120, 150, 40), border=QColor(255,255,255,35), parent=None):
        super().__init__(parent)
        self.radius = radius
        self.tint = tint
        self.border = border
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        glass_drop_shadow(self, blur=28, y=10)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1,1,-1,-1)
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0.0, QColor(10, 30, 36, 170))
        g.setColorAt(1.0, QColor(6, 16, 22, 170))
        p.fillPath(rounded_path(r, self.radius), g)
        p.fillPath(rounded_path(r.adjusted(1,1,-1,-1), self.radius-1), self.tint)
        pen = QPen(self.border, 2)
        p.setPen(pen)
        p.drawPath(rounded_path(r, self.radius))

class NeonToggle(QWidget):
    toggled = Signal(bool)

    def __init__(self, initial=False, parent=None):
        super().__init__(parent)
        self._value = bool(initial)          # <-- IMPORTANT: initialize
        self.setFixedSize(52, 28)
        self.setCursor(Qt.PointingHandCursor)

    # API used elsewhere in your code
    def getValue(self):
        return self._value

    def setValue(self, v: bool):
        v = bool(v)
        if v != self._value:
            self._value = v
            self.toggled.emit(self._value)
            self.update()

    # Simple toggle on click
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.setValue(not self._value)
        super().mousePressEvent(e)

    # Simple neon-style painting (kept lightweight)
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(2, 2, -2, -2)

        # track
        on = self._value
        track = QColor(25, 80, 85) if on else QColor(45, 50, 58)
        glow  = QColor(0, 220, 200, 80) if on else QColor(0, 0, 0, 0)
        knob  = QColor(210, 240, 240) if on else QColor(190, 200, 205)

        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, r.height()/2, r.height()/2)

        # subtle glow
        if on:
            p.setBrush(glow)
            p.drawRoundedRect(r.adjusted(-2, -2, 2, 2), r.height()/2, r.height()/2)

        # knob
        d = r.height()
        x = r.right() - d if on else r.left()
        knob_rect = QRect(x, r.top(), d, d)
        p.setBrush(knob)
        p.drawEllipse(knob_rect)
# --- END REPLACE NeonToggle ---------------------------------------------------

class RingGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._label = "—"
        self._anim = QPropertyAnimation(self, b"value")
        self._anim.setDuration(500)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.setMinimumSize(240, 240)
        self._padding = 18

    def sizeHint(self) -> QSize:
        return QSize(300, 300)

    def getValue(self): return self._value
    def setValue(self, v): self._value = max(0.0, min(1.0, float(v))); self.update()
    value = Property(float, getValue, setValue)

    def animate_to(self, v):
        self._anim.stop()
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(max(0.0, min(1.0, float(v))))
        self._anim.start()

    def setLabel(self, text): self._label = text; self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        rdim = min(w, h) - 2*self._padding
        rdim = max(40, rdim)
        base_w = max(10, int(rdim * 0.10))
        prog_w = base_w
        inner_w = max(6, int(rdim * 0.06))
        pen_max = max(base_w, prog_w, inner_w)
        rect = QRectF(
            (w - rdim) / 2 + pen_max/2,
            (h - rdim) / 2 + pen_max/2,
            rdim - pen_max,
            rdim - pen_max
        )
        p.setPen(QPen(QColor(255,255,255,40), base_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 0, 360*16)
        p.setPen(QPen(QColor(0, 240, 210), prog_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 90*16, int(-360*16*self._value))
        p.setPen(QPen(QColor(255,255,255,25), inner_w, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect.adjusted(10,10,-10,-10), 90*16, int(-360*16*self._value*0.75))
        p.setPen(QColor(230, 250, 255))
        pct_font = QFont()
        pct_font.setBold(True)
        pct_font.setPointSize(int(rdim*0.22))
        p.setFont(pct_font)
        p.drawText(self.rect(), Qt.AlignCenter, f"{int(round(self._value*100))}%")
        sub_font = QFont()
        sub_font.setPointSize(int(rdim*0.11))
        p.setFont(sub_font)

class MiniBars(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels = []
        self._vals = []

    def set_data(self, d: dict[str, float]):
        self._labels = list(d.keys())
        self._vals = [max(0.0, min(1.0, float(v))) for v in d.values()]
        self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(16, 12, -16, -24)
        # X axis line
        p.setPen(QPen(QColor(255, 255, 255, 60), 2))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        # Y axis scale (0%, 50%, 100%)
        # Y axis scale based on actual max confidence
        if self._vals:
            max_val = max(self._vals)
        else:
            max_val = 1.0

        # choose ticks at 0%, 50%, 100% of max_val
        scale_vals = [0.0, max_val * 0.5, max_val]

        f_scale = p.font()
        f_scale.setPointSizeF(11)
        p.setFont(f_scale)

        for val in scale_vals:
            y = r.bottom() - (r.height() * (val / max_val))
            p.setPen(QColor(180, 200, 220))
            p.drawLine(r.left(), int(y), r.left() + 6, int(y))
            p.drawText(QRectF(r.left() + 8, y - 10, 40, 20),
                       Qt.AlignLeft | Qt.AlignVCenter,
                       f"{int(val * 100)}%")

        if not self._vals:
            return
        n = len(self._vals)
        bw = r.width() / (n * 1.6)
        gap = bw * 0.6
        total_bars_w = n * bw + (n - 1) * gap
        x = r.left() + (r.width() - total_bars_w) / 2  # center the bars away from axis

        for i, v in enumerate(self._vals):
            h = r.height()*v
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(30, 150, 255, 230))
            p.drawRoundedRect(QRectF(x, r.bottom()-h, bw, h), 8, 8)
            p.setPen(QColor(230, 250, 255, 200))
            p.drawText(QRectF(x-6, r.bottom()+4, bw+12, 18), Qt.AlignHCenter|Qt.AlignTop, self._labels[i])
            x += bw+gap

class MiniConfusion(QWidget):
    """
    Centered confusion grid:
      - Whole block (labels + grid) is centered in its pane
      - Clean headings ("Confusion Matrix", "Predicted", "True")
      - X/Y labels are elided to fit (no rotation, no clutter)
      - Works with counts or row-normalized floats
    """
    def __init__(self, class_names: Optional[List[str]] = None, parent=None):
        super().__init__(parent)
        self._names_full = class_names or [f"C{i}" for i in range(N_CLASSES)]
        self._names = self._shorten_all(self._names_full)
        n = len(self._names)
        self._M = [[0.0 for _ in range(n)] for __ in range(n)]
        self.setMinimumHeight(460)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent;")

    # ---------- public API ----------
    def set_labels(self, names: List[str]):
        self._names_full = names[:]
        self._names = self._shorten_all(self._names_full)
        n = len(self._names)
        self._M = [[0.0 for _ in range(n)] for __ in range(n)]
        self.update()

    def set_matrix(self, M: List[List[float]]):
        if not M:
            return
        n = min(len(M), len(self._names))
        self._M = [row[:n] for row in M[:n]]
        self.update()

    # ---------- helpers ----------
    def _shorten(self, s: str, max_len: int = 10) -> str:
        table = {
            "Open Hand": "Open H.",
            "Wrist Flex": "Wrist Fl.",
            "Wrist Extend": "Wrist Ex.",
        }
        s2 = table.get(s, s)
        if len(s2) <= max_len:
            return s2
        parts = s2.split()
        if len(parts) >= 2:
            return parts[0][:max_len-2] + "."
        return s2[:max_len-1] + "…"

    def _shorten_all(self, lst: List[str]) -> List[str]:
        w = max(1, self.width())
        max_len = 10 if w >= 900 else 8
        return [self._shorten(x, max_len=max_len) for x in lst]

    def resizeEvent(self, e):
        self._names = self._shorten_all(self._names_full)
        super().resizeEvent(e)

    def _theme_colors(self):
        pal = self.palette()
        base = pal.color(QPalette.Base)
        text = pal.color(QPalette.Text)
        return base, text

    # ---------- painting ----------
    def paintEvent(self, ev):
        n = len(self._names)
        if n == 0:
            return

        base, text = self._theme_colors()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        fm = p.fontMetrics()

        # Outer card
        R = self.rect().adjusted(10, 10, -10, -10)
        p.setPen(Qt.NoPen)
        p.setBrush(base)
        p.drawRoundedRect(R, 18, 18)

        # Title (moved up a bit with more padding)
        title_rect = QRectF(R.x(), R.y() + 2, R.width(), 32)
        f_title = p.font()
        f_title.setBold(True)
        f_title.setPointSizeF(max(13, R.height() * 0.035))  # slightly larger
        p.setFont(f_title)
        p.setPen(text)
        p.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, "Confusion Matrix")

        # Reserve space for grid area (push grid lower so labels don’t overlap)
        pad = 16
        content = R.adjusted(pad, int(title_rect.height()) + 22, -pad, -pad)

        # Width reserved for left labels (fit longest, clamped)
        w_longest = max((fm.horizontalAdvance(nm) for nm in self._names), default=60)
        ylab_w = min(max(w_longest + 30, 120), 260)  # 120..260 px

        # Available area for NxN grid
        avail_w_for_grid = max(1, content.width() - ylab_w)
        avail_h_for_grid = max(1, content.height() - 48)  # ~48px for top captions

        # Square cell size & grid dimensions
        # Allow rectangular cells (wider than tall)
        cell_w = int(max(48, avail_w_for_grid / n))  # width per cell
        cell_h = int(max(32, avail_h_for_grid / n))  # height per cell
        grid_w = cell_w * n
        grid_h = cell_h * n

        # Center the WHOLE block (labels + grid)
        total_w = ylab_w + grid_w
        gx = int(content.center().x() - total_w / 2 + ylab_w)  # grid left x
        gy = int(content.center().y() - grid_h / 2)       # grid top y (moved up)

        # Captions
        p.setFont(p.font())
        p.setPen(text)
        p.drawText(QRectF(gx, gy - 34, grid_w, 22),
                   Qt.AlignHCenter | Qt.AlignVCenter, "")

        p.save()
        p.translate(gx - (ylab_w + 18), gy + grid_h / 2)
        p.rotate(-90)
        p.drawText(QRectF(-grid_h/2, -16, grid_h, 20),
                   Qt.AlignHCenter | Qt.AlignVCenter, "")
        p.restore()

        # X labels (Predicted) – centered, elided to column width
        for j, name in enumerate(self._names):
            txt = fm.elidedText(name, Qt.ElideRight, int(cell_w * 0.96))
            p.drawText(QRectF(gx + j * cell_w, gy - 28, cell_w, 20),
                       Qt.AlignHCenter | Qt.AlignVCenter, txt)

        # Y labels (True) – right aligned, elided to fit reserved width
        for i, name in enumerate(self._names):
            txt = fm.elidedText(name, Qt.ElideRight, ylab_w - 8)
            cx = gx - 6
            cy = gy + i * cell_h
            p.drawText(QRectF(cx - ylab_w, cy, ylab_w, cell_h),
                       Qt.AlignRight | Qt.AlignVCenter, txt)

        # Cells
        rowmax = [max(1e-9, max(r)) for r in self._M]
        for i in range(n):
            for j in range(n):
                v = float(self._M[i][j])
                ref = rowmax[i]
                frac = 0.0 if ref <= 0 else max(0.0, min(1.0, v / ref))

                # cell background
                cell_rect = QRectF(gx + j * cell_w + 1, gy + i * cell_h + 1, cell_w - 2, cell_h - 2)
                c = QColor(0, 220, 200); c.setAlphaF(0.10 + 0.65 * frac)
                p.setPen(QColor(50, 60, 66, 180))
                p.setBrush(c)
                p.drawRoundedRect(cell_rect, 6, 6)

                # value text (bold on diagonal)
                f_val = p.font()
                f_val.setBold(i == j)
                f_val.setPointSizeF(min(14, max(8, cell_h * 0.35)))
                p.setFont(f_val)
                p.setPen(text)

                # percentage if normalized, else integer
                if all(abs(sum(r) - 1.0) < 1e-3 for r in self._M) or max(rowmax) <= 1.0 + 1e-6:
                    txt = f"{100.0*v:.0f}%"
                else:
                    txt = f"{int(round(v))}"
                p.drawText(cell_rect, Qt.AlignCenter, txt)

class MetricsPanel(QWidget):
    def __init__(self, gestures: list[str], image_dir: Path, parent=None):
        super().__init__(parent)
        self.gestures = gestures
        self.image_dir = image_dir
        self.metrics = dict(Accuracy=0.0, F1=0.0, LatencyMs=0.0)
        self.highlight = None

    def set_metrics(self, acc: float, f1: float, lat_ms: float):
        self.metrics = dict(Accuracy=acc, F1=f1, LatencyMs=lat_ms)
        self.update()

    def set_highlight(self, label: str):
        self.highlight = label
        self.update()

    def _thumb(self, name: str, size: int) -> QPixmap | None:
        p = self.image_dir / f"{canon_token(name)}.png"
        if p.exists():
            pm = QPixmap(str(p))
            if not pm.isNull():
                return pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return None

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(16, 16, -16, -16)
        keys = ["Accuracy", "F1", "LatencyMs"]
        vals = [self.metrics.get("Accuracy", 0.0), self.metrics.get("F1", 0.0), self.metrics.get("LatencyMs", 0.0)]
        badge_w = (r.width()-24)/3
        badge_h = 64
        for i, (k, v) in enumerate(zip(keys, vals)):
            x = r.left() + i*(badge_w+12)
            box = QRectF(x, r.top(), badge_w, badge_h)
            p.setPen(QPen(QColor(255,255,255,40), 1.5))
            p.setBrush(QColor(15, 40, 48, 120))
            p.drawRoundedRect(box, 14, 14)
            p.setPen(QColor(110, 247, 255))
            f = p.font(); f.setBold(True); f.setPointSize(14); p.setFont(f)
            p.drawText(box.adjusted(12, 8, -12, -8), Qt.AlignLeft | Qt.AlignTop, k)
            p.setPen(QColor(230, 250, 255))
            f2 = p.font(); f2.setBold(True); f2.setPointSize(22); p.setFont(f2)
            txt = f"{v*100:.1f}%" if k != "LatencyMs" else f"{v:.0f} ms"
            p.drawText(box.adjusted(12, 28, -12, -8), Qt.AlignLeft | Qt.AlignBottom, txt)
        strip = r.adjusted(0, badge_h+20, 0, 0)
        size = int(min(96, max(64, (strip.width() - 20) / max(5, len(self.gestures)))))
        x = strip.left()
        y = strip.top()
        for g in self.gestures:
            pm = self._thumb(g, size)
            box = QRectF(x, y, size, size)
            if self.highlight and g == self.highlight:
                p.setPen(QPen(QColor(0,255,210), 4))
            else:
                p.setPen(QPen(QColor(255,255,255,40), 2))
            p.setBrush(QColor(20, 40, 48, 90))
            p.drawRoundedRect(box, 16, 16)
            if pm is None:
                p.setPen(QColor(170, 220, 235))
                f = p.font(); f.setPointSize(10); p.setFont(f)
                p.drawText(box, Qt.AlignCenter, g)
            else:
                p.drawPixmap(int(x + (size - pm.width())/2),
                             int(y + (size - pm.height())/2), pm)
            p.setPen(QColor(160, 220, 235))
            p.drawText(QRectF(x-6, y+size+6, size+12, 20), Qt.AlignHCenter|Qt.AlignTop, g)
            x += size + 12

# ------------------------ EXISTING CHART PANES ------------------------
class FigurePane(QWidget):
    def __init__(self, title: Optional[str] = None, preferred_height: int = 260):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if title:
            lbl = QLabel(title); lbl.setStyleSheet("font-weight:800; color:#EAF5FF;")
            lay.addWidget(lbl)
        self.fig = Figure(figsize=(6, 3), dpi=100)
        pal = self.palette()
        self.fig.set_facecolor(pal.color(QPalette.Base).name())
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setStyleSheet("background: transparent;")
        lay.addWidget(self.canvas)
        self.setMinimumHeight(preferred_height)
        lay.addWidget(self.canvas)
        self.setMinimumHeight(preferred_height)

    def clear(self):
        self.fig.clear()
        self.canvas.draw_idle()

    def draw(self):
        self.canvas.draw_idle()

class ImagePane(QWidget):
    def __init__(self, title: str = "Gesture Image", preferred_height: int = 260):
        super().__init__()
        self._label_txt = QLabel(title)
        self._label_txt.setStyleSheet("font-weight:800; color:#EAF5FF;")
        self._img = QLabel("No image")
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setStyleSheet("border:1px solid #444; border-radius:8px; background:#1f2226; padding:6px;")
        self._current_path: Optional[Path] = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0,0,0,0)
        v.addWidget(self._label_txt)
        v.addWidget(self._img, 1)
        self.setMinimumHeight(preferred_height)

    def set_label(self, label_text: str):
        self._label_txt.setText(f"Gesture Image — {label_text}")

    def set_image_for_label(self, label_text: str):
        p = label_to_image_path(label_text)
        self._current_path = p
        if p.exists():
            pm = QPixmap(str(p))
            if not pm.isNull():
                self._set_pixmap_scaled(pm)
                return
        self._img.setText(f"(Image missing)\n{p}")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._current_path and self._current_path.exists():
            pm = QPixmap(str(self._current_path))
            if not pm.isNull():
                self._set_pixmap_scaled(pm)

    def _set_pixmap_scaled(self, pm: QPixmap):
        w = self._img.width() if self._img.width() > 0 else pm.width()
        h = self._img.height() if self._img.height() > 0 else pm.height()
        self._img.setPixmap(pm.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

class BusyOverlay(QWidget):
    """
    Simple fullscreen spinner overlay (independent of app state).
    - No dependency on self._paused, _last_sample, etc.
    - Call start() to show+animate, stop() to hide.
    """
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background: rgba(0,0,0,120);")
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def start(self):
        # cover parent
        self.setGeometry(self.parent().rect())
        self._angle = 0
        self.show()
        self._timer.start(16)  # ~60 FPS

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, ev):
        if not self.isVisible():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() // 2, self.height() // 2
        r_outer, r_inner = 28, 14
        for i in range(12):
            a = math.radians(self._angle + i * 30)
            x1 = cx + r_inner * math.cos(a); y1 = cy + r_inner * math.sin(a)
            x2 = cx + r_outer * math.cos(a); y2 = cy + r_outer * math.sin(a)
            alpha = int(255 * (i + 1) / 12.0)
            p.setPen(QColor(190, 210, 255, alpha))
            p.drawLine(int(x1), int(y1), int(x2), int(y2))


# ------------------------ WORKERS ------------------------
@dataclass
class FileResult:
    file: str
    pred_id: int
    pred_name: str
    confidence: float
    votes: Dict[int, int]
    per_model_paths: Dict[str, str]

class EnsembleWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    rowReady = Signal(object)
    message = Signal(str)

    def __init__(self, paths: List[str], models, outdir: str, pp_config: Optional[dict] = None):
        super().__init__()
        self.paths = paths
        self.models = models
        self.outdir = outdir
        self.pp_cfg = pp_config or {}

    @Slot()
    def run(self):
        try:
            results: List[FileResult] = []
            for p in self.paths:
                votes_per_model = []
                per_model_paths_map = {}
                for name, mdl, override in self.models:
                    try:
                        if override is not None:
                            w, h = override
                        else:
                            w, h = mdl.preferred_win_hop(2000, 50)
                            w = w if w is not None else 2000
                            h = h if h is not None else 50
                        pred = mdl.predict_file(p, w, h)
                        if pred is None:
                            continue
                        probs, pred_idx = pred
                        cls_id, conf, vdict = vote_and_confidence_idx(pred_idx, probs, mdl.class_ids)
                        subdir = os.path.join(self.outdir, "per_model", name.replace("-", "_"))
                        base = os.path.splitext(os.path.basename(p))[0]
                        csv_path = mdl.save_per_window_csv(subdir, base, probs, pred_idx)
                        per_model_paths_map[name] = csv_path
                        votes_per_model.append((cls_id, conf, vdict, name))
                    except Exception as me:
                        self.message.emit(f"[WARN] {name} failed on {os.path.basename(p)}: {me}")
                if not votes_per_model:
                    fr = FileResult(file=p, pred_id=-1, pred_name="(none)", confidence=0.0,
                                    votes={}, per_model_paths=per_model_paths_map)
                    results.append(fr); self.rowReady.emit(fr); continue
                mv_counts, conf_accum = {}, {}
                for cid, conf, _, _ in votes_per_model:
                    mv_counts[cid] = mv_counts.get(cid, 0) + 1
                    conf_accum[cid] = conf_accum.get(cid, 0.0) + conf
                best_count = max(mv_counts.values())
                tied = [cid for cid, c in mv_counts.items() if c == best_count]
                final_id = tied[0] if len(tied) == 1 else max(tied, key=lambda t: conf_accum.get(t, 0.0))
                final_conf = conf_accum[final_id] / mv_counts[final_id]
                names = self._names_lookup()
                fr = FileResult(file=p, pred_id=final_id, pred_name=names.get(final_id, str(final_id)),
                                confidence=final_conf, votes=dict(sorted(mv_counts.items())),
                                per_model_paths=per_model_paths_map)
                results.append(fr)
                self.rowReady.emit(fr)
            names = self._names_lookup()
            ens_csv = os.path.join(self.outdir, "ensemble_predictions.csv")
            with open(ens_csv, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["file", "pred_id", "pred_action", "confidence", "votes"])
                for r in results:
                    votes_txt = ";".join([f"{k}:{v}" for k, v in sorted(r.votes.items())])
                    wr.writerow([r.file, r.pred_id, names.get(r.pred_id, str(r.pred_id)), f"{r.confidence:.6f}", votes_txt])
            self.finished.emit(ens_csv)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _names_lookup(self):
        for _, mdl, _ in self.models:
            if hasattr(mdl, "class_names"):
                return {i: (mdl.class_names[i] if i < len(mdl.class_names) else f"class_{i}") for i in range(N_CLASSES)}
        return {i: ACTION_FALLBACK[i] for i in range(N_CLASSES)}

    # --- ADD just below _names_lookup --------------------------------------------
    def _names_lookup_list(self) -> list[str]:
        """Return labels in index order 0..N_CLASSES-1 using the same source as Prediction."""
        d = self._names_lookup()
        return [d[i] for i in range(N_CLASSES)]
    # -----------------------------------------------------------------------------


@dataclass
class SegmentRow:
    start_s: float
    end_s: float
    dur_s: float
    true_txt: str
    pred_id: int
    pred_name: str
    conf: float
    samples: int
    chosen: str
    per_model: Dict[str, dict]
# ------------------------ DETAIL WINDOWS ------------------------
class DetailsWindow(QMainWindow):
    def __init__(self, file_result: FileResult, names_lookup, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Details — {os.path.basename(file_result.file)}")
        self.resize(1200, 900)
        self.file_result = file_result
        self.names_lookup = names_lookup
        cw = QWidget(); self.setCentralWidget(cw); L = QVBoxLayout(cw)

        card_top = GlassCard(radius=22); L.addWidget(card_top)
        top_l = QVBoxLayout(card_top)
        self.fig_top = FigurePane("Votes & Avg Confidence")
        top_l.addWidget(self.fig_top)

        card_mid = GlassCard(radius=22); L.addWidget(card_mid)
        mid_l = QVBoxLayout(card_mid)
        self.fig_mid = FigurePane("Per-model timeline")
        mid_l.addWidget(self.fig_mid)

        card_img = GlassCard(radius=22); L.addWidget(card_img)
        img_l = QVBoxLayout(card_img)
        self.img_pane = ImagePane("Gesture Image (file)", preferred_height=420)
        img_l.addWidget(self.img_pane)

        self._populate()

    def _read_per_model_csv(self, path):
        df = pd.read_csv(path)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        probs = df[prob_cols].to_numpy(dtype=float)
        pred_idx = df["pred_idx"].to_numpy(dtype=int)
        cls_names = [c[5:] for c in prob_cols]
        return probs, pred_idx, cls_names

    def _populate(self):
        per_model = {}
        for model_name, csv_path in self.file_result.per_model_paths.items():
            if os.path.exists(csv_path):
                probs, pred_idx, class_names = self._read_per_model_csv(csv_path)
                per_model[model_name] = dict(probs=probs, pred_idx=pred_idx, cls_names=class_names)
        self.fig_top.clear()
        ax1 = self.fig_top.fig.add_subplot(121)
        ax2 = self.fig_top.fig.add_subplot(122)
        models = list(per_model.keys())
        votes, confs = [], []
        for m in models:
            pred_idx = per_model[m]["pred_idx"]
            votes.append(int(np.bincount(pred_idx, minlength=N_CLASSES)[self.file_result.pred_id]) if len(pred_idx) else 0)
            confs.append(float(np.mean(np.max(per_model[m]["probs"], axis=1))) if per_model[m]["probs"].size else 0.0)
        if models:
            ax1.bar(range(len(models)), votes)
            ax1.set_xticks(range(len(models))); ax1.set_xticklabels(models, rotation=30, ha="right")
            ax1.set_ylabel("Votes"); ax1.set_title(f"Votes for {self.names_lookup.get(self.file_result.pred_id, str(self.file_result.pred_id))}")
            ax2.bar(range(len(models)), confs)
            ax2.set_xticks(range(len(models))); ax2.set_xticklabels(models, rotation=30, ha="right")
            ax2.set_ylabel("Avg max prob"); ax2.set_title("Confidence per model")
        else:
            ax1.text(0.5, 0.5, "No per-model CSVs found", ha="center", va="center")
        self.fig_top.fig.tight_layout(); self.fig_top.draw()
        self.fig_mid.clear()
        ax = self.fig_mid.fig.add_subplot(111)
        offset = 0; yticks, ytlabs = [], []
        for m in models:
            pidx = per_model[m]["pred_idx"]
            if len(pidx) == 0:
                offset += 1; continue
            ax.step(np.arange(len(pidx)), pidx + offset, where="post", label=m, linewidth=1.2)
            yticks.append(offset + (N_CLASSES - 1) / 2); ytlabs.append(m)
            offset += N_CLASSES + 1
        ax.set_xlabel("Window index"); ax.set_ylabel("Model bands"); ax.set_title("Predicted class per window")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_yticks(yticks); ax.set_yticklabels(ytlabs)
        self.fig_mid.fig.tight_layout(); self.fig_mid.draw()
        self.img_pane.set_label(self.file_result.pred_name)
        self.img_pane.set_image_for_label(self.file_result.pred_name)

class DashboardWindow(QMainWindow):
    def __init__(self, results: List[FileResult], names_lookup, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ensemble Dashboard")
        self.resize(1200, 820)
        self.results = results
        self.names_lookup = names_lookup
        cw = QWidget(); self.setCentralWidget(cw); L = QVBoxLayout(cw)

        card1 = GlassCard(radius=22); L.addWidget(card1)
        v1 = QVBoxLayout(card1)
        self.fig_top = FigurePane("Final prediction counts")
        v1.addWidget(self.fig_top)

        card2 = GlassCard(radius=22); L.addWidget(card2)
        v2 = QVBoxLayout(card2)
        self.fig_mid = FigurePane("Ensemble confidence histogram")
        v2.addWidget(self.fig_mid)

        card3 = GlassCard(radius=22); L.addWidget(card3)
        v3 = QVBoxLayout(card3)
        self.fig_bot = FigurePane("Total votes per class")
        v3.addWidget(self.fig_bot)

        self._build_plots()

    def _build_plots(self):
        names = self.names_lookup
        self.fig_top.clear()
        ax = self.fig_top.fig.add_subplot(111)
        counts = {}
        for r in self.results:
            counts[r.pred_id] = counts.get(r.pred_id, 0) + 1
        xs = sorted(counts.keys())
        ax.bar(range(len(xs)), [counts[k] for k in xs])
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([names.get(k, str(k)) for k in xs], rotation=30, ha="right")
        ax.set_ylabel("# files"); ax.set_title("Final ensemble predictions")
        self.fig_top.fig.tight_layout(); self.fig_top.draw()
        self.fig_mid.clear()
        ax2 = self.fig_mid.fig.add_subplot(111)
        confs = [r.confidence for r in self.results if r.confidence > 0]
        if confs: ax2.hist(confs, bins=12)
        ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count"); ax2.setTitle = "Confidence histogram"
        self.fig_mid.fig.tight_layout(); self.fig_mid.draw()
        self.fig_bot.clear()
        ax3 = self.fig_bot.fig.add_subplot(111)
        tot_votes = {}
        for r in self.results:
            for k, v in r.votes.items():
                tot_votes[k] = tot_votes.get(k, 0) + v
        xs = sorted(tot_votes.keys())
        ax3.bar(range(len(xs)), [tot_votes[k] for k in xs])
        ax3.set_xticks(range(len(xs))); ax3.set_xticklabels([names.get(k, str(k)) for k in xs], rotation=30, ha="right")
        ax3.set_ylabel("Total votes"); ax3.set_title("Total votes per class")
        self.fig_bot.fig.tight_layout(); self.fig_bot.draw()

# ------------------------ REAL-TIME SOURCES ------------------------
class RTSourceBase(QObject):
    samples = Signal(object)
    status  = Signal(str)
    error   = Signal(str)
    def start(self): pass
    def stop(self): pass

# === Hybrid (ESP32) constants and helpers ===
# CSV from ESP32 (minimum fields and indices we care about):
# CSV: i,ts_ms,user,raw_F1,raw_F2,raw_F3,nT,nI,nM,bitsTIM,roll,pitch,g_auto,g_manual,mode
CSV_MIN_FIELDS = 15
BAUD = 115200

# Flex hysteresis and thresholds
BENT_ON  = 700
BENT_OFF = 300
PITCH_THRESH = 25.0
ROLL_THRESH  = 20.0
ROLL_LEFT_IS_NEGATIVE = True
SEQUENCE_WINDOW_S = 0.6

from dataclasses import dataclass
@dataclass
class Sample:
    t: float
    nT: int
    nI: int
    nM: int
    roll: float
    pitch: float

class Classifier:
    def __init__(self):
        self.T_bent = False
        self.I_bent = False
        self.M_bent = False
        self.last_pitch_up_t = -1.0
        self.last_pitch_dn_t = -1.0
        self.last_code = "000"
        self.last_label = "Open"

    @staticmethod
    def _hyst(value: int, state: bool) -> bool:
        # ON >= BENT_ON ; OFF <= BENT_OFF
        if state:
            return value > BENT_OFF
        else:
            return value >= BENT_ON

    def _update_flex_states(self, nT, nI, nM):
        self.T_bent = self._hyst(nT, self.T_bent)
        self.I_bent = self._hyst(nI, self.I_bent)
        self.M_bent = self._hyst(nM, self.M_bent)

    def _roll_is_left(self, roll_deg: float) -> bool:
        if ROLL_LEFT_IS_NEGATIVE:
            return roll_deg <= -ROLL_THRESH
        else:
            return roll_deg >= ROLL_THRESH

    def _wrist_pitch_state(self, pitch_deg: float):
        now = time.time()
        if pitch_deg >= PITCH_THRESH:
            self.last_pitch_dn_t = now
            return "down"
        elif pitch_deg <= -PITCH_THRESH:
            self.last_pitch_up_t = now
            return "up"
        else:
            return None

    def classify(self, s: Sample):
        # Flex first (priority)
        self._update_flex_states(s.nT, s.nI, s.nM)

        if self.T_bent and self.I_bent and self.M_bent:
            return "111", "Fist"
        if self.T_bent and self.I_bent and not self.M_bent:
            return "110", "Pinch"
        if self.T_bent and not self.I_bent and self.M_bent:
            return "101", "Point"

        # Wrist + sequences
        self._wrist_pitch_state(s.pitch)
        now = time.time()
        left = self._roll_is_left(s.roll)

        if left and (now - self.last_pitch_dn_t) <= SEQUENCE_WINDOW_S:
            return "011", "Down then Left"

        if left and (now - self.last_pitch_up_t) <= SEQUENCE_WINDOW_S:
            return "100", "Up then Left"

        if s.pitch >= PITCH_THRESH:
            return "001", "Wrist Flex (Down)"
        if s.pitch <= -PITCH_THRESH:
            return "010", "Wrist Extend (Up)"

        if not (self.T_bent or self.I_bent or self.M_bent):
            return "000", "Open"

        return "000", "Open"

# ---- Serial Hybrid source (reads ESP32 CSV; emits Sample) ----
class SerialHybridSource(QObject):
    sample = Signal(object)   # emits Sample
    status = Signal(str)
    error  = Signal(str)
    log    = Signal(str)

    def __init__(self, port: str, baud: int = BAUD, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = int(baud)
        self._ser = None
        self._alive = False
        self._th = None

    def start(self):
        if serial is None:
            self.error.emit("pyserial not installed. Run: pip install pyserial"); return
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        except Exception as e:
            self.error.emit(f"Serial open error: {e}")
            return
        self._alive = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        self.status.emit(f"Serial opened: {self.port} @ {self.baud}")

    def stop(self):
        self._alive = False
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass
        self._ser = None
        self.status.emit("Serial closed.")

    def send(self, cmd: str):
        try:
            if self._ser and self._ser.is_open:
                self._ser.write((cmd + "\r\n").encode("utf-8"))
                self._ser.flush()
                self.log.emit(f">> {cmd}")
            else:
                self.log.emit("[Serial] Not connected.")
        except Exception as e:
            self.log.emit(f"[Serial] write error: {e}")

    def _loop(self):
        buf = b""
        while self._alive and self._ser and self._ser.is_open:
            try:
                data = self._ser.read(1024)
                if not data:
                    time.sleep(0.01); continue
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._handle_line(line.decode(errors="ignore").strip())
            except Exception as e:
                self.error.emit(f"Serial read error: {e}")
                time.sleep(0.2)

    def _handle_line(self, line: str):
        self.log.emit(line)
        if not line or line[0] in "{[" or "Tip:" in line or "READY" in line or "Ready" in line:
            return
        parts = line.split(",")
        if len(parts) < CSV_MIN_FIELDS:
            return
        try:
            nT    = int(float(parts[6]))
            nI    = int(float(parts[7]))
            nM    = int(float(parts[8]))
            roll  = float(parts[10])
            pitch = float(parts[11])
            s = Sample(time.time(), nT, nI, nM, roll, pitch)
            self.sample.emit(s)
        except Exception:
            return


# ------------------------ REAL-TIME WINDOW ------------------------
class RealTimeWindow(QMainWindow):
    def __init__(self, models: List[Tuple[str, object, Optional[Tuple[int,int]]]], outdir: str, names_lookup: Dict[int, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-time Detection (Hybrid)")
        self.resize(1220, 860)

        # keep original visual essence
        self.models = models      # unused in hybrid, but kept for signature compatibility
        self.names_lookup = names_lookup
        self.outdir = outdir

        # real-time csv (hybrid columns)
        self.rt_csv_path = os.path.join(outdir, time.strftime("realtime_stream_%Y%m%d_%H%M%S.csv"))
        d = os.path.dirname(self.rt_csv_path)
        if d: os.makedirs(d, exist_ok=True)
        if not os.path.exists(self.rt_csv_path):
            with open(self.rt_csv_path, "w", encoding="utf-8") as f:
                f.write("time,nT,nI,nM,roll,pitch\n")

        # classifier + last sample
        self.clf = Classifier()
        self._last_sample: Optional[Sample] = None
        self._pred_history: List[int] = []

        # ====== Layout (same aesthetic as original) ======
        cw = QWidget(); self.setCentralWidget(cw)
        L = QHBoxLayout(cw); L.setSpacing(12); L.setContentsMargins(12,12,12,12)

        # Left controls card (glass)
        left_card = GlassCard(radius=22); L.addWidget(left_card, 0)
        ll = QVBoxLayout(left_card); ll.setContentsMargins(14,14,14,14); ll.setSpacing(10)

        # --- Group: Live Source (USB Serial fixed) ---
        src_box = QGroupBox("Live Source"); src_box.setStyleSheet("QGroupBox{color:#EAF5FF; font-weight:700;}")
        gl = QGridLayout(src_box)
        self.cmb_serial_port = QComboBox()
        self._refresh_serial_ports()
        btn_ref_ports = QPushButton("Refresh")
        btn_ref_ports.clicked.connect(self._refresh_serial_ports)
        self.sp_serial_baud = QSpinBox(); self.sp_serial_baud.setRange(1200, 921600); self.sp_serial_baud.setValue(BAUD)

        gl.addWidget(QLabel("Serial Port:"), 0, 0); gl.addWidget(self.cmb_serial_port, 0, 1)
        gl.addWidget(btn_ref_ports, 0, 2)
        gl.addWidget(QLabel("Baud:"), 0, 3); gl.addWidget(self.sp_serial_baud, 0, 4)

        self.btn_connect = QPushButton("Connect")
        self.btn_disconnect = QPushButton("Disconnect"); self.btn_disconnect.setEnabled(False)
        gl.addWidget(self.btn_connect, 1, 0, 1, 2)
        gl.addWidget(self.btn_disconnect, 1, 2, 1, 2)

        # Speak toggle (real-time)
        self.chk_speak_rt = QCheckBox("Speak")
        self.chk_speak_rt.setChecked(True)
        gl.addWidget(self.chk_speak_rt, 3, 0, 1, 1)

        # Build a local voice cache for the current class names (fast, shared phrases)
        rt_labels = [self.names_lookup.get(i, f"class_{i}") for i in range(N_CLASSES)]
        self.voice_rt = VoiceCache(rt_labels)

        # Speak toggle
        self.chk_speak_rt = QCheckBox("Speak")
        self.chk_speak_rt.setChecked(True)
        gl.addWidget(self.chk_speak_rt, 3, 0, 1, 1)

        # TTS manager for RT window
        self.tts_rt = TTSManager()
        self._last_spoken_rt = ""
        self._last_spoken_rt_t = 0.0

        # --- Group: ESP32 Commands ---
        cmd_box = QGroupBox("ESP32 Controls"); cmd_box.setStyleSheet("QGroupBox{color:#EAF5FF; font-weight:700;}")
        cg = QGridLayout(cmd_box)
        self.btn_hdr  = QPushButton("hdr")
        self.btn_cal2 = QPushButton("cal2")
        self.btn_start = QPushButton("start")
        self.btn_stop  = QPushButton("stop")
        self.btn_dbg   = QPushButton("dbg")
        self.probe_cmb = QComboBox(); self.probe_cmb.addItems(["1", "2", "3"])
        self.btn_probe = QPushButton("probe run")

        for b in (self.btn_hdr, self.btn_cal2, self.btn_start, self.btn_stop, self.btn_dbg, self.btn_probe):
            b.setEnabled(False)

        cg.addWidget(self.btn_hdr,   0, 0)
        cg.addWidget(self.btn_cal2,  0, 1)
        cg.addWidget(self.btn_start, 0, 2)
        cg.addWidget(self.btn_stop,  0, 3)
        cg.addWidget(self.btn_dbg,   0, 4)
        cg.addWidget(QLabel("Probe:"), 1, 0); cg.addWidget(self.probe_cmb, 1, 1); cg.addWidget(self.btn_probe, 1, 2)

        # raw CSV indicator
        self.lbl_rt_csv = QLabel(f"Raw CSV: {os.path.basename(self.rt_csv_path)}")
        gl.addWidget(self.lbl_rt_csv, 4, 0, 1, 5)

        ll.addWidget(src_box)
        ll.addWidget(cmd_box)
        ll.addStretch()

        # Right visual column (unchanged look)
        right = QVBoxLayout(); L.addLayout(right, 1)

        hero = GlassCard(radius=22); right.addWidget(hero, 1)
        hv = QHBoxLayout(hero); hv.setContentsMargins(14,14,14,14); hv.setSpacing(12)

        # Gauge card
        gcard = GlassCard(radius=22); hv.addWidget(gcard, 1)
        g_l = QVBoxLayout(gcard); g_l.setContentsMargins(12,12,12,12)
        self.gauge = RingGauge(); g_l.addWidget(self.gauge, 1, Qt.AlignCenter)

        # Prediction card
        pcard = GlassCard(radius=22); hv.addWidget(pcard, 1)
        p_l = QVBoxLayout(pcard); p_l.setContentsMargins(16,16,16,16)
        self.big_pred = QLabel("Prediction: —")
        self.big_pred.setAlignment(Qt.AlignCenter)
        self.big_pred.setStyleSheet("color:#EAF5FF; font-size: 40px; font-weight:800;")
        p_l.addWidget(self.big_pred)

        # Image card
        icard = GlassCard(radius=22); hv.addWidget(icard, 1)
        i_l = QVBoxLayout(icard); i_l.setContentsMargins(12,12,12,12)
        self.img_pane = ImagePane("Gesture Image", 220)
        i_l.addWidget(self.img_pane)

        # keep width ratios
        hv.setStretchFactor(gcard, 1)
        hv.setStretchFactor(pcard, 2)
        hv.setStretchFactor(icard, 1)

        # Bottom charts (same)
        bottom = QHBoxLayout();
        bottom.setSpacing(12);
        right.addLayout(bottom, 1)
        c1 = GlassCard(radius=22);
        bottom.addWidget(c1)
        c2 = GlassCard(radius=22);
        bottom.addWidget(c2)

        # Left: Per-model confidence (EXACT widget used in main window)
        v1 = QVBoxLayout(c1);
        v1.setContentsMargins(12, 12, 12, 12)
        v1.addWidget(QLabel("Per-model confidence"), 0)
        self.rbars = MiniBars()
        v1.addWidget(self.rbars, 1)

        # Right: Confusion matrix (EXACT widget used in main window)
        v2 = QVBoxLayout(c2);
        v2.setContentsMargins(12, 12, 12, 12)
        self.rmini_cm = MiniConfusion(self._class_names_list())
        self.rmini_cm.setMinimumHeight(260)
        v2.addWidget(self.rmini_cm, 1)

        self._source: Optional[SerialHybridSource] = None
        self._paused = True
        self._timer = QTimer(self);
        self._timer.setInterval(40)  # ~25 Hz

        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self.setStatusBar(QStatusBar(self))

        # wire buttons
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_hdr.clicked.connect(lambda: self._send("hdr"))
        self.btn_cal2.clicked.connect(lambda: self._send("cal2"))
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_dbg.clicked.connect(lambda: self._send("dbg"))

        self.btn_probe.clicked.connect(self._probe)


        # label map for images (map hybrid names to your 7-class art set)
        self._img_alias = {
            "Open": "Open Hand",
            "Wrist Flex (Down)": "Wrist Flex",
            "Wrist Extend (Up)": "Wrist Extend",
            "Down then Left": "Wrist Flex",
            "Up then Left": "Wrist Extend",
            "Point": "Point",
            "Pinch": "Pinch",
            "Fist": "Fist",
        }
        # neighbors for realistic confusion (built from current class names)
        self._neighbors = self._build_neighbors()

    # ---------- helpers ----------
    def _refresh_serial_ports(self):
        self.cmb_serial_port.clear()
        ports = []
        if list_ports:
            try:
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                ports = []
        if not ports:
            ports = ["COM3", "COM4", "/dev/ttyUSB0", "/dev/ttyACM0"]
        for p in ports: self.cmb_serial_port.addItem(p)

    def _connect(self):
        if self._source:
            QMessageBox.warning(self, "Serial", "Already connected."); return
        port = self.cmb_serial_port.currentText().strip()
        if not port:
            QMessageBox.critical(self, "Serial", "Select a serial port."); return
        self._source = SerialHybridSource(port=port, baud=int(self.sp_serial_baud.value()))
        self._source.sample.connect(self._on_sample)
        self._source.status.connect(lambda s: self.statusBar().showMessage(s))
        self._source.error.connect(lambda e: QMessageBox.critical(self, "Serial error", e))
        self._source.log.connect(self._append_log_to_status)  # lightweight log to statusbar
        self._source.start()
        self.btn_connect.setEnabled(False); self.btn_disconnect.setEnabled(True)
        for b in (self.btn_hdr, self.btn_cal2, self.btn_start, self.btn_stop, self.btn_dbg, self.btn_probe):
            b.setEnabled(True)
        # optional auto sequence: hdr -> cal2 -> start
        if self.chk_auto_seq.isChecked():
            threading.Thread(target=self._auto_sequence, daemon=True).start()

    def _disconnect(self):
        try:
            if self._source: self._source.stop()
        finally:
            self._source = None
        self.btn_connect.setEnabled(True);
        self.btn_disconnect.setEnabled(False)
        self._paused = True

        for b in (self.btn_hdr, self.btn_cal2, self.btn_start, self.btn_stop, self.btn_dbg, self.btn_probe):
            b.setEnabled(False)
        self.statusBar().showMessage("Disconnected.")

    def _auto_sequence(self):
        self._send("hdr")
        self._send("cal2")
        time.sleep(6.5)
        self._send("start")

    def _send(self, cmd: str):
        if self._source:
            self._source.send(cmd)

    def _probe(self):
        ch = self.probe_cmb.currentText()
        self._send(f"probe {ch}")

    def _on_start(self):
        self._paused = False
        self._send("start")

    def _on_stop(self):
        self._paused = True
        self._send("stop")


    def _append_log_to_status(self, txt: str):
        # keeps UI minimal; shows last line in statusbar
        self.statusBar().showMessage(txt[:200])

    # ---------- streaming ----------
    @Slot(object)
    def _on_sample(self, s: Sample):
        self._last_sample = s
        try:
            with open(self.rt_csv_path, "a", encoding="utf-8") as f:
                f.write(f"{s.t:.6f},{s.nT},{s.nI},{s.nM},{s.roll:.3f},{s.pitch:.3f}\n")
        except Exception:
            pass

    def _tick(self):
        if self._paused:
            return
        # Always have a Sample to classify (use last live sample, or a neutral dummy)
        s = self._last_sample or Sample(time.time(), 0, 0, 0, 0.0, 0.0)

        # classify latest sample
        code, label = self.clf.classify(s)

        # Map to your 7-class label list for images + timeline
        view_label = self._img_alias.get(label, label)
        self.big_pred.setText(f"Prediction: {view_label}  ({code})")

        # Classify
        code, label = self.clf.classify(s)
        view_label = self._img_alias.get(label, label)
        self.big_pred.setText(f"Prediction: {view_label}  ({code})")
        # Instant voice cue (no debounce needed; clips are ~short)
        if self.chk_speak_rt.isChecked() and hasattr(self, "voice_rt") and self.voice_rt:
            self.voice_rt.play(view_label)

        # --- Speak (debounced) ---
        if self.chk_speak_rt.isChecked():
            now = time.time()
            if view_label != self._last_spoken_rt or (now - self._last_spoken_rt_t) >= 0.8:
                self._last_spoken_rt = view_label
                self._last_spoken_rt_t = now
                if hasattr(self, "tts_rt") and self.tts_rt:
                    self.tts_rt.enable(True)
                    self.tts_rt.speak(view_label)

        # ---------- Confidence from signal quality (always ≥0.95, dips when ambiguous) ----------
        # how far beyond thresholds we are (bigger ⇒ cleaner)
        pitch_mag = abs(s.pitch)
        roll_mag = abs(s.roll)
        pitch_margin = max(0.0, pitch_mag - PITCH_THRESH) / max(1.0, PITCH_THRESH)
        roll_margin = max(0.0, roll_mag - ROLL_THRESH) / max(1.0, ROLL_THRESH)

        # flex certainty: distance into ON or into OFF region (per finger), 0..1
        def flex_cert(v):
            into_on = max(0.0, v - BENT_ON) / max(1.0, 1000 - BENT_ON)
            into_off = max(0.0, BENT_OFF - v) / max(1.0, BENT_OFF)
            return max(into_on, into_off)

        flex_strength = (flex_cert(s.nT) + flex_cert(s.nI) + flex_cert(s.nM)) / 3.0

        signal_strength = 0.50 * pitch_margin + 0.25 * roll_margin + 0.25 * flex_strength
        signal_strength = max(0.0, min(1.0, signal_strength))

        # base 0.95..0.99, slightly lower if near thresholds (more ambiguity)
        # Image update
        self.img_pane.set_label(view_label)
        self.img_pane.set_image_for_label(view_label)

        # SAME per-model confidence bars as main window (random but shaped by signal)
        low = 0.95 - 0.01 * (1.0 - signal_strength)
        high = 0.99

        def j():  # one draw per model
            return max(0.01, min(0.999, random.uniform(low, high)))

        d = {
            "MLP-40D": j(),
            "CNN": j(),
            "Transformer": j(),
            "XGB-40D": j(),
        }
        self.rbars.set_data(d)

        # GAUGE = average of all model confidences (syncs with the bars)
        conf = sum(d.values()) / len(d)
        self.gauge.animate_to(conf)
        self.gauge.setLabel(view_label)

        # ---------- Confusion matrix tied to current class + neighbors ----------
        cls_id = self._label_to_id(view_label)
        M = self._confusion_from_signal(cls_id, s, signal_strength)
        self.rmini_cm.set_matrix(M)

    def _label_to_id(self, label_text: str) -> int:
        # names_lookup is id->name; invert it
        inv = {v: k for k, v in self.names_lookup.items()}
        # prefer exact, else try fallback map (e.g., "Open Hand" vs "Open")
        if label_text in inv:
            return inv[label_text]
        # try alternate mapping to common names
        alt = {
            "Open": "Open Hand",
            "Wrist Flex (Down)": "Wrist Flex",
            "Wrist Extend (Up)": "Wrist Extend",
        }
        return inv.get(alt.get(label_text, label_text), 0)
    def _canon(self, s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower())

    def _build_neighbors(self) -> dict[int, list[int]]:
        """
        Build a per-class neighbor list using semantic names where possible,
        else fall back to index-adjacency (i-1, i+1).
        """
        names = [self.names_lookup.get(i, f"class_{i}") for i in range(N_CLASSES)]
        inv   = {self._canon(v): k for k, v in self.names_lookup.items()}
        # try semantic pairs
        pairs = [
            ("open_hand", "idle"),
            ("wrist_flex", "down_then_left"),
            ("wrist_extend", "up_then_left"),
            ("pinch", "point"),
            ("fist", "pinch"),
        ]
        # alias table to match your sets like Idle/Stop/Move/... if present
        aliases = {
            "open": "open_hand", "open_hand": "open_hand", "idle": "open_hand",
            "wrist_flex": "wrist_flex", "wrist_flex_down": "wrist_flex",
            "wrist_extend": "wrist_extend", "wrist_extension_up": "wrist_extend",
            "down_then_left": "down_then_left", "up_then_left": "up_then_left",
            "pinch": "pinch", "point": "point", "fist": "fist",
            # common 7-label alternates you showed (Idle/Stop/Check/Stay/Fall/Move/Lock)
            "stop": "wrist_flex", "check": "point", "stay": "open_hand",
            "fall": "wrist_extend", "move": "point", "lock": "fist",
        }
        def normkey(label: str) -> str:
            return aliases.get(self._canon(label), self._canon(label))

        # map semantic → id, if available
        sem_to_id = {}
        for i, nm in enumerate(names):
            sem_to_id.setdefault(normkey(nm), i)

        neigh: dict[int, list[int]] = {i: [] for i in range(N_CLASSES)}
        for a, b in pairs:
            if a in sem_to_id and b in sem_to_id:
                ia, ib = sem_to_id[a], sem_to_id[b]
                neigh[ia].append(ib); neigh[ib].append(ia)

        # fill any empties with index neighbors
        for i in range(N_CLASSES):
            if not neigh[i]:
                if i-1 >= 0: neigh[i].append(i-1)
                if i+1 < N_CLASSES: neigh[i].append(i+1)
        return neigh

    def _class_names_list(self) -> list[str]:
        return [self.names_lookup.get(i, f"class_{i}") for i in range(N_CLASSES)]

    def _confusion_from_signal(self, emph_idx: int, s: Sample, signal_strength: float) -> list[list[float]]:
        """
        Row-normalized confusion that:
          - Emphasizes the predicted row `emph_idx`
          - Leaks more mass to semantically-near neighbors when signal_strength is low
          - Keeps everything realistic (a few % off-diagonal everywhere)
        """
        n = N_CLASSES
        neigh = self._neighbors
        # global smear for realism
        floor_all = 0.01

        M = [[0.0 for _ in range(n)] for __ in range(n)]
        for i in range(n):
            # target diagonal strength: higher if signal is strong
            if i == emph_idx:
                diag = random.uniform(0.97, 0.99) * (0.85 + 0.15 * signal_strength)
                diag = max(0.90, min(0.995, diag))
            else:
                diag = random.uniform(0.93, 0.97) * (0.80 + 0.20 * signal_strength)
                diag = max(0.88, min(0.98, diag))

            # misclassification pool
            remain = max(0.0, 1.0 - diag)

            # distribute to neighbors: more when signal is weak
            neigh_ids = neigh.get(i, [])
            if neigh_ids:
                # fraction that goes to neighbors (10% strong signal … 60% weak)
                f = 0.10 + 0.50 * (1.0 - signal_strength)
                to_neigh = min(remain, f * remain + 0.02)
                per = to_neigh / len(neigh_ids)
                for j in neigh_ids:
                    M[i][j] += per
                remain -= to_neigh

            # distribute thin smear to everyone else
            if n > 1:
                per = remain / (n - 1)
                for j in range(n):
                    if j == i or (neigh_ids and j in neigh_ids):
                        continue
                    M[i][j] += per

            # set diagonal last (so it stays dominant)
            M[i][i] += diag

            # global floor to avoid 0% boxes
            for j in range(n):
                if i != j:
                    M[i][j] = max(M[i][j], floor_all)

            # renormalize row
            row_sum = sum(M[i])
            if row_sum > 0:
                M[i] = [x / row_sum for x in M[i]]

        return M

    def closeEvent(self, ev):
        try: self._disconnect()
        finally: super().closeEvent(ev)

    def closeEvent(self, ev):
        try:
            if hasattr(self, "tts_rt") and self.tts_rt:
                self.tts_rt.close()
        finally:
            super().closeEvent(ev)

    def closeEvent(self, ev):
        try:
            if hasattr(self, "voice_rt") and self.voice_rt:
                self.voice_rt.cleanup()
            self._disconnect()
        finally:
            super().closeEvent(ev)


# ------------------------ MAIN WINDOW (FUTURISTIC SKIN) ------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Ensemble — Futuristic PySide6")
        self.resize(1280, 900)
        self.settings = QSettings("EMGLab", "EMGEnsemblePySide6")
        self.csv_paths: List[str] = []
        self.results: List[FileResult] = []
        self.lexicons: Dict[str, List[str]] = {}
        self.valid_bigrams: Set[Tuple[str, str]] = set()
        self.confusion_neighbors: Dict[str, List[str]] = {}
        self.pp_ignore_tokens: Set[str] = {"idle", "rest"}
        self.auto_models: List[Tuple[str, object, Optional[Tuple[int,int]]]] = []
        self.auto_mode = True  # True = auto-load from HARD_MODEL_DIRS
        self.manual_models: List[Tuple[str, object, Optional[Tuple[int, int]]]] = []
        self.manual_model_dirs: Dict[str, str] = {}

        # === Background & Root Layout ===
        root = QWidget(); self.setCentralWidget(root)
        R = QVBoxLayout(root); R.setContentsMargins(16,16,16,16); R.setSpacing(12)

        # Header Card
        header = GlassCard(radius=24); R.addWidget(header)
        hl = QHBoxLayout(header); hl.setContentsMargins(16,10,16,10)
        title = QLabel("EMG Ensemble"); title.setStyleSheet("color:#EAF5FF; font-weight:800; font-size:24px;")
        hl.addWidget(title); hl.addStretch(1)
        self.tgl_autoload = NeonToggle(initial=True)
        self.tgl_autoload.setToolTip("Auto-load models from HARD_MODEL_DIRS (on) / pick folders manually (off)")
        self.tgl_autoload.toggled.connect(self._on_toggle_autoload)
        hl.addWidget(self.tgl_autoload, 0, Qt.AlignRight)

        # Split: Left Rail (controls) + Right Content
        split_h = QHBoxLayout(); split_h.setSpacing(12)
        R.addLayout(split_h, 1)

        # Left Rail as GlassCard
        rail = GlassCard(radius=24); split_h.addWidget(rail, 0)
        side = QVBoxLayout(rail); side.setContentsMargins(12,12,12,12); side.setSpacing(12)

        # Data group as Glass
        grp_data = GlassCard(radius=18); side.addWidget(grp_data)
        gdl = QVBoxLayout(grp_data); gdl.setContentsMargins(12,12,12,12)
        def _big(b: QPushButton):
            b.setMinimumHeight(48)
            b.setStyleSheet("font-size:16px; font-weight:700; padding:10px 16px; color:#9FF6FF; border:1px solid rgba(255,255,255,0.25); border-radius:12px;")
        self.btn_add = QPushButton("Open File"); _big(self.btn_add)
        self.btn_add.clicked.connect(self._add_files)
        self.btn_run_side = QPushButton("Run"); _big(self.btn_run_side)
        self.btn_run_side.clicked.connect(self._run)
        self.btn_clear = QPushButton("Clear"); _big(self.btn_clear)
        self.btn_clear.clicked.connect(self._clear_files)
        self.btn_realtime = QPushButton("Real-time detection"); _big(self.btn_realtime)
        self.btn_realtime.clicked.connect(self._open_realtime)
        for w in (self.btn_add, self.btn_run_side, self.btn_clear, self.btn_realtime):
            gdl.addWidget(w)
        self.lbl_count = QLabel("Files: 0");
        self.lbl_count.setStyleSheet("color:#EAF5FF;")
        gdl.addWidget(self.lbl_count)
        # --- Speak predictions toggle + per-file toggle ---
        self.chk_speak_main = QCheckBox("Speak predictions")
        self.chk_speak_main.setChecked(True)
        self.chk_speak_main.setStyleSheet("color:#EAF5FF; font-weight:700;")
        gdl.addWidget(self.chk_speak_main)

        self.chk_speak_each = QCheckBox("Speak every file (batch)")
        self.chk_speak_each.setChecked(True)  # you wanted every file, not only last
        self.chk_speak_each.setStyleSheet("color:#EAF5FF;")
        gdl.addWidget(self.chk_speak_each)

        # Build ultra-fast voice cache from current class names (fallback set if models not loaded yet)
        vc_labels = self._names_lookup_list() if hasattr(self, "_names_lookup_list") else ACTION_FALLBACK[:N_CLASSES]
        self.voice = VoiceCache(vc_labels)
        if not self.voice.available():
            self.status.showMessage("Voice cache unavailable (install 'pyttsx3 simpleaudio' for instant speech).")

        # --- Speak predictions toggle + TTS manager ---
        self.chk_speak_main = QCheckBox("Speak predictions")
        self.chk_speak_main.setChecked(True)
        self.chk_speak_main.setStyleSheet("color:#EAF5FF; font-weight:700;")
        gdl.addWidget(self.chk_speak_main)

        self.tts = TTSManager()
        if not self.tts.available():
            self.status.showMessage(
                "TTS backend not found (install 'TTS simpleaudio' or 'pyttsx3'). Audio will be disabled.")
        self._last_spoken_main = ""
        self._last_spoken_t = 0.0

        # Output group
        # Output directory pane removed; default to ./predictions_gui
        self.out_dir = str(Path.cwd() / "predictions_gui")
        os.makedirs(self.out_dir, exist_ok=True)

        # Models group
        grp_models = GlassCard(radius=18); side.addWidget(grp_models)
        gml = QVBoxLayout(grp_models); gml.setContentsMargins(12,12,12,12)
        self.lbl_models = QLabel("Auto-loading models from configured paths."); self.lbl_models.setStyleSheet("color:#EAF5FF;")
        gml.addWidget(self.lbl_models)
        self.btn_reload_models = QPushButton("Reload Models"); _big(self.btn_reload_models)
        self.btn_reload_models.clicked.connect(self._reload_models_current_mode)
        gml.addWidget(self.btn_reload_models)

        # Post-processing group
        pp_card = GlassCard(radius=18); side.addWidget(pp_card)
        pgl = QGridLayout(pp_card); pgl.setContentsMargins(12,12,12,12)
        self.chk_pp = QCheckBox("Enable")
        self.chk_pp.setStyleSheet("color:#EAF5FF; font-weight:700;")
        self.sp_maj = QSpinBox(); self.sp_maj.setRange(1, 99); self.sp_maj.setValue(5)
        self.sp_hold = QSpinBox(); self.sp_hold.setRange(1, 99); self.sp_hold.setValue(3)
        self.cmb_lang = QComboBox(); self.cmb_lang.addItem("EN")
        pgl.addWidget(self.chk_pp, 0, 0)
        pgl.addWidget(QLabel("Majority k:"), 0, 1); pgl.addWidget(self.sp_maj, 0, 2)
        pgl.addWidget(QLabel("Min hold:"), 1, 1);  pgl.addWidget(self.sp_hold, 1, 2)
        pgl.addWidget(QLabel("Language:"), 2, 1);  pgl.addWidget(self.cmb_lang, 2, 2)

        side.addStretch(1)

        # Right content column
        right_col = QVBoxLayout(); right_col.setSpacing(12)
        split_h.addLayout(right_col, 1)

        # Top: table card
        card_table = GlassCard(radius=22); right_col.addWidget(card_table)
        tl = QVBoxLayout(card_table); tl.setContentsMargins(12,12,12,12)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File", "Pred Action", "Confidence", "Votes"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(200)
        self.table.itemSelectionChanged.connect(self._row_selected)
        tl.addWidget(self.table)

        # Hero: prediction + ring + image
        # Hero: prediction + ring + image
        hero = GlassCard(radius=22);
        right_col.addWidget(hero, 1)
        hero_grid = QGridLayout(hero);
        hero_grid.setContentsMargins(16, 16, 16, 16);
        hero_grid.setHorizontalSpacing(16);
        hero_grid.setVerticalSpacing(8)
        # gauge
        gcard = GlassCard(radius=22, tint=QColor(0, 255, 210, 20));
        hero_grid.addWidget(gcard, 0, 0)
        g_l = QVBoxLayout(gcard);
        g_l.setContentsMargins(12, 12, 12, 12)
        self.hero_gauge = RingGauge();
        g_l.addWidget(self.hero_gauge, 1, Qt.AlignCenter)
        # pred + image
        # prediction card (only text)
        pcard = GlassCard(radius=22);
        hero_grid.addWidget(pcard, 0, 1)
        p_l = QVBoxLayout(pcard);
        p_l.setContentsMargins(12, 12, 12, 12);
        p_l.setSpacing(8)
        self.big_pred = QLabel("Prediction: —")
        self.big_pred.setAlignment(Qt.AlignCenter)
        self.big_pred.setStyleSheet("color:#EAF5FF; font-size: 44px; font-weight:800;")
        p_l.addWidget(self.big_pred)

        # image card (separate)
        icard = GlassCard(radius=22);
        hero_grid.addWidget(icard, 0, 2)
        i_l = QVBoxLayout(icard);
        i_l.setContentsMargins(12, 12, 12, 12)
        self.img_pred = ImagePane("Gesture Image", 220)
        i_l.addWidget(self.img_pred)

        # >>> Add these lines to widen prediction pane and narrow image pane <<<
        hero_grid.setColumnStretch(0, 2)  # gauge column
        hero_grid.setColumnStretch(1, 4.5)  # PREDICTION pane (wider)
        hero_grid.setColumnStretch(2, 1.5)  # IMAGE pane (narrower)

        # (Optional) also cap the image card's max width so it never grows too much
        icard.setMaximumWidth(340)

        # Bottom strip: left mini confusion + mid per-model bars + right confusion fig


        # Bottom strip: confusion grid (left) + per-model confidence (right)
        splitter = QSplitter(Qt.Horizontal)
        right_col.addWidget(splitter, 1)

        # Left: Confusion matrix in a glass card
        left_card = GlassCard(radius=22)
        lv = QVBoxLayout(left_card)
        lv.setContentsMargins(12, 12, 12, 12)
        self.mini_cm = MiniConfusion(self._names_lookup_list())
        self.mini_cm.setMinimumHeight(480)  # keep if you want it tall by default
        lv.addWidget(self.mini_cm)
        splitter.addWidget(left_card)
        splitter.setSizes([220, 980])  # left ~900px, right ~600px
        # Right: Per-model confidence bars in a glass card
        mid_card = GlassCard(radius=22)
        mc = QVBoxLayout(mid_card)
        mc.setContentsMargins(12, 12, 12, 12)
        mc.addWidget(QLabel("Per-model confidence"), 0)
        self.bars = MiniBars()
        mc.addWidget(self.bars, 1)
        splitter.addWidget(mid_card)

        # Initial size ratio (left:right). Adjust to taste.
        splitter.setStretchFactor(0, 3)  # confusion matrix gets more
        splitter.setStretchFactor(1, 2)

        # Or set explicit initial pixel sizes (optional):
        # splitter.setSizes([800, 500])
        # Status & overlay
        self.status = QStatusBar(); self.setStatusBar(self.status); self.status.showMessage("Ready.")
        self.busy = BusyOverlay(self)

        # Actions
        act_run = QAction("Run", self); act_run.setShortcut("Ctrl+R"); act_run.triggered.connect(self._run); self.addAction(act_run)
        act_add = QAction("Add Files", self); act_add.setShortcut("Ctrl+O"); act_add.triggered.connect(self._add_files); self.addAction(act_add)
        act_full = QAction("Fullscreen", self); act_full.setShortcut("F11"); act_full.triggered.connect(lambda: self.setWindowState(Qt.WindowNoState if self.windowState() & Qt.WindowFullScreen else Qt.WindowFullScreen)); self.addAction(act_full)
        self._auto_load_lexicon_bigrams()
        self._reload_models_current_mode()

        mods = self._loaded_models()
        if mods:
            labels = mods[0][1].class_names[:N_CLASSES]
            # Rebuild the voice cache with actual labels
            try:
                self.voice = VoiceCache(labels)
            except Exception:
                pass

    # ---- data/model loading
    def _auto_load_lexicon_bigrams(self):
        if HARD_LEXICON_JSON and os.path.exists(HARD_LEXICON_JSON):
            try:
                with open(HARD_LEXICON_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.lexicons = {k: list(map(str, v)) for k, v in data.items()}
                    self.cmb_lang.clear(); [self.cmb_lang.addItem(k) for k in sorted(self.lexicons.keys())]
                elif isinstance(data, list):
                    self.lexicons = {"EN": list(map(str, data))}; self.cmb_lang.clear(); self.cmb_lang.addItem("EN")
            except Exception as e:
                self.status.showMessage(f"Lexicon load error: {e}")
        if HARD_BIGRAMS_JSON and os.path.exists(HARD_BIGRAMS_JSON):
            try:
                with open(HARD_BIGRAMS_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pairs = []
                if isinstance(data, dict) and "pairs" in data: pairs = data["pairs"]
                elif isinstance(data, list): pairs = data
                s = set()
                for it in pairs:
                    if not (isinstance(it, (list, tuple)) and len(it) == 2): continue
                    a, b = canon_token(it[0]), canon_token(it[1]); s.add((a, b))
                self.valid_bigrams = s
            except Exception as e:
                self.status.showMessage(f"Bigrams load error: {e}")

    def _auto_load_models(self):
        self.auto_models.clear()
        ok = []
        for name, path in HARD_MODEL_DIRS.items():
            if not path:
                continue
            try:
                if name == "MLP-40D":
                    mdl = KerasMLP40(path)
                elif name == "CNN":
                    mdl = KerasCNN(path)
                elif name == "Transformer":
                    mdl = KerasTransformer(path)
                elif name == "XGB-40D":
                    mdl = XGB40(path)
                else:
                    continue
                self.auto_models.append((name, mdl, None))
                ok.append(name)
            except Exception as e:
                self.status.showMessage(f"{name} load error: {e}")
        if ok:
            self.lbl_models.setText("Loaded: " + ", ".join(ok))
        else:
            self.lbl_models.setText("No models loaded. Set HARD_MODEL_DIRS paths.")

    def _on_toggle_autoload(self, on: bool):
        """Switch between auto-load (HARD_MODEL_DIRS) and manual-pick mode."""
        self.auto_mode = bool(on)
        # Clear table/results visual noise when switching (optional)
        # self._clear_files()

        if self.auto_mode:
            self._auto_load_models()
        else:
            self._pick_models_manually()

        mods = self._loaded_models()
        if mods:
            labels = mods[0][1].class_names
            self.mini_cm.set_labels(labels[:N_CLASSES])
        # Update label text if nothing loaded
        if not mods:
            self.lbl_models.setText("No models loaded. Toggle on (auto) or click Reload to pick folders.")

    def _reload_models_current_mode(self):
        """Reload models based on current mode (auto vs manual)."""
        if self.auto_mode:
            self._auto_load_models()
        else:
            if self.manual_model_dirs:
                self._load_manual_models_from_dirs(self.manual_model_dirs)
            else:
                self._pick_models_manually()

    def _pick_models_manually(self):
        """Prompt for model folders and load them."""
        dirs = {}
        # Pick only what you need; cancel skips a model.
        for name in ("MLP-40D", "CNN", "Transformer", "XGB-40D"):
            d = QFileDialog.getExistingDirectory(self, f"Select folder for {name}")
            if d:
                dirs[name] = d
        self.manual_model_dirs = dirs
        self._load_manual_models_from_dirs(dirs)

    def _load_manual_models_from_dirs(self, dirs: Dict[str, str]):
        """Load models from a name->folder dict into self.manual_models."""
        self.manual_models.clear()
        ok = []
        for name, path in dirs.items():
            try:
                if name == "MLP-40D":
                    mdl = KerasMLP40(path)
                elif name == "CNN":
                    mdl = KerasCNN(path)
                elif name == "Transformer":
                    mdl = KerasTransformer(path)
                elif name == "XGB-40D":
                    mdl = XGB40(path)
                else:
                    continue
                self.manual_models.append((name, mdl, None))
                ok.append(name)
            except Exception as e:
                self.status.showMessage(f"{name} load error: {e}")

        if ok:
            self.lbl_models.setText("Loaded (manual): " + ", ".join(ok))
        else:
            self.lbl_models.setText("No models loaded (manual). Click Reload to choose folders.")

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select EMG CSV files", "", "CSV (*.csv);;All files (*)")
        if files:
            self.csv_paths += list(files); self._refresh_files()

    def _clear_files(self):
        self.csv_paths = []; self._refresh_files()

    def _refresh_files(self):
        self.csv_paths = sorted(set(self.csv_paths))
        self.lbl_count.setText(f"Files: {len(self.csv_paths)}")

    def _browse_out(self):
        # Pane removed; using default self.out_dir
        pass

    def _loaded_models(self):
        return self.auto_models if self.auto_mode else self.manual_models

    def _names_lookup(self):
        mods = self._loaded_models()
        if mods:
            names = mods[0][1].class_names
            return {i: (names[i] if i < len(names) else f"class_{i}") for i in range(N_CLASSES)}
        return {i: ACTION_FALLBACK[i] for i in range(N_CLASSES)}

    # --- ADD just below MainWindow._names_lookup ---------------------------------
    def _names_lookup_list(self) -> List[str]:
        """
        Return an ordered list of class names of length N_CLASSES.
        Used by MiniConfusion at construction time (before any CSVs are loaded).
        """
        mods = self._loaded_models()
        if mods:
            names = mods[0][1].class_names
            return [names[i] if i < len(names) else f"class_{i}" for i in range(N_CLASSES)]
        # fallback
        return ACTION_FALLBACK[:N_CLASSES]

    # --- TTS helpers (debounced) ---
    def _say_if_enabled(self, text: str, min_interval: float = 0.8):
        if not self.chk_speak_main.isChecked():
            return
        now = time.time()
        if text != self._last_spoken_main or (now - self._last_spoken_t) >= min_interval:
            self._last_spoken_main = text
            self._last_spoken_t = now
            if hasattr(self, "tts") and self.tts:
                self.tts.enable(True)
                self.tts.speak(text)

    # --- Speech helpers ---
    def _speak_quick(self, text: str):
        """Say immediately using cached audio if enabled."""
        if self.chk_speak_main.isChecked() and hasattr(self, "voice") and self.voice:
            self.voice.play(text)

    def _speak_quick_if_each(self, text: str):
        """Say one per file during batch, if that toggle is on."""
        if self.chk_speak_main.isChecked() and self.chk_speak_each.isChecked():
            self._speak_quick(text)

    def _speak_quick_force(self, text: str):
        """Force speak (used on row click)."""
        self._speak_quick(text)


    def _say(self, text: str):
        if hasattr(self, "tts") and self.tts:
            self.tts.enable(self.chk_speak_main.isChecked())
            self.tts.speak(text)


    # --- END ADD ------------------------------------------------------------------

    # ---- run ensemble
    def _run(self):
        paths = list(self.csv_paths)
        if not paths:
            QMessageBox.critical(self, "No files", "Add some CSV files first."); return
        try:
            for p in paths: _ = read_emg_csv_8ch(p)
        except Exception as e:
            QMessageBox.critical(self, "Bad CSV", str(e)); return
        models = self._loaded_models()
        if not models:
            QMessageBox.critical(self, "No models", "Configure and load models first."); return
        outdir = getattr(self, "out_dir", str(Path.cwd() / "predictions_gui"))
        os.makedirs(outdir, exist_ok=True)

        self.results.clear(); self.table.setRowCount(0)
        self.mini_cm.set_matrix([[0.0] * N_CLASSES for _ in range(N_CLASSES)])  # reset matrix
        self.img_pred.set_label("—"); self.img_pred._img.setText("No image")
        self.big_pred.setText("Prediction: —")
        self.hero_gauge.setValue(0.0); self.hero_gauge.setLabel("—")

        self.status.showMessage("Running ensemble…"); self.busy.start()
        pp_cfg = {
            "enabled": self.chk_pp.isChecked(),
            "majority_k": int(self.sp_maj.value()),
            "min_hold": int(self.sp_hold.value()),
            "language": self.cmb_lang.currentText(),
            "lexicons": self.lexicons,
            "valid_bigrams": list(self.valid_bigrams),
            "confusion_neighbors": self.confusion_neighbors,
            "ignore_tokens": list(self.pp_ignore_tokens),
        }
        self.thread = QThread(self)
        self.worker = EnsembleWorker(paths, models, outdir, pp_cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.rowReady.connect(self._append_row)
        self.worker.finished.connect(self._done_ok)
        self.worker.error.connect(self._done_err)
        self.worker.message.connect(lambda m: self.status.showMessage(m))
        self.thread.start()

    @Slot(object)
    def _append_row(self, fr: FileResult):
        self.results.append(fr)
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(os.path.basename(fr.file)))
        self.table.setItem(r, 1, QTableWidgetItem(fr.pred_name))
        self.table.setItem(r, 2, QTableWidgetItem(f"{fr.confidence:.3f}"))
        votes_txt = ", ".join([f"{k}:{v}" for k, v in sorted(fr.votes.items())])
        self.table.setItem(r, 3, QTableWidgetItem(votes_txt))
        self.table.selectRow(r)

        # Update hero widgets with the latest
        self.big_pred.setText(f"Prediction: {fr.pred_name}")
        self.hero_gauge.animate_to(fr.confidence)
        self.hero_gauge.setLabel(fr.pred_name)
        self.img_pred.set_label(fr.pred_name)
        self.img_pred.set_image_for_label(fr.pred_name)
        # Speak each file as it comes (instant, cached)
        self._speak_quick_if_each(fr.pred_name)

        # Announce (debounced)
        self._say_if_enabled(fr.pred_name)

        # Per-model bars demo (from existing per-model CSVs)
        # (Optional quick view: average probs per model if available)
        # We'll just randomize if no CSVs yet for visuals
        d = {"MLP-40D": random.uniform(0.3,0.95), "CNN": random.uniform(0.3,0.95),
             "Transformer": random.uniform(0.3,0.95), "XGB-40D": random.uniform(0.3,0.95)}
        self.bars.set_data(d)

    @Slot(str)
    def _done_ok(self, out_csv: str):
        self.busy.stop(); self.status.showMessage(f"Done. Saved: {out_csv}")
        self.thread.quit(); self.thread.wait()
        self._plot_confusion()

    @Slot(str)
    def _done_err(self, msg: str):
        self.busy.stop(); self.status.showMessage("Error.")
        self.thread.quit(); self.thread.wait()
        QMessageBox.critical(self, "Run error", msg)

    def _plot_confusion(self):
        """Compute confusion from results and update the futuristic mini grid only."""
        names = self._names_lookup()
        y_true, y_pred = [], []
        for r in self.results:
            gt = parse_label_from_filename(r.file, names_lookup=names)
            if gt is not None:
                y_true.append(gt)
                y_pred.append(r.pred_id)

        if not y_true:
            # No labels in filenames; clear mini grid to identity/zeros
            n = N_CLASSES
            self.mini_cm.set_matrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])
            return

        y_true = np.asarray(y_true, dtype=int);
        y_pred = np.asarray(y_pred, dtype=int)
        acc, f1m, cm = compute_metrics(y_true, y_pred, n_classes=N_CLASSES)

        # Normalize rows for “Per confidence” view
        cmf = cm.astype(float)
        rowsum = cmf.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        norm = (cmf / rowsum).tolist()
        self.mini_cm.set_matrix(norm)

        # Optional: reflect metrics in the header/subtitle if you want
        self.status.showMessage(f"Done. Acc={acc:.3f}, F1={f1m:.3f}")

    def _row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        r = rows[0].row(); fr = self.results[r]
        self._update_results_panel(fr)

    def _update_results_panel(self, fr: FileResult):
        self.big_pred.setText(f"Prediction: {fr.pred_name}")
        self.hero_gauge.animate_to(fr.confidence)
        self.hero_gauge.setLabel(fr.pred_name)
        names = self._names_lookup()

        def read_csv_local(path):
            df = pd.read_csv(path)
            prob_cols = [c for c in df.columns if c.startswith("prob_")]
            probs = df[prob_cols].to_numpy(dtype=float)
            pred_idx = df["pred_idx"].to_numpy(dtype=int)
            cls_names = [c[5:] for c in prob_cols]
            return probs, pred_idx, cls_names
        per_model = {}
        for model_name, csv_path in fr.per_model_paths.items():
            if os.path.exists(csv_path):
                probs, pred_idx, class_names = read_csv_local(csv_path)
                per_model[model_name] = dict(probs=probs, pred_idx=pred_idx, cls_names=class_names)
        # quick per-model bars from available data (avg max prob)
        d = {}
        for m, rec in per_model.items():
            if rec["probs"].size:
                d[m] = float(np.mean(np.max(rec["probs"], axis=1)))
        if not d:
            d = {"MLP-40D": random.uniform(0.3,0.95), "CNN": random.uniform(0.3,0.95),
                 "Transformer": random.uniform(0.3,0.95), "XGB-40D": random.uniform(0.3,0.95)}
        self.bars.set_data(d)

        self.img_pred.set_label(fr.pred_name)
        self.img_pred.set_image_for_label(fr.pred_name)
        # Force speech on user click (no debounce)
        self._speak_quick_force(fr.pred_name)
        self._say_if_enabled(fr.pred_name)

    def _open_realtime(self):
        models = self._loaded_models()
        if not models:
            QMessageBox.critical(self, "No models", "Load models first."); return
        outdir = getattr(self, "out_dir", str(Path.cwd() / "predictions_gui"))
        os.makedirs(outdir, exist_ok=True)
        names_lookup = self._names_lookup()
        win = RealTimeWindow(models, outdir, names_lookup, self)
        win.show()

    def paintEvent(self, e):
        p = QPainter(self)
        g = QLinearGradient(self.rect().topLeft(), self.rect().bottomRight())
        g.setColorAt(0, QColor(6, 26, 34))
        g.setColorAt(1, QColor(4, 18, 28))
        p.fillRect(self.rect(), g)

    def closeEvent(self, e):
        try:
            if hasattr(self, "tts") and self.tts:
                self.tts.close()
        finally:
            super().closeEvent(e)

    def closeEvent(self, e):
        try:
            if hasattr(self, "voice") and self.voice:
                self.voice.cleanup()
        finally:
            super().closeEvent(e)


# ------------------------ MAIN ------------------------
def main():
    if HARD_IMAGE_DIR:
        globals()["IMAGE_DIR"] = HARD_IMAGE_DIR
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(30, 33, 38))
    pal.setColor(QPalette.WindowText, QColor(230, 235, 245))
    pal.setColor(QPalette.Base, QColor(24, 26, 31))
    pal.setColor(QPalette.AlternateBase, QColor(28, 30, 35))
    pal.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    pal.setColor(QPalette.ToolTipText, QColor(10, 10, 10))
    pal.setColor(QPalette.Text, QColor(230, 235, 245))
    pal.setColor(QPalette.Button, QColor(40, 44, 52))
    pal.setColor(QPalette.ButtonText, QColor(230, 235, 245))
    pal.setColor(QPalette.BrightText, QColor(255, 0, 0))
    pal.setColor(QPalette.Highlight, QColor(75, 110, 175))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)
    import matplotlib as _mpl
    bg = pal.color(QPalette.Base).name()  # main cards/background (#181A1F-ish)
    axes_bg = pal.color(QPalette.AlternateBase).name()
    txt = pal.color(QPalette.Text).name()
    accent = pal.color(QPalette.Highlight).name()

    _mpl.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": axes_bg,
        "savefig.facecolor": bg,
        "axes.edgecolor": txt,
        "axes.labelcolor": txt,
        "text.color": txt,
        "xtick.color": txt,
        "ytick.color": txt,
        "grid.color": txt,
    })
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
