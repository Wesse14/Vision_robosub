from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

DEFAULT_FOREGROUND_LABEL = "pipeline"
DEFAULT_BACKGROUND_LABEL = "background"
DEFAULT_LABELS = (DEFAULT_FOREGROUND_LABEL, DEFAULT_BACKGROUND_LABEL)
REQUIRED_SAMPLE_COLUMNS = (
    "label",
    "x",
    "y",
    "r",
    "g",
    "b",
    "b_bgr",
    "g_bgr",
    "r_bgr",
    "L",
    "a",
    "b_lab",
)
DEFAULT_ENHANCED_RESULTS_DIR = Path("data/test_results")
DEFAULT_ENHANCED_IMAGE_NAME = "01_enhanced.png"


@dataclass
class SamplingState:
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    image_lab: np.ndarray
    display: np.ndarray
    samples: list[dict[str, int | str]]
    labels: tuple[str, ...]
    current_label: str
    brush_radius: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively sample image colors and train a GMM mask model.")
    parser.add_argument(
        "--image",
        type=Path,
        help="Image to sample from. Defaults to the newest 01_enhanced.png in data/test_results.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_ENHANCED_RESULTS_DIR,
        type=Path,
        help="Directory searched for enhanced batch outputs when --image is omitted.",
    )
    parser.add_argument(
        "--result-name",
        help="Name of a folder under data/test_results to annotate, using its 01_enhanced.png.",
    )
    parser.add_argument(
        "--samples",
        default=Path("data/color_samples.csv"),
        type=Path,
        help="CSV path for sampled pixels.",
    )
    parser.add_argument(
        "--fresh-samples",
        action="store_true",
        help="Start with an empty sample set instead of appending to the existing CSV.",
    )
    parser.add_argument(
        "--model",
        default=Path("data/color_classifier_gmm.joblib"),
        type=Path,
        help="Output joblib path for the binary GMM model.",
    )
    parser.add_argument(
        "--foreground-label",
        default=DEFAULT_FOREGROUND_LABEL,
        help="Label to train as foreground/query class.",
    )
    parser.add_argument(
        "--background-label",
        default=DEFAULT_BACKGROUND_LABEL,
        help="Label to train as background/non-query class.",
    )
    parser.add_argument(
        "--preview-mask",
        type=Path,
        help="Output path for a preview mask after training.",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Posterior threshold used for the preview mask.",
    )
    parser.add_argument(
        "--components",
        default=3,
        type=int,
        help="Maximum Gaussian components per class.",
    )
    parser.add_argument(
        "--brush-radius",
        default=5,
        type=int,
        help="Initial brush radius in pixels.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Brush labels assigned to number keys. Defaults to foreground and background.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Use the browser-based sampler instead of OpenCV GUI.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --web mode.",
    )
    parser.add_argument(
        "--port",
        default=8765,
        type=int,
        help="Port for --web mode.",
    )
    return parser.parse_args(argv)


def find_latest_enhanced_image(results_dir: Path) -> Path:
    candidates = list(results_dir.glob(f"*/{DEFAULT_ENHANCED_IMAGE_NAME}"))
    if not candidates:
        raise FileNotFoundError(
            f"No {DEFAULT_ENHANCED_IMAGE_NAME} files found under {results_dir}. "
            "Run `uv run python image_batch.py --pipeline enhance` or pass --image."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_image_path(args: argparse.Namespace) -> Path:
    if args.image is not None:
        return args.image
    if args.result_name is not None:
        image_path = args.results_dir / args.result_name / DEFAULT_ENHANCED_IMAGE_NAME
        if not image_path.exists():
            raise FileNotFoundError(f"Could not find enhanced result image: {image_path}")
        return image_path
    image_path = find_latest_enhanced_image(args.results_dir)
    print(f"Using latest enhanced image: {image_path}")
    return image_path


def resolve_preview_mask_path(args: argparse.Namespace, image_path: Path) -> Path:
    if args.preview_mask is not None:
        return args.preview_mask
    if image_path.name == DEFAULT_ENHANCED_IMAGE_NAME:
        return image_path.with_name("04_color_mask_preview.png")
    return Path("data/color_mask_preview.png")


def save_training_preview(
    model: dict[str, object],
    image_bgr: np.ndarray,
    preview_mask_path: Path,
    *,
    threshold: float,
) -> None:
    save_preview_mask(model, image_bgr, preview_mask_path, threshold=threshold)
    if preview_mask_path != Path("data/color_mask_preview.png"):
        save_preview_mask(
            model,
            image_bgr,
            Path("data/color_mask_preview.png"),
            threshold=threshold,
        )


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a BGR color image.")
    return image


def sample_patch_rows(
    image_bgr: np.ndarray,
    image_rgb: np.ndarray,
    image_lab: np.ndarray,
    *,
    x: int,
    y: int,
    brush_radius: int,
    label: str,
) -> list[dict[str, int | str]]:
    if brush_radius < 1:
        raise ValueError("brush_radius must be at least 1.")
    height, width = image_bgr.shape[:2]
    x0, x1 = max(0, x - brush_radius), min(width, x + brush_radius + 1)
    y0, y1 = max(0, y - brush_radius), min(height, y + brush_radius + 1)

    rows: list[dict[str, int | str]] = []
    for py in range(y0, y1):
        for px in range(x0, x1):
            if (px - x) ** 2 + (py - y) ** 2 > brush_radius**2:
                continue
            bgr = image_bgr[py, px]
            rgb = image_rgb[py, px]
            lab = image_lab[py, px]
            rows.append(
                {
                    "label": label,
                    "x": int(px),
                    "y": int(py),
                    "r": int(rgb[0]),
                    "g": int(rgb[1]),
                    "b": int(rgb[2]),
                    "b_bgr": int(bgr[0]),
                    "g_bgr": int(bgr[1]),
                    "r_bgr": int(bgr[2]),
                    "L": int(lab[0]),
                    "a": int(lab[1]),
                    "b_lab": int(lab[2]),
                }
            )
    return rows


def save_samples(samples: list[dict[str, int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples, columns=REQUIRED_SAMPLE_COLUMNS).to_csv(path, index=False)


def load_samples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_SAMPLE_COLUMNS) - set(df.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Sample CSV is missing column(s): {missing_columns}")
    return df


def load_existing_sample_rows(path: Path, *, fresh: bool) -> list[dict[str, int | str]]:
    if fresh or not path.exists():
        return []
    return load_samples(path).to_dict("records")


def _fit_gmm(samples_lab: np.ndarray, n_components: int, covariance_type: str) -> GaussianMixture:
    if len(samples_lab) == 0:
        raise ValueError("Cannot fit a GMM without samples.")
    component_count = min(n_components, len(samples_lab))
    return GaussianMixture(
        n_components=component_count,
        covariance_type=covariance_type,
        random_state=0,
    ).fit(samples_lab)


def sample_mask(samples: pd.DataFrame, image_shape: tuple[int, int], label: str) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=bool)
    labeled = samples[samples["label"] == label]
    if labeled.empty:
        return mask

    xs = labeled["x"].to_numpy(dtype=np.int64)
    ys = labeled["y"].to_numpy(dtype=np.int64)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    mask[ys[valid], xs[valid]] = True
    return mask


def lab_sample_values(samples: pd.DataFrame, label: str) -> np.ndarray:
    labeled = samples[samples["label"] == label]
    if labeled.empty:
        return np.empty((0, 3), dtype=np.float64)
    return labeled[["L", "a", "b_lab"]].to_numpy(dtype=np.float64)


def posterior_from_model(model: dict[str, object], image_bgr: np.ndarray) -> np.ndarray:
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lab_flat = image_lab.reshape(-1, 3)
    query_gmm = model["query_gmm"]
    non_query_gmm = model["non_query_gmm"]
    query_prior = float(model["query_prior"])
    non_query_prior = float(model["non_query_prior"])

    log_prob_query = query_gmm.score_samples(lab_flat)
    log_prob_non = non_query_gmm.score_samples(lab_flat)
    p_lab_given_query = np.exp(log_prob_query)
    p_lab_given_non = np.exp(log_prob_non)
    p_lab = p_lab_given_query * query_prior + p_lab_given_non * non_query_prior
    p_lab = np.maximum(p_lab, np.finfo(float).eps)
    posterior = (p_lab_given_query * query_prior) / p_lab
    return posterior.reshape(image_bgr.shape[:2])


def save_preview_mask(
    model: dict[str, object],
    image_bgr: np.ndarray,
    path: Path,
    *,
    threshold: float,
) -> None:
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    mask[posterior_from_model(model, image_bgr) > threshold] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write preview mask: {path}")


def train_binary_gmm_model(
    samples: pd.DataFrame,
    *,
    image_bgr: np.ndarray,
    foreground_label: str = DEFAULT_FOREGROUND_LABEL,
    background_label: str = DEFAULT_BACKGROUND_LABEL,
    n_components: int = 3,
    covariance_type: str = "full",
) -> dict[str, object]:
    if n_components <= 0:
        raise ValueError("n_components must be greater than zero.")

    query_pixels = lab_sample_values(samples, foreground_label)
    if len(query_pixels) == 0:
        raise ValueError(f"No foreground samples found for label: {foreground_label}")

    explicit_background_pixels = lab_sample_values(samples, background_label)
    if len(explicit_background_pixels) > 0:
        non_query_pixels = explicit_background_pixels
        total = len(query_pixels) + len(non_query_pixels)
        return {
            "query_gmm": _fit_gmm(query_pixels, n_components, covariance_type),
            "non_query_gmm": _fit_gmm(non_query_pixels, n_components, covariance_type),
            "query_prior": len(query_pixels) / total,
            "non_query_prior": len(non_query_pixels) / total,
            "n_components": n_components,
            "covariance_type": covariance_type,
        }

    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    height, width = image_bgr.shape[:2]
    foreground_mask = sample_mask(samples, (height, width), foreground_label)
    background_mask = ~foreground_mask
    if not np.any(background_mask):
        raise ValueError("No background pixels remain outside the marked foreground.")

    query_pixels = image_lab[foreground_mask].reshape(-1, 3).astype(np.float64)
    non_query_pixels = image_lab[background_mask].reshape(-1, 3).astype(np.float64)
    total = len(query_pixels) + len(non_query_pixels)

    return {
        "query_gmm": _fit_gmm(query_pixels, n_components, covariance_type),
        "non_query_gmm": _fit_gmm(non_query_pixels, n_components, covariance_type),
        "query_prior": len(query_pixels) / total,
        "non_query_prior": len(non_query_pixels) / total,
        "n_components": n_components,
        "covariance_type": covariance_type,
    }


def train_and_save_model(
    samples_csv: Path,
    model_path: Path,
    *,
    image_bgr: np.ndarray,
    foreground_label: str = DEFAULT_FOREGROUND_LABEL,
    background_label: str = DEFAULT_BACKGROUND_LABEL,
    n_components: int = 3,
    covariance_type: str = "full",
) -> dict[str, object]:
    samples = load_samples(samples_csv)
    model = train_binary_gmm_model(
        samples,
        image_bgr=image_bgr,
        foreground_label=foreground_label,
        background_label=background_label,
        n_components=n_components,
        covariance_type=covariance_type,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def print_controls(labels: Sequence[str]) -> None:
    label_lines = "\n".join(
        f"  {idx + 1} = label {label}" for idx, label in enumerate(labels[:9])
    )
    print(
        f"""
Controls:
  left drag = paint segmentation samples
{label_lines}
  + / - = brush size
  s = save CSV
  t = train model and save preview mask
  q = quit
"""
    )


def label_color(label: str, foreground_label: str, background_label: str) -> tuple[int, int, int]:
    if label == foreground_label:
        return (0, 255, 255)
    if label == background_label:
        return (255, 80, 80)
    return (80, 255, 80)



WEB_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Color Sampler</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #101214; color: #f3f3f3; }
    header { display: flex; gap: 12px; align-items: center; padding: 10px 12px; background: #1b1f24; position: sticky; top: 0; z-index: 2; }
    button, input { font: inherit; }
    button { border: 1px solid #59616b; background: #252b32; color: #f3f3f3; padding: 6px 10px; border-radius: 4px; cursor: pointer; }
    button.active { border-color: #ffd166; background: #4b3f22; }
    label { display: inline-flex; gap: 6px; align-items: center; }
    #wrap { padding: 12px; }
    #canvas { max-width: 100%; height: auto; image-rendering: auto; cursor: crosshair; border: 1px solid #363d45; }
    #status { margin-left: auto; color: #cfd7df; }
  </style>
</head>
<body>
<header>
  <div id="labels"></div>
  <label>Brush <input id="brush" type="number" min="1" value="5" style="width: 72px"></label>
  <button id="save">Save CSV</button>
  <button id="train">Train model</button>
  <button id="reset">Reset samples</button>
  <span id="status">0 samples</span>
</header>
<div id="wrap"><canvas id="canvas"></canvas></div>
<script>
const config = __CONFIG__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const brushInput = document.getElementById('brush');
const statusEl = document.getElementById('status');
let currentLabel = config.foregroundLabel;
let image = new Image();
let drawing = false;
let pending = Promise.resolve();

function setStatus(text) { statusEl.textContent = text; }
function brushRadius() { return Math.max(1, parseInt(brushInput.value || '1', 10)); }
function canvasPoint(evt) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return { x: Math.round((evt.clientX - rect.left) * sx), y: Math.round((evt.clientY - rect.top) * sy) };
}
async function postJson(path, body) {
  const response = await fetch(path, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {}) });
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || response.statusText);
  return data;
}
function sampleAt(evt) {
  const p = canvasPoint(evt);
  const radius = brushRadius();
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.strokeStyle = currentLabel === config.backgroundLabel ? '#50a7ff' : '#ffd166';
  ctx.lineWidth = 2;
  ctx.stroke();
  pending = pending.then(() => postJson('/sample', { x: p.x, y: p.y, radius, label: currentLabel }))
    .then(data => setStatus(`${data.samples} samples`))
    .catch(err => setStatus(err.message));
}
function buildLabels() {
  const labels = document.getElementById('labels');
  config.labels.forEach((label, idx) => {
    const btn = document.createElement('button');
    btn.textContent = `${idx + 1}: ${label}`;
    btn.onclick = () => {
      currentLabel = label;
      document.querySelectorAll('#labels button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    };
    if (label === currentLabel) btn.classList.add('active');
    labels.appendChild(btn);
  });
}
canvas.addEventListener('mousedown', evt => { drawing = true; sampleAt(evt); });
canvas.addEventListener('mousemove', evt => { if (drawing) sampleAt(evt); });
window.addEventListener('mouseup', () => { drawing = false; });
window.addEventListener('keydown', evt => {
  const idx = Number(evt.key) - 1;
  if (idx >= 0 && idx < config.labels.length) document.querySelectorAll('#labels button')[idx].click();
  if (evt.key === '+' || evt.key === '=') brushInput.value = brushRadius() + 1;
  if (evt.key === '-' || evt.key === '_') brushInput.value = Math.max(1, brushRadius() - 1);
});
document.getElementById('save').onclick = async () => {
  try { const data = await postJson('/save'); setStatus(`saved ${data.samples} samples`); } catch (err) { setStatus(err.message); }
};
document.getElementById('train').onclick = async () => {
  try { const data = await postJson('/train'); setStatus(`saved model: ${data.model}`); } catch (err) { setStatus(err.message); }
};
document.getElementById('reset').onclick = async () => {
  try { const data = await postJson('/reset'); ctx.drawImage(image, 0, 0); setStatus(`${data.samples} samples`); } catch (err) { setStatus(err.message); }
};
image.onload = () => { canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; ctx.drawImage(image, 0, 0); };
image.src = '/image';
buildLabels();
</script>
</body>
</html>
"""


@dataclass
class WebSamplingState:
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    image_lab: np.ndarray
    samples: list[dict[str, int | str]]
    labels: tuple[str, ...]
    samples_path: Path
    model_path: Path
    preview_mask_path: Path = Path("data/color_mask_preview.png")
    foreground_label: str = DEFAULT_FOREGROUND_LABEL
    background_label: str = DEFAULT_BACKGROUND_LABEL
    n_components: int = 3
    threshold: float = 0.5


def make_web_handler(state: WebSamplingState) -> type[BaseHTTPRequestHandler]:
    class ColorSamplerHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/":
                config = {
                    "labels": state.labels,
                    "foregroundLabel": state.foreground_label,
                    "backgroundLabel": state.background_label,
                }
                html = WEB_HTML.replace("__CONFIG__", json.dumps(config))
                self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/image":
                ok, encoded = cv2.imencode(".png", state.image_bgr)
                if not ok:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "Could not encode image"})
                    return
                self._send(HTTPStatus.OK, encoded.tobytes(), "image/png")
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

        def do_POST(self) -> None:
            try:
                if self.path == "/sample":
                    payload = self._read_json()
                    rows = sample_patch_rows(
                        state.image_bgr,
                        state.image_rgb,
                        state.image_lab,
                        x=int(payload["x"]),
                        y=int(payload["y"]),
                        brush_radius=int(payload["radius"]),
                        label=str(payload["label"]),
                    )
                    state.samples.extend(rows)
                    self._send_json(HTTPStatus.OK, {"ok": True, "added": len(rows), "samples": len(state.samples)})
                    return
                if self.path == "/save":
                    save_samples(state.samples, state.samples_path)
                    self._send_json(HTTPStatus.OK, {"ok": True, "samples": len(state.samples), "samplesPath": str(state.samples_path)})
                    return
                if self.path == "/train":
                    save_samples(state.samples, state.samples_path)
                    model = train_and_save_model(
                        state.samples_path,
                        state.model_path,
                        image_bgr=state.image_bgr,
                        foreground_label=state.foreground_label,
                        background_label=state.background_label,
                        n_components=state.n_components,
                    )
                    save_training_preview(
                        model,
                        state.image_bgr,
                        state.preview_mask_path,
                        threshold=state.threshold,
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "samples": len(state.samples),
                            "model": str(state.model_path),
                            "previewMask": str(state.preview_mask_path),
                        },
                    )
                    return
                if self.path == "/reset":
                    state.samples.clear()
                    self._send_json(HTTPStatus.OK, {"ok": True, "samples": 0})
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})

    return ColorSamplerHandler


def run_web(args: argparse.Namespace) -> None:
    image_path = resolve_image_path(args)
    preview_mask_path = resolve_preview_mask_path(args, image_path)
    image_bgr = load_image(image_path)
    samples = load_existing_sample_rows(args.samples, fresh=args.fresh_samples)
    labels = tuple(args.labels)
    if not labels:
        raise ValueError("At least one label is required.")
    if args.foreground_label not in labels:
        raise ValueError("foreground-label must be one of the configured labels.")
    if args.background_label not in labels:
        raise ValueError("background-label must be one of the configured labels.")

    state = WebSamplingState(
        image_bgr=image_bgr,
        image_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        image_lab=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB),
        samples=samples,
        labels=labels,
        samples_path=args.samples,
        model_path=args.model,
        preview_mask_path=preview_mask_path,
        foreground_label=args.foreground_label,
        background_label=args.background_label,
        n_components=args.components,
        threshold=args.threshold,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_web_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"Browser color sampler running at {url}")
    print(f"Loaded {len(samples)} existing sample(s) from {args.samples}")
    print("Use SSH/VS Code port forwarding if this is a remote machine.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def should_use_web(args: argparse.Namespace) -> bool:
    return bool(args.web or (os.name != "nt" and not os.environ.get("DISPLAY")))


def run_gui(args: argparse.Namespace) -> None:
    image_path = resolve_image_path(args)
    preview_mask_path = resolve_preview_mask_path(args, image_path)
    image_bgr = load_image(image_path)
    samples = load_existing_sample_rows(args.samples, fresh=args.fresh_samples)
    labels = tuple(args.labels)
    if not labels:
        raise ValueError("At least one label is required.")
    if args.foreground_label not in labels:
        raise ValueError("foreground-label must be one of the configured labels.")
    if args.background_label not in labels:
        raise ValueError("background-label must be one of the configured labels.")

    state = SamplingState(
        image_bgr=image_bgr,
        image_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        image_lab=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB),
        display=image_bgr.copy(),
        samples=samples,
        labels=labels,
        current_label=args.foreground_label,
        brush_radius=max(1, args.brush_radius),
    )

    def mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del param
        drawing = event == cv2.EVENT_LBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        )
        if not drawing:
            return
        rows = sample_patch_rows(
            state.image_bgr,
            state.image_rgb,
            state.image_lab,
            x=x,
            y=y,
            brush_radius=state.brush_radius,
            label=state.current_label,
        )
        state.samples.extend(rows)
        color = label_color(
            state.current_label,
            args.foreground_label,
            args.background_label,
        )
        cv2.circle(state.display, (x, y), state.brush_radius, color, 1)
        cv2.imshow("color-sampler", state.display)

    cv2.namedWindow("color-sampler")
    cv2.setMouseCallback("color-sampler", mouse)
    print_controls(labels)
    print(f"loaded {len(samples)} existing sample(s) from {args.samples}")
    print("label:", state.current_label)

    while True:
        cv2.imshow("color-sampler", state.display)
        key = cv2.waitKey(20) & 0xFF

        if ord("1") <= key <= ord("9"):
            label_idx = key - ord("1")
            if label_idx < len(labels):
                state.current_label = labels[label_idx]
                print("label:", state.current_label)
        elif key in (ord("+"), ord("=")):
            state.brush_radius += 1
            print("brush:", state.brush_radius)
        elif key in (ord("-"), ord("_")):
            state.brush_radius = max(1, state.brush_radius - 1)
            print("brush:", state.brush_radius)
        elif key == ord("s"):
            save_samples(state.samples, args.samples)
            print(f"saved {len(state.samples)} samples to {args.samples}")
        elif key == ord("t"):
            try:
                save_samples(state.samples, args.samples)
                model = train_and_save_model(
                    args.samples,
                    args.model,
                    image_bgr=state.image_bgr,
                    foreground_label=args.foreground_label,
                    background_label=args.background_label,
                    n_components=args.components,
                )
                save_training_preview(
                    model,
                    state.image_bgr,
                    preview_mask_path,
                    threshold=args.threshold,
                )
                print(f"saved GMM model to {args.model}")
                print(f"saved preview mask to {preview_mask_path}")
            except ValueError as exc:
                print(f"could not train model: {exc}")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if should_use_web(args):
        run_web(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
