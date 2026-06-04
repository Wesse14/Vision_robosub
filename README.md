# async-processor

A lightweight `asyncio.Queue` based processing loop for underwater marker and ArUco navigation detection. The main path enhances frames, rectifies marker cutouts, detects navigation IDs with mask/grid matching plus OpenCV ArUco fallback, and logs processing FPS.

## Run

All commands should go through `uv`:

```sh
uv run python main.py
uv run pytest
uv run python -m src
```

Useful batch commands:

```sh
uv run python image_batch.py --pipeline full
uv run python image_batch.py --pipeline aruco
uv run python image_batch.py --pipeline marker
uv run python image_batch.py --pipeline enhance
```

`full` writes the root `navigation_signals.txt` summary. `aruco` keeps per-image inspection outputs, including the enhanced image and ArUco/mask-grid result images.

## Main Pipeline

```text
frames -> image-enhancer -> enhanced_frames -> marker-rectifier -> marker_cutouts -> aruco-detector -> frame-rate-logger
```

## Available Modules

### `ImageEnhancementModule`

Applies the underwater image enhancement pipeline to OpenCV-style BGR `uint8` color images. If the payload is a `VideoFrame`, frame metadata is preserved and only `.image` is replaced.

### `MarkerRectificationModule`

Detects the square marker region and warps it to a normalized cutout. It uses yellow/blue suppression masks internally so pipe/water colored regions interfere less with marker-edge detection.

With `debug=True`, marker debug images are written under the configured debug directory.

### `ArucoDetectionModule`

Detects navigation marker IDs from rectified marker cutouts. It first tries mask/grid matching and falls back to OpenCV ArUco detection when the mask/grid result is not strong enough.

### `FrameRateLoggerModule`

Logs average processing FPS for messages flowing through the configured queue.

## Minimal Module

```python
from src import BaseModule, Message, ModuleContext, RoutedMessage

class UppercaseModule(BaseModule[str]):
    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> RoutedMessage[str]:
        return RoutedMessage.from_payload('out', message.payload.upper())
```
