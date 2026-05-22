from __future__ import annotations

from dataclasses import replace
from typing import Literal

import cv2
import numpy as np

from ..messages import Message, RoutedMessage
from ..video import VideoFrame
from .base import BaseModule, ModuleContext

A_SHIFT = 0.0
B_SHIFT = 0.0
RED_STRENGTH = 25     #35
CLAHE_CLIP = 2.0
OMEGA = 0.75
T_MIN = 0.15
GAMMA = 1.1

EnhancementMode = Literal["underwater"] | None


def white_balance_underwater(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    a_float = a_channel.astype(np.float32)
    b_float = b_channel.astype(np.float32)
    a_float = a_float - (float(np.mean(a_float)) - 128.0) + A_SHIFT
    b_float = b_float - (float(np.mean(b_float)) - 128.0) + B_SHIFT
    a_balanced = np.clip(a_float, 0, 255).astype(np.uint8)
    b_balanced = np.clip(b_float, 0, 255).astype(np.uint8)
    return cv2.cvtColor(
        cv2.merge([l_channel, a_balanced, b_balanced]),
        cv2.COLOR_LAB2BGR,
    )


def restore_red_channel(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image)
    boosted = cv2.equalizeHist(red)
    strength = RED_STRENGTH / 100.0
    restored = cv2.addWeighted(red, 1.0 - strength, boosted, strength, 0.0)
    return cv2.merge([blue, green, restored])


def clahe_enhance_bgr(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(CLAHE_CLIP, 0.1), tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    return cv2.cvtColor(
        cv2.merge([enhanced_l, a_channel, b_channel]),
        cv2.COLOR_LAB2BGR,
    )


def dehaze_underwater(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    atmospheric_light = max(float(np.percentile(gray, 95)), 1.0)
    transmission = 1.0 - OMEGA * (gray / atmospheric_light)
    transmission = np.clip(transmission, T_MIN, 1.0)
    transmission_3c = cv2.merge([transmission, transmission, transmission])
    scene = (image.astype(np.float32) - atmospheric_light) / transmission_3c
    scene = scene + atmospheric_light
    return np.clip(scene, 0, 255).astype(np.uint8)


def sharpen_underwater(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.addWeighted(image, 1.2, blurred, -0.2, 0)


def gamma_correct(image: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def enhance_underwater(image: np.ndarray) -> np.ndarray:
    enhanced = white_balance_underwater(image)
    enhanced = restore_red_channel(enhanced)
    enhanced = clahe_enhance_bgr(enhanced)
    enhanced = dehaze_underwater(enhanced)
    enhanced = sharpen_underwater(enhanced)
    enhanced = gamma_correct(enhanced)
    return enhanced


def apply_enhancement(image: np.ndarray, mode: EnhancementMode) -> np.ndarray:
    validate_color_image(image)
    if mode is None:
        return image.copy()
    if mode == "underwater":
        return enhance_underwater(image)
    raise ValueError(f"Unsupported enhancement mode: {mode}")


def validate_color_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("Image payload must be a numpy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image payload must be a BGR color image with 3 channels.")
    if image.dtype != np.uint8:
        raise ValueError("Image payload must use uint8 dtype.")


class ImageEnhancementModule(BaseModule[VideoFrame | np.ndarray]):
    def __init__(
        self,
        name: str,
        input_queue: str,
        output_queue: str,
        *,
        mode: EnhancementMode = "underwater",
    ) -> None:
        if not output_queue:
            raise ValueError("Module output_queue cannot be empty.")
        super().__init__(name=name, input_queue=input_queue)
        self.output_queue = output_queue
        self.mode = mode

    async def process(
        self,
        message: Message[VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[VideoFrame | np.ndarray]:
        payload = message.payload
        if isinstance(payload, VideoFrame):
            enhanced_image = apply_enhancement(payload.image, self.mode)
            enhanced_payload: VideoFrame | np.ndarray = replace(
                payload,
                image=enhanced_image,
            )
        else:
            enhanced_payload = apply_enhancement(payload, self.mode)

        return RoutedMessage(
            destination=self.output_queue,
            message=Message(
                payload=enhanced_payload,
                metadata=message.metadata,
            ),
        )
