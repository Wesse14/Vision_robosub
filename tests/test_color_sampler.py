from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from color_sampler import (
    WebSamplingState,
    REQUIRED_SAMPLE_COLUMNS,
    load_samples,
    sample_patch_rows,
    save_samples,
    train_and_save_model,
    train_binary_gmm_model,
    make_web_handler,
    parse_args,
    should_use_web,
)


def make_test_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_bgr = np.zeros((5, 5, 3), dtype=np.uint8)
    for y in range(5):
        for x in range(5):
            image_bgr[y, x] = [x * 10, y * 20, 100]
    import cv2

    return (
        image_bgr,
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB),
    )


def make_samples_frame() -> pd.DataFrame:
    rows = []
    for idx in range(4):
        rows.append(
            {
                "label": "pipeline",
                "x": idx,
                "y": 0,
                "r": 10 + idx,
                "g": 120,
                "b": 20,
                "b_bgr": 20,
                "g_bgr": 120,
                "r_bgr": 10 + idx,
                "L": 120 + idx,
                "a": 100,
                "b_lab": 140,
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_SAMPLE_COLUMNS)


def test_sample_patch_rows_records_rgb_bgr_lab_values() -> None:
    image_bgr, image_rgb, image_lab = make_test_images()

    rows = sample_patch_rows(
        image_bgr,
        image_rgb,
        image_lab,
        x=2,
        y=2,
        brush_radius=1,
        label="pipeline",
    )

    assert len(rows) == 5
    center = next(row for row in rows if row["x"] == 2 and row["y"] == 2)
    assert center["label"] == "pipeline"
    assert center["b_bgr"] == int(image_bgr[2, 2, 0])
    assert center["g_bgr"] == int(image_bgr[2, 2, 1])
    assert center["r_bgr"] == int(image_bgr[2, 2, 2])
    assert center["r"] == int(image_rgb[2, 2, 0])
    assert center["L"] == int(image_lab[2, 2, 0])
    assert center["a"] == int(image_lab[2, 2, 1])
    assert center["b_lab"] == int(image_lab[2, 2, 2])


def test_save_and_load_samples_preserves_required_columns(tmp_path) -> None:
    path = tmp_path / "samples.csv"
    samples = make_samples_frame().to_dict("records")

    save_samples(samples, path)
    loaded = load_samples(path)

    assert list(loaded.columns) == list(REQUIRED_SAMPLE_COLUMNS)
    assert len(loaded) == len(samples)


def test_train_and_save_model_writes_runtime_gmm_dict(tmp_path) -> None:
    samples_path = tmp_path / "samples.csv"
    model_path = tmp_path / "color_classifier_gmm.joblib"
    make_samples_frame().to_csv(samples_path, index=False)

    image_bgr, _, _ = make_test_images()

    train_and_save_model(samples_path, model_path, image_bgr=image_bgr, n_components=3)

    model = joblib.load(model_path)
    assert set(model) == {
        "query_gmm",
        "non_query_gmm",
        "query_prior",
        "non_query_prior",
        "n_components",
        "covariance_type",
    }
    assert model["query_gmm"].n_components == 3
    assert model["non_query_gmm"].n_components == 3
    assert model["query_prior"] == 4 / 25
    assert model["non_query_prior"] == 21 / 25


def test_train_binary_gmm_requires_foreground_samples() -> None:
    image_bgr, _, _ = make_test_images()
    samples = make_samples_frame()
    samples = samples[samples["label"] != "pipeline"]

    with pytest.raises(ValueError, match="No foreground samples"):
        train_binary_gmm_model(samples, image_bgr=image_bgr, foreground_label="pipeline")


def test_train_binary_gmm_requires_background_pixels() -> None:
    image_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    rows = []
    for y in range(2):
        for x in range(2):
            rows.append(
                {
                    "label": "pipeline",
                    "x": x,
                    "y": y,
                    "r": 0,
                    "g": 0,
                    "b": 0,
                    "b_bgr": 0,
                    "g_bgr": 0,
                    "r_bgr": 0,
                    "L": 0,
                    "a": 0,
                    "b_lab": 0,
                }
            )
    samples = pd.DataFrame(rows, columns=REQUIRED_SAMPLE_COLUMNS)

    with pytest.raises(ValueError, match="No background pixels"):
        train_binary_gmm_model(samples, image_bgr=image_bgr, foreground_label="pipeline")


def test_train_binary_gmm_uses_explicit_background_sample_rows() -> None:
    image_bgr, _, _ = make_test_images()
    rows = make_samples_frame().to_dict("records")
    for idx in range(6):
        rows.append(
            {
                "label": "background",
                "x": idx,
                "y": 4,
                "r": 200,
                "g": 20,
                "b": 20,
                "b_bgr": 20,
                "g_bgr": 20,
                "r_bgr": 200,
                "L": 40 + idx,
                "a": 150,
                "b_lab": 120,
            }
        )
    samples = pd.DataFrame(rows, columns=REQUIRED_SAMPLE_COLUMNS)

    model = train_binary_gmm_model(
        samples,
        image_bgr=image_bgr,
        foreground_label="pipeline",
        background_label="background",
        n_components=3,
    )

    assert model["query_prior"] == 4 / 10
    assert model["non_query_prior"] == 6 / 10



def test_should_use_web_when_requested() -> None:
    args = parse_args(["--image", "image.png", "--web"])

    assert should_use_web(args) is True


def test_web_handler_samples_pixels_and_saves_csv(tmp_path) -> None:
    import json
    from http import HTTPStatus
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request, urlopen

    image_bgr, image_rgb, image_lab = make_test_images()
    state = WebSamplingState(
        image_bgr=image_bgr,
        image_rgb=image_rgb,
        image_lab=image_lab,
        samples=[],
        labels=("pipeline",),
        samples_path=tmp_path / "samples.csv",
        model_path=tmp_path / "model.joblib",
        foreground_label="pipeline",
        n_components=1,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_web_handler(state))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        sample_payload = json.dumps(
            {"x": 2, "y": 2, "radius": 1, "label": "pipeline"}
        ).encode("utf-8")
        request = Request(
            f"{base_url}/sample",
            data=sample_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
        assert response.status == HTTPStatus.OK
        assert data["samples"] == 5
        assert len(state.samples) == 5

        request = Request(f"{base_url}/save", data=b"{}", method="POST")
        with urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
        assert response.status == HTTPStatus.OK
        assert data["samples"] == 5
        assert state.samples_path.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
