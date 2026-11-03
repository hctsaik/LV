"""Local, dependency-free Streamlit region selection component."""
from __future__ import annotations

import base64
import io
from pathlib import Path


def image_region_picker(image, *, boxes=(), selected_id=None, roi=None, key=None, allow_draw=True):
    import streamlit.components.v1 as components
    from PIL import Image
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            display = source.convert("RGB")
    else:
        display = image.convert("RGB")
    display.thumbnail((1600, 1600))
    buf = io.BytesIO()
    display.save(buf, format="JPEG", quality=88)
    component = components.declare_component("lv_region_picker", path=str(Path(__file__).with_name("region_picker")))
    return component(image="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                     boxes=list(boxes), selected_id=selected_id, roi=roi,
                     allow_draw=allow_draw, key=key, default=None)
