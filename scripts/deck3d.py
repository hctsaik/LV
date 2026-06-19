"""3D scatter via deck.gl inside a Streamlit Components-v2 component.

Gives what plotly 3D in Streamlit cannot:
  • HOVER a point  → its thumbnail appears instantly (client-side, NO rerun)
  • Shift+drag BOX → selected GLOBAL record indices return to Python (one rerun)

Fully offline (deck.gl UMD vendored at scripts/vendor/deck.gl.min.js). Validated
against Streamlit 1.58 `st.components.v2.component`. See scratch/deck3d_demo.py for
the standalone prototype this was lifted from.

Usage:
    import deck3d
    picked = deck3d.render_deck3d(points, key="...")   # -> list[int] global indices
where each point = {"index": int, "position": [x,y,z], "color": [r,g,b],
                    "thumb": "data:image/...", "label": str}.

`render_deck3d` returns [] when there's no new selection, or None if Components v2
is unavailable (caller should fall back to plotly).
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).parent
_VENDOR = _HERE / "vendor" / "deck.gl.min.js"
_CDN = "https://unpkg.com/deck.gl@9.0.38/dist.min.js"

_HTML = """
<div id="root" style="position:relative;width:100%;height:520px;">
  <canvas id="deck-canvas" style="width:100%;height:100%;display:block;"></canvas>
  <div id="hint" style="position:absolute;left:8px;top:8px;z-index:5;
       font:12px/1.4 system-ui,sans-serif;color:#ddd;background:rgba(0,0,0,.45);
       padding:4px 8px;border-radius:6px;pointer-events:none;">
    拖曳=旋轉 · 滾輪=縮放 · 滑點=看縮圖 · Shift+拖框=選取
  </div>
  <div id="tip" style="position:absolute;display:none;z-index:10;
       background:#111;border:1px solid #555;border-radius:8px;padding:6px;
       box-shadow:0 4px 16px rgba(0,0,0,.5);pointer-events:none;">
    <img id="tip-img" width="96" height="96"
         style="display:block;object-fit:contain;border-radius:4px;background:#222;"/>
    <div id="tip-label" style="font:12px system-ui,sans-serif;color:#eee;
         text-align:center;margin-top:4px;"></div>
  </div>
  <div id="selbox" style="position:absolute;display:none;z-index:8;
       border:1.5px dashed #4fc3f7;background:rgba(79,195,247,.15);
       pointer-events:none;"></div>
  <div id="status" style="position:absolute;right:8px;bottom:8px;z-index:5;
       font:11px system-ui,sans-serif;color:#9fe;background:rgba(0,0,0,.45);
       padding:3px 7px;border-radius:6px;pointer-events:none;">deck: loading…</div>
</div>
"""

_CSS = """
:host { display:block; width:100%; }
#root { background:#0d1117; border-radius:10px; overflow:hidden; }
"""

_JS_TEMPLATE = r"""
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const $ = (sel) => parentElement.querySelector(sel);
  const root = $('#root'), canvas = $('#deck-canvas');
  const tip = $('#tip'), tipImg = $('#tip-img'), tipLab = $('#tip-label');
  const selbox = $('#selbox'), status = $('#status');
  const points = (data && data.points) ? data.points : [];

  const DECK_SRC = __DECK_SRC__;
  const CDN = __CDN_URL__;
  function haveDeck() { return typeof window !== 'undefined' && window.deck && window.deck.Deck; }
  function ensureDeck() {
    return new Promise((resolve, reject) => {
      if (haveDeck()) return resolve();
      if (DECK_SRC && DECK_SRC.length > 0) {
        try { const s = document.createElement('script'); s.textContent = DECK_SRC;
              document.head.appendChild(s); } catch (e) { return reject(e); }
        let n = 0; const iv = setInterval(() => {
          if (haveDeck()) { clearInterval(iv); resolve(); }
          else if (++n > 50) { clearInterval(iv); reject(new Error('deck global missing')); }
        }, 20);
      } else {
        const s = document.createElement('script'); s.src = CDN; s.async = true;
        s.onload = () => haveDeck() ? resolve() : reject(new Error('deck missing after CDN'));
        s.onerror = () => reject(new Error('CDN failed')); document.head.appendChild(s);
      }
    });
  }

  let deckInstance = null; const cleanupFns = [];
  function start() {
    const D = window.deck;
    status.textContent = 'deck v' + (D.VERSION || '?') + ' · ' + points.length + ' 點';
    const layer = new D.ScatterplotLayer({
      id: 'pts', data: points, pickable: true,
      getPosition: d => d.position, getFillColor: d => d.color,
      getRadius: 2.2, radiusUnits: 'common', radiusMinPixels: 3, radiusMaxPixels: 16,
    });
    deckInstance = new D.Deck({
      canvas: canvas,
      views: new D.OrbitView({ orbitAxis: 'Y', fovy: 50 }),
      initialViewState: { target: [0, 0, 0], rotationX: 25, rotationOrbit: 30, zoom: 2.6 },
      controller: { dragRotate: true, scrollZoom: true, dragPan: true },
      layers: [layer],
      onHover: (info) => {
        if (info && info.object && info.object.thumb) {
          tipImg.src = info.object.thumb;
          tipLab.textContent = (info.object.label || '') + ' (#' + info.object.index + ')';
          const rect = root.getBoundingClientRect();
          tip.style.left = Math.min(info.x + 14, rect.width - 116) + 'px';
          tip.style.top  = Math.min(info.y + 14, rect.height - 130) + 'px';
          tip.style.display = 'block';
        } else { tip.style.display = 'none'; }
      },
    });

    let dragging = false, sx = 0, sy = 0;
    const onDown = (e) => {
      if (!e.shiftKey) return;
      dragging = true; const rect = canvas.getBoundingClientRect();
      sx = e.clientX - rect.left; sy = e.clientY - rect.top;
      selbox.style.left = sx+'px'; selbox.style.top = sy+'px';
      selbox.style.width = '0px'; selbox.style.height = '0px'; selbox.style.display = 'block';
      e.preventDefault(); e.stopPropagation();
    };
    const onMove = (e) => {
      if (!dragging) return; const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      selbox.style.left = Math.min(sx,cx)+'px'; selbox.style.top = Math.min(sy,cy)+'px';
      selbox.style.width = Math.abs(cx-sx)+'px'; selbox.style.height = Math.abs(cy-sy)+'px';
      e.preventDefault(); e.stopPropagation();
    };
    const onUp = (e) => {
      if (!dragging) return; dragging = false; selbox.style.display = 'none';
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      const x = Math.round(Math.min(sx,cx)), y = Math.round(Math.min(sy,cy));
      const w = Math.round(Math.abs(cx-sx)), h = Math.round(Math.abs(cy-sy));
      if (w < 3 || h < 3) return;
      let picked = [];
      try { picked = deckInstance.pickObjects({ x, y, width: w, height: h }); } catch (err) {}
      const idx = picked.map(p => p.object && p.object.index).filter(v => v !== undefined);
      setTriggerValue('selection', Array.from(new Set(idx)).sort((a,b)=>a-b));
      e.preventDefault(); e.stopPropagation();
    };
    canvas.addEventListener('pointerdown', onDown, true);
    window.addEventListener('pointermove', onMove, true);
    window.addEventListener('pointerup', onUp, true);
    cleanupFns.push(() => canvas.removeEventListener('pointerdown', onDown, true));
    cleanupFns.push(() => window.removeEventListener('pointermove', onMove, true));
    cleanupFns.push(() => window.removeEventListener('pointerup', onUp, true));
  }
  ensureDeck().then(start).catch((err) => { status.textContent = 'deck 失敗: ' + err.message; });
  return () => {
    try { if (deckInstance) deckInstance.finalize(); } catch (e) {}
    cleanupFns.forEach(fn => { try { fn(); } catch (e) {} });
  };
}
"""


@st.cache_resource
def _component():
    """Register the component once (cached across reruns; deck.gl read once)."""
    src = _VENDOR.read_text(encoding="utf-8", errors="ignore") if _VENDOR.exists() else ""
    js = (_JS_TEMPLATE.replace("__DECK_SRC__", json.dumps(src))
                      .replace("__CDN_URL__", json.dumps(_CDN)))
    return st.components.v2.component("lv_deck3d", html=_HTML, css=_CSS, js=js,
                                     isolate_styles=True)


def available() -> bool:
    return hasattr(getattr(st, "components", None), "v2")


def render_deck3d(points: list[dict], *, key: str, height: int = 540):
    """Render the 3D scatter; return list[int] of selected global indices (or [])."""
    if not available():
        return None
    comp = _component()
    result = comp(data={"points": points}, height=height + 24, key=key,
                  on_selection_change=lambda: None)
    sel = getattr(result, "selection", None)
    return [int(i) for i in sel] if sel else []
