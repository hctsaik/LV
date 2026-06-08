from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk

_GOAL_DIR = Path(__file__).parent / "goal"
_CANVAS_W, _CANVAS_H = 480, 360
_DOT_R = 6


def _paste_centered(reference: Image.Image, target: Image.Image, cx: int, cy: int) -> Image.Image:
    result = target.copy().convert("RGBA")
    ref = reference.convert("RGBA")
    x = cx - ref.width // 2
    y = cy - ref.height // 2
    result.paste(ref, (x, y), mask=ref)
    return result


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Region Synthesis")
        self.root.resizable(False, False)

        self._ref_img: Image.Image | None = None
        self._tgt_img: Image.Image | None = None
        self._result:  Image.Image | None = None

        # display transform for target canvas (original → canvas coords)
        self._tgt_scale = 1.0
        self._tgt_off = (0, 0)

        # keep tk image refs alive
        self._tk_tgt:    ImageTk.PhotoImage | None = None
        self._tk_result: ImageTk.PhotoImage | None = None

        self._build()
        self._x_var.trace_add("write", self._on_xy_typed)
        self._y_var.trace_add("write", self._on_xy_typed)

    # ── UI construction ──────────────────────────────────────────────

    def _build(self) -> None:
        P = {"padx": 6, "pady": 3}

        # ── file pickers ────────────────────────────────────────────
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)

        self._ref_var = tk.StringVar()
        self._tgt_var = tk.StringVar()
        self._x_var   = tk.StringVar(value="0")
        self._y_var   = tk.StringVar(value="0")

        def file_row(parent, label, var, cmd, r):
            tk.Label(parent, text=label, width=16, anchor="w").grid(row=r, column=0, **P)
            tk.Entry(parent, textvariable=var, width=40).grid(row=r, column=1, **P)
            tk.Button(parent, text="開啟", width=6, command=cmd).grid(row=r, column=2, **P)

        file_row(top, "Reference Image", self._ref_var, self._browse_ref, 0)
        file_row(top, "Target Image",    self._tgt_var, self._browse_tgt, 1)

        # ── x / y inputs ────────────────────────────────────────────
        xy = tk.Frame(self.root)
        xy.pack(fill="x", padx=8)
        tk.Label(xy, text="中心點 X").grid(row=0, column=0, **P)
        tk.Entry(xy, textvariable=self._x_var, width=7).grid(row=0, column=1, **P)
        tk.Label(xy, text="Y").grid(row=0, column=2, **P)
        tk.Entry(xy, textvariable=self._y_var, width=7).grid(row=0, column=3, **P)

        # ── dual canvas ─────────────────────────────────────────────
        canvases = tk.Frame(self.root)
        canvases.pack(padx=8, pady=6)

        def labeled_canvas(parent, title, col):
            tk.Label(parent, text=title, font=("Arial", 10, "bold")).grid(row=0, column=col, pady=(0, 2))
            c = tk.Canvas(parent, width=_CANVAS_W, height=_CANVAS_H, bg="#222",
                          highlightthickness=1, highlightbackground="#555")
            c.grid(row=1, column=col, padx=4)
            return c

        self._canvas_tgt    = labeled_canvas(canvases, "Target（點擊設定中心點）", 0)
        self._canvas_result = labeled_canvas(canvases, "結果預覽", 1)

        self._canvas_tgt.bind("<Button-1>", self._on_canvas_click)

        _placeholder = lambda c, t: c.create_text(
            _CANVAS_W // 2, _CANVAS_H // 2, text=t, fill="#666", font=("Arial", 11))
        _placeholder(self._canvas_tgt,    "開啟 Target Image 後顯示")
        _placeholder(self._canvas_result, "按 Run 後顯示結果")

        # ── buttons ─────────────────────────────────────────────────
        btns = tk.Frame(self.root)
        btns.pack(fill="x", padx=8, pady=6)
        tk.Button(btns, text="▶ Run",   width=14, command=self._run,
                  bg="#3498db", fg="white").pack(side="left",  padx=8)
        tk.Button(btns, text="💾 Save", width=14, command=self._save,
                  bg="#2ecc71", fg="white").pack(side="right", padx=8)

    # ── file browsing ────────────────────────────────────────────────

    def _browse_ref(self) -> None:
        p = filedialog.askopenfilename(title="選擇 Reference Image",
                                       filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if p:
            self._ref_var.set(p)
            self._ref_img = Image.open(p)

    def _browse_tgt(self) -> None:
        p = filedialog.askopenfilename(title="選擇 Target Image",
                                       filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if p:
            self._tgt_var.set(p)
            self._tgt_img = Image.open(p)
            self._show_target()

    # ── target canvas ────────────────────────────────────────────────

    def _show_target(self) -> None:
        if self._tgt_img is None:
            return
        display = self._tgt_img.copy().convert("RGB")
        display.thumbnail((_CANVAS_W, _CANVAS_H), Image.LANCZOS)
        dw, dh = display.size
        ox = (_CANVAS_W - dw) // 2
        oy = (_CANVAS_H - dh) // 2
        self._tgt_scale = self._tgt_img.width / dw   # canvas px → original px
        self._tgt_off   = (ox, oy)

        self._tk_tgt = ImageTk.PhotoImage(display)
        self._canvas_tgt.delete("all")
        self._canvas_tgt.create_image(ox, oy, anchor="nw", image=self._tk_tgt)
        self._redraw_dot()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self._tgt_img is None:
            return
        ox, oy = self._tgt_off
        img_x = round((event.x - ox) * self._tgt_scale)
        img_y = round((event.y - oy) * self._tgt_scale)
        img_x = max(0, min(img_x, self._tgt_img.width  - 1))
        img_y = max(0, min(img_y, self._tgt_img.height - 1))
        # update fields (suppress trace re-entry via flag)
        self._x_var.set(str(img_x))
        self._y_var.set(str(img_y))
        self._redraw_dot(event.x, event.y)

    def _on_xy_typed(self, *_) -> None:
        """Sync red dot when user types into X/Y entries."""
        if self._tgt_img is None:
            return
        try:
            ix = int(self._x_var.get())
            iy = int(self._y_var.get())
        except ValueError:
            return
        ox, oy = self._tgt_off
        cx = round(ix / self._tgt_scale) + ox
        cy = round(iy / self._tgt_scale) + oy
        self._redraw_dot(cx, cy)

    def _redraw_dot(self, cx: float | None = None, cy: float | None = None) -> None:
        self._canvas_tgt.delete("dot")
        if cx is None or cy is None:
            try:
                ix = int(self._x_var.get())
                iy = int(self._y_var.get())
            except ValueError:
                return
            ox, oy = self._tgt_off
            cx = round(ix / self._tgt_scale) + ox
            cy = round(iy / self._tgt_scale) + oy
        self._canvas_tgt.create_oval(
            cx - _DOT_R, cy - _DOT_R, cx + _DOT_R, cy + _DOT_R,
            fill="red", outline="white", width=1.5, tags="dot",
        )

    # ── run / save ───────────────────────────────────────────────────

    def _run(self) -> None:
        if self._ref_img is None or self._tgt_img is None:
            messagebox.showerror("錯誤", "請先選擇 Reference 和 Target 圖片")
            return
        try:
            cx = int(self._x_var.get())
            cy = int(self._y_var.get())
        except ValueError:
            messagebox.showerror("錯誤", "X / Y 必須是整數")
            return

        self._result = _paste_centered(self._ref_img, self._tgt_img, cx, cy)

        preview = self._result.copy().convert("RGB")
        preview.thumbnail((_CANVAS_W, _CANVAS_H), Image.LANCZOS)
        pw, ph = preview.size
        self._tk_result = ImageTk.PhotoImage(preview)
        self._canvas_result.delete("all")
        self._canvas_result.create_image(
            (_CANVAS_W - pw) // 2, (_CANVAS_H - ph) // 2,
            anchor="nw", image=self._tk_result,
        )

    def _save(self) -> None:
        if self._result is None:
            messagebox.showwarning("提示", "請先按 Run 產生結果")
            return
        _GOAL_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="儲存結果", initialdir=str(_GOAL_DIR),
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("所有檔案", "*.*")],
        )
        if not path:
            return
        img = self._result.convert("RGB") if Path(path).suffix.lower() in (".jpg", ".jpeg") else self._result
        img.save(path)
        messagebox.showinfo("完成", f"已儲存至：{path}")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
