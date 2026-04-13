"""
ASL Inference Launcher — desktop GUI for infer.py

Launches infer.py as a subprocess and streams its output live into a log panel.
No webcam option; video files only.
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import shutil
from tkinter import filedialog

import customtkinter as ctk

ROOT         = Path(__file__).resolve().parent.parent
INFER_SCRIPT = Path(__file__).resolve().parent / "infer.py"
VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
    ("All files",   "*.*"),
]
POLL_INTERVAL_MS = 50   # ms between queue drain ticks


class InferApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("ASL Inference Launcher")
        self.geometry("860x700")
        self.minsize(720, 580)
        self.resizable(True, True)

        # ── State ─────────────────────────────────────────────────────────
        self._process: subprocess.Popen | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._poll_job = None

        # ── Tk variables ───────────────────────────────────────────────────
        self._input_path  = ctk.StringVar()
        self._output_dir  = ctk.StringVar()
        self._output_name = ctk.StringVar()
        self._last_output_dir: str = ""   # persists across clears
        self._mode        = ctk.StringVar(value="wordsign")
        self._conf        = ctk.DoubleVar(value=0.4)
        self._headless    = ctk.BooleanVar(value=True)

        self._input_path.trace_add("write", self._on_input_path_changed)

        self._build_layout()

    # ══════════════════════════════════════════════════════════════════════
    #  Layout construction
    # ══════════════════════════════════════════════════════════════════════

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # ── Row 0: Input / Output card ────────────────────────────────────
        io_frame = ctk.CTkFrame(self, corner_radius=10)
        io_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 6))
        io_frame.grid_columnconfigure(1, weight=1)
        self._build_input_section(io_frame)
        self._build_output_section(io_frame)

        # ── Row 1: Options card ───────────────────────────────────────────
        opts_frame = ctk.CTkFrame(self, corner_radius=10)
        opts_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=6)
        opts_frame.grid_columnconfigure((1, 3), weight=1)
        self._build_options_section(opts_frame)

        # ── Row 2: Action bar ─────────────────────────────────────────────
        action_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        action_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=6)
        action_frame.grid_columnconfigure(0, weight=1)
        self._build_action_section(action_frame)

        # ── Row 3: Log panel ──────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self, corner_radius=10)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=16, pady=(6, 16))
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        self._build_log_section(log_frame)

    def _build_input_section(self, parent):
        ctk.CTkLabel(
            parent, text="Input / Output",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=14, pady=(12, 4), sticky="w")

        ctk.CTkLabel(parent, text="Input Video").grid(
            row=1, column=0, padx=(14, 8), pady=4, sticky="w")
        ctk.CTkEntry(parent, textvariable=self._input_path,
                     placeholder_text="Select a video file…"
                     ).grid(row=1, column=1, padx=(0, 8), pady=4, sticky="ew")
        ctk.CTkButton(parent, text="Browse", width=80,
                      command=self._on_browse_input
                      ).grid(row=1, column=2, padx=(0, 14), pady=4)

    def _build_output_section(self, parent):
        ctk.CTkLabel(parent, text="Output Folder").grid(
            row=2, column=0, padx=(14, 8), pady=4, sticky="w")
        ctk.CTkEntry(parent, textvariable=self._output_dir,
                     placeholder_text="Browse to select output folder"
                     ).grid(row=2, column=1, padx=(0, 8), pady=4, sticky="ew")
        ctk.CTkButton(parent, text="Browse", width=80,
                      command=self._on_browse_output_dir
                      ).grid(row=2, column=2, padx=(0, 14), pady=4)

        ctk.CTkLabel(parent, text="Output Filename").grid(
            row=3, column=0, padx=(14, 8), pady=(4, 12), sticky="w")
        self._name_entry = ctk.CTkEntry(
            parent, textvariable=self._output_name,
            placeholder_text="Auto-filled from input filename",
        )
        self._name_entry.grid(
            row=3, column=1, columnspan=2, padx=(0, 14), pady=(4, 12), sticky="ew")

    def _build_options_section(self, parent):
        ctk.CTkLabel(
            parent, text="Options",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, padx=14, pady=(12, 4), sticky="w")

        # Mode
        ctk.CTkLabel(parent, text="Mode:").grid(
            row=1, column=0, padx=(14, 8), pady=6, sticky="w")
        self._mode_selector = ctk.CTkSegmentedButton(
            parent,
            values=["fingerspell", "wordsign"],
            variable=self._mode,
            command=self._on_mode_change,
        )
        self._mode_selector.grid(row=1, column=1, padx=(0, 24), pady=6, sticky="w")

        # Confidence threshold
        ctk.CTkLabel(parent, text="Confidence:").grid(
            row=1, column=2, padx=(0, 8), pady=6, sticky="e")
        conf_inner = ctk.CTkFrame(parent, fg_color="transparent")
        conf_inner.grid(row=1, column=3, padx=(0, 14), pady=6, sticky="w")
        self._conf_slider = ctk.CTkSlider(
            conf_inner, from_=0.05, to=1.0, number_of_steps=19,
            variable=self._conf, width=150,
        )
        self._conf_slider.pack(side="left")
        self._conf_label = ctk.CTkLabel(conf_inner, text="0.40", width=44)
        self._conf_label.pack(side="left", padx=(6, 0))
        self._conf.trace_add("write", lambda *_: self._conf_label.configure(
            text=f"{self._conf.get():.2f}"))

        # Headless
        ctk.CTkCheckBox(
            parent, text="Headless  (suppress CV2 preview window)",
            variable=self._headless,
        ).grid(row=2, column=0, columnspan=4, padx=14, pady=(0, 12), sticky="w")

        # Set initial sensitivity
        self._on_mode_change("wordsign")

    def _build_action_section(self, parent):
        self._run_btn = ctk.CTkButton(
            parent, text="Run Inference",
            command=self._on_run,
            fg_color="#2a7d4f", hover_color="#1f5e3a",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
        )
        self._run_btn.grid(row=0, column=0, padx=(0, 8), pady=6, sticky="ew")

        self._cancel_btn = ctk.CTkButton(
            parent, text="Cancel",
            command=self._on_cancel,
            fg_color="#8b2020", hover_color="#641818",
            state="disabled", width=100, height=40,
        )
        self._cancel_btn.grid(row=0, column=1, padx=(0, 8), pady=6)

        self._clear_btn = ctk.CTkButton(
            parent, text="Clear",
            command=self._on_clear,
            fg_color="transparent", border_width=2,
            width=90, height=40,
        )
        self._clear_btn.grid(row=0, column=2, pady=6)

        self._progress = ctk.CTkProgressBar(parent, mode="indeterminate", width=180)
        self._progress.grid(row=0, column=3, padx=(12, 0), pady=6)
        self._progress.grid_remove()   # hidden until running

    def _build_log_section(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 0))
        ctk.CTkLabel(
            header, text="Output Log",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")

        self._log_text = ctk.CTkTextbox(
            parent,
            state="disabled",
            font=ctk.CTkFont(family="monospace", size=12),
            wrap="word",
        )
        self._log_text.grid(row=1, column=0, sticky="nsew", padx=14, pady=(4, 14))

    # ══════════════════════════════════════════════════════════════════════
    #  Event handlers — file browsing
    # ══════════════════════════════════════════════════════════════════════

    # ── File dialog helpers ───────────────────────────────────────────────
    # Use zenity (native GTK picker) when available — it handles large
    # directories without blocking the UI. Falls back to tkinter filedialog.

    @staticmethod
    def _zenity_available() -> bool:
        return shutil.which("zenity") is not None

    def _pick_file(self) -> str:
        """Return selected file path, or '' if cancelled."""
        if self._zenity_available():
            result = subprocess.run(
                ["zenity", "--file-selection",
                 "--title=Select input video",
                 "--file-filter=Video files (mp4 avi mov mkv)|*.mp4 *.avi *.mov *.mkv",
                 "--file-filter=All files|*"],
                capture_output=True, text=True,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        # Fallback
        return filedialog.askopenfilename(
            title="Select input video",
            filetypes=VIDEO_FILETYPES,
        ) or ""

    def _pick_directory(self) -> str:
        """Return selected directory path, or '' if cancelled."""
        start = self._last_output_dir or str(Path.home())
        if self._zenity_available():
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory",
                 "--title=Select output folder",
                 f"--filename={start}/"],
                capture_output=True, text=True,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        # Fallback
        return filedialog.askdirectory(
            title="Select output folder", initialdir=start) or ""

    def _on_browse_input(self):
        path = self._pick_file()
        if path:
            self._input_path.set(path)

    def _on_browse_output_dir(self):
        d = self._pick_directory()
        if d:
            self._last_output_dir = d
            self._output_dir.set(d)

    def _on_input_path_changed(self, *_):
        p = self._input_path.get().strip()
        if not p:
            return
        self._output_name.set(f"{Path(p).stem}_annotated.mp4")

    # ══════════════════════════════════════════════════════════════════════
    #  Event handlers — options
    # ══════════════════════════════════════════════════════════════════════

    def _on_mode_change(self, mode: str):
        state = "normal" if mode == "wordsign" else "disabled"
        self._conf_slider.configure(state=state)
        dim = ("gray50" if state == "disabled" else ("gray14", "gray84"))
        if isinstance(dim, tuple):
            self._conf_label.configure(text_color=dim)
        else:
            self._conf_label.configure(text_color=dim)

    # ══════════════════════════════════════════════════════════════════════
    #  Event handlers — run / cancel / clear
    # ══════════════════════════════════════════════════════════════════════

    def _on_run(self):
        cmd = self._build_command()
        if cmd is None:
            return
        self._log("=" * 64)
        self._log("Command: " + " ".join(cmd))
        self._log("=" * 64)
        self._start_subprocess(cmd)

    def _on_cancel(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log("--- Cancelled by user ---")

    def _on_clear(self):
        if self._process and self._process.poll() is None:
            return   # don't clear while running
        self._input_path.set("")
        self._output_dir.set("")
        self._output_name.set("")
        self._mode.set("wordsign")
        self._conf.set(0.4)
        self._headless.set(True)
        self._on_mode_change("wordsign")
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════════
    #  Command assembly
    # ══════════════════════════════════════════════════════════════════════

    def _build_command(self) -> list[str] | None:
        input_path = self._input_path.get().strip()
        if not input_path:
            self._log("ERROR: No input video selected.")
            return None
        if not Path(input_path).is_file():
            self._log(f"ERROR: Input file not found: {input_path}")
            return None

        out_dir = self._output_dir.get().strip()
        if not out_dir:
            self._log("ERROR: No output folder selected. Please browse for one.")
            return None
        out_name = self._output_name.get().strip() or f"{Path(input_path).stem}_annotated.mp4"

        # Ensure output dir exists
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        output_path = str(Path(out_dir) / out_name)

        cmd = [
            sys.executable,
            str(INFER_SCRIPT),
            "--no-ui",
            "--mode",   self._mode.get(),
            "--video",  input_path,
            "--output", output_path,
        ]
        if self._headless.get():
            cmd.append("--headless")
        if self._mode.get() == "wordsign":
            cmd.extend(["--conf", f"{self._conf.get():.2f}"])

        return cmd

    # ══════════════════════════════════════════════════════════════════════
    #  Subprocess management (non-blocking I/O via thread + queue)
    # ══════════════════════════════════════════════════════════════════════

    def _start_subprocess(self, cmd: list[str]):
        self._set_running_state(True)
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                cwd=str(ROOT),
            )
        except OSError as exc:
            self._log(f"ERROR: Failed to start process: {exc}")
            self._set_running_state(False)
            return

        threading.Thread(target=self._reader_thread, daemon=True).start()
        self._poll_job = self.after(POLL_INTERVAL_MS, self._poll_queue)

    def _reader_thread(self):
        proc = self._process
        if proc is None or proc.stdout is None:
            self._stdout_queue.put(None)
            return
        for line in proc.stdout:
            self._stdout_queue.put(line.rstrip("\n"))
        proc.wait()
        self._stdout_queue.put(None)

    def _poll_queue(self):
        try:
            while True:
                line = self._stdout_queue.get_nowait()
                if line is None:
                    self._on_process_done()
                    return
                self._log(line)
        except queue.Empty:
            pass
        self._poll_job = self.after(POLL_INTERVAL_MS, self._poll_queue)

    def _on_process_done(self):
        rc = self._process.returncode if self._process else -1
        self._log(f"{'=' * 64}")
        self._log(f"Process finished  (exit code {rc})")
        self._process = None
        self._set_running_state(False)

    # ══════════════════════════════════════════════════════════════════════
    #  UI helpers
    # ══════════════════════════════════════════════════════════════════════

    def _set_running_state(self, running: bool):
        if running:
            self._run_btn.configure(state="disabled")
            self._cancel_btn.configure(state="normal")
            self._clear_btn.configure(state="disabled")
            self._progress.grid()
            self._progress.start()
        else:
            self._run_btn.configure(state="normal")
            self._cancel_btn.configure(state="disabled")
            self._clear_btn.configure(state="normal")
            self._progress.stop()
            self._progress.grid_remove()

        widget_state = "disabled" if running else "normal"
        for w in (self._mode_selector, self._conf_slider,
                  self._name_entry):
            w.configure(state=widget_state)

    def _log(self, line: str):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", line + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def launch_ui():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = InferApp()
    app.mainloop()


if __name__ == "__main__":
    launch_ui()
