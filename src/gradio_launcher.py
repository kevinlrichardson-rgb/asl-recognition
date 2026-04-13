"""
gradio_launcher.py — Gradio web launcher for ASL inference.

Mirrors the Tkinter launcher UI: video file upload, output folder/filename,
mode toggle, confidence slider, headless flag, Run/Cancel/Clear, live log.

Usage:
    python src/gradio_launcher.py
    python src/gradio_launcher.py --port 7861
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

# Absolute path to infer.py — unambiguous regardless of working directory
_INFER_PY = Path(__file__).resolve().parent / "infer.py"


def _build_cmd(
    video_path: str,
    out_folder: str,
    out_name: str,
    mode: str,
    conf: float,
    headless: bool,
) -> list:
    """Build the infer.py subprocess argument list."""
    cmd = [
        sys.executable, str(_INFER_PY),
        "--mode", mode,
        "--video", video_path,
        "--conf", f"{conf:.2f}",
    ]
    if headless:
        cmd.append("--headless")
    out_folder = (out_folder or "").strip()
    out_name   = (out_name   or "").strip()
    if out_folder:
        filename = out_name if out_name else Path(video_path).stem + "_out.mp4"
        cmd += ["--output", os.path.join(out_folder, filename)]
    return cmd


def _run_inference(video_file, out_folder, out_name, mode, conf, headless, proc_state):
    """
    Generator event handler — streams subprocess stdout to the log textbox.
    Yields (log_text, proc_state) on every new line.
    """
    if not video_file:
        yield "[ERROR] Please select an input video file.\n", proc_state
        return

    video_path = str(video_file)
    if not os.path.isfile(video_path):
        yield f"[ERROR] File not found: {video_path}\n", proc_state
        return

    cmd = _build_cmd(video_path, out_folder, out_name, mode, conf, headless)
    log = "$ " + " ".join(cmd) + "\n\n"
    yield log, proc_state

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"   # prevent child stdout buffering through the pipe

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        yield log + f"[ERROR] Failed to start process: {exc}\n", proc_state
        return

    proc_state = dict(proc_state)   # new dict so Gradio detects state change
    proc_state["proc"] = proc
    yield log, proc_state

    for line in proc.stdout:
        log += line
        yield log, proc_state

    proc.wait()
    log += f"\n[Done — exit code {proc.returncode}]\n"
    proc_state["proc"] = None
    yield log, proc_state


def _cancel(log_text, proc_state):
    """Terminate the running subprocess if one is active."""
    proc = proc_state.get("proc")
    if proc is not None:
        proc.terminate()
        proc_state = dict(proc_state)
        proc_state["proc"] = None
        return log_text + "\n[Cancelled by user]\n", proc_state
    return log_text, proc_state


def _clear():
    """Reset the output log and subprocess state."""
    return "", {}


def _pick_folder(selected):
    """
    FileExplorer select callback — extract a directory path from the selection
    and hide the explorer panel.
    """
    if not selected:
        return gr.update(), gr.update(visible=True)
    path = selected if isinstance(selected, str) else selected[0]
    folder = path if os.path.isdir(path) else str(Path(path).parent)
    return folder, gr.update(visible=False)


def _default_out_name(video_file):
    """Auto-fill output filename with the source video stem."""
    if not video_file:
        return ""
    return Path(str(video_file)).stem + "_out.mp4"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ASL Inference Launcher") as demo:
        proc_state = gr.State({})

        gr.Markdown("## ASL Inference Launcher")

        # ── Input / Output ─────────────────────────────────────────────────────
        gr.Markdown("### Input / Output")
        video_file = gr.File(
            file_types=[".mp4", ".avi", ".mov", ".mkv"],
            type="filepath",
            label="Input Video",
        )
        _DATA_DIR = str(Path(_INFER_PY).resolve().parent.parent / "data")
        with gr.Row():
            out_folder = gr.Textbox(
                label="Output Folder",
                value=_DATA_DIR,
                scale=4,
            )
            browse_folder_btn = gr.Button("Browse", scale=0, min_width=80)

        with gr.Group(visible=False) as folder_explorer_panel:
            gr.Markdown("*Select a folder (or any file inside it) then click to confirm.*")
            folder_explorer = gr.FileExplorer(
                glob="**",
                root_dir="/",
                file_count="single",
                label="Browse for Output Folder",
                max_height=260,
            )

        out_name = gr.Textbox(
            label="Output Filename",
            placeholder="auto-named from input video",
        )

        # ── Options ────────────────────────────────────────────────────────────
        gr.Markdown("### Options")
        with gr.Row():
            mode = gr.Radio(
                choices=["fingerspell", "wordsign"],
                value="wordsign",
                label="Mode",
                scale=1,
            )
            conf = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01,
                value=0.40,
                label="Confidence Threshold",
                scale=2,
            )
        headless = gr.Checkbox(
            value=True,
            label="Headless  (suppress CV2 preview window)",
            interactive=False,
        )

        # ── Action buttons ─────────────────────────────────────────────────────
        with gr.Row():
            run_btn    = gr.Button("Run Inference", variant="primary",   scale=3)
            cancel_btn = gr.Button("Cancel",        variant="stop",      scale=1)
            clear_btn  = gr.Button("Clear",         variant="secondary", scale=1)

        # ── Output log ─────────────────────────────────────────────────────────
        gr.Markdown("### Output Log")
        log_box = gr.Textbox(
            label="",
            lines=18,
            max_lines=500,
            interactive=False,
            autoscroll=True,
        )

        # ── Event wiring ───────────────────────────────────────────────────────

        # Auto-fill output filename when a video is selected
        video_file.change(
            fn=_default_out_name,
            inputs=[video_file],
            outputs=[out_name],
        )

        # Browse button shows the file explorer
        browse_folder_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[folder_explorer_panel],
        )

        # Selecting an entry populates the folder textbox and hides the explorer
        folder_explorer.change(
            fn=_pick_folder,
            inputs=[folder_explorer],
            outputs=[out_folder, folder_explorer_panel],
        )

        run_btn.click(
            fn=_run_inference,
            inputs=[video_file, out_folder, out_name, mode, conf, headless, proc_state],
            outputs=[log_box, proc_state],
        )
        cancel_btn.click(
            fn=_cancel,
            inputs=[log_box, proc_state],
            outputs=[log_box, proc_state],
        )
        clear_btn.click(
            fn=_clear,
            inputs=[],
            outputs=[log_box, proc_state],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="ASL Inference Gradio Launcher")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue()   # required for generator-based streaming to work
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
