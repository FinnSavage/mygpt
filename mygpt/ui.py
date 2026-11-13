"""Gradio browser UI: Train, Generate, and System check tabs."""

import time

import gradio as gr
import pandas as pd

from . import data, sample
from . import train as train_mod
from .utils import doctor as run_doctor
from .utils import fmt_time

UPDATE_EVERY = 0.3  # seconds, throttles UI redraws during training


def _train_fn(dataset_choice, uploaded_file, size, steps, run_name,
              time_minutes, use_autotune, fit_data, early_stop):
    source = uploaded_file if uploaded_file else dataset_choice
    steps = int(steps) if steps else None
    run_name = (run_name or "").strip() or None
    time_budget = float(time_minutes) * 60 if time_minutes else None
    options = dict(autotune=use_autotune, fit_data=fit_data, time_budget=time_budget, early_stop=early_stop)

    log = ""
    history = {"step": [], "loss": []}
    last_sample = ""
    last_update = 0.0

    for event in train_mod.train(dataset=source, preset=size, run=run_name, max_steps=steps, **options):
        t = event["type"]
        if t == "info":
            log += event["msg"] + "\n"
        elif t == "step":
            history["step"].append(event["step"])
            history["loss"].append(event["loss"])
            log = (
                f"step {event['step'] + 1}/{event['max_steps']}  loss {event['loss']:.4f}  "
                f"elapsed {fmt_time(event['elapsed'])}  eta {fmt_time(event['eta'])}\n" + log
            )
        elif t == "eval":
            log = f"eval @ {event['step'] + 1}: train {event['train_loss']:.4f}  val {event['val_loss']:.4f}\n" + log
        elif t == "sample":
            last_sample = event["text"]
        elif t == "done":
            log = f"Done. Saved to {event['run_dir']}.\n" + log

        now = time.time()
        if t in ("info", "eval", "sample", "done") or now - last_update > UPDATE_EVERY:
            last_update = now
            yield log, pd.DataFrame(history), last_sample

    yield log, pd.DataFrame(history), last_sample


def _generate_fn(run, prompt, max_tokens, temperature, top_k):
    if not run:
        yield "Pick a run first."
        return
    model, tok, device = sample.load_run(run)
    text = ""
    for event in sample.generate(model, tok, device, prompt, int(max_tokens), temperature, int(top_k)):
        if event["type"] == "warning":
            text += f"({event['msg']})\n"
        else:
            text += event["text"]
        yield text


def _doctor_df():
    rows = run_doctor()
    return pd.DataFrame(
        [[label, value, "OK" if ok else "!!", hint] for label, value, ok, hint in rows],
        columns=["Check", "Value", "Status", "Hint"],
    )


def _refresh_runs():
    return gr.update(choices=sample.list_runs())


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="mygpt") as demo:
        gr.Markdown("# mygpt")

        with gr.Tab("Train"):
            with gr.Row():
                dataset = gr.Dropdown(
                    [(f"{k}: {v[1]}", k) for k, v in data.DATASETS.items()],
                    value="names", label="Dataset",
                )
                upload = gr.File(label="...or upload a .txt file", file_types=[".txt"], type="filepath")
            with gr.Row():
                size = gr.Dropdown(["auto", "tiny", "small", "medium"], value="auto", label="Model size")
                steps = gr.Number(label="Steps (blank = preset default)", precision=0, value=None)
                run_name = gr.Textbox(label="Run name (blank = auto)")
            with gr.Accordion("Options", open=False):
                with gr.Row():
                    time_minutes = gr.Number(label="Time budget in minutes (blank = off)", value=None)
                    use_autotune = gr.Checkbox(value=True, label="Autotune for this computer")
                    fit_data = gr.Checkbox(value=True, label="Size the model to the dataset")
                    early_stop = gr.Checkbox(value=False, label="Stop early when it stops improving")
            with gr.Row():
                start_btn = gr.Button("Start training", variant="primary")
                stop_btn = gr.Button("Stop")
            loss_plot = gr.LinePlot(
                value=pd.DataFrame({"step": [], "loss": []}),
                x="step", y="loss", label="Training loss",
            )
            sample_box = gr.Textbox(label="Latest sample", lines=6)
            log_box = gr.Textbox(label="Log", lines=10, autoscroll=True)

            train_event = start_btn.click(
                _train_fn, inputs=[dataset, upload, size, steps, run_name,
                                   time_minutes, use_autotune, fit_data, early_stop],
                outputs=[log_box, loss_plot, sample_box],
            )
            stop_btn.click(None, inputs=None, outputs=None, cancels=[train_event])

        with gr.Tab("Generate"):
            with gr.Row():
                run_dd = gr.Dropdown(sample.list_runs(), label="Run")
                refresh_btn = gr.Button("Refresh runs")
            prompt_box = gr.Textbox(label="Prompt", lines=2)
            with gr.Row():
                max_tokens = gr.Slider(10, 2000, value=300, step=10, label="Characters to generate")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_k = gr.Slider(1, 500, value=200, step=1, label="Top-k")
            gen_btn = gr.Button("Generate", variant="primary")
            output_box = gr.Textbox(label="Output", lines=10)

            refresh_btn.click(_refresh_runs, outputs=run_dd)
            gen_btn.click(
                _generate_fn, inputs=[run_dd, prompt_box, max_tokens, temperature, top_k],
                outputs=output_box,
            )

        with gr.Tab("System check"):
            doctor_df = gr.Dataframe(value=_doctor_df(), label="Checks")
            doctor_btn = gr.Button("Re-check")
            doctor_btn.click(_doctor_df, outputs=doctor_df)

    return demo


def main():
    # footer_links=[] hides Gradio's developer footer (API docs, "Built with Gradio", settings and their icons)
    build_demo().launch(footer_links=[])
