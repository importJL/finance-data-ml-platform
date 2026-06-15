import logging
import time
from datetime import datetime

import streamlit as st


def _setup_console_logger(name="ML_PIPELINE"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


class PipelineLogger:
    def __init__(self, total_steps, title="ML Pipeline Progress"):
        self.total_steps = total_steps
        self.step_times = {}
        self._start_time = time.time()
        self._console = _setup_console_logger()

        self._status = st.status(f"**{title}**", expanded=True)
        self._progress = st.progress(0.0)
        self._step_labels = {}

        self._status.markdown(
            f"<div style='margin-bottom:0.5rem;color:#666;font-size:0.85rem;'>"
            f"Started at {datetime.now().strftime('%H:%M:%S')}</div>",
            unsafe_allow_html=True,
        )
        self._console.info("─" * 50)
        self._console.info("Pipeline started — %d steps", total_steps)

    def start_step(self, step_num, label, detail=""):
        self.step_times[step_num] = time.time()
        self._step_labels[step_num] = label
        msg = f"**Step {step_num}/{self.total_steps}:** {label}"
        if detail:
            msg += f"  —  {detail}"
        self._status.markdown(f"🔄 {msg}")
        self._console.info("[%d/%d] %s %s", step_num, self.total_steps, label, detail)

    def complete_step(self, step_num, detail=""):
        elapsed = time.time() - self.step_times.get(step_num, self._start_time)
        label = self._step_labels.get(step_num, f"Step {step_num}")
        msg = f"**Step {step_num}/{self.total_steps}:** {label}"
        if detail:
            msg += f"  —  {detail}"
        msg += f"  :green[✓]  ({elapsed:.1f}s)"
        self._status.markdown(f"✅ {msg}")
        self._progress.progress(min(step_num / self.total_steps, 1.0))
        self._console.info(
            "[%d/%d] ✅ %s %s (%.1fs)",
            step_num, self.total_steps, label, detail, elapsed,
        )

    def fail_step(self, step_num, error_msg):
        label = self._step_labels.get(step_num, f"Step {step_num}")
        msg = (
            f"**Step {step_num}/{self.total_steps}:** {label}"
            f"  —  :red[Failed: {error_msg}]"
        )
        self._status.markdown(f"❌ {msg}")
        self._console.error(
            "[%d/%d] ❌ %s — %s", step_num, self.total_steps, label, error_msg,
        )

    def log_info(self, msg):
        self._status.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_({msg})_")
        self._console.info("  %s", msg)

    def log_metric(self, label, value, delta=None):
        self._status.metric(label, value, delta=delta)
        self._console.info("  Metric — %s: %s", label, value)

    def log_table(self, title, data):
        self._status.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{title}**")
        self._status.dataframe(data, use_container_width=True, hide_index=True)
        if hasattr(data, "shape"):
            self._console.info("  Table — %s: %d rows", title, data.shape[0])

    def close(self):
        elapsed = time.time() - self._start_time
        elapsed_str = (
            f"{elapsed:.1f}s" if elapsed < 60
            else f"{elapsed // 60:.0f}m {elapsed % 60:.0f}s"
        )
        self._status.markdown(
            f"<div style='margin-top:0.5rem;color:#666;font-size:0.85rem;'>"
            f"Total elapsed: **{elapsed_str}**</div>",
            unsafe_allow_html=True,
        )
        self._progress.progress(1.0)
        self._console.info("Pipeline completed in %s", elapsed_str)
        self._console.info("─" * 50)
