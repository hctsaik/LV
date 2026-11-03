"""Provider dispatch without changing legacy DINO contracts or importing heavy engines on entry."""
from pathlib import Path


def runtime_for(config, *, trusted=False):
    import os
    from model_runtimes import LEGACY, runtime_id
    selected = runtime_id(config)
    if selected != os.environ.get("LV_MODEL_RUNTIME_ID", LEGACY):
        from model_runtime_rpc import RuntimeProxy
        return RuntimeProxy(config, trusted=trusted)
    if config.get("backend") == "pytorch" or Path(config.get("model_path", "")).suffix.lower() == ".pt":
        from yolo_attribution import YoloRuntime
        return YoloRuntime(config, trusted=trusted)
    from fp_external_runtime import Runtime, DEFAULT_CONFIG
    return Runtime({k: v for k, v in config.items() if k in DEFAULT_CONFIG})


def run_dino_gradcam(*args, **kwargs):
    """Explicit legacy provider; its target/reference and resolution behavior stay unchanged."""
    from explain_ui import _run_gradcam
    return _run_gradcam(*args, **kwargs)


def external_context(session, query):
    return bool(session.get("investigation_diagnostic") or query.get("diagnostic_case"))
