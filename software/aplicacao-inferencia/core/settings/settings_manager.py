"""
Gestão de configurações do sistema.

Fonte de verdade em runtime: tabela system_config (SQLite).
config.yaml é usado apenas para bootstrapar valores padrão na primeira execução.
Nunca ler config.yaml em runtime após o bootstrap.
"""
from db.repositories import config_repo, audit_repo

_DEFAULTS: dict[str, str] = {
    # Realtime
    "realtime.conf_thres":       "0.50",
    "realtime.iou_thres":        "0.45",
    "realtime.source":           "0",
    "realtime.device":           "cpu",
    "realtime.roi_x_start":      "100",
    "realtime.roi_x_end":        "480",
    "realtime.roi_y_start":      "70",
    "realtime.roi_y_end":        "450",
    "arduino.port":              "COM5",
    "arduino.baudrate":          "9600",
    "training.img_size":         "640",
    "roi_timer.seconds":         "3",
    "conveyor.delay_vidro_ms":   "4700",
    "conveyor.delay_papel_ms":   "6260",
    "conveyor.delay_plastico_ms":"7900",
    "conveyor.delay_metal_ms":        "9000",
    "conveyor.delay_nao_identificado_ms": "10000",
}


def bootstrap_defaults():
    """Grava padrões no banco para chaves ainda não presentes. Idempotente."""
    for key, default in _DEFAULTS.items():
        if config_repo.get(key) is None:
            config_repo.set(key, default)


def get(key: str) -> str:
    value = config_repo.get(key)
    return value if value is not None else _DEFAULTS.get(key, "")


def get_int(key: str) -> int:
    return int(get(key))


def get_float(key: str) -> float:
    return float(get(key))


def get_all() -> dict[str, str]:
    return config_repo.get_all()


def set(key: str, value: str, user_id: int):
    old = get(key)
    config_repo.record_history(key, old, value, user_id)
    config_repo.set(key, value)
    audit_repo.record(
        "CONFIG_CHANGE",
        param_key=key,
        old_value=old,
        new_value=value,
        user_id=user_id,
    )
