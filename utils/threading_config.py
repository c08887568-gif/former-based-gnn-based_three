import os


DEFAULT_NUM_THREADS = 8
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_default_threads():
    value = str(DEFAULT_NUM_THREADS)
    for name in THREAD_ENV_VARS:
        os.environ[name] = value
    return DEFAULT_NUM_THREADS


def get_runtime_thread_count():
    return DEFAULT_NUM_THREADS


def apply_torch_thread_config(torch_module):
    torch_module.set_num_threads(DEFAULT_NUM_THREADS)
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return DEFAULT_NUM_THREADS
