#!/usr/bin/env bash
# Bootstrap Ubuntu with Conda and the project's TensorFlow GPU dependencies.
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${HOME}/miniforge3"
ENV_NAME="fl-byzantine-defense"
INSTALL_DRIVER=0
SKIP_SYSTEM=0
DOWNLOAD_DATA=1
IS_WSL=0

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
log() { printf '\n[setup] %s\n' "$*"; }

usage() {
    cat <<'HELP'
Usage: bash setup.sh [options]

Target: Ubuntu 22.04/24.04, x86-64, NVIDIA GPU (native Linux or WSL2).
Run as your normal user; sudo is used only for system packages/drivers.

  --install-driver        Install Ubuntu's recommended driver if nvidia-smi fails.
                          Native Ubuntu only. Stops for a manual reboot afterward.
  --conda-prefix PATH     Conda installation to use/create (default: ~/miniforge3).
  --env-name NAME         Dedicated environment (default: fl-byzantine-defense).
  --skip-system-packages  Skip apt when system prerequisites are already installed.
  --skip-data             Leave CIFAR-10 download to the first training run.
  -h, --help              Show this help without changing anything.

Installs Conda using Miniforge, Python 3.11, requirements.txt, and TensorFlow
2.20 CUDA/cuDNN wheels. Verifies imports and an actual GPU operation. Creates
run_config.local.json with a portable dataset path; existing configs are retained.
Does not start training or reboot automatically. Internet access is required.
HELP
}

parse_args() {
    while (($#)); do
        case "$1" in
            --install-driver) INSTALL_DRIVER=1; shift ;;
            --skip-system-packages) SKIP_SYSTEM=1; shift ;;
            --skip-data) DOWNLOAD_DATA=0; shift ;;
            --conda-prefix|--env-name)
                (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 needs a value"
                if [[ "$1" == --conda-prefix ]]; then CONDA_ROOT="$2"; else ENV_NAME="$2"; fi
                shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown option: $1 (see --help)" ;;
        esac
    done
    [[ "$ENV_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && "$ENV_NAME" != base ]] || die "Use a dedicated environment name (letters, digits, underscores, hyphens)."
    [[ "$CONDA_ROOT" == /* && "$CONDA_ROOT" != / && ! "$CONDA_ROOT" =~ [[:space:]] ]] || die "--conda-prefix must be an absolute path without whitespace."
}

check_platform() {
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "This installer targets x86-64 Ubuntu Linux."
    # shellcheck disable=SC1091
    source /etc/os-release
    [[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) ]] || die "Supported targets: Ubuntu 22.04 and 24.04; found ${PRETTY_NAME:-unknown}."
    ((EUID != 0)) || die "Run as your normal user, without sudo; the script requests sudo only for apt/drivers."
    if grep -qi microsoft /proc/sys/kernel/osrelease; then IS_WSL=1; fi
    [[ -f "$PROJECT_DIR/requirements.txt" && -f "$PROJECT_DIR/run_config.json" ]] || die "Copy the full repository to this machine first."
}

install_system_packages() {
    if ((SKIP_SYSTEM)); then return; fi
    log "Installing Ubuntu prerequisites (sudo may ask for your password)"
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl bzip2 build-essential libgomp1 pciutils
}

ensure_driver() {
    local smi
    smi="$(command -v nvidia-smi || true)"
    if [[ -z "$smi" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then smi=/usr/lib/wsl/lib/nvidia-smi; fi
    if [[ -n "$smi" ]] && "$smi"; then return; fi
    if ((IS_WSL)); then
        die "WSL needs WSL2 and an NVIDIA Windows host driver. Install/update them in Windows and rerun. Never install a Linux display driver inside WSL."
    fi
    ((INSTALL_DRIVER)) || die "No working NVIDIA driver. Rerun with --install-driver to install Ubuntu's recommended driver, or repair your existing driver first."
    command -v lspci >/dev/null || die "lspci is missing; rerun without --skip-system-packages."
    lspci -nn | grep -Ei '(VGA|3D|Display).*NVIDIA' >/dev/null || die "No NVIDIA GPU detected; check hardware or VM GPU passthrough."
    log "Installing Ubuntu's recommended NVIDIA driver"
    sudo apt-get update
    sudo apt-get install -y ubuntu-drivers-common "linux-headers-$(uname -r)"
    sudo ubuntu-drivers install
    printf '\nDriver installation finished. Reboot manually, complete Secure Boot key enrollment if prompted,\nthen rerun this same setup command. Setup is not complete yet.\n'
    exit 2
}

ensure_conda() {
    local staging installer release url
    if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
        [[ ! -e "$CONDA_ROOT" ]] || die "$CONDA_ROOT exists but has no executable bin/conda; choose another --conda-prefix."
        command -v curl >/dev/null || die "curl is missing; rerun without --skip-system-packages."
        release=26.9.4-2
        installer="Miniforge3-$release-Linux-x86_64.sh"
        url="https://github.com/conda-forge/miniforge/releases/download/$release"
        log "Downloading Miniforge $release and verifying its SHA-256 checksum"
        (
            staging="$(mktemp -d)"
            # Only this script's private temporary download directory is removed.
            trap 'rm -rf -- "$staging"' EXIT
            curl --fail --location --retry 3 "$url/$installer" -o "$staging/$installer"
            curl --fail --location --retry 3 "$url/$installer.sha256" -o "$staging/$installer.sha256"
            cd "$staging"
            sha256sum --check "$installer.sha256"
            bash "$staging/$installer" -b -p "$CONDA_ROOT"
        )
    fi
    CONDA_BIN="$CONDA_ROOT/bin/conda"
    ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
    if [[ ! -d "$ENV_PREFIX/conda-meta" ]]; then
        [[ ! -e "$ENV_PREFIX" ]] || die "$ENV_PREFIX exists but is not a Conda environment."
        log "Creating Python 3.11 environment: $ENV_PREFIX"
        "$CONDA_BIN" create -y --prefix "$ENV_PREFIX" --override-channels -c conda-forge python=3.11 pip
    fi
    PYTHON="$ENV_PREFIX/bin/python"
    "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Choose a new --env-name; the existing environment must use Python 3.11"'
}

install_project() {
    log "Installing project dependencies and TensorFlow CUDA/cuDNN libraries"
    cd "$PROJECT_DIR"
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install -r requirements.txt 'tensorflow[and-cuda]==2.20.0'
    "$PYTHON" -m pip check
}

verify_environment() {
    log "Checking application imports and GPU computation"
    KERAS_BACKEND=tensorflow "$PYTHON" - <<'PY_VERIFY'
import importlib
import tensorflow as tf
for name in (
    "flwr", "ray", "flwr_datasets", "hdbscan", "sklearn", "openpyxl",
    "matplotlib", "PIL", "filelock", "tensorboard",
    "fl_defense.client_app", "fl_defense.server_app",
):
    importlib.import_module(name)
assert tf.__version__ == "2.20.0", tf.__version__
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise SystemExit("TensorFlow cannot see a GPU. Check the driver, CUDA_VISIBLE_DEVICES, and TensorFlow loader errors above. Setup is incomplete.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(False)
with tf.device("/GPU:0"):
    result = tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
    convolution = tf.nn.conv2d(tf.ones((1, 8, 8, 1)), tf.ones((3, 3, 1, 2)), strides=1, padding="VALID")
assert "GPU" in result.device and float(result[0, 0].numpy()) == 32.0
assert "GPU" in convolution.device and float(convolution[0, 0, 0, 0].numpy()) == 9.0
print("TensorFlow", tf.__version__, "GPU computation passed:", result.device)
print("Build details:", tf.sysconfig.get_build_info())
PY_VERIFY
    "$PYTHON" run.py --help >/dev/null
}

prepare_local_config() {
    log "Preparing machine-local configuration and CIFAR-10 cache"
    "$PYTHON" - "$PROJECT_DIR" "$DOWNLOAD_DATA" <<'PY_CONFIG'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
destination = root / "run_config.local.json"
if not destination.exists():
    config = json.loads((root / "run_config.json").read_text())
    config["training"]["dataset-dir"] = str(root / "datasets")
    # Exclusive creation preserves an existing local experiment configuration.
    with destination.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
else:
    print("Keeping existing", destination)
    config = json.loads(destination.read_text())
dataset_dir = Path(config["training"]["dataset-dir"]).expanduser()
if not dataset_dir.is_absolute():
    raise SystemExit(f"Set training.dataset-dir to an absolute path in {destination}")
from run import prepare_run_config
prepare_run_config(config, root)  # Existing experiment validation; no training.
if sys.argv[2] == "1":
    from fl_defense.data.cifar10 import _load_local_training_data
    images, labels = _load_local_training_data(str(dataset_dir))
    print(f"CIFAR-10 cache ready: {len(labels)} training examples at {dataset_dir}")
print("Use configuration:", destination)
PY_CONFIG
}

main() {
    parse_args "$@"
    check_platform
    install_system_packages
    ensure_driver
    ensure_conda
    # Activation also makes installed CLI tools available to Flower subprocesses.
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$ENV_PREFIX"
    export KERAS_BACKEND=tensorflow
    install_project
    verify_environment
    prepare_local_config
    log "Setup complete. To run in a new terminal:"
    printf 'source %q\nconda activate %q\ncd %q\npython run.py --config run_config.local.json\n' \
        "$CONDA_ROOT/etc/profile.d/conda.sh" "$ENV_PREFIX" "$PROJECT_DIR"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    trap 'printf "\nSetup stopped at line %s. Fix the error above and rerun; existing data/configs are retained.\n" "$LINENO" >&2' ERR
    main "$@"
fi
