#!/usr/bin/env bash
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "run this installer with sudo" >&2
    exit 1
fi
if [[ $# -ne 2 ]]; then
    echo "usage: sudo $0 SERVICE_USER CAMERA_CONFIG" >&2
    exit 2
fi

service_user="$1"
config_path="$(realpath "$2")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_python="${repo_root}/.venv/bin/python"
install_root="/opt/piper-control"
venv_root="${install_root}/venv"
config_destination="/etc/piper-control/camera.yaml"
unit_destination="/etc/systemd/system/camera-bridge.service"

if [[ ! "$service_user" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    echo "invalid service user name" >&2
    exit 1
fi
if ! id "$service_user" >/dev/null 2>&1; then
    echo "unknown service user: ${service_user}" >&2
    exit 1
fi
if [[ ! -f "$config_path" ]]; then
    echo "camera configuration does not exist: ${config_path}" >&2
    exit 1
fi
if [[ ! -x "$source_python" ]]; then
    echo "source environment does not exist: ${source_python}" >&2
    exit 1
fi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${repo_root}/src" \
    "$source_python" -c \
    'import sys; sys.excepthook = lambda _kind, value, _traceback: print(f"invalid configuration: {value}", file=sys.stderr); from pathlib import Path; from piper_control.camera_bridge.config import load_camera_config; load_camera_config(Path(sys.argv[1]))' \
    "$config_path"

install -d -o root -g root -m 0755 "$install_root" /etc/piper-control
if [[ ! -x "${venv_root}/bin/python" ]]; then
    python3 -m venv "$venv_root"
fi
"${venv_root}/bin/pip" install --upgrade "${repo_root}[camera]"
chown -R root:root "$install_root"
chmod -R go-w "$install_root"
install -o root -g root -m 0644 "$config_path" "$config_destination"

escape_sed() {
    printf '%s' "$1" | sed 's/[&|]/\\&/g'
}

sed -e "s|@SERVICE_USER@|$(escape_sed "$service_user")|g" \
    "${repo_root}/deploy/camera-bridge.service.in" \
    > "/etc/systemd/system/camera-bridge.service"
chown root:root "$unit_destination"
chmod 0644 "$unit_destination"
if command -v systemd-analyze >/dev/null 2>&1; then
    systemd-analyze verify "$unit_destination"
fi
systemctl daemon-reload
echo "Installed camera-bridge.service without enabling or starting it."
