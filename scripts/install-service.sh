#!/usr/bin/env bash
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "run this installer with sudo" >&2
    exit 1
fi
if [[ $# -ne 3 ]]; then
    echo "usage: sudo $0 SERVICE_USER CONTROL_CONFIG BRIDGE_CONFIG" >&2
    exit 2
fi

service_user="$1"
config_path="$(realpath "$2")"
bridge_config_path="$(realpath "$3")"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_python="${repo_root}/.venv/bin/python"
install_root="/opt/piper-control"
venv_root="${install_root}/venv"
config_destination="/etc/piper-control/config.yaml"
bridge_config_destination="/etc/piper-control/piper-control-bridge.yaml"
udev_rules_source="${repo_root}/deploy/70-piper-can.rules"
udev_rules_destination="/etc/udev/rules.d/70-piper-can.rules"

if [[ ! "$service_user" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
    echo "invalid service user name" >&2
    exit 1
fi
if ! id "$service_user" >/dev/null 2>&1; then
    echo "unknown service user: ${service_user}" >&2
    exit 1
fi
if [[ ! -f "$config_path" ]]; then
    echo "configuration does not exist: ${config_path}" >&2
    exit 1
fi
if [[ ! -f "$bridge_config_path" ]]; then
    echo "Bridge configuration does not exist: ${bridge_config_path}" >&2
    exit 1
fi
if [[ ! -x "$source_python" ]]; then
    echo "source environment does not exist: ${source_python}" >&2
    exit 1
fi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${repo_root}/src" \
    "$source_python" -c \
    'import sys; sys.excepthook = lambda _kind, value, _traceback: print(f"invalid configuration: {value}", file=sys.stderr); from pathlib import Path; from piper_control.config import load_config; load_config(Path(sys.argv[1]))' \
    "$config_path"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${repo_root}/src" \
    "$source_python" -c \
    'import sys; sys.excepthook = lambda _kind, value, _traceback: print(f"invalid configuration: {value}", file=sys.stderr); from pathlib import Path; from piper_control.bridge.config import load_bridge_config; load_bridge_config(Path(sys.argv[1]))' \
    "$bridge_config_path"

install -d -o root -g root -m 0755 "$install_root" /etc/piper-control
install -d -o root -g root -m 0755 /etc/udev/rules.d
if [[ ! -x "${venv_root}/bin/python" ]]; then
    python3 -m venv "$venv_root"
fi
"${venv_root}/bin/pip" install --upgrade "$repo_root"
chown -R root:root "$install_root"
chmod -R go-w "$install_root"
install -o root -g root -m 0644 "$config_path" "$config_destination"
install -o root -g root -m 0644 "$bridge_config_path" "$bridge_config_destination"
install -o root -g root -m 0644 "$udev_rules_source" "$udev_rules_destination"

escape_sed() {
    printf '%s' "$1" | sed 's/[&|]/\\&/g'
}

install -o root -g root -m 0644 \
    "${repo_root}/deploy/piper-control-can.service.in" \
    "/etc/systemd/system/piper-control-can.service"
sed -e "s|@SERVICE_USER@|$(escape_sed "$service_user")|g" \
    "${repo_root}/deploy/piper-control@.service.in" \
    > "/etc/systemd/system/piper-control@.service"
sed -e "s|@SERVICE_USER@|$(escape_sed "$service_user")|g" \
    "${repo_root}/deploy/piper-control-bridge@.service.in" \
    > "/etc/systemd/system/piper-control-bridge@.service"
chown root:root \
    "/etc/systemd/system/piper-control@.service" \
    "/etc/systemd/system/piper-control-bridge@.service"
chmod 0644 \
    "/etc/systemd/system/piper-control@.service" \
    "/etc/systemd/system/piper-control-bridge@.service"
if command -v systemd-analyze >/dev/null 2>&1; then
    systemd-analyze verify \
        "/etc/systemd/system/piper-control-can.service" \
        "/etc/systemd/system/piper-control@.service" \
        "/etc/systemd/system/piper-control-bridge@.service"
fi
systemctl daemon-reload
udevadm control --reload-rules
echo "Installed units without enabling or starting the robot. Review configuration before manual startup."
echo "Stable CAN names apply after the adapters are replugged or the host is rebooted."
