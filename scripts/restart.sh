#!/usr/bin/env bash
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
sudo systemctl stop \
    piper-control-bridge@left.service \
    piper-control-bridge@right.service
sudo systemctl stop \
    piper-control@left.service \
    piper-control@right.service
sudo systemctl stop camera-bridge.service
sudo systemctl restart piper-control-can.service
sudo systemctl start camera-bridge.service
sudo systemctl start \
    piper-control@left.service \
    piper-control@right.service
exec sudo systemctl start \
    piper-control-bridge@left.service \
    piper-control-bridge@right.service
