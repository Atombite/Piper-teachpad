#!/usr/bin/env bash
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
sudo systemctl start camera-bridge.service
sudo systemctl start \
    piper-control@left.service \
    piper-control@right.service
exec sudo systemctl start \
    piper-control-bridge@left.service \
    piper-control-bridge@right.service
