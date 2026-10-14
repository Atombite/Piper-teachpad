#!/usr/bin/env bash
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
exec systemctl status \
    camera-bridge.service \
    piper-control-can.service \
    piper-control@left.service \
    piper-control@right.service \
    piper-control-bridge@left.service \
    piper-control-bridge@right.service
