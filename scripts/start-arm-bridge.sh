#!/usr/bin/env bash
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

exec sudo systemctl start \
    piper-control-bridge@left.service \
    piper-control-bridge@right.service
