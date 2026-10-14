# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from piper_control.api import create_app
from piper_control.config import ArmConfig, ArmSide
from piper_control.types import ArmStateSnapshot


class DashboardController:
    def snapshot(self) -> ArmStateSnapshot:
        return ArmStateSnapshot(side=ArmSide.RIGHT)

    def emergency_stop(self, _reason: str) -> None:
        pass

    def request_recovery(self) -> None:
        pass

    def request_homing(self) -> None:
        pass


def dashboard_config() -> ArmConfig:
    return ArmConfig.model_validate(
        {
            "server": {"host": "127.0.0.1", "port": 8765},
            "control_hz": 100,
            "feedback_timeout_s": 0.5,
            "shutdown_home_timeout_s": 30,
            "name": "right_arm",
            "can_port": "can0",
            "auto_initialize": False,
            "receive_latency_s": 0.1,
            "calibration_joints_pos_deg": [0] * 6,
            "reset_joints_pos_deg": [0] * 6,
            "min_joints_deg": [-150, 0, -170, -100, -70, -120],
            "max_joints_deg": [150, 180, 0, 100, 70, 120],
            "max_joints_speed_deg_s": [30] * 6,
            "default_joints_speed_deg_s": [10] * 6,
            "gripper": {
                "enabled": False,
                "min_position_m": 0,
                "max_position_m": 0.07,
                "max_speed_m_s": 0.05,
                "default_speed_m_s": 0.02,
                "effort_nm": 1,
            },
        }
    )


def test_dashboard_registers_routes_and_has_mode_and_safety_controls() -> None:
    app = create_app(DashboardController(), ArmSide.RIGHT, dashboard_config())
    paths = {route.path for route in app.routes}
    page = Path("src/piper_control/web/index.html").read_text(encoding="utf-8")

    assert "/" in paths
    assert "/static/app.js" in paths
    assert "/static/style.css" in paths
    assert 'data-action="emergency-stop"' in page
    assert 'data-action="recover"' in page
    assert 'data-action="homing"' in page
    assert 'data-action="hold"' in page
    assert 'data-action="mode-local"' in page
    assert 'data-action="mode-remote"' in page
    assert Path("src/piper_control/web/app.js").is_file()
    assert Path("src/piper_control/web/style.css").is_file()


def test_dashboard_script_polls_without_overlap_and_keeps_failures_visible() -> None:
    script = Path("src/piper_control/web/app.js").read_text(encoding="utf-8")

    assert "setInterval(refreshState, 200)" in script
    assert "if (refreshInFlight)" in script
    assert "refreshInFlight = true" in script
    assert "refreshInFlight = false" in script
    assert 'fetchJson("/api/state"' in script
    assert 'fetchJson("/api/base_info"' in script
    assert "poll-error" in script
    assert 'setText("control-readiness", "UNKNOWN")' in script
    assert "State request failed:" in script


def test_dashboard_script_keeps_recovery_and_homing_independent() -> None:
    script = Path("src/piper_control/web/app.js").read_text(encoding="utf-8")

    assert 'confirm("Recover from emergency stop?' in script
    assert 'confirm("Start manual homing?' in script
    assert 'changeState("recover")' in script
    assert 'changeState("homing")' in script
    assert 'fetchJson("/api/emergency_stop"' in script


def test_dashboard_mode_switch_is_explicit_and_server_state_driven() -> None:
    script = Path("src/piper_control/web/app.js").read_text(encoding="utf-8")
    page = Path("src/piper_control/web/index.html").read_text(encoding="utf-8")

    assert 'changeState("hold")' in script
    assert 'fetchJson("/control_mode"' in script
    assert 'JSON.stringify({ mode })' in script
    assert 'latestState?.state !== "HOLD"' in script
    assert 'state.control_mode' in script
    assert 'state.ws_controller_identity' in script
    assert 'id="control-mode"' in page
    assert 'id="ws-identity"' in page


def test_emergency_stop_bypasses_pending_state_action_lock() -> None:
    script = Path("src/piper_control/web/app.js").read_text(encoding="utf-8")
    page = Path("src/piper_control/web/index.html").read_text(encoding="utf-8")
    emergency_handler = script.split(
        "document.querySelector('[data-action=\"emergency-stop\"]')", 1
    )[1].split("document.querySelector('[data-action=\"recover\"]')", 1)[0]
    emergency_function = "async function runEmergencyStop()" + script.split(
        "async function runEmergencyStop()", 1
    )[1].split("async function changeState", 1)[0]

    assert re.sub(r"\s+", "", emergency_function) == (
        "asyncfunctionrunEmergencyStop(){"
        'setText("action-status","Emergencystoprequestinprogress…");'
        'try{awaitfetchJson("/api/emergency_stop");'
        'setText("action-status","Emergencystoprequestaccepted.");'
        "awaitrefreshState();}catch(error){"
        "constmessage=errorinstanceofError?error.message:String(error);"
        'setText("action-status",`Emergencystopfailed:${message}`);'
        "showError(message);}}"
    )
    assert re.sub(r"\s+", "", emergency_handler) == (
        '.addEventListener("click",()=>{runEmergencyStop();});'
    )
    assert (
        'querySelectorAll(\'[data-action="recover"], [data-action="homing"]\')'
        in script
    )
    assert "if (stateActionInFlight) return;" in script
    emergency_button = re.search(
        r'<button[^>]*data-action="emergency-stop"[^>]*>', page
    )
    assert emergency_button is not None
    assert "disabled" not in emergency_button.group(0).lower()


def test_dashboard_has_one_arm_and_single_arm_telemetry_targets() -> None:
    page = Path("src/piper_control/web/index.html").read_text(encoding="utf-8")

    for element_id in (
        "overall-state",
        "control-readiness",
        "control-block-reason",
        "emergency-state",
        "first-fault",
        "ws-occupancy",
        "arm",
        "arm-label",
        "actual-joints",
        "target-joints",
        "grippers",
        "receive-timestamp",
        "robot-timestamp",
        "poll-error",
    ):
        assert f'id="{element_id}"' in page
    assert page.count('class="panel arm-card"') == 1
    assert 'id="right-arm"' not in page
    assert 'id="left-arm"' not in page


def test_dashboard_script_uses_protocol_2_single_arm_payloads() -> None:
    script = Path("src/piper_control/web/app.js").read_text(encoding="utf-8")

    assert "base.side" in script
    assert "base.config.server.port" in script
    assert "index < 6" in script
    assert "state.ActualQ" in script
    assert "state.control_ready" in script
    assert "state.control_block_reason" in script
    assert "state.ActualGripper" in script
    assert "side_order" not in script
    assert "ActualQ.slice(6)" not in script
    assert "ActualGripper[1]" not in script


def test_dashboard_resources_are_declared_as_package_data() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    patterns = project["tool"]["setuptools"]["package-data"]["piper_control"]

    assert {"web/*.html", "web/*.js", "web/*.css"}.issubset(patterns)


def test_dashboard_has_visible_company_footer_after_controls() -> None:
    page = Path("src/piper_control/web/index.html").read_text(encoding="utf-8")
    footer = re.search(r'<footer class="company-footer">([^<]+)</footer>', page)
    assert footer is not None
    assert footer.group(1) == "Copyright © 2026-2026 atombite.ai"
    assert page.index(footer.group(0)) > page.index("</main>")
