# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import subprocess
import shutil

import yaml


ROOT = Path(__file__).parents[1]


def test_piper_can_udev_rules_use_documented_serial_placeholders() -> None:
    rules = (ROOT / "deploy/70-piper-can.rules").read_text().splitlines()
    active_rules = [line for line in rules if line and not line.startswith("#")]

    assert active_rules == [
        'ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", '
        'ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", '
        'ATTRS{serial}=="REPLACE_WITH_LEFT_PIPER_CAN_ADAPTER_SERIAL", '
        'NAME="can_left"',
        'ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", '
        'ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", '
        'ATTRS{serial}=="REPLACE_WITH_RIGHT_PIPER_CAN_ADAPTER_SERIAL", '
        'NAME="can_right"',
    ]
    assert "udevadm info" in "\n".join(rules)


def test_hardware_example_uses_native_piper_grippers_and_stable_can_names() -> None:
    payload = yaml.safe_load((ROOT / "config/piper-control.example.yaml").read_text())

    assert payload["arms"]["left"]["can_port"] == "can_left"
    assert payload["arms"]["right"]["can_port"] == "can_right"
    assert payload["arms"]["left"]["gripper"]["backend"] == "piper"
    assert payload["arms"]["right"]["gripper"]["backend"] == "piper"
    assert {
        path.name for path in (ROOT / "config").glob("piper-control*.example.yaml")
    } == {
        "piper-control.example.yaml",
        "piper-control-bridge.example.yaml",
    }


def test_mujoco_example_uses_native_piper_grippers_on_both_arms() -> None:
    payload = yaml.safe_load((ROOT / "config/piper-control.mujoco.yaml").read_text())

    assert payload["arms"]["left"]["gripper"]["backend"] == "piper"
    assert payload["arms"]["right"]["gripper"]["backend"] == "piper"


def test_mujoco_submodule_uses_exact_public_upstream() -> None:
    result = subprocess.run(
        [
            "git",
            "config",
            "-f",
            str(ROOT / ".gitmodules"),
            "--get",
            "submodule.thirdparty/mujoco_menagerie.url",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == (
        "https://github.com/google-deepmind/mujoco_menagerie.git"
    )


def test_public_camera_example_uses_only_explicit_placeholders() -> None:
    payload = yaml.safe_load((ROOT / "config/camera.yaml").read_text())
    left_device = payload["cameras"]["left_wrist"]["device"]
    identifiers = (
        left_device.removeprefix("/dev/v4l/by-id/").removesuffix(
            "-video-index0"
        ),
        payload["cameras"]["right_wrist"]["serial"],
        payload["cameras"]["head"]["serial"],
    )

    assert all(
        identifier.startswith("REPLACE_WITH_") for identifier in identifiers
    )
    assert all(
        not any(character.isdigit() for character in identifier)
        for identifier in identifiers
    )


def test_can_unit_is_root_oneshot_without_automatic_down() -> None:
    text = (ROOT / "deploy/piper-control-can.service.in").read_text()
    assert "Type=oneshot" in text
    assert "User=root" in text
    assert "RemainAfterExit=yes" in text
    assert "ExecStop" not in text
    assert "can-down" not in text


def test_controller_instance_has_side_specific_runtime_and_manual_restart_policy() -> None:
    text = (ROOT / "deploy/piper-control@.service.in").read_text()
    assert "Requires=piper-control-can.service" in text
    assert "After=piper-control-can.service" in text
    assert "KillSignal=SIGTERM" in text
    assert "SendSIGKILL=no" in text
    assert "RuntimeDirectory=piper-control-%i" in text
    assert "--side %i" in text
    assert "/run/piper-control-%i/piper-control.lock" in text
    assert "TimeoutStopSec=infinity" in text
    assert "Restart=" not in text
    assert "can-down" not in text


def test_deployment_has_independent_controller_and_bridge_templates() -> None:
    service_templates = sorted(
        path.name for path in (ROOT / "deploy").glob("piper-control*.service.in")
    )

    assert service_templates == [
        "piper-control-bridge@.service.in",
        "piper-control-can.service.in",
        "piper-control@.service.in",
    ]
    assert not (ROOT / "deploy/piper-control.service.in").exists()


def test_bridge_instance_is_bound_after_matching_controller_without_restart() -> None:
    text = (ROOT / "deploy/piper-control-bridge@.service.in").read_text()
    assert "Requires=piper-control@%i.service" in text
    assert "After=piper-control@%i.service" in text
    assert "--side %i" in text
    assert "piper-control-bridge.yaml" in text
    assert "Restart=" not in text


def test_management_scripts_are_shell_syntax_valid() -> None:
    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    expected = {"can-up.sh", "can-status.sh", "can-down.sh", "install-service.sh", "start.sh", "stop.sh", "restart.sh", "status.sh"}
    assert expected <= {path.name for path in scripts}
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_restart_reconfigures_can_after_a_usb_adapter_is_replugged(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "sudo-calls"
    sudo = fake_bin / "sudo"
    sudo.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    sudo.chmod(0o755)

    subprocess.run(
        [str(ROOT / "scripts/restart.sh")],
        check=True,
        env={"PATH": f"{fake_bin}:/usr/bin", "CALL_LOG": str(call_log)},
    )

    assert call_log.read_text().splitlines() == [
        "systemctl stop piper-control-bridge@left.service piper-control-bridge@right.service",
        "systemctl stop piper-control@left.service piper-control@right.service",
        "systemctl stop camera-bridge.service",
        "systemctl restart piper-control-can.service",
        "systemctl start camera-bridge.service",
        "systemctl start piper-control@left.service piper-control@right.service",
        "systemctl start piper-control-bridge@left.service piper-control-bridge@right.service",
    ]


def test_can_scripts_forward_a_config_path_with_spaces_and_one_command(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo with spaces"
    scripts = repo / "scripts"
    python = repo / ".venv/bin/python"
    scripts.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    config = tmp_path / "arm config.yaml"

    for command in ("up", "status", "down"):
        source = ROOT / "scripts" / f"can-{command}.sh"
        script = scripts / source.name
        shutil.copy2(source, script)
        result = subprocess.run(
            [str(script), str(config)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.splitlines() == [
            "-m",
            "piper_control.can_cli",
            "--config",
            str(config),
            command,
        ]


def test_installer_does_not_start_service_implicitly() -> None:
    text = (ROOT / "scripts/install-service.sh").read_text()
    assert "daemon-reload" in text
    assert "systemctl start" not in text
    assert "systemctl enable --now" not in text


def test_installer_deploys_and_reloads_stable_can_naming_rules() -> None:
    text = (ROOT / "scripts/install-service.sh").read_text()

    assert 'udev_rules_source="${repo_root}/deploy/70-piper-can.rules"' in text
    assert 'udev_rules_destination="/etc/udev/rules.d/70-piper-can.rules"' in text
    assert 'install -d -o root -g root -m 0755 /etc/udev/rules.d' in text
    assert '"$udev_rules_source" "$udev_rules_destination"' in text
    assert "udevadm control --reload-rules" in text
    assert "udevadm trigger" not in text


def test_units_only_reference_fixed_root_owned_runtime_paths() -> None:
    repo_text = str(ROOT)
    can_unit = (ROOT / "deploy/piper-control-can.service.in").read_text()
    main_unit = (ROOT / "deploy/piper-control@.service.in").read_text()
    bridge_unit = (ROOT / "deploy/piper-control-bridge@.service.in").read_text()

    assert repo_text not in can_unit + main_unit + bridge_unit
    assert "@REPO_ROOT@" not in can_unit + main_unit + bridge_unit
    assert "/opt/piper-control/venv/bin/python" in can_unit
    assert "/opt/piper-control/venv/bin/python" in main_unit
    assert "/opt/piper-control/venv/bin/python" in bridge_unit
    assert "/etc/piper-control/config.yaml" in can_unit
    assert "/etc/piper-control/config.yaml" in main_unit
    assert "/etc/piper-control/piper-control-bridge.yaml" in bridge_unit


def test_installer_hardens_fixed_runtime_tree_and_copied_config() -> None:
    text = (ROOT / "scripts/install-service.sh").read_text()

    assert 'install_root="/opt/piper-control"' in text
    assert 'config_destination="/etc/piper-control/config.yaml"' in text
    assert 'bridge_config_destination="/etc/piper-control/piper-control-bridge.yaml"' in text
    assert 'install -o root -g root -m 0644' in text
    assert 'chown -R root:root "$install_root"' in text
    assert 'chmod -R go-w "$install_root"' in text
    assert "systemd-analyze verify" in text


def test_installer_validates_config_before_any_deployment_mutation() -> None:
    text = (ROOT / "scripts/install-service.sh").read_text()
    validation = 'from piper_control.config import load_config; load_config(Path(sys.argv[1]))'
    bridge_validation = 'from piper_control.bridge.config import load_bridge_config; load_bridge_config(Path(sys.argv[1]))'

    assert validation in text
    assert bridge_validation in text
    assert 'source_python="${repo_root}/.venv/bin/python"' in text
    assert 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${repo_root}/src"' in text
    assert "invalid configuration:" in text
    validation_index = text.index(validation)
    bridge_validation_index = text.index(bridge_validation)
    mutation_commands = (
        'install -d -o root -g root -m 0755 "$install_root" /etc/piper-control',
        'python3 -m venv "$venv_root"',
        '"${venv_root}/bin/pip" install --upgrade "$repo_root"',
        'chown -R root:root "$install_root"',
        'chmod -R go-w "$install_root"',
        'install -o root -g root -m 0644 "$config_path" "$config_destination"',
        '"${repo_root}/deploy/piper-control-can.service.in"',
        '> "/etc/systemd/system/piper-control@.service"',
        '> "/etc/systemd/system/piper-control-bridge@.service"',
        'systemctl daemon-reload',
    )

    for command in mutation_commands:
        assert validation_index < text.index(command), command
        assert bridge_validation_index < text.index(command), command


def test_recursive_mode_hardening_removes_group_and_world_write(tmp_path: Path) -> None:
    install_root = tmp_path / "opt/piper-control"
    executable = install_root / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("python")
    executable.chmod(0o777)
    install_root.chmod(0o777)

    subprocess.run(["chmod", "-R", "go-w", str(install_root)], check=True)

    assert install_root.stat().st_mode & 0o022 == 0
    assert executable.stat().st_mode & 0o022 == 0


def test_rendered_units_are_valid_and_cannot_contain_user_injection(
    tmp_path: Path,
) -> None:
    main_template = (ROOT / "deploy/piper-control@.service.in").read_text()
    bridge_template = (ROOT / "deploy/piper-control-bridge@.service.in").read_text()
    can_template = (ROOT / "deploy/piper-control-can.service.in").read_text()
    rendered_main = main_template.replace("@SERVICE_USER@", "robot_user")
    rendered_bridge = bridge_template.replace("@SERVICE_USER@", "robot_user")
    assert "@SERVICE_USER@" not in rendered_main + rendered_bridge + can_template
    assert "\n" not in rendered_main.split("User=", 1)[1].splitlines()[0]
    assert "%" not in rendered_main.split("User=", 1)[1].splitlines()[0]

    if shutil.which("systemd-analyze"):
        # Verification is about unit syntax; use an existing executable boundary.
        rendered_main = rendered_main.replace(
            "/opt/piper-control/venv/bin/python", "/bin/true"
        )
        rendered_bridge = rendered_bridge.replace(
            "/opt/piper-control/venv/bin/python", "/bin/true"
        )
        can_template = can_template.replace(
            "/opt/piper-control/venv/bin/python", "/bin/true"
        )
        main_path = tmp_path / "piper-control@left.service"
        bridge_path = tmp_path / "piper-control-bridge@left.service"
        can_path = tmp_path / "piper-control-can.service"
        main_path.write_text(rendered_main)
        bridge_path.write_text(rendered_bridge)
        can_path.write_text(can_template)
        subprocess.run(
            ["systemd-analyze", "verify", str(can_path), str(main_path), str(bridge_path)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_installer_rejects_unsafe_service_user_before_rendering() -> None:
    text = (ROOT / "scripts/install-service.sh").read_text()
    assert "^[a-z_][a-z0-9_-]*$" in text


def test_package_exposes_calibration_command() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    assert 'piper-calibrate = "piper_control.calibration_cli:main"' in text


def test_readme_documents_reproducible_dual_arm_mujoco_operation() -> None:
    text = (ROOT / "README.md").read_text()

    required_operator_text = (
        "git submodule update --init --recursive",
        ".venv/bin/pip install -e '.[mujoco]'",
        ".venv/bin/piper-control --config config/piper-control.mujoco.yaml --mujoco",
        ".venv/bin/piper-control --config config/piper-control.mujoco.yaml --mujoco-headless",
        "ws://127.0.0.1:8765/ws/control",
        "ws://127.0.0.1:9876/ws/control",
        "52.6 cm",
        "100 mm -> 70 mm",
        "物理安全认证",
        "DISPLAY",
        "不打开 CAN",
        "关闭 Viewer",
    )
    for expected in required_operator_text:
        assert expected in text, expected


def test_readme_mujoco_acceptance_holds_both_clients_until_bounded_convergence() -> None:
    text = (ROOT / "README.md").read_text()
    snippet = text.split("PYTHONPATH=. .venv/bin/python - <<'PY'\n", 1)[1].split(
        "\nPY\n", 1
    )[0]

    compile(snippet, "<README MuJoCo acceptance>", "exec")
    assert "async with (" in snippet
    assert ') as left,' in snippet
    assert ') as right,' in snippet
    command_start = snippet.index("await asyncio.gather(")
    polling_start = snippet.index("loop = asyncio.get_running_loop()")
    command_block = snippet[command_start:polling_start]
    for command in (
        "left.servo_j(LEFT_TARGET)",
        "right.servo_j(RIGHT_TARGET)",
        "left.move_gripper(LEFT_GRIPPER)",
        "right.move_gripper(RIGHT_GRIPPER)",
    ):
        assert command in command_block
    assert "loop = asyncio.get_running_loop()" in snippet
    assert "deadline = loop.time() +" in snippet
    assert "while loop.time() < deadline:" in snippet
    assert snippet.count("await asyncio.gather(") >= 2
    assert "left.get_state(), right.get_state()" in snippet
    assert 'left_state["ActualQ"]' in snippet
    assert 'right_state["ActualQ"]' in snippet
    assert 'left_state["ActualGripper"]' in snippet
    assert 'right_state["ActualGripper"]' in snippet
    assert "print(states)" in snippet
    assert "raise TimeoutError" in snippet


def test_service_installer_remains_hardware_default() -> None:
    installer = (ROOT / "scripts/install-service.sh").read_text()
    main_unit = (ROOT / "deploy/piper-control@.service.in").read_text()

    assert '"${venv_root}/bin/pip" install --upgrade "$repo_root"' in installer
    assert "--mujoco" not in installer
    assert "--mujoco" not in main_unit


def test_management_scripts_name_all_four_instances_in_safe_order() -> None:
    start = (ROOT / "scripts/start.sh").read_text()
    stop = (ROOT / "scripts/stop.sh").read_text()
    restart = (ROOT / "scripts/restart.sh").read_text()
    status = (ROOT / "scripts/status.sh").read_text()
    controllers = ("piper-control@left.service", "piper-control@right.service")
    bridges = (
        "piper-control-bridge@left.service",
        "piper-control-bridge@right.service",
    )
    for unit in (*controllers, *bridges):
        assert unit in start
        assert unit in stop
        assert unit in restart
        assert unit in status
    assert max(start.index(unit) for unit in controllers) < min(
        start.index(unit) for unit in bridges
    )
    assert max(stop.index(unit) for unit in bridges) < min(
        stop.index(unit) for unit in controllers
    )


def test_camera_unit_is_fixed_non_root_runtime_with_no_restart() -> None:
    text = (ROOT / "deploy/camera-bridge.service.in").read_text()

    assert "User=@SERVICE_USER@" in text
    assert "User=root" not in text
    assert "/opt/piper-control/venv/bin/python" in text
    assert "-m piper_control.camera_bridge.main" in text
    assert "--config /etc/piper-control/camera.yaml" in text
    assert "Restart=no" in text
    assert "Type=notify" in text
    assert "NotifyAccess=main" in text
    assert "TimeoutStartSec=45s" in text
    assert "KillSignal=SIGTERM" in text
    assert str(ROOT) not in text
    assert "@REPO_ROOT@" not in text


def test_camera_installer_validates_before_mutation_and_never_starts() -> None:
    text = (ROOT / "scripts/install-camera-service.sh").read_text()
    validation = (
        "from piper_control.camera_bridge.config import load_camera_config; "
        "load_camera_config(Path(sys.argv[1]))"
    )

    assert "SERVICE_USER CAMERA_CONFIG" in text
    assert "^[a-z_][a-z0-9_-]*$" in text
    assert validation in text
    assert 'source_python="${repo_root}/.venv/bin/python"' in text
    assert '"${venv_root}/bin/pip" install --upgrade "${repo_root}[camera]"' in text
    assert 'config_destination="/etc/piper-control/camera.yaml"' in text
    assert 'install -o root -g root -m 0644 "$config_path" "$config_destination"' in text
    assert 'chown -R root:root "$install_root"' in text
    assert 'chmod -R go-w "$install_root"' in text
    assert "systemd-analyze verify" in text
    assert "systemctl daemon-reload" in text
    assert "systemctl start" not in text
    assert "systemctl enable" not in text

    validation_index = text.index(validation)
    for mutation in (
        'install -d -o root -g root -m 0755 "$install_root" /etc/piper-control',
        'python3 -m venv "$venv_root"',
        'install -o root -g root -m 0644 "$config_path" "$config_destination"',
        '> "/etc/systemd/system/camera-bridge.service"',
        "systemctl daemon-reload",
    ):
        assert validation_index < text.index(mutation), mutation


def test_rendered_camera_unit_has_valid_systemd_syntax(tmp_path: Path) -> None:
    template = (ROOT / "deploy/camera-bridge.service.in").read_text()
    rendered = template.replace("@SERVICE_USER@", "robot_user")
    assert "@SERVICE_USER@" not in rendered
    assert "\n" not in rendered.split("User=", 1)[1].splitlines()[0]
    assert "%" not in rendered.split("User=", 1)[1].splitlines()[0]

    if shutil.which("systemd-analyze"):
        path = tmp_path / "camera-bridge.service"
        path.write_text(
            rendered.replace("/opt/piper-control/venv/bin/python", "/bin/true")
        )
        subprocess.run(
            ["systemd-analyze", "verify", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_management_scripts_order_camera_around_robot_services() -> None:
    start = (ROOT / "scripts/start.sh").read_text()
    stop = (ROOT / "scripts/stop.sh").read_text()
    restart = (ROOT / "scripts/restart.sh").read_text()
    status = (ROOT / "scripts/status.sh").read_text()
    camera = "camera-bridge.service"
    controllers = ("piper-control@left.service", "piper-control@right.service")
    bridges = (
        "piper-control-bridge@left.service",
        "piper-control-bridge@right.service",
    )

    for text in (start, stop, restart, status):
        assert camera in text
    assert start.index(camera) < min(start.index(unit) for unit in controllers)
    assert max(stop.index(unit) for unit in (*bridges, *controllers)) < stop.index(camera)
    restart_stop, restart_start = restart.split("systemctl restart", 1)
    assert max(restart_stop.index(unit) for unit in (*bridges, *controllers)) < restart_stop.index(camera)
    assert restart_start.index(camera) < min(
        restart_start.index(unit) for unit in controllers
    )


def test_readme_documents_camera_bridge_operation_and_fault_policy() -> None:
    text = (ROOT / "README.md").read_text()
    required = (
        "scripts/install-camera-service.sh",
        "camera/left_wrist/color",
        "camera/right_wrist/depth",
        "camera/head/infrared_2",
        "topic + JSON metadata + payload",
        "host_monotonic_ns",
        "RCVHWM=14",
        "journalctl -u camera-bridge.service",
        "Restart=no",
        "连续 3 次",
        "唯一一次重连",
        "examples/camera_zmq_client.py",
    )
    for expected in required:
        assert expected in text, expected


def test_readme_documents_bridge_operation_and_hardware_acceptance() -> None:
    text = (ROOT / "README.md").read_text()
    required = (
        "RUNNING → HOLD → local/remote → RUNNING",
        '"identity":"local"',
        '"identity":"bridge"',
        '"cmd":"release"',
        "priority",
        "1 秒",
        "3 秒",
        "控制频率 × 5",
        "piper-control@left",
        "piper-control@right",
        "piper-control-bridge@left",
        "piper-control-bridge@right",
        "低速",
        "急停",
    )
    for expected in required:
        assert expected in text, expected
