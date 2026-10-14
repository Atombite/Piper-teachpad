# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
COPYRIGHT = "Copyright © 2026-2026 atombite.ai"
SPDX = "SPDX-License-Identifier: Apache-2.0"
COMMENT_SUFFIXES = {
    ".py", ".js", ".mjs", ".html", ".css", ".sh", ".yaml",
    ".rules", ".toml", ".xml", ".urdf",
}
COMMENT_NAMES = {".gitignore", ".gitmodules"}
EXEMPT_NAMES = {
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "README.md", "README.zh-CN.md",
    "package.json", "package-lock.json", "src/piper_control/web/teach.bundle.js",
    "thirdparty/mujoco_menagerie",
}
EXEMPT_PREFIXES = (
    "src/piper_control/web/vendor/", "src/piper_control/web/models/",
    "thirdparty/",
)


def retained_paths() -> list[str]:
    """Return the public tree in both the curation and published repositories."""
    manifest = ROOT / "PUBLIC_FILES.txt"
    if manifest.is_file():
        return manifest.read_text(encoding="utf-8").splitlines()

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        name.decode("utf-8")
        for name in result.stdout.split(b"\0")
        if name and name != b"PUBLIC_FILES.txt"
    )


RETAINED = retained_paths()
HEADER_FILES = [
    name for name in RETAINED
    if name not in EXEMPT_NAMES and not name.startswith(EXEMPT_PREFIXES)
]
PROTECTED_LICENSES = {
    "PIPER-MODEL-LICENSE.txt": "e32d78fd3761052ab0b2bb42fef8a7a9ad3f298d315d305333c349d2aff1f134",
    "THREE-LICENSE.txt": "8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc",
    "URDF-LOADER-LICENSE.txt": "6e953d21f4cff0ec408189fd3f49eb4e062719d4011d18fd9d5d9932a06efe55",
}

FORBIDDEN_PATH_PATTERNS = (
    "src/piper_control/workflows/**/*",
    "src/piper_control/box_turn*.py",
    "config/artifacts/**/*",
    "config/*box-turn*",
    "config/*fixed-trajectory*",
    "examples/*five_marker*",
    "examples/*fixed_trajectory*",
    "examples/*box_turn*",
    "tools/*box_turn*",
    "tests/test_box_turn*.py",
    "tests/test_five_marker*.py",
    "tests/test_marker_backtest*.py",
    "tests/test_workflow*.py",
    "tests/test_piper_sdk_guide.py",
    "artifacts/**/*",
    "docs/piper_sdk_api_guide.html",
    "docs/*" + "RS" + "00*",
    "docs/*" + "rs" + "00*",
    "docs/superpowers/**/*",
)

RUNTIME_PREFIXES = ("config/", "deploy/", "examples/", "src/", "tools/")
PRIVATE_IPV4 = re.compile(
    r"(?<![\d.])(?:"
    r"10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}"
    r")(?![\d.])"
)
SERIAL_DEVICE_PATH = re.compile(r"/dev/serial/by-id/[^\s\"'`,)]+")
YAML_SERIAL_VALUE = re.compile(
    r"^\s*serial:\s*[\"']?([^\"'#\s]+)", re.MULTILINE
)
LEGACY_GRIPPER_TOKEN = re.compile(r"rs[0-9]{2}", re.IGNORECASE)
SCP_GIT_REMOTE = re.compile(
    r"(?<![\w.-])(?:git@)?[a-z0-9_.-]+(?:\.[a-z0-9_-]+)+:"
    r"[^\s`'\"]+\.git\b",
    re.IGNORECASE,
)
INTERNAL_COMPANY_HOST = re.compile(
    r"\b(?:[a-z0-9-]+\.)*atombite\.(?!ai\b)[a-z]{2,}\b",
    re.IGNORECASE,
)
PUBLIC_SSH_REMOTE = "git@github.com:Atombite/Piper-teachpad.git"
PUBLIC_BROWSER_URL = "https://github.com/Atombite/Piper-teachpad"
PUBLIC_GUIDES = ("README.md", "README.zh-CN.md")


def test_public_tree_excludes_application_workflows_and_internal_artifacts() -> None:
    forbidden = {
        path.relative_to(ROOT).as_posix()
        for pattern in FORBIDDEN_PATH_PATTERNS
        for path in ROOT.glob(pattern)
        if path.is_file()
    }
    if (ROOT / ".git").exists():
        # Ignore local orchestration scratch; tracked or publishable records are forbidden.
        records = subprocess.run(
            [
                "git", "ls-files", "-z", "--cached", "--others",
                "--exclude-standard", "--", ".superpowers",
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.split("\0")
        forbidden.update(name for name in records if name and (ROOT / name).is_file())
    else:
        forbidden.update(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / ".superpowers").rglob("*") if path.is_file()
        )
    forbidden = sorted(forbidden)
    assert not forbidden, "Non-public files remain:\n" + "\n".join(forbidden)


def test_public_package_excludes_vision_workflow_extra() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert not re.search(r"^\s*[\"']?vision-workflow[\"']?\s*=", metadata, re.MULTILINE)


def test_public_text_excludes_private_endpoints_and_legacy_identifiers() -> None:
    offenders: list[str] = []
    for name in RETAINED:
        path = ROOT / name
        if not path.is_file():
            continue
        contents = path.read_bytes()
        if b"\0" in contents[:8192]:
            continue
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if PRIVATE_IPV4.search(text):
            offenders.append(f"{name}: private IPv4 endpoint")
        if LEGACY_GRIPPER_TOKEN.search(text):
            offenders.append(f"{name}: legacy gripper identifier")
        if INTERNAL_COMPANY_HOST.search(text):
            offenders.append(f"{name}: internal company host")
        remotes = SCP_GIT_REMOTE.findall(text)
        if any(remote != PUBLIC_SSH_REMOTE for remote in remotes):
            offenders.append(f"{name}: non-public scp-style Git remote")
        if name.startswith(RUNTIME_PREFIXES):
            if any(
                "REPLACE_WITH_" not in value
                for value in SERIAL_DEVICE_PATH.findall(text)
            ):
                offenders.append(f"{name}: concrete serial device path")
            if path.suffix in {".yaml", ".yml"} and any(
                not value.startswith("REPLACE_WITH_")
                for value in YAML_SERIAL_VALUE.findall(text)
            ):
                offenders.append(f"{name}: concrete YAML serial")

    assert offenders == []


def test_retained_python_files_do_not_import_removed_modules() -> None:
    forbidden = []
    for name in RETAINED:
        if not name.endswith(".py"):
            continue
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""] + [
                    f"{node.module}.{alias.name}" for alias in node.names
                ]
            else:
                continue
            if any(
                module == "piper_control.workflows"
                or module.startswith(("piper_control.workflows.", "piper_control.box_turn"))
                for module in modules
            ):
                forbidden.append(f"{name}:{node.lineno}")
    assert not forbidden, "Retained imports of non-public modules: " + ", ".join(forbidden)


def test_retained_file_formats_require_explicit_attribution_review() -> None:
    unknown = [
        name for name in HEADER_FILES
        if Path(name).suffix not in COMMENT_SUFFIXES
        and name not in COMMENT_NAMES and not name.endswith(".service.in")
    ]
    assert not unknown, f"Classify new retained formats for safe attribution: {unknown}"


@pytest.mark.parametrize("name", HEADER_FILES)
def test_first_party_files_carry_syntax_safe_attribution(name: str) -> None:
    path = ROOT / name
    lines = path.read_text(encoding="utf-8").splitlines()
    offset = int(bool(lines and (lines[0].startswith("#!") or lines[0].startswith("<?xml"))))
    if path.suffix in {".html", ".xml", ".urdf"}:
        expected = [f"<!-- {COPYRIGHT}", f"     {SPDX} -->"]
    elif path.suffix == ".css":
        expected = [f"/* {COPYRIGHT}", f" * {SPDX} */"]
    elif path.suffix in {".js", ".mjs"}:
        expected = [f"// {COPYRIGHT}", f"// {SPDX}"]
    else:
        expected = [f"# {COPYRIGHT}", f"# {SPDX}"]
    assert lines[offset:offset + 2] == expected, name


def test_release_includes_official_license_and_company_notice() -> None:
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "NOTICE").is_file()
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    # Debian's canonical license copy, independently pinned before attribution edits.
    assert hashlib.sha256(license_text.encode()).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )
    assert COPYRIGHT in (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Apache-2.0" in (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "https://github.com/Atombite/Piper-teachpad" in (
        ROOT / "NOTICE"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("name, expected", PROTECTED_LICENSES.items())
def test_vendor_licenses_keep_original_notices(name: str, expected: str) -> None:
    contents = (ROOT / "src/piper_control/web/vendor" / name).read_bytes()
    assert hashlib.sha256(contents).hexdigest() == expected
    assert COPYRIGHT.encode() not in contents


def test_generated_bundle_is_attributed_through_its_build_sources() -> None:
    bundle = (ROOT / "src/piper_control/web/teach.bundle.js").read_text(encoding="utf-8")
    assert COPYRIGHT not in bundle[:256]
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for reference in (
        "teach.bundle.js", "tools/build_teach_frontend.mjs", "teach-main.js",
        "THREE-LICENSE.txt", "URDF-LOADER-LICENSE.txt", "PIPER-MODEL-LICENSE.txt",
        "thirdparty/mujoco_menagerie",
    ):
        assert reference in notices


@pytest.mark.parametrize(
    "name, counterpart",
    (("README.md", "README.zh-CN.md"), ("README.zh-CN.md", "README.md")),
)
def test_public_guides_cross_link_and_describe_supported_entry_points(
    name: str, counterpart: str
) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    required = (
        counterpart,
        PUBLIC_SSH_REMOTE,
        PUBLIC_BROWSER_URL,
        "piper-control",
        "piper-control-bridge",
        "piper-control-can",
        "piper-calibrate",
        "camera-bridge",
        "Safety Console",
        "Teach Pad",
        "config/piper-control.example.yaml",
        "config/piper-control.mujoco.yaml",
        "config/piper-control-bridge.example.yaml",
        "config/camera.yaml",
        "examples/gello_ws_teleop.py",
        "examples/lerobot_ws_replay.py",
        "examples/camera_zmq_client.py",
        "Copyright © 2026-2026 atombite.ai",
        "Apache-2.0",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
    )
    missing = [item for item in required if item not in text]
    assert not missing, f"{name} misses public guide contracts: {missing}"


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_public_guides_cover_safety_units_and_retained_interfaces(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    required = (
        "8765",
        "9876",
        "18765",
        "19876",
        "15555",
        "/teach",
        "/ws/control",
        "servoJ",
        "move_gripper",
        "camera/left_wrist/color",
        "camera/right_wrist/depth",
        "camera/head/infrared_2",
        "Piper",
        "mm",
        "m/s",
        "E-stop",
    )
    missing = [item for item in required if item not in text]
    assert not missing, f"{name} misses safety/interface details: {missing}"

    folded = text.casefold()
    assert "authentication" in folded
    assert "trusted" in folded
    assert "certif" in folded
    forbidden = (
        "docs/" + "rs" + "00",
        "box" + "-turn",
        "fixed" + "-trajectory",
    )
    assert not any(item in folded for item in forbidden)


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_public_guides_name_the_actual_teach_build_scripts(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "npm ci" in text
    assert "npm run build:teach" in text
    assert "npm run build:teach-tests" in text
    assert "npm run build:test\n" not in text


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_public_guides_distinguish_controller_and_prioritized_bridge_urls(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    for value in (
        "ws://HOST:8765/ws/control",
        "ws://HOST:9876/ws/control",
        "ws://HOST:18765/ws/control",
        "ws://HOST:19876/ws/control",
        '"cmd": "servoJ"',
        '"cmd": "move_gripper"',
        '"cmd": "release"',
        '"priority": 100',
    ):
        assert value in text, f"{name}: {value}"
    assert "PiperControlClient" not in text.split("prioritized bridge client", 1)[-1].split("```", 2)[1]


def test_chinese_guide_contains_copyable_bounded_dual_arm_mujoco_check() -> None:
    text = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    snippet = text.split("PYTHONPATH=. .venv/bin/python - <<'PY'\n", 1)[1].split(
        "\nPY\n", 1
    )[0]
    compile(snippet, "<Chinese README MuJoCo acceptance>", "exec")
    for value in (
        "async with (",
        "left.servo_j(LEFT_TARGET)",
        "right.servo_j(RIGHT_TARGET)",
        "left.move_gripper(LEFT_GRIPPER)",
        "right.move_gripper(RIGHT_GRIPPER)",
        "deadline = loop.time() +",
        "left.get_state(), right.get_state()",
        'left_state["ActualQ"]',
        'right_state["ActualGripper"]',
        "raise TimeoutError",
    ):
        assert value in snippet


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_public_guides_install_all_full_verification_extras(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    assert ".venv/bin/pip install -e '.[dev,mujoco,camera,dataset]'" in text


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_public_guides_keep_gello_and_lerobot_on_exclusive_local_controllers(
    name: str,
) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    before_command = text.split("examples/gello_ws_teleop.py", 1)[0]
    examples = text[before_command.rfind("\n## ") :]
    for value in (
        "PiperControlClient",
        'identity="local"',
        "ws://127.0.0.1:8765/ws/control",
        "ws://127.0.0.1:9876/ws/control",
        "local mode",
        "exclusive control",
        "18765/19876",
    ):
        assert value in examples, f"{name}: {value}"


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_public_guides_do_not_start_uninstalled_camera_from_arm_only_flow(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    arm_systemd = text.split("### systemd", 1)[1].split("## ", 1)[0]
    assert "scripts/start.sh" not in arm_systemd
    for unit in (
        "piper-control@left.service",
        "piper-control@right.service",
        "piper-control-bridge@left.service",
        "piper-control-bridge@right.service",
    ):
        assert unit in arm_systemd
    all_stack = text.split("scripts/start.sh", 1)[0].rsplit("camera-bridge.service", 1)
    assert len(all_stack) == 2, f"{name}: camera prerequisite must precede scripts/start.sh"


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_every_public_guide_python_fence_is_standalone_valid_syntax(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    snippets = re.findall(r"^```python\n(.*?)^```$", text, re.MULTILINE | re.DOTALL)
    assert snippets, f"{name}: expected at least one Python example"
    for index, snippet in enumerate(snippets, 1):
        compile(snippet, f"<{name} Python fence {index}>", "exec")


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_manual_controller_commands_use_an_explicit_writable_lock(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    commands = re.findall(r"\.venv/bin/piper-control --config [^\n]+", text)
    assert commands, name
    assert all("--lock-path /tmp/" in command for command in commands), commands


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_public_guides_distinguish_controller_and_bridge_disconnect_safety(
    name: str,
) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    for value in (
        "controller control-lease disconnect",
        "downstream piper-control-bridge client disconnect",
        "queued priority lane",
        "may remain RUNNING",
        "Safety Console",
    ):
        assert value in normalized, f"{name}: {value}"


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_system_install_examples_define_an_existing_non_root_service_user(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    assert 'SERVICE_USER="$USER"' in text
    assert 'install-service.sh "$SERVICE_USER"' in text
    assert 'install-camera-service.sh "$SERVICE_USER"' in text
    assert "existing non-root" in text
    assert not re.search(r"install-(?:camera-)?service\.sh robot\b", text)


@pytest.mark.parametrize("name", PUBLIC_GUIDES)
def test_public_guides_discover_and_label_each_hardware_identifier_type(name: str) -> None:
    text = (ROOT / name).read_text(encoding="utf-8")
    for value in (
        "GELLO/serial devices",
        "camera devices",
        "USB-CAN adapters",
        "ip -details link show type can",
        "udevadm info --attribute-walk --path=/sys/class/net/can0",
        "udevadm info --attribute-walk --path=/sys/class/net/can1",
        "ATTRS{serial}",
        "REPLACE_WITH_LEFT_PIPER_CAN_ADAPTER_SERIAL",
        "REPLACE_WITH_RIGHT_PIPER_CAN_ADAPTER_SERIAL",
    ):
        assert value in text, f"{name}: {value}"
