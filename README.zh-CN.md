# Piper Teach Pad

[English](README.md)

Piper Teach Pad 是面向 AgileX Piper 双臂的公开控制栈。核心场景由 **Safety Console**
安全管理终端、浏览器 **Teach Pad** 示教器、供算法或遥操作接入的
`piper-control-bridge`（arm bridge）与 `camera-bridge`，以及使用同一 API 的 MuJoCo
验证环境组成。

## 安全边界

> **机械臂运动可能造成人身伤害或设备损坏。** 仅允许经过培训的人员在现场监护下使用。
> 使能前必须清空完整工作区，确认物理 E-stop（急停）、机械臂就绪与新鲜反馈，检查关节/
> 夹爪限位和左右 CAN 映射，并用保守速度与短行程开始首次运动。操作员应始终能触达急停。

网页和 Bridge **没有 authentication（身份认证）或传输加密**。只能部署在 trusted、隔离的
局域网，不能把端口暴露到不可信网络或互联网。本项目没有 safety certification（安全认证），
也不是经过生产加固的系统。MuJoCo 只验证软件接口与场景，不构成风险评估或真机运动许可。
直接 controller control-lease disconnect 会进入 HOLD；downstream piper-control-bridge client
disconnect 只释放该客户端的优先级 lane，另一个 queued priority lane 可能接管，而 Bridge 的
上游连接和 controller may remain RUNNING 并保持/命令当前位置。不能把下游断连当作停止；
应使用 Safety Console 或 controller 的 E-stop/HOLD。不要用 `kill -9`、断电或关闭 CAN 代替
受控停止。

## 组件和默认端点

| 组件 | 程序/路由 | 默认端点 | 用途 |
|---|---|---|---|
| 控制器 | `piper-control` | 左 `8765`，右 `9876` | 安全状态机、REST/WebSocket、网页 |
| Safety Console | `GET /` | `http://HOST:8765/`、`http://HOST:9876/` | 状态、就绪、E-stop、恢复、归零、模式 |
| Teach Pad | `GET /teach` | `http://HOST:8765/teach`、`http://HOST:9876/teach` | 单臂示教与夹爪控制 |
| arm bridge | `piper-control-bridge` | 左 `18765`，右 `19876` | 带优先级的多客户端 WebSocket |
| 相机桥 | `camera-bridge` | `tcp://HOST:15555` | 七路最新图像主题 |
| CAN 工具 | `piper-control-can` | 本机 SocketCAN | 固定 1 Mbit/s 的 `up/status/down` |
| 标定 | `piper-calibrate` | 本机 SocketCAN | 只读原始关节标定报告 |

每只臂独立接收六个关节量和一个夹爪标量。左右臂都使用 Piper 原生夹爪。关节位置/速度
单位为度和度/秒；夹爪 wire/API 命令单位为米和 `m/s`，Teach Pad 按 `mm`、`mm/s` 显示。

## 前置条件与安装

需要 Linux 和 Python 3.10 以上；真机需要 SocketCAN。网页需要浏览器。只有重建/测试
Teach Pad bundle 时需要 Node.js/npm。交互式 MuJoCo Viewer 需要 GLFW 和 `DISPLAY`；
headless 不需要显示器。

```bash
git clone --recurse-submodules git@github.com:Atombite/Piper-teachpad.git
cd Piper-teachpad
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

可选依赖可以组合：

```bash
.venv/bin/pip install -e '.[dev,mujoco,camera,dataset]'
npm ci
```

`mujoco` 提供动力学；`camera` 提供 V4L2/RealSense/ZeroMQ；`dataset` 提供 LeRobot
所需 Parquet；`dev` 提供测试依赖。硬件驱动基于
[AgileX Piper SDK](https://github.com/agilexrobotics/piper_sdk)，模型来自
[Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper)。

## 配置与设备发现

从以下保留且经过校验的示例开始：

- `config/piper-control.example.yaml`：左右 Piper 真机控制器；
- `config/piper-control-bridge.example.yaml`：左右 arm bridge；
- `config/piper-control.mujoco.yaml`：双臂 MuJoCo；
- `config/camera.yaml`：一台 V4L2 腕部相机和两台 RealSense。

编辑前先复制。控制配置保留 `arms.left` 和/或 `arms.right`；名称、CAN 接口、HTTP 端口不能
重复，映射键就是 side。每个启用的夹爪必须是 `backend: piper`。按实际硬件审核标定、
逻辑限位、速度限位和 reset 姿态。

稳定设备标识必须在本机发现，不能复制其他机器人的值。严格按下面标注的设备类型使用：

```bash
# GELLO/serial devices（不是 CAN udev 规则的来源）：
ls -l /dev/serial/by-id/

# camera devices：
ls -l /dev/v4l/by-id/
rs-enumerate-devices

# USB-CAN adapters：先列出当前 CAN netdev，再逐个检查。
ip -details link show type can
udevadm info --attribute-walk --path=/sys/class/net/can0
udevadm info --attribute-walk --path=/sys/class/net/can1
```

在每个 USB-CAN attribute walk 中找到适配器的 `ATTRS{serial}`。必须先通过逐个连接等方式，
物理确认哪个适配器对应左臂、哪个对应右臂，再替换 `deploy/70-piper-can.rules` 中的
`REPLACE_WITH_LEFT_PIPER_CAN_ADAPTER_SERIAL` 和
`REPLACE_WITH_RIGHT_PIPER_CAN_ADAPTER_SERIAL`。相机占位符单独替换到
`config/camera.yaml`。安装 udev 规则后重新插拔 USB-CAN 或重启，并在运动前把
`can_left`/`can_right` 与实体机械臂再次核对。CAN 固定为 1,000,000 bit/s：

```bash
sudo .venv/bin/piper-control-can --config config/piper-control.example.yaml up
.venv/bin/piper-control-can --config config/piper-control.example.yaml status
```

`scripts/can-up.sh`、`scripts/can-status.sh`、`scripts/can-down.sh` 是相同 CLI 的包装器。

### 只读标定

先停止 controller，把每只臂放到目标逻辑 reset 姿态。`piper-calibrate` 只读原始反馈，不会
使能、归零、HOLD、急停、复位、失能或发送运动。最初 50 个新鲜帧保持静止，然后缓慢覆盖
审核过的范围，按 Ctrl+C：

```bash
.venv/bin/piper-calibrate --config config/piper-control.example.yaml > calibration-result.yaml
```

审核后再复制报告中的标定和 min/max。

## 运行真机和 Bridge

配置并确认 CAN 后启动全部配置臂；也可以加 `--side left` 或 `--side right`：

```bash
sudo .venv/bin/piper-control-can --config config/piper-control.example.yaml up
.venv/bin/piper-control --config config/piper-control.example.yaml --lock-path /tmp/piper-control-demo.lock
```

手工运行显式使用 `/tmp` 下当前用户可写的锁。systemd unit 会创建每臂运行目录，并使用配置
好的 `/run/piper-control-ARM/piper-control.lock`，不需要照搬手工锁路径。

每次重启从 `local` 开始。先检查两个 Safety Console、反馈和物理急停。进入 Bridge 控制的
明确顺序是 `RUNNING → HOLD → local/remote → RUNNING`，只有 HOLD 才能切换模式。

```bash
.venv/bin/piper-control-bridge --config config/piper-control-bridge.example.yaml --side left
.venv/bin/piper-control-bridge --config config/piper-control-bridge.example.yaml --side right
```

Teach Pad 首帧是 `{"id":"h1","cmd":"hello","identity":"local"}`；Bridge 上游首帧是
`{"id":"h1","cmd":"hello","identity":"bridge"}`。对应臂进入 `remote` 后，算法才应连接
Bridge 端口。

### systemd

安装器会先校验 YAML，再复制 root 管理的环境；它不会 enable 或 start：

```bash
SERVICE_USER="$USER"  # 必须是已存在的 existing non-root 服务账户。
sudo scripts/install-service.sh "$SERVICE_USER" config/piper-control.example.yaml \
  config/piper-control-bridge.example.yaml
sudo systemctl start piper-control-can.service
sudo systemctl start piper-control@left.service piper-control@right.service
sudo systemctl start piper-control-bridge@left.service piper-control-bridge@right.service
systemctl status piper-control@left.service piper-control@right.service \
  piper-control-bridge@left.service piper-control-bridge@right.service
```

四个 arm unit 是 `piper-control@left`、`piper-control@right`、
`piper-control-bridge@left`、`piper-control-bridge@right`。启动时 controller 在 Bridge 前；
停止/重启时 Bridge 先停。

## Safety Console 与 Teach Pad

访问 controller 的 `/` 打开 Safety Console。它显示状态、反馈和故障原因，急停始终可用，
并提供明确的恢复、归零、local/remote 操作。不能靠重启绕过锁存故障。

同端口 `/teach` 是 Teach Pad，离线渲染 Piper URDF，只控制该端口的机械臂：

- **Preview** 只改目标，不获取控制权、不发送运动；
- **Reviewed execution** 重新检查状态、身份、限位后发送冻结目标；
- **Live follow** 要求显式获取控制权并持续按住；松手、失焦、反馈过期、状态变化或网络
  故障都会断开并 HOLD；
- 夹爪卡片共享同一排他控制权，使用 Piper 原生限位。

每个 controller 端口一次只允许一个控制客户端；冲突返回 `CLIENT_CONFLICT`，左右端口独立。

## Controller 与 arm bridge 接口

controller WebSocket 是 `ws://HOST:8765/ws/control` 和
`ws://HOST:9876/ws/control`，支持 `get_state`、`get_base_info`、`servoJ`、
`move_gripper`，第一帧必须是正确的 `hello`。严格客户端示例：

```python
import asyncio
from examples.ws_client import PiperControlClient

async def main():
    async with PiperControlClient("ws://127.0.0.1:8765/ws/control") as client:
        state = await client.get_state()
        info = await client.get_base_info()
        await client.servo_j([0.0] * 6, speed=[5.0] * 6)
        await client.move_gripper(0.02, speed=0.01)
        print({"state": state, "base_info": info})

asyncio.run(main())
```

arm bridge 提供相同运动/状态命令，但不需要 `hello`；`servoJ`/`move_gripper` 增加整数
`priority`，并支持 `{"id":"r1","cmd":"release","priority":100}`。关节和夹爪共享整臂
所有权：高优先级抢占，同优先级保持当前所有者。命令 1 秒过期，3 秒无新命令释放所有者，
每个 lane 容量为控制频率 × 5。断连释放该客户端全部 lane。ACK 只表示接受/排队，不表示到位。

算法与遥操作客户端应连接下游 prioritized Bridge，而不是 controller 直连端口：左侧
`ws://HOST:18765/ws/control`，右侧 `ws://HOST:19876/ws/control`。`8765`/`9876` 是
controller 端口，需要 `hello` 并检查 local/bridge 身份；Bridge 端口不需要 `hello`，支持
`get_state`、`get_base_info`、`servoJ`、`move_gripper`、`release`。下面是可复制的
prioritized bridge client；关节使用度/度每秒，Piper 夹爪使用米/米每秒：

```python
import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://HOST:18765/ws/control") as bridge:
        requests = (
            {"id": "j1", "cmd": "servoJ", "target_joints": [0.0] * 6,
             "speed": [5.0] * 6, "priority": 100},
            {"id": "g1", "cmd": "move_gripper", "target_position": 0.02,
             "speed": 0.01, "priority": 100},
            {"id": "r1", "cmd": "release", "priority": 100},
        )
        for request in requests:
            await bridge.send(json.dumps(request))
            response = json.loads(await bridge.recv())
            if response.get("id") != request["id"] or response.get("ok") is not True:
                raise RuntimeError(response)
            print(response)

asyncio.run(main())
```

## 双臂 MuJoCo 验证

MuJoCo 不构造硬件驱动、不打开 CAN：

```bash
git submodule update --init --recursive
.venv/bin/pip install -e '.[mujoco]'
.venv/bin/piper-control --config config/piper-control.mujoco.yaml --mujoco --lock-path /tmp/piper-control-demo.lock
.venv/bin/piper-control --config config/piper-control.mujoco.yaml --mujoco-headless --lock-path /tmp/piper-control-demo.lock
```

没有 `DISPLAY` 时使用 headless；关闭 Viewer 会请求正常停机。两个基座相距 52.6 cm，Piper
API 的 0–100 mm 映射到模型 0–70 mm（`100 mm -> 70 mm`）。服务运行后，可使用
`examples/ws_client.py` 同时保持左右客户端。以下完整脚本向两侧发送 `servoJ` 和
`move_gripper`，轮询 `ActualQ`/`ActualGripper`，并在 20 秒后有界失败：

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
import asyncio
from examples.ws_client import PiperControlClient

LEFT_TARGET = [0, 45, -45, 5, 5, 5]
RIGHT_TARGET = [0, 55, -55, -5, -5, -5]
LEFT_GRIPPER = 0.10
RIGHT_GRIPPER = 0.05

def close_enough(actual, target, tolerance):
    return all(abs(a - b) <= tolerance for a, b in zip(actual, target, strict=True))

async def main():
    async with (
        PiperControlClient("ws://127.0.0.1:8765/ws/control") as left,
        PiperControlClient("ws://127.0.0.1:9876/ws/control") as right,
    ):
        await asyncio.gather(
            left.servo_j(LEFT_TARGET),
            right.servo_j(RIGHT_TARGET),
            left.move_gripper(LEFT_GRIPPER),
            right.move_gripper(RIGHT_GRIPPER),
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 20.0
        states = None
        while loop.time() < deadline:
            left_state, right_state = await asyncio.gather(
                left.get_state(), right.get_state()
            )
            states = {"left": left_state, "right": right_state}
            if (
                close_enough(left_state["ActualQ"], LEFT_TARGET, 2.0)
                and close_enough(right_state["ActualQ"], RIGHT_TARGET, 2.0)
                and abs(left_state["ActualGripper"] - LEFT_GRIPPER) <= 0.015
                and abs(right_state["ActualGripper"] - RIGHT_GRIPPER) <= 0.015
            ):
                print(states)
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"MuJoCo targets did not converge within 20 s: {states}")

asyncio.run(main())
PY
```

轻量模拟同样不访问 CAN：

```bash
.venv/bin/piper-control --config config/piper-control.example.yaml --simulate --lock-path /tmp/piper-control-sim.lock
```

## 通用算法和遥操作示例

保留的 GELLO 和 LeRobot 示例是 prioritized arm Bridge 的替代入口。它们使用
`PiperControlClient` 默认的 `identity="local"`，直接连接 controller 的 `8765`/`9876`。
运动前必须让两侧 controller 保持 `local mode`，并释放 Teach Pad 和其他客户端，保证示例
拥有 exclusive control。绝不能把这些示例指向 Bridge 端口 `18765/19876`；需要优先级仲裁
的客户端应使用上面的原始 Bridge 协议。

从[上游仓库](https://github.com/wuphilipp/gello_software)安装 GELLO。清空工作区，让 leader
与对应 Piper 前六轴在默认 5° 内，从 10°/s 低速开始。运动需要两道确认：

```bash
.venv/bin/python examples/gello_ws_teleop.py \
  --left-port /dev/serial/by-id/REPLACE_WITH_LEFT_GELLO_SERIAL_DEVICE \
  --right-port /dev/serial/by-id/REPLACE_WITH_RIGHT_GELLO_SERIAL_DEVICE \
  --left-url ws://127.0.0.1:8765/ws/control \
  --right-url ws://127.0.0.1:9876/ws/control \
  --allow-motion --confirm-motion I_UNDERSTAND
```

它以 50 Hz 独立发送六关节命令，并忽略 GELLO 夹爪通道。越限、串口、状态或 WebSocket
错误会关闭对应侧，使 controller HOLD。

LeRobot 需要 `.[dataset]`。没有运动开关时，只校验并打印 episode，不连接 WebSocket：

```bash
.venv/bin/python examples/lerobot_ws_replay.py /path/to/piper-dataset \
  --source actions --episode-index 0
```

监护下回放要使用保守 rate 和两道确认：

```bash
.venv/bin/python examples/lerobot_ws_replay.py /path/to/piper-dataset \
  --source actions --rate 0.5 \
  --left-url ws://127.0.0.1:8765/ws/control \
  --right-url ws://127.0.0.1:9876/ws/control \
  --joint-speed-deg-s 10 \
  --gripper-speed-m-s 0.02 --allow-motion --confirm-motion I_UNDERSTAND
```

接受的 v2 布局是 16 个 float32：左 J1–J6、左夹爪、右 J1–J6、右夹爪、左右夹爪副本。
关节在数据中为弧度、API 中为度；夹爪保持米。

## Camera bridge

在 `config/camera.yaml` 填入本机标识，安装 extra 后人工启动：

```bash
.venv/bin/pip install -e '.[camera]'
.venv/bin/camera-bridge --config config/camera.yaml
```

systemd 安装器先校验再修改系统，且不会自动启动：

```bash
SERVICE_USER="$USER"  # 必须是已存在的 existing non-root 服务账户。
sudo scripts/install-camera-service.sh "$SERVICE_USER" config/camera.yaml
sudo systemctl start camera-bridge.service
systemctl status camera-bridge.service
```

两个安装器都会在修改系统前确认 `SERVICE_USER` 存在。安装后的服务不能以 root 运行；如果
`$USER` 是 root，请显式填写另一个已存在的非 root 专用账户。

只有 arm service 与 `camera-bridge.service` 都已安装、配置已审核后，才能使用全栈助手
`scripts/start.sh`；它依次启动 camera、controller、arm bridge。仅完成 arm 安装时不要调用，
因为 camera unit 尚不存在。`scripts/status.sh`、`scripts/restart.sh`、`scripts/stop.sh` 同样
面向完整安装后的全栈。

七路主题包括 `camera/left_wrist/color`、`camera/right_wrist/depth`、
`camera/head/infrared_2`。每幅图是 `topic + JSON metadata + payload` 三段消息。使用
`host_monotonic_ns` 计算帧龄/软件配对。客户端使用 `RCVHWM=14`，排空到最新帧：

```bash
PYTHONPATH=src .venv/bin/python examples/camera_zmq_client.py \
  --endpoint tcp://127.0.0.1:15555 --duration 10 --max-age-ms 500
PYTHONPATH=src .venv/bin/python examples/camera_preview_gui.py \
  --endpoint 127.0.0.1:15555
```

`Restart=no` 用来保留故障。允许一次超时；连续三次失败进入全局 fault。明确掉线会暂停全部
图像，只允许一次重连。查看详情：

```bash
journalctl -u camera-bridge.service -b
```

## 验证与排障

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src examples tools
for f in scripts/*.sh; do bash -n "$f"; done
npm ci
npm run build:teach
npm run build:teach-tests
git diff --check
```

- 配置：对照四个公开 YAML，检查端点唯一、Piper 夹爪 backend、标定和限位；
- CAN：检查稳定映射、1 Mbit/s、终端电阻和接线；
- `EMERGENCY_STOP`：排除 `fault_reason`，确认安全后明确恢复，不能绕过锁存；
- WebSocket：检查所有权、身份/模式和 `RUNNING` 状态；
- MuJoCo：初始化子模块并检查 `thirdparty/mujoco_menagerie/agilex_piper/piper.xml`；
- Viewer：没有 `DISPLAY` 时使用 `--mujoco-headless`；
- 相机：核对标识并查看 `journalctl -u camera-bridge.service -b`，不要用重启循环掩盖失明。

## 仓库结构

- `src/piper_control/`：controller、两个页面、arm bridge、camera bridge；
- `config/`：真机、Bridge、相机、MuJoCo 示例；
- `examples/`：WebSocket、GELLO、LeRobot、相机客户端；
- `scripts/`、`deploy/`：CAN 与 systemd；
- `tests/`：Python、浏览器侧 JS、仿真检查；
- `thirdparty/mujoco_menagerie/`：固定版本的公开模型子模块。

## 许可证与署名

Copyright © 2026-2026 atombite.ai

项目采用 Apache License 2.0（`Apache-2.0`），见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
模型与浏览器依赖保留各自条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。再分发
修改版时必须保留要求的版权、许可证、NOTICE 和第三方声明。

权威源码：<https://github.com/Atombite/Piper-teachpad>
