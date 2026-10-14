// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import {
  assert,
  assertDeepEqual,
  assertEqual,
  scheduleTests,
  test,
} from "./teach-test-harness.js";
import {
  PiperWebSocketClient,
  controlAvailability,
  createControlLifecycle,
  createLiveControlSession,
  createLiveFollower,
  executeReviewedTarget,
  failClosedOnTelemetryFailure,
  readPageExpectedIdentity,
} from "../../src/piper_control/web/teach-control.js";
import {
  createTeachingState,
  motionBlockReason,
  reduceTeachingState,
  resolveSynchronousIk,
} from "../../src/piper_control/web/teach-state.js";

const BASE_INFO = {
  protocol_version: "3.0",
  control_mode: "local",
  side: "left",
  supported_commands: ["hello", "servoJ", "get_state", "get_base_info"],
  units: { joint_position: "degree", joint_speed: "degree/second" },
  arm: {
    server: { port: 8765 },
    joint_limits_deg: {
      min: [-150, 0, -154, -100, -69, -120],
      max: [150, 179, 0, 100, 69, 120],
    },
    joint_speed_deg_s: {
      default: [10, 10, 10, 10, 10, 10],
      max: [30, 30, 30, 30, 30, 30],
    },
  },
};
const EXPECTED_IDENTITY = Object.freeze({ side: "left", port: 8765 });
const TARGET = [0, 80, -80, 5, 5, 5];
const SPEED = [10, 10, 10, 10, 10, 10];

function runningState(joints = [0, 90, -77, 0, 0, 0], side = "left") {
  return {
    side,
    state: "RUNNING",
    emergency_stopped: false,
    control_mode: "local",
    control_ready: true,
    control_block_reason: null,
    ActualQ: [...joints],
    arm: { connected: true, enabled: true, feedback_fresh: true },
  };
}

class FakeSocket extends EventTarget {
  constructor({ autoOpen = true, autoClose = true, autoHello = true } = {}) {
    super();
    this.readyState = 0;
    this.sent = [];
    this.closeCalls = [];
    this.autoClose = autoClose;
    this.autoHello = autoHello;
    if (autoOpen) queueMicrotask(() => this.open());
  }

  open() {
    this.readyState = 1;
    this.dispatchEvent(new Event("open"));
  }

  send(raw) {
    const request = JSON.parse(raw);
    this.sent.push(request);
    if (this.autoHello && request.cmd === "hello") {
      queueMicrotask(() => this.reply({ identity: "local" }));
    }
  }

  reply(data, overrides = {}) {
    const request = this.sent.at(-1);
    const payload = { id: request.id, cmd: request.cmd, ok: true, data, ...overrides };
    this.dispatchEvent(new MessageEvent("message", { data: JSON.stringify(payload) }));
  }

  reject(code = "STATE_CONFLICT", message = "rejected") {
    const request = this.sent.at(-1);
    this.dispatchEvent(new MessageEvent("message", { data: JSON.stringify({
      id: request?.id,
      cmd: request?.cmd || "connect",
      ok: false,
      error: { code, message },
    }) }));
  }

  close(code = 1000, reason = "") {
    if (this.readyState >= 2) return;
    this.closeCalls.push({ code, reason });
    if (!this.autoClose) {
      this.readyState = 2;
      return;
    }
    this.closeNow(code, reason);
  }

  closeNow(code = 1000, reason = "") {
    this.readyState = 3;
    this.dispatchEvent(new CloseEvent("close", { code, reason }));
  }

  commands(name) {
    return this.sent.filter(({ cmd }) => cmd === name);
  }
}

class FakeClock {
  constructor() {
    this.nowMs = 0;
    this.nextId = 1;
    this.jobs = new Map();
  }

  setTimeout = (callback, delay) => {
    const id = this.nextId++;
    this.jobs.set(id, { callback, at: this.nowMs + delay, interval: 0 });
    return id;
  };

  clearTimeout = (id) => this.jobs.delete(id);

  setInterval = (callback, delay) => {
    const id = this.nextId++;
    this.jobs.set(id, { callback, at: this.nowMs + delay, interval: delay });
    return id;
  };

  clearInterval = (id) => this.jobs.delete(id);

  now = () => this.nowMs;

  tick(ms) {
    const end = this.nowMs + ms;
    for (;;) {
      const due = [...this.jobs.entries()]
        .filter(([, job]) => job.at <= end)
        .sort((a, b) => a[1].at - b[1].at || a[0] - b[0])[0];
      if (!due) break;
      const [id, job] = due;
      this.nowMs = job.at;
      if (job.interval) job.at += job.interval;
      else this.jobs.delete(id);
      job.callback();
    }
    this.nowMs = end;
  }
}

async function nextTask() {
  for (let index = 0; index < 8; index += 1) await Promise.resolve();
}

async function replyAcquire(socket) {
  await nextTask();
  assertEqual(socket.sent[0]?.cmd, "hello", "hello must be the first frame");
  await nextTask();
  assertEqual(socket.sent[1]?.cmd, "get_base_info", "base info must be read first");
  socket.reply(BASE_INFO);
  await nextTask();
  assertEqual(socket.sent[2]?.cmd, "get_state", "state must be read second");
  socket.reply(runningState());
  await nextTask();
}

test("client uses the same-origin endpoint and strictly correlates one request", async () => {
  const socket = new FakeSocket();
  let openedUrl = null;
  const client = new PiperWebSocketClient({
    socketFactory: (url) => { openedUrl = url; return socket; },
    location: { protocol: "https:", host: "robot.local:8765" },
  });
  await client.open();
  assertDeepEqual(socket.sent[0], { id: "teach-1", cmd: "hello", identity: "local" },
    "client did not complete the local identity handshake first");
  assertEqual(openedUrl, "wss://robot.local:8765/ws/control", "wrong websocket URL");
  const pending = client.request("get_state");
  await assertRejects(client.request("get_base_info"), /in flight/i);
  socket.reply(runningState(), { id: "wrong" });
  await assertRejects(pending, /correlation/i);
  client.close();
});

test("client open waits for hello and fails closed when identity is rejected", async () => {
  const socket = new FakeSocket({ autoHello: false });
  const client = new PiperWebSocketClient({ socketFactory: () => socket });
  const opening = client.open();
  await nextTask();
  assertEqual(socket.sent[0]?.cmd, "hello", "open did not send hello first");
  let settled = false;
  opening.then(() => { settled = true; }, () => { settled = true; });
  await nextTask();
  assertEqual(settled, false, "open resolved before hello response");
  socket.reject("IDENTITY_MISMATCH", "local identity required");
  await assertRejects(opening, /local identity required/i);
  assertEqual(socket.readyState, 3, "rejected hello retained the controller socket");
});

test("client rejects command mismatch and structured protocol errors", async () => {
  const socketA = new FakeSocket();
  const clientA = new PiperWebSocketClient({ socketFactory: () => socketA });
  await clientA.open();
  const mismatched = clientA.request("get_state");
  socketA.reply(runningState(), { cmd: "get_base_info" });
  await assertRejects(mismatched, /correlation/i);

  const socketB = new FakeSocket();
  const clientB = new PiperWebSocketClient({ socketFactory: () => socketB });
  await clientB.open();
  const rejected = clientB.request("servoJ", { target_joints: TARGET, speed: SPEED });
  socketB.reject("OUT_OF_RANGE", "J1 outside limits");
  const error = await rejection(rejected);
  assertEqual(error.code, "OUT_OF_RANGE", "structured error code was lost");
  assert(/J1/.test(error.message), "structured error message was lost");
});

test("client times out and closes so a late response cannot be reused", async () => {
  const clock = new FakeClock();
  const socket = new FakeSocket();
  const client = new PiperWebSocketClient({ socketFactory: () => socket, clock });
  await client.open();
  const pending = client.request("get_state", {}, 40);
  clock.tick(41);
  await assertRejects(pending, /timed out/i);
  assertEqual(socket.readyState, 3, "timeout did not close the lease");
  socket.reply(runningState());
  assertEqual(socket.commands("servoJ").length, 0, "late response caused motion");
});

test("client clears request state when socket.send throws synchronously", async () => {
  const clock = new FakeClock();
  const socket = new FakeSocket();
  const client = new PiperWebSocketClient({ socketFactory: () => socket, clock });
  await client.open();
  socket.send = () => { throw new Error("send exploded"); };
  await assertRejects(client.request("get_state"), /send exploded/);
  assertEqual(clock.jobs.size, 0, "failed send retained request timeout");
  assertEqual(socket.readyState, 3, "failed send retained controller lease");
});

test("closing during open rejects and removes the open timeout and listener", async () => {
  const clock = new FakeClock();
  const socket = new FakeSocket({ autoOpen: false });
  const client = new PiperWebSocketClient({ socketFactory: () => socket, clock });
  const opening = client.open();
  client.abort("page hidden");
  await assertRejects(opening, /page hidden/i);
  assertEqual(clock.jobs.size, 0, "aborted open retained timeout");
  socket.open();
  assertEqual(socket.commands("get_base_info").length, 0, "late open resumed control");
});

test("preview creates no socket and sends no command", () => {
  const socket = new FakeSocket({ autoOpen: false });
  assertEqual(socket.sent.length, 0, "preview sent a command");
  assertEqual(socket.readyState, 0, "preview opened the controller socket");
});

test("button availability requires both a clear motion gate and the matching lifecycle", () => {
  assertDeepEqual(
    controlAvailability({ mode: "reviewed", blockReason: null, lifecycle: "read-only" }),
    { acquire: false, execute: true, deadman: false },
    "reviewed controls were wrong",
  );
  assertDeepEqual(
    controlAvailability({ mode: "live", blockReason: null, lifecycle: "read-only" }),
    { acquire: true, execute: false, deadman: false },
    "live acquisition was wrong",
  );
  assertDeepEqual(
    controlAvailability({ mode: "live", blockReason: null, lifecycle: "live-armed" }),
    { acquire: false, execute: false, deadman: true },
    "armed deadman was wrong",
  );
  assertDeepEqual(
    controlAvailability({ mode: "live", blockReason: "stale", lifecycle: "live-armed" }),
    { acquire: false, execute: false, deadman: false },
    "blocked state left motion enabled",
  );
});

test("the first telemetry failure immediately stops every active control path", () => {
  const reviewedAbortController = new AbortController();
  const gripperStops = [];
  const liveStops = [];
  const reason = "Telemetry request failed: Request timed out after 1000ms.";

  failClosedOnTelemetryFailure({
    reviewedAbortController,
    gripperSession: { stop: (value) => gripperStops.push(value) },
    liveFollower: {
      state: "live-active",
      deactivate: (value) => liveStops.push(value),
    },
  }, reason);

  assertEqual(reviewedAbortController.signal.aborted, true,
    "reviewed execution continued after telemetry failed");
  assertEqual(reviewedAbortController.signal.reason, reason,
    "reviewed execution lost the telemetry failure reason");
  assertDeepEqual(gripperStops, [reason],
    "gripper control did not stop immediately after telemetry failed");
  assertDeepEqual(liveStops, [reason],
    "live following did not stop immediately after telemetry failed");
});

test("page identity requires valid injected side and a port matching the current origin", () => {
  for (const [side, port] of [["left", 8765], ["right", 9876]]) {
    const identity = readPageExpectedIdentity(
      { body: { dataset: { armSide: side, serverPort: String(port) } } },
      { protocol: "http:", port: String(port) },
    );
    assertDeepEqual(identity, { side, port }, `valid ${side} page identity changed`);
  }
  for (const [side, datasetPort, originPort] of [
    [undefined, "8765", "8765"], ["center", "8765", "8765"],
    ["left", undefined, "8765"], ["left", "8765.0", "8765"],
    ["left", "8765", "9876"],
  ]) {
    let error = null;
    try {
      readPageExpectedIdentity(
        { body: { dataset: { armSide: side, serverPort: datasetPort } } },
        { protocol: "http:", port: originPort },
      );
    } catch (caught) {
      error = caught;
    }
    assert(error, `invalid page identity ${side}/${datasetPort}/${originPort} was accepted`);
  }
});

test("reviewed execution freezes its target, revalidates, sends once, and closes", async () => {
  const clock = new FakeClock();
  const socket = new FakeSocket();
  const target = [...TARGET];
  const pending = executeReviewedTarget({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    targetJointsDeg: target,
    speedDegS: SPEED,
    timeoutMs: 30000,
    pollIntervalMs: 50,
    clock,
  });
  target[0] = 99;
  await replyAcquire(socket);
  assertEqual(socket.sent[3]?.cmd, "servoJ", "servoJ did not follow validation");
  assertDeepEqual(socket.sent[3].target_joints, TARGET, "reviewed target was mutable");
  socket.reply({ accepted: true });
  await nextTask();
  clock.tick(50);
  await nextTask();
  assertEqual(socket.sent[4]?.cmd, "get_state", "execution did not poll state");
  socket.reply(runningState([1, 81, -79, 4, 5, 6]));
  const result = await pending;
  assertDeepEqual(result.targetJointsDeg, TARGET, "result target was changed");
  assertEqual(socket.commands("servoJ").length, 1, "reviewed execution sent more than once");
  assertEqual(socket.readyState, 3, "reviewed execution did not release lease");
});

test("reviewed execution resolves only after the websocket lease is closed", async () => {
  const socket = new FakeSocket({ autoClose: false });
  const pending = executeReviewedTarget({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    targetJointsDeg: runningState().ActualQ,
    speedDegS: SPEED,
  });
  await replyAcquire(socket);
  socket.reply({ accepted: true });
  let settled = false;
  void pending.finally(() => { settled = true; });
  await nextTask();
  assertEqual(socket.readyState, 2, "reviewed execution did not begin releasing its lease");
  assertEqual(settled, false, "reviewed execution finished before the close event");
  socket.closeNow();
  await pending;
  assertEqual(settled, true, "reviewed execution did not finish after lease release");
});

test("reviewed execution rejects backend-blocked and out-of-limit targets without servo", async () => {
  for (const [target, state, pattern] of [
    [[999, 80, -80, 5, 5, 5], runningState(), /limit/i],
    [TARGET, {
      ...runningState(),
      state: "HOMING",
      control_ready: false,
      control_block_reason: "Service is HOMING; RUNNING is required.",
    }, /RUNNING/i],
  ]) {
    const socket = new FakeSocket();
    const pending = executeReviewedTarget({ socketFactory: () => socket, expectedIdentity: EXPECTED_IDENTITY, targetJointsDeg: target, speedDegS: SPEED });
    await nextTask();
    socket.reply(BASE_INFO);
    await nextTask();
    socket.reply(state);
    await assertRejects(pending, pattern);
    assertEqual(socket.commands("servoJ").length, 0, "invalid review sent servoJ");
    assertEqual(socket.readyState, 3, "invalid review retained lease");
  }
});

test("reviewed execution rejects a racing external controller without servo", async () => {
  const socket = new FakeSocket();
  const pending = executeReviewedTarget({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    targetJointsDeg: TARGET,
    speedDegS: SPEED,
  });
  await nextTask();
  socket.dispatchEvent(new MessageEvent("message", { data: JSON.stringify({
    cmd: "connect", ok: false, error: {
      code: "CLIENT_CONFLICT", message: "controller already connected",
    },
  }) }));
  socket.close(4409, "controller already connected");

  await assertRejects(pending, /controller already connected/i);
  assertEqual(socket.commands("servoJ").length, 0, "conflicting reviewed execution sent servoJ");
  assertEqual(socket.readyState, 3, "conflicting reviewed execution retained a lease");
});

test("fresh websocket revalidation rejects incomplete base and state schemas", async () => {
  const baseMutations = [
    (base) => { delete base.side; },
    (base) => { base.side = "right"; },
    (base) => { base.side = 1; },
    (base) => { delete base.arm.server.port; },
    (base) => { base.arm.server.port = 9876; },
    (base) => { base.arm.server.port = "8765"; },
    (base) => { base.supported_commands = ["get_state"]; },
    (base) => { base.units.joint_position = "radian"; },
    (base) => { base.units.joint_speed = "radian/second"; },
    (base) => { base.arm.joint_speed_deg_s.default[2] = 0; },
    (base) => { base.arm.joint_speed_deg_s.default[3] = 31; },
  ];
  for (const mutate of baseMutations) {
    const socket = new FakeSocket();
    const pending = executeReviewedTarget({ socketFactory: () => socket, expectedIdentity: EXPECTED_IDENTITY, targetJointsDeg: TARGET, speedDegS: SPEED });
    await nextTask();
    const bad = structuredClone(BASE_INFO);
    mutate(bad);
    socket.reply(bad);
    await assertRejects(pending, /identity|side|port|command|unit|speed/i);
    assertEqual(socket.commands("servoJ").length, 0, "invalid base schema sent servoJ");
  }

  for (const [path, value] of [
    ["side", undefined], ["side", "right"], ["side", 1],
    ["control_ready", undefined], ["control_ready", null], ["control_ready", "true"],
    ["control_block_reason", undefined], ["control_block_reason", "unexpected block"],
  ]) {
    const socket = new FakeSocket();
    const pending = executeReviewedTarget({ socketFactory: () => socket, expectedIdentity: EXPECTED_IDENTITY, targetJointsDeg: TARGET, speedDegS: SPEED });
    await nextTask();
    socket.reply(BASE_INFO);
    await nextTask();
    const bad = runningState();
    if (value === undefined) delete bad[path];
    else bad[path] = value;
    socket.reply(bad);
    await assertRejects(pending, /side|readiness/i);
    assertEqual(socket.commands("servoJ").length, 0, `invalid ${path} sent servoJ`);
  }
});

test("failed live acquisition remains in progress until its lease is closed", async () => {
  const socket = new FakeSocket({ autoClose: false });
  const follower = createLiveFollower({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    getTargetSnapshot: () => TARGET,
    getTelemetrySnapshot: () => ({ payload: runningState(), receivedAtMs: 0 }),
    speedDegS: SPEED,
    eventTarget: new EventTarget(),
    documentTarget: new EventTarget(),
    clock: new FakeClock(),
  });
  const acquiring = follower.acquire();
  await nextTask();
  socket.reply(BASE_INFO);
  await nextTask();
  socket.reply({
    ...runningState(),
    control_ready: false,
    control_block_reason: "Robot feedback is stale.",
  });
  let settled = false;
  void acquiring.catch(() => {}).finally(() => { settled = true; });
  await nextTask();
  assertEqual(socket.readyState, 2, "failed acquisition did not begin releasing its lease");
  assertEqual(settled, false, "failed acquisition returned before lease release");
  socket.closeNow();
  await assertRejects(acquiring, /stale/i);
  assertEqual(follower.state, "read-only", "released failed acquisition did not become retryable");
  follower.dispose();
});

test("live acquire rejects fresh cross-arm identity before any servo", async () => {
  for (const [badBase, badState] of [
    [{ ...structuredClone(BASE_INFO), side: "right" }, null],
    [structuredClone(BASE_INFO), runningState(undefined, "right")],
  ]) {
    const socket = new FakeSocket();
    const follower = createLiveFollower({
      socketFactory: () => socket,
      expectedIdentity: EXPECTED_IDENTITY,
      getTargetSnapshot: () => TARGET,
      getTelemetrySnapshot: () => ({ payload: runningState(), receivedAtMs: 0 }),
      speedDegS: SPEED,
      eventTarget: new EventTarget(),
      documentTarget: new EventTarget(),
    });
    const acquiring = follower.acquire();
    await nextTask();
    socket.reply(badBase);
    if (badState) {
      await nextTask();
      socket.reply(badState);
    }
    await assertRejects(acquiring, /side|identity/i);
    assertEqual(socket.commands("servoJ").length, 0, "identity mismatch sent servoJ");
    assertEqual(follower.state, "read-only", "identity rejection did not remain read-only");
    follower.dispose();
  }
});

test("fresh left and right service identities acquire independently", async () => {
  for (const [side, port] of [["left", 8765], ["right", 9876]]) {
    const socket = new FakeSocket();
    const base = structuredClone(BASE_INFO);
    base.side = side;
    base.arm.server.port = port;
    const follower = createLiveFollower({
      socketFactory: () => socket,
      expectedIdentity: { side, port },
      getTargetSnapshot: () => TARGET,
      getTelemetrySnapshot: () => ({ payload: runningState(undefined, side), receivedAtMs: 0 }),
      speedDegS: SPEED,
      eventTarget: new EventTarget(),
      documentTarget: new EventTarget(),
    });
    const acquiring = follower.acquire();
    await nextTask();
    socket.reply(base);
    await nextTask();
    socket.reply(runningState(undefined, side));
    await acquiring;
    assertEqual(follower.state, "live-armed", `${side} identity did not acquire`);
    follower.dispose();
  }
});

for (const eventName of ["emergency-stop", "teaching-mode-change", "blur", "visibilitychange", "pagehide", "dispose"]) {
  test(`reviewed execution closes immediately and cannot resume on ${eventName}`, async () => {
    const socket = new FakeSocket();
    const eventTarget = new EventTarget();
    const documentTarget = new EventTarget();
    Object.defineProperty(documentTarget, "hidden", { value: eventName === "visibilitychange", configurable: true });
    const abortController = new AbortController();
    const lifecycle = createControlLifecycle({
      eventTarget,
      documentTarget,
      stop: (reason) => abortController.abort(reason),
    });
    const pending = executeReviewedTarget({
      socketFactory: () => socket,
      expectedIdentity: EXPECTED_IDENTITY,
      targetJointsDeg: TARGET,
      speedDegS: SPEED,
      signal: abortController.signal,
    });
    await replyAcquire(socket);
    assertEqual(socket.commands("servoJ").length, 1, "reviewed servo was not pending");
    if (eventName === "dispose") lifecycle.dispose();
    else (eventName === "visibilitychange" ? documentTarget : eventTarget).dispatchEvent(new Event(eventName));
    assertEqual(socket.readyState, 3, `${eventName} did not synchronously close reviewed socket`);
    const countAtStop = socket.sent.length;
    socket.reply({ accepted: true });
    await assertRejects(pending, /abort|hidden|focus|mode|emergency|dispose|page/i);
    assertEqual(socket.sent.length, countAtStop, `${eventName} allowed a post-stop request`);
    lifecycle.dispose();
  });
}

test("reviewed cancellation during poll delay prevents a later get_state", async () => {
  const clock = new FakeClock();
  const socket = new FakeSocket();
  const eventTarget = new EventTarget();
  const abortController = new AbortController();
  const lifecycle = createControlLifecycle({
    eventTarget,
    documentTarget: new EventTarget(),
    stop: (reason) => abortController.abort(reason),
  });
  const pending = executeReviewedTarget({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    targetJointsDeg: TARGET,
    speedDegS: SPEED,
    signal: abortController.signal,
    clock,
    pollIntervalMs: 50,
  });
  await replyAcquire(socket);
  socket.reply({ accepted: true });
  await nextTask();
  eventTarget.dispatchEvent(new Event("blur"));
  clock.tick(1000);
  await assertRejects(pending, /focus|abort/i);
  assertEqual(socket.commands("get_state").length, 1, "cancelled poll sent another state request");
  assertEqual(socket.commands("servoJ").length, 1, "cancelled poll repeated servoJ");
  lifecycle.dispose({ stop: false });
});

for (const interruption of ["blur", "visibilitychange", "emergency-stop", "socket-close", "dispose"]) {
  test(`${interruption} cannot revive a deferred opening acquire`, async () => {
    const socket = new FakeSocket({ autoOpen: false });
    const eventTarget = new EventTarget();
    const documentTarget = new EventTarget();
    Object.defineProperty(documentTarget, "hidden", { value: interruption === "visibilitychange", configurable: true });
    const clock = new FakeClock();
    const follower = createLiveFollower({
      socketFactory: () => socket,
      expectedIdentity: EXPECTED_IDENTITY,
      getTargetSnapshot: () => TARGET,
      getTelemetrySnapshot: () => ({ payload: runningState(), receivedAtMs: 0 }),
      speedDegS: SPEED,
      eventTarget,
      documentTarget,
      clock,
    });
    const acquiring = follower.acquire();
    if (interruption === "dispose") follower.dispose();
    else if (interruption === "socket-close") socket.close(1006, "network lost");
    else (interruption === "visibilitychange" ? documentTarget : eventTarget).dispatchEvent(new Event(interruption));
    socket.open();
    await assertRejects(acquiring, /abort|closed|lost|focus|hidden|emergency|dispose/i);
    assertEqual(follower.state, interruption === "dispose" ? "disposed" : "stopping", "late open revived acquire");
    assertEqual(socket.sent.length, 0, "late open continued acquisition");
    assertEqual(clock.jobs.size, 0, "interrupted opening retained a timer");
  });
}

test("reviewed execution times out, closes, and never repeats servoJ", async () => {
  const clock = new FakeClock();
  const socket = new FakeSocket();
  const pending = executeReviewedTarget({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    targetJointsDeg: TARGET,
    speedDegS: SPEED,
    timeoutMs: 100,
    pollIntervalMs: 50,
    clock,
  });
  await replyAcquire(socket);
  socket.reply({ accepted: true });
  await nextTask();
  clock.tick(50);
  await nextTask();
  socket.reply(runningState());
  await nextTask();
  clock.tick(50);
  await nextTask();
  socket.reply(runningState());
  await assertRejects(pending, /timed out/i);
  assertEqual(socket.commands("servoJ").length, 1, "timeout repeated servoJ");
  assertEqual(socket.readyState, 3, "timeout retained the lease");
});

test("reviewed execution allows the speed-derived duration beyond thirty seconds", async () => {
  const clock = new FakeClock();
  const socket = new FakeSocket();
  const target = [0, 179, -80, 0, 0, 0];
  const speed = [5, 5, 5, 5, 5, 5];
  const initial = runningState([0, 0, -80, 0, 0, 0]);
  const pending = executeReviewedTarget({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    targetJointsDeg: target,
    speedDegS: speed,
    pollIntervalMs: 30000,
    clock,
  });
  await nextTask();
  socket.reply(BASE_INFO);
  await nextTask();
  socket.reply(initial);
  await nextTask();
  socket.reply({ accepted: true });
  await nextTask();

  clock.tick(30000);
  await nextTask();
  socket.reply(initial);
  await nextTask();
  assertEqual(socket.readyState, 1, "valid long move was cut off at thirty seconds");

  clock.tick(30000);
  await nextTask();
  socket.reply(runningState(target));
  const result = await pending;
  assertDeepEqual(result.actualJointsDeg, target, "long move did not complete");
});

async function activeFollower({ eventTarget = new EventTarget(), documentTarget = new EventTarget() } = {}) {
  const clock = new FakeClock();
  const socket = new FakeSocket();
  let target = [...TARGET];
  let telemetry = { payload: runningState(), receivedAtMs: clock.now() };
  const follower = createLiveFollower({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    getTargetSnapshot: () => target,
    speedDegS: SPEED,
    getTelemetrySnapshot: () => telemetry,
    eventTarget,
    documentTarget,
    clock,
  });
  const acquiring = follower.acquire();
  await replyAcquire(socket);
  await acquiring;
  await follower.activate();
  return {
    clock, socket, follower, eventTarget, documentTarget,
    setTarget: (next) => { target = next; },
    setTelemetry: (next) => { telemetry = next; },
  };
}

test("a thrown live IK solve closes before any stale target can be sent", async () => {
  const harness = await activeFollower();
  let state = createTeachingState(BASE_INFO, EXPECTED_IDENTITY);
  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: runningState(),
    receivedAtMs: harness.clock.now(),
  });
  state = reduceTeachingState(state, { type: "control-state", controlState: "live-armed" });
  state = resolveSynchronousIk(state, "requested-pose", () => {
    throw new Error("solver exploded");
  });
  const reason = motionBlockReason(state, harness.clock.now());
  assert(/solver exploded/i.test(reason), "solver exception did not block motion");
  harness.follower.deactivate(reason);
  harness.clock.tick(1000);
  await nextTask();
  assertEqual(harness.socket.readyState, 3, "solver exception retained the live lease");
  assertEqual(harness.socket.commands("servoJ").length, 0, "old target was sent after solver exception");
});

test("live following coalesces immutable targets at 20Hz with no backlog", async () => {
  const harness = await activeFollower();
  harness.setTarget([1, 80, -80, 5, 5, 5]);
  harness.clock.tick(50);
  await nextTask();
  assertEqual(harness.socket.commands("servoJ").length, 1, "first live target not sent");
  assertDeepEqual(harness.socket.commands("servoJ")[0].target_joints, [1, 80, -80, 5, 5, 5], "first target wrong");
  harness.setTarget([2, 80, -80, 5, 5, 5]);
  harness.clock.tick(200);
  assertEqual(harness.socket.commands("servoJ").length, 1, "live follower queued while request was pending");
  harness.socket.reply({ accepted: true });
  await nextTask();
  harness.clock.tick(50);
  await nextTask();
  assertEqual(harness.socket.commands("servoJ").length, 2, "latest coalesced target was not sent");
  assertDeepEqual(harness.socket.commands("servoJ")[1].target_joints, [2, 80, -80, 5, 5, 5], "stale target was queued");
  harness.follower.dispose();
});

test("deadman release retires the follower and reacquires a fresh lease", async () => {
  const clock = new FakeClock();
  const eventTarget = new EventTarget();
  const sockets = [];
  const session = createLiveControlSession({
    socketFactory: () => {
      const socket = new FakeSocket();
      sockets.push(socket);
      return socket;
    },
    expectedIdentity: EXPECTED_IDENTITY,
    getTargetSnapshot: () => TARGET,
    getTelemetrySnapshot: () => ({ payload: runningState(), receivedAtMs: clock.now() }),
    speedDegS: SPEED,
    eventTarget,
    documentTarget: new EventTarget(),
    clock,
  });

  const firstAcquire = session.acquire();
  await replyAcquire(sockets[0]);
  await firstAcquire;
  await session.activate();
  clock.tick(50);
  await nextTask();
  assertEqual(sockets[0].commands("servoJ").length, 1, "first lease sent no live target");
  sockets[0].reply({ accepted: true });
  await nextTask();

  eventTarget.dispatchEvent(new Event("pointerup"));
  await nextTask();
  assertEqual(sockets[0].readyState, 3, "release retained the first lease");
  assertEqual(session.state, "read-only", "released session did not become reacquirable");

  const secondAcquire = session.acquire();
  await replyAcquire(sockets[1]);
  await secondAcquire;
  await session.activate();
  clock.tick(50);
  await nextTask();
  assertEqual(sockets[1].commands("servoJ").length, 1, "fresh lease sent no second live target");
  eventTarget.dispatchEvent(new Event("pointerup"));
  await nextTask();
  assertEqual(sockets[1].readyState, 3, "release retained the second lease");
  assertEqual(session.state, "read-only", "second release did not return to read-only");
  session.dispose();
});

test("live session stopAndWait resolves only after the websocket lease is closed", async () => {
  const clock = new FakeClock();
  const socket = new FakeSocket({ autoClose: false });
  const session = createLiveControlSession({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    getTargetSnapshot: () => TARGET,
    getTelemetrySnapshot: () => ({ payload: runningState(), receivedAtMs: clock.now() }),
    speedDegS: SPEED,
    eventTarget: new EventTarget(),
    documentTarget: new EventTarget(),
    clock,
  });
  const acquiring = session.acquire();
  await replyAcquire(socket);
  await acquiring;
  let released = false;
  const waiting = session.stopAndWait("Switching to gripper control.").then(() => { released = true; });
  await nextTask();
  assertEqual(released, false, "stopAndWait resolved before the socket close event");
  assertEqual(session.state, "stopping", "session did not enter stopping state");
  socket.closeNow();
  await waiting;
  assertEqual(session.state, "read-only", "closed lease did not retire the follower");
  session.dispose();
});

for (const eventName of ["pointerup", "pointercancel", "keyup", "blur", "visibilitychange", "emergency-stop"]) {
  test(`live following closes before another send on ${eventName}`, async () => {
    const eventTarget = new EventTarget();
    const documentTarget = new EventTarget();
    Object.defineProperty(documentTarget, "hidden", { value: eventName === "visibilitychange", configurable: true });
    const harness = await activeFollower({ eventTarget, documentTarget });
    harness.clock.tick(50);
    await nextTask();
    const countAtEvent = harness.socket.commands("servoJ").length;
    const target = eventName === "visibilitychange" ? documentTarget : eventTarget;
    target.dispatchEvent(new Event(eventName));
    harness.clock.tick(1000);
    await nextTask();
    assertEqual(harness.socket.readyState, 3, `${eventName} did not close socket`);
    assertEqual(harness.socket.commands("servoJ").length, countAtEvent, `${eventName} allowed a later send`);
  });
}

for (const eventName of ["error", "close"]) {
  test(`socket ${eventName} irreversibly stops live following`, async () => {
    const harness = await activeFollower();
    harness.socket.dispatchEvent(eventName === "close"
      ? new CloseEvent("close", { code: 1006 })
      : new Event("error"));
    harness.clock.tick(1000);
    await nextTask();
    assertEqual(harness.socket.commands("servoJ").length, 0, `socket ${eventName} allowed motion`);
    assertEqual(harness.follower.state, "stopping", `socket ${eventName} was reversible`);
  });
}

test("real telemetry age crossing stale threshold closes before the next live send", async () => {
  const harness = await activeFollower();
  harness.clock.tick(50);
  await nextTask();
  assertEqual(harness.socket.commands("servoJ").length, 1, "live target missing");
  harness.clock.tick(1001);
  await nextTask();
  harness.socket.reply({ accepted: true });
  await nextTask();
  assertEqual(harness.socket.commands("servoJ").length, 1, "late response restarted live sending");
  assertEqual(harness.socket.readyState, 3, "stale feedback did not close socket");
});

test("network-hung telemetry snapshot stops live motion with zero extra servo", async () => {
  const harness = await activeFollower();
  harness.setTelemetry(null);
  harness.clock.tick(50);
  await nextTask();
  assertEqual(harness.socket.commands("servoJ").length, 0, "missing telemetry allowed servoJ");
  assertEqual(harness.socket.readyState, 3, "missing telemetry retained lease");
});

test("live tick rejects malformed or unsafe real telemetry before servo", async () => {
  const mutations = [
    (payload) => { delete payload.control_ready; },
    (payload) => { payload.control_ready = "true"; },
    (payload) => { payload.control_ready = false; payload.control_block_reason = "Emergency stop is active."; },
    (payload) => { payload.control_block_reason = "unexpected block"; },
    (payload) => { payload.ActualQ[5] = Number.NaN; },
  ];
  for (const mutate of mutations) {
    const harness = await activeFollower();
    const payload = runningState();
    mutate(payload);
    harness.setTelemetry({ payload, receivedAtMs: harness.clock.now() });
    harness.clock.tick(50);
    await nextTask();
    assertEqual(harness.socket.commands("servoJ").length, 0, "unsafe telemetry allowed servoJ");
    assertEqual(harness.socket.readyState, 3, "unsafe telemetry retained lease");
  }
});

for (const stage of ["opening", "base", "state"]) {
  test(`stop during deferred ${stage} acquire is irreversible`, async () => {
    const socket = new FakeSocket({ autoOpen: false });
    const eventTarget = new EventTarget();
    const follower = createLiveFollower({
      socketFactory: () => socket,
      expectedIdentity: EXPECTED_IDENTITY,
      getTargetSnapshot: () => TARGET,
      getTelemetrySnapshot: () => ({ payload: runningState(), receivedAtMs: 0 }),
      speedDegS: SPEED,
      eventTarget,
      documentTarget: new EventTarget(),
      clock: new FakeClock(),
    });
    const acquiring = follower.acquire();
    if (stage !== "opening") {
      socket.open();
      await nextTask();
    }
    if (stage === "state") {
      socket.reply(BASE_INFO);
      await nextTask();
    }
    eventTarget.dispatchEvent(new Event("blur"));
    const sentAtStop = socket.sent.length;
    if (stage === "opening") socket.open();
    else socket.reply(stage === "base" ? BASE_INFO : runningState());
    await assertRejects(acquiring, /focus|abort|closed/i);
    await nextTask();
    assertEqual(follower.state, "stopping", "late acquire overwrote stopping state");
    assertEqual(socket.sent.length, sentAtStop, "late acquire continued protocol requests");
  });
}

test("dispose during deferred acquire remains disposed after late socket events", async () => {
  const socket = new FakeSocket({ autoOpen: false });
  const follower = createLiveFollower({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    getTargetSnapshot: () => TARGET,
    getTelemetrySnapshot: () => ({ payload: runningState(), receivedAtMs: 0 }),
    speedDegS: SPEED,
    eventTarget: new EventTarget(),
    documentTarget: new EventTarget(),
    clock: new FakeClock(),
  });
  const acquiring = follower.acquire();
  follower.dispose();
  socket.open();
  await assertRejects(acquiring, /dispose|abort|closed/i);
  assertEqual(follower.state, "disposed", "late acquire overwrote disposed state");
  assertEqual(socket.sent.length, 0, "disposed acquire continued requests");
});

test("a racing second tab remains read-only after 4409 conflict", async () => {
  const socket = new FakeSocket();
  const follower = createLiveFollower({
    socketFactory: () => socket,
    expectedIdentity: EXPECTED_IDENTITY,
    getTargetSnapshot: () => TARGET,
    getTelemetrySnapshot: () => ({ payload: runningState(), receivedAtMs: 0 }),
    speedDegS: SPEED,
    eventTarget: new EventTarget(),
    documentTarget: new EventTarget(),
  });
  const pending = follower.acquire();
  await nextTask();
  socket.dispatchEvent(new MessageEvent("message", { data: JSON.stringify({
    cmd: "connect", ok: false, error: { code: "CLIENT_CONFLICT", message: "controller already connected" },
  }) }));
  socket.close(4409, "controller already connected");
  await assertRejects(pending, /controller already connected/i);
  assertEqual(socket.commands("servoJ").length, 0, "conflicting tab sent servoJ");
  assertEqual(follower.state, "read-only", "conflicting tab did not remain read-only");
});

test("dispose removes lifecycle listeners and timers", async () => {
  const harness = await activeFollower();
  harness.follower.dispose();
  const closeCount = harness.socket.closeCalls.length;
  harness.eventTarget.dispatchEvent(new Event("pointerup"));
  harness.eventTarget.dispatchEvent(new Event("emergency-stop"));
  harness.clock.tick(1000);
  assertEqual(harness.socket.closeCalls.length, closeCount, "disposed listeners remained registered");
  assertEqual(harness.clock.jobs.size, 0, "disposed follower retained timers");
});

async function rejection(promise) {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected promise to reject");
}

async function assertRejects(promise, pattern) {
  const error = await rejection(promise);
  assert(pattern.test(error?.message || String(error)), `unexpected rejection: ${error?.message || error}`);
}

scheduleTests();
