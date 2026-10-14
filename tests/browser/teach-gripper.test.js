// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import {
  assert,
  assertEqual,
  scheduleTests,
  test,
} from "./teach-test-harness.js";
import { createGripperControlSession } from "../../src/piper_control/web/teach-gripper.js";

const EXPECTED_IDENTITY = Object.freeze({ side: "left", port: 8765 });
const BASE_INFO = {
  protocol_version: "3.0",
  control_mode: "local",
  side: "left",
  supported_commands: ["hello", "servoJ", "move_gripper", "get_state", "get_base_info"],
  units: {
    joint_position: "degree",
    joint_speed: "degree/second",
    gripper_position: "meter",
    gripper_speed: "meter/second",
  },
  arm: {
    server: { port: 8765 },
    joint_limits_deg: { min: [-150, 0, -154, -100, -69, -120], max: [150, 179, 0, 100, 69, 120] },
    joint_speed_deg_s: { default: [10, 10, 10, 10, 10, 10], max: [30, 30, 30, 30, 30, 30] },
    gripper: {
      backend: "piper",
      enabled: true,
      position_m: { min: 0, max: 0.1 },
      speed_m_s: { default: 0.02, max: 0.05 },
      effort_nm: 1,
    },
  },
};

function runningState(positionM = 0.02) {
  return {
    side: "left",
    state: "RUNNING",
    emergency_stopped: false,
    control_mode: "local",
    control_ready: true,
    control_block_reason: null,
    ActualQ: [0, 90, -77, 0, 0, 0],
    ActualGripper: positionM,
    arm: { connected: true, enabled: true, feedback_fresh: true },
  };
}

class FakeSocket extends EventTarget {
  constructor() {
    super();
    this.readyState = 0;
    this.sent = [];
    queueMicrotask(() => {
      this.readyState = 1;
      this.dispatchEvent(new Event("open"));
    });
  }

  send(raw) {
    const request = JSON.parse(raw);
    this.sent.push(request);
    if (request.cmd === "hello") queueMicrotask(() => this.reply({ identity: "local" }));
  }

  reply(data) {
    const request = this.sent.at(-1);
    this.dispatchEvent(new MessageEvent("message", { data: JSON.stringify({
      id: request.id,
      cmd: request.cmd,
      ok: true,
      data,
    }) }));
  }

  close(code = 1000, reason = "") {
    if (this.readyState >= 2) return;
    this.readyState = 3;
    this.dispatchEvent(new CloseEvent("close", { code, reason }));
  }

  commands(name) { return this.sent.filter(({ cmd }) => cmd === name); }
}

class FakeClock {
  constructor() { this.nowMs = 0; this.nextId = 1; this.jobs = new Map(); }
  setTimeout = (callback, delay) => { const id = this.nextId++; this.jobs.set(id, { callback, at: this.nowMs + delay, interval: 0 }); return id; };
  clearTimeout = (id) => this.jobs.delete(id);
  setInterval = (callback, delay) => { const id = this.nextId++; this.jobs.set(id, { callback, at: this.nowMs + delay, interval: delay }); return id; };
  clearInterval = (id) => this.jobs.delete(id);
  now = () => this.nowMs;
  tick(ms) {
    const end = this.nowMs + ms;
    for (;;) {
      const due = [...this.jobs.entries()].filter(([, job]) => job.at <= end)
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

async function acquire(session, sockets, base = BASE_INFO, initial = runningState()) {
  const pending = session.start();
  await nextTask();
  const socket = sockets.at(-1);
  assertEqual(socket.sent[0]?.cmd, "hello", "gripper did not send hello first");
  await nextTask();
  assertEqual(socket.sent[1]?.cmd, "get_base_info", "gripper did not read base info first");
  socket.reply(base);
  await nextTask();
  assertEqual(socket.sent[2]?.cmd, "get_state", "gripper did not read state second");
  socket.reply(initial);
  await pending;
}

function makeHarness(overrides = {}) {
  const clock = new FakeClock();
  const sockets = [];
  let targetPositionM = 0.02;
  let speedMps = 0.02;
  let editRevision = 0;
  let telemetry = { payload: runningState(), receivedAtMs: clock.now() };
  const session = createGripperControlSession({
    socketFactory: () => { const socket = new FakeSocket(); sockets.push(socket); return socket; },
    expectedIdentity: EXPECTED_IDENTITY,
    getTargetSnapshot: () => ({ targetPositionM, speedMps, editRevision }),
    getTelemetrySnapshot: () => telemetry,
    beforeAcquire: overrides.beforeAcquire,
    eventTarget: overrides.eventTarget || new EventTarget(),
    documentTarget: overrides.documentTarget || new EventTarget(),
    clock,
  });
  return {
    clock, sockets, session,
    setTarget: (positionM, speed = speedMps) => {
      targetPositionM = positionM;
      speedMps = speed;
      editRevision += 1;
    },
    followTelemetryTarget: (positionM) => { targetPositionM = positionM; },
    setTelemetry: (value) => { telemetry = value; },
  };
}

test("gripper uses configured metadata and coalesces only the latest target at 20 Hz", async () => {
  const harness = makeHarness();
  await acquire(harness.session, harness.sockets);
  harness.setTarget(0.03, 0.015);
  harness.clock.tick(50);
  await nextTask();
  const socket = harness.sockets[0];
  assertEqual(socket.commands("move_gripper").length, 1, "first gripper target was not sent");
  assertEqual(socket.commands("move_gripper")[0].target_position, 0.03, "wrong gripper position");
  assertEqual(socket.commands("move_gripper")[0].speed, 0.015, "wrong gripper speed");
  assertEqual(socket.commands("servoJ").length, 0, "gripper control sent a joint command");

  harness.setTarget(0.08, 0.025);
  harness.clock.tick(200);
  assertEqual(socket.commands("move_gripper").length, 1, "pending gripper command created a backlog");
  socket.reply({ accepted: true });
  await nextTask();
  harness.clock.tick(50);
  await nextTask();
  assertEqual(socket.commands("move_gripper").length, 2, "latest gripper target was not sent");
  assertEqual(socket.commands("move_gripper")[1].target_position, 0.08, "stale gripper position was queued");
  harness.session.stop("pointer released");
});

test("gripper acquisition rejects non-Piper and incompatible-unit metadata without motion", async () => {
  const unsupportedBackend = structuredClone(BASE_INFO);
  unsupportedBackend.units.gripper_position = "radian";
  unsupportedBackend.units.gripper_speed = "radian/second";
  unsupportedBackend.arm.gripper = {
    backend: "unsupported_backend", enabled: true,
    position_rad: { min: 0, max: 0.7 },
    physical_position_rad: { min: 0, max: 1.014287304878 },
    speed_rad_s: { default: 0.5, max: 2.0 },
  };
  const missingBackend = structuredClone(BASE_INFO);
  delete missingBackend.arm.gripper.backend;
  const unknownBackend = structuredClone(BASE_INFO);
  unknownBackend.arm.gripper.backend = "other";
  const radianPosition = structuredClone(BASE_INFO);
  radianPosition.units.gripper_position = "radian";
  const radianSpeed = structuredClone(BASE_INFO);
  radianSpeed.units.gripper_speed = "radian/second";
  for (const base of [unsupportedBackend, missingBackend, unknownBackend, radianPosition, radianSpeed]) {
    const harness = makeHarness();
    const result = harness.session.start().then(() => null, (error) => error);
    await nextTask();
    const socket = harness.sockets[0];
    socket.reply(base);
    await nextTask();
    if (socket.sent.at(-1).cmd === "get_state") socket.reply(runningState());
    const error = await result;
    harness.session.stop();
    assert(error instanceof Error, "unsupported metadata acquired gripper control");
    assert(/incompatible/i.test(error.message), `unexpected rejection: ${error.message}`);
    assertEqual(socket.commands("move_gripper").length, 0, "unsupported metadata sent gripper motion");
    assertEqual(socket.readyState, 3, "rejected metadata retained the control lease");
  }
});

test("gripper sends nothing until the target changes after acquisition", async () => {
  const harness = makeHarness();
  await acquire(harness.session, harness.sockets);

  harness.clock.tick(200);
  await nextTask();
  assertEqual(harness.sockets[0].commands("move_gripper").length, 0,
    "acquiring control sent an unedited gripper target");

  harness.setTarget(0.04, 0.02);
  harness.clock.tick(50);
  await nextTask();
  assertEqual(harness.sockets[0].commands("move_gripper").length, 1,
    "editing the gripper target did not start motion");
  harness.session.stop();
});

test("telemetry target following is not treated as a user gripper edit", async () => {
  const harness = makeHarness();
  await acquire(harness.session, harness.sockets);

  harness.followTelemetryTarget(0.03);
  harness.clock.tick(100);
  await nextTask();
  assertEqual(harness.sockets[0].commands("move_gripper").length, 0,
    "telemetry target following started gripper motion");

  harness.setTarget(0.04);
  harness.clock.tick(50);
  await nextTask();
  assertEqual(harness.sockets[0].commands("move_gripper").length, 1,
    "a user edit did not start gripper motion");
  harness.session.stop();
});

test("an edit during acquisition is sent after control becomes active", async () => {
  let releaseAcquire;
  const acquireGate = new Promise((resolve) => { releaseAcquire = resolve; });
  const harness = makeHarness({ beforeAcquire: () => acquireGate });
  const pending = harness.session.start();
  await nextTask();
  harness.setTarget(0.03);
  releaseAcquire();
  await nextTask();

  const socket = harness.sockets[0];
  await nextTask();
  socket.reply(BASE_INFO);
  await nextTask();
  socket.reply(runningState());
  await pending;
  harness.clock.tick(50);
  await nextTask();
  assertEqual(socket.commands("move_gripper").length, 1,
    "an edit made after acquisition started was discarded");
  assertEqual(socket.commands("move_gripper")[0].target_position, 0.03,
    "the acquisition-time edit sent the wrong target");
  harness.session.stop();
});

test("release closes the gripper lease before another send and permits a fresh drag", async () => {
  const harness = makeHarness();
  await acquire(harness.session, harness.sockets);
  harness.clock.tick(50);
  await nextTask();
  const firstCount = harness.sockets[0].commands("move_gripper").length;
  harness.session.stop("pointer released");
  harness.clock.tick(1000);
  await nextTask();
  assertEqual(harness.sockets[0].readyState, 3, "release retained the gripper lease");
  assertEqual(harness.sockets[0].commands("move_gripper").length, firstCount, "release allowed another gripper send");
  assertEqual(harness.session.state, "read-only", "released session was not reusable");

  harness.setTelemetry({ payload: runningState(), receivedAtMs: harness.clock.now() });
  await acquire(harness.session, harness.sockets);
  harness.setTarget(0.04, 0.02);
  harness.clock.tick(50);
  await nextTask();
  assertEqual(harness.sockets[1].commands("move_gripper").length, 1, "fresh drag sent no gripper command");
  harness.session.stop("pointer released");
});

test("gripper waits for arm control release and cancellation cannot acquire later", async () => {
  let releaseArm;
  const armReleased = new Promise((resolve) => { releaseArm = resolve; });
  const harness = makeHarness({ beforeAcquire: () => armReleased });
  const pending = harness.session.start();
  await nextTask();
  assertEqual(harness.sockets.length, 0, "gripper opened before arm control released");
  harness.session.stop("pointer released");
  releaseArm();
  await assertRejects(pending, /released|abort|cancel/i);
  assertEqual(harness.sockets.length, 0, "cancelled gripper opened a late lease");
});

test("unsafe telemetry and out-of-range targets close without another gripper send", async () => {
  const harness = makeHarness();
  await acquire(harness.session, harness.sockets);
  harness.setTarget(0.101, 0.02);
  harness.clock.tick(50);
  await nextTask();
  assertEqual(harness.sockets[0].commands("move_gripper").length, 0, "out-of-range gripper command was sent");
  assertEqual(harness.sockets[0].readyState, 3, "invalid gripper target retained the lease");

  const unsafe = makeHarness();
  await acquire(unsafe.session, unsafe.sockets);
  unsafe.setTelemetry({
    payload: { ...runningState(), control_ready: false, control_block_reason: "Emergency stop is active." },
    receivedAtMs: 0,
  });
  unsafe.clock.tick(50);
  await nextTask();
  assertEqual(unsafe.sockets[0].commands("move_gripper").length, 0, "emergency telemetry allowed gripper motion");
  assertEqual(unsafe.sockets[0].readyState, 3, "emergency telemetry retained the lease");
});

test("malformed canonical readiness closes gripper control before sending", async () => {
  const mutations = [
    (payload) => { delete payload.control_ready; },
    (payload) => { payload.control_ready = "true"; },
    (payload) => { payload.control_block_reason = "unexpected block"; },
    (payload) => { payload.control_ready = false; payload.control_block_reason = null; },
  ];
  for (const mutate of mutations) {
    const harness = makeHarness();
    await acquire(harness.session, harness.sockets);
    const payload = runningState();
    mutate(payload);
    harness.setTelemetry({ payload, receivedAtMs: harness.clock.now() });
    harness.clock.tick(50);
    await nextTask();
    assertEqual(harness.sockets[0].commands("move_gripper").length, 0,
      "malformed readiness allowed a gripper command");
    assertEqual(harness.sockets[0].readyState, 3, "malformed readiness retained the gripper lease");
  }
});

async function rejection(promise) {
  try { await promise; } catch (error) { return error; }
  throw new Error("expected promise to reject");
}

async function assertRejects(promise, pattern) {
  const error = await rejection(promise);
  assert(pattern.test(error?.message || String(error)), `unexpected rejection: ${error?.message || error}`);
}

scheduleTests();
