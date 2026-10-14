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
  createTeachingState,
  gripperBlockReason,
  motionBlockReason,
  reduceTeachingState,
  resolveSynchronousIk,
  safetyStatusMessage,
} from "../../src/piper_control/web/teach-state.js";

const BASE_INFO = {
  protocol_version: "3.0",
  control_mode: "local",
  side: "left",
  supported_commands: ["hello", "servoJ", "move_gripper", "get_state", "get_base_info"],
  units: { joint_position: "degree", joint_speed: "degree/second" },
  arm: {
    server: { port: 8765 },
    joint_limits_deg: {
      min: [-150, 0, -154, -100, -69, -120],
      max: [150, 179, 0, 100, 69, 120],
    },
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
BASE_INFO.units.gripper_position = "meter";
BASE_INFO.units.gripper_speed = "meter/second";

function runningState(joints, gripper = 0.02) {
  return {
    state: "RUNNING",
    emergency_stopped: false,
    control_mode: "local",
    control_ready: true,
    control_block_reason: null,
    ws_controller_connected: false,
    ActualQ: joints,
    ActualGripper: gripper,
    arm: { connected: true, enabled: true, feedback_fresh: true },
  };
}

test("telemetry trusts the backend control readiness verdict", () => {
  let state = createTeachingState(BASE_INFO);
  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: {
      ...runningState([0, 90, -77, 0, 0, 0]),
      control_ready: false,
      control_block_reason: "Robot feedback is stale.",
    },
    receivedAtMs: 1000,
  });

  assertEqual(state.telemetryValid, false, "backend block verdict was ignored");
  assertEqual(motionBlockReason(state, 1000), "Robot feedback is stale.",
    "backend block reason was replaced by a page-local verdict");
});

test("telemetry reports remote mode and HOLD as distinct canonical block reasons", () => {
  for (const [update, expected] of [
    [{ control_mode: "remote" }, /remote control mode/i],
    [{ state: "HOLD", control_ready: false, control_block_reason: "Service is HOLD; RUNNING is required." }, /HOLD/i],
  ]) {
    let state = createTeachingState(BASE_INFO, { side: "left", port: 8765 });
    state = reduceTeachingState(state, {
      type: "telemetry",
      payload: { ...runningState([0, 90, -77, 0, 0, 0]), ...update },
      receivedAtMs: 1000,
    });
    assert(expected.test(motionBlockReason(state, 1000)),
      `canonical block reason was lost for ${JSON.stringify(update)}`);
  }
});

test("telemetry never overwrites an edited target until reset", () => {
  let state = createTeachingState(BASE_INFO);
  state = reduceTeachingState(state, { type: "telemetry", payload: runningState([0, 90, -77, 0, 0, 0]) });
  state = reduceTeachingState(state, { type: "target-joints", jointsDeg: [5, 80, -70, 1, 2, 3] });
  state = reduceTeachingState(state, { type: "telemetry", payload: runningState([1, 91, -76, 0, 0, 0]) });
  assertDeepEqual(state.targetJointsDeg, [5, 80, -70, 1, 2, 3], "edited target changed");
  state = reduceTeachingState(state, { type: "reset-target" });
  assertDeepEqual(state.targetJointsDeg, [1, 91, -76, 0, 0, 0], "reset did not copy actual");
});

test("malformed gripper preserves last finite fingers and valid arm telemetry", () => {
  let validState = createTeachingState(BASE_INFO);
  validState = reduceTeachingState(validState, { type: "telemetry", payload: runningState([0, 90, -77, 0, 0, 0], 0.025) });
  const next = reduceTeachingState(validState, {
    type: "telemetry",
    payload: runningState([1, 2, -3, 4, 5, 6], Number.NaN),
  });
  assertDeepEqual(next.actualJointsDeg, [1, 2, -3, 4, 5, 6], "valid arm telemetry was rejected");
  assertEqual(next.actualGripperM, 0.025, "last finite gripper was not preserved");
  assertEqual(motionBlockReason(next), null, "malformed optional gripper blocked arm motion");
});

test("gripper state follows telemetry until edited and validates configured metres", () => {
  let state = createTeachingState(BASE_INFO, { side: "left", port: 8765 });
  assertDeepEqual(state.gripperConfig, {
    enabled: true,
    backend: "piper",
    positionUnit: "meter",
    speedUnit: "meter/second",
    minPosition: 0,
    maxPosition: 0.1,
    physicalMinPosition: 0,
    physicalMaxPosition: 0.1,
    defaultSpeed: 0.02,
    maxSpeed: 0.05,
    displayScale: 1000,
    positionLabel: "mm",
    speedLabel: "mm/s",
    displayDecimals: 1,
    minPositionM: 0,
    maxPositionM: 0.1,
    defaultSpeedMps: 0.02,
    maxSpeedMps: 0.05,
  }, "gripper metadata was not normalized");
  assertEqual(state.gripperConfig.maxPosition * state.gripperConfig.displayScale, 100,
    "Piper opening was not displayed in millimetres");
  assertEqual(state.gripperConfig.defaultSpeed * state.gripperConfig.displayScale, 20,
    "Piper speed was not displayed in millimetres per second");
  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: runningState([0, 90, -77, 0, 0, 0], 0.025),
    receivedAtMs: 1000,
  });
  assertEqual(state.targetGripperM, 0.025, "initial gripper target did not follow feedback");
  assertEqual(state.gripperTargetRevision, 0, "telemetry was recorded as a user edit");
  state = reduceTeachingState(state, { type: "gripper-target", positionM: 0.08 });
  assertEqual(state.gripperTargetRevision, 1, "user gripper edit did not advance its revision");
  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: runningState([1, 90, -77, 0, 0, 0], 0.03),
    receivedAtMs: 1001,
  });
  assertEqual(state.actualGripperM, 0.03, "actual gripper did not update");
  assertEqual(state.targetGripperM, 0.08, "edited gripper target was overwritten");
  state = reduceTeachingState(state, { type: "gripper-control-released" });
  assertEqual(state.targetGripperM, 0.03, "released gripper target did not resync to feedback");
  assertEqual(state.gripperTargetEdited, false, "released gripper target remained edited");
  assertEqual(state.gripperTargetRevision, 1, "release was recorded as a user edit");
});

test("non-Piper and incompatible-unit metadata provide no gripper controls", () => {
  const unsupportedBackend = structuredClone(BASE_INFO);
  unsupportedBackend.units.gripper_position = "radian";
  unsupportedBackend.units.gripper_speed = "radian/second";
  unsupportedBackend.arm.gripper = {
    backend: "unsupported_backend",
    enabled: true,
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
    let state = createTeachingState(base, { side: "left", port: 8765 });
    assertEqual(state.gripperConfig, null, "unsupported metadata enabled gripper configuration");
    state = reduceTeachingState(state, {
      type: "telemetry",
      payload: runningState([0, 90, -77, 0, 0, 0], 0.02),
      receivedAtMs: 1000,
    });
    assert(gripperBlockReason(state, 1000), "unsupported metadata allowed gripper control");
    assertEqual(motionBlockReason(state, 1000), null, "optional gripper metadata blocked arm teaching");
  }
});

test("Piper physical feedback initializes a target within configured metre bounds", () => {
  const base = structuredClone(BASE_INFO);
  base.arm.gripper.position_m = { min: 0.01, max: 0.05 };
  base.arm.gripper.physical_position_m = { min: 0, max: 0.07 };
  let state = createTeachingState(base, { side: "left", port: 8765 });
  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: runningState([0, 90, -77, 0, 0, 0], 0.06),
    receivedAtMs: 1000,
  });
  assertEqual(state.actualGripperM, 0.06, "physical Piper feedback was rejected");
  assertEqual(state.targetGripperM, 0.05, "Piper target was initialized outside its command range");
  assertEqual(gripperBlockReason(state, 1000), null,
    "physical Piper feedback disabled gripper control");
});

test("gripper safety ignores joint IK but fails closed on gripper and lease hazards", () => {
  let ready = createTeachingState(BASE_INFO, { side: "left", port: 8765 });
  ready = reduceTeachingState(ready, {
    type: "telemetry",
    payload: runningState([0, 90, -77, 0, 0, 0], 0.02),
    receivedAtMs: 1000,
  });
  ready = reduceTeachingState(ready, { type: "requested-tcp", pose: "unvalidated" });
  assertEqual(gripperBlockReason(ready, 1000), null, "joint IK incorrectly blocked the gripper");

  const occupied = reduceTeachingState(ready, {
    type: "telemetry",
    payload: { ...runningState([0, 90, -77, 0, 0, 0], 0.02), ws_controller_connected: true },
    receivedAtMs: 1000,
  });
  assert(/controller/i.test(gripperBlockReason(occupied, 1000)), "foreign lease did not block gripper");
  assertEqual(gripperBlockReason(occupied, 1000, { ownsControl: true }), null,
    "page-owned gripper lease was treated as foreign");

  const disabledBase = structuredClone(BASE_INFO);
  disabledBase.arm.gripper.enabled = false;
  let disabled = createTeachingState(disabledBase, { side: "left", port: 8765 });
  disabled = reduceTeachingState(disabled, {
    type: "telemetry",
    payload: runningState([0, 90, -77, 0, 0, 0], 0),
    receivedAtMs: 1000,
  });
  assert(/disabled/i.test(gripperBlockReason(disabled, 1000)), "disabled gripper was enabled");
});

test("motion gate rejects service, telemetry, target, IK, lease, request, and model faults", () => {
  let ready = createTeachingState(BASE_INFO);
  ready = reduceTeachingState(ready, { type: "telemetry", payload: runningState([0, 90, -77, 0, 0, 0]) });
  ready = reduceTeachingState(ready, { type: "model-ready" });
  assertEqual(motionBlockReason(ready), null, "valid preview should permit motion");

  const cases = [
    ["model-error", { type: "model-error", message: "link4.stl failed" }, /model/i],
    ["ik-error", { type: "ik-result", requestRevision: 0, result: { ok: false, reason: "unreachable" } }, /IK/i],
    ["request", { type: "control-state", controlState: "acquiring" }, /progress/i],
  ];
  for (const [name, event, expected] of cases) {
    const blocked = reduceTeachingState(ready, event);
    assert(expected.test(motionBlockReason(blocked)), `${name} did not fail closed`);
  }

  const occupied = reduceTeachingState(ready, {
    type: "telemetry",
    payload: { ...runningState([0, 90, -77, 0, 0, 0]), ws_controller_connected: true },
  });
  assert(/controller/i.test(motionBlockReason(occupied)), "occupied lease did not block");

  const owned = reduceTeachingState(occupied, { type: "control-state", controlState: "live-armed" });
  assertEqual(motionBlockReason(owned), null, "the page treated its own acquired lease as a conflict");

  const acquiring = reduceTeachingState(occupied, { type: "control-state", controlState: "acquiring" });
  const acquiringReason = motionBlockReason(acquiring);
  assert(/progress/i.test(acquiringReason), "lease acquisition did not report its in-flight request");
  assert(!/controller/i.test(acquiringReason),
    "lease acquisition treated its own pending WebSocket lease as a foreign controller");

  const reviewed = reduceTeachingState(occupied, { type: "control-state", controlState: "executing" });
  const reviewedReason = motionBlockReason(reviewed);
  assert(/progress/i.test(reviewedReason), "reviewed execution did not report its in-flight request");
  assert(!/controller/i.test(reviewedReason),
    "reviewed execution treated its own WebSocket lease as a foreign controller");
});

test("model failure is persistent across telemetry and can only clear on reload", () => {
  let state = createTeachingState(BASE_INFO);
  state = reduceTeachingState(state, { type: "model-error", message: "link4.stl failed" });
  state = reduceTeachingState(state, { type: "telemetry", payload: runningState([0, 90, -77, 0, 0, 0]) });
  assert(/link4\.stl/.test(motionBlockReason(state)), "telemetry cleared the model failure");
});

test("missing or malformed canonical readiness fields fail closed", () => {
  for (const [field, value] of [
    ["control_ready", undefined],
    ["control_ready", null],
    ["control_ready", 1],
    ["control_ready", "true"],
    ["control_block_reason", undefined],
    ["control_block_reason", "unexpected block"],
  ]) {
    const payload = runningState([0, 90, -77, 0, 0, 0]);
    if (value === undefined) delete payload[field];
    else payload[field] = value;
    let state = createTeachingState(BASE_INFO, { side: "left", port: 8765 });
    state = reduceTeachingState(state, { type: "telemetry", payload, receivedAtMs: 1000 });
    assert(/readiness|unavailable|invalid/i.test(motionBlockReason(state, 1000)),
      `${field}=${value} was accepted`);
  }
});

test("missing and malformed joint telemetry fail closed", () => {
  for (const joints of [undefined, [0, 1], [0, 90, -77, 0, 0, Number.NaN]]) {
    const payload = runningState([0, 90, -77, 0, 0, 0]);
    payload.ActualQ = joints;
    let state = createTeachingState(BASE_INFO);
    state = reduceTeachingState(state, { type: "telemetry", payload, receivedAtMs: 1000 });
    assert(/six finite/i.test(motionBlockReason(state, 1000)), `malformed joints ${joints} were accepted`);
  }
});

test("client receipt age makes otherwise valid telemetry stale", () => {
  let state = createTeachingState(BASE_INFO, { side: "left", port: 8765 });
  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: runningState([0, 90, -77, 0, 0, 0]),
    receivedAtMs: 1000,
  });
  assertEqual(motionBlockReason(state, 1500), null, "fresh client telemetry was blocked");
  assert(/stale/i.test(motionBlockReason(state, 2501)), "aged client telemetry remained valid");
});

test("a transient poll failure blocks motion without replacing the last service state", () => {
  let state = createTeachingState(BASE_INFO, { side: "left", port: 8765 });
  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: runningState([0, 90, -77, 0, 0, 0]),
    receivedAtMs: 1000,
  });
  state = reduceTeachingState(state, {
    type: "telemetry-error",
    message: "Request timed out after 1000 ms.",
  });

  assertEqual(state.serviceState, "RUNNING", "poll failure replaced the real service state");
  assertEqual(state.telemetryFailureCount, 1, "poll failure was not tracked independently");
  assert(/timed out/i.test(motionBlockReason(state, 2000)), "poll failure did not fail closed");
  assert(/RUNNING.*feedback delayed.*motion paused/i.test(safetyStatusMessage(state, 2000)),
    "first poll failure was presented as a controller state change");
  assert(!/1000 ms/i.test(safetyStatusMessage(state, 2000)),
    "first poll failure exposed a transient timeout as the primary safety state");

  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: runningState([0, 90, -77, 0, 0, 0]),
    receivedAtMs: 2100,
  });
  assertEqual(state.telemetryFailureCount, 0, "successful telemetry did not clear communication failures");
  assertEqual(safetyStatusMessage(state, 2100), "Target is valid for motion.",
    "successful telemetry did not restore the safety display");
});

test("base info verifies protocol, side, port, commands, units, speeds, and limits", () => {
  const mutations = [
    (base) => { delete base.protocol_version; },
    (base) => { base.protocol_version = "1.0"; },
    (base) => { base.side = "right"; },
    (base) => { base.arm.server.port = 9876; },
    (base) => { base.supported_commands = ["get_state"]; },
    (base) => { base.units.joint_position = "radian"; },
    (base) => { delete base.arm.joint_speed_deg_s; },
    (base) => { base.arm.joint_limits_deg.min = [0]; },
  ];
  for (const mutate of mutations) {
    const bogus = structuredClone(BASE_INFO);
    mutate(bogus);
    const state = createTeachingState(bogus, { side: "left", port: 8765 });
    assert(/base|protocol|side|port|command|unit|speed|limit/i.test(motionBlockReason(state, 0)), "bogus base info was accepted");
  }
});

test("requested TCP invalidates IK until a matching finite in-limit solve arrives", () => {
  let state = createTeachingState(BASE_INFO);
  state = reduceTeachingState(state, { type: "telemetry", payload: runningState([0, 90, -77, 0, 0, 0]), receivedAtMs: 1000 });
  state = reduceTeachingState(state, { type: "requested-tcp", pose: "pose-a" });
  const revisionA = state.tcpRequestRevision;
  assert(/IK/i.test(motionBlockReason(state, 1000)), "new TCP request retained old IK validation");

  state = reduceTeachingState(state, { type: "requested-tcp", pose: "pose-b" });
  const revisionB = state.tcpRequestRevision;
  state = reduceTeachingState(state, {
    type: "ik-result",
    requestRevision: revisionA,
    result: { ok: true, jointsDeg: [1, 90, -77, 0, 0, 0] },
  });
  assert(/IK/i.test(motionBlockReason(state, 1000)), "stale solve result validated a newer target");

  state = reduceTeachingState(state, {
    type: "ik-result",
    requestRevision: revisionB,
    result: { ok: true, jointsDeg: [1, 2] },
  });
  assert(/IK/i.test(motionBlockReason(state, 1000)), "malformed successful solve opened the gate");

  state = reduceTeachingState(state, {
    type: "ik-result",
    requestRevision: revisionB,
    result: { ok: true, jointsDeg: [1, 90, -77, 0, 0, 0] },
  });
  assertEqual(motionBlockReason(state, 1000), null, "matching valid solve did not restore gate");
});

test("a thrown synchronous IK solve becomes a persistent invalid result", () => {
  let state = createTeachingState(BASE_INFO);
  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: runningState([0, 90, -77, 0, 0, 0]),
    receivedAtMs: 1000,
  });
  const previous = [...state.targetJointsDeg];
  state = resolveSynchronousIk(state, "requested-pose", () => {
    throw new Error("jacobian exploded");
  });
  assertDeepEqual(state.targetJointsDeg, previous, "failed IK replaced the last safe target");
  assert(/jacobian exploded/i.test(state.ik.reason), "thrown IK reason was hidden");
  assert(/IK/i.test(motionBlockReason(state, 1000)), "thrown IK left motion enabled");
});

test("a fresh read-only lifecycle clears a prior control error for explicit reacquire", () => {
  let state = createTeachingState(BASE_INFO);
  state = reduceTeachingState(state, { type: "control-error", message: "connection lost" });
  state = reduceTeachingState(state, { type: "control-state", controlState: "read-only", message: null });
  assertEqual(state.controlError, null, "fresh lifecycle retained an obsolete motion fault");
});

test("a finished control request returns to read-only without blocking retry", () => {
  let state = createTeachingState(BASE_INFO);
  state = reduceTeachingState(state, {
    type: "telemetry",
    payload: runningState([0, 90, -77, 0, 0, 0]),
    receivedAtMs: 1000,
  });
  state = reduceTeachingState(state, { type: "control-state", controlState: "executing" });
  state = reduceTeachingState(state, { type: "control-finished", message: "Page focus lost." });

  assertEqual(state.controlState, "read-only", "finished request remained in progress");
  assertEqual(motionBlockReason(state, 1000), null, "finished request blocked the next action");
});

scheduleTests();
