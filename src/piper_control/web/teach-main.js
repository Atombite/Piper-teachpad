// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";

import { readToolPose, setRobotJoints, solveIk } from "./teach-kinematics.js";
import {
  controlAvailability,
  createControlLifecycle,
  createLiveControlSession,
  executeReviewedTarget,
  failClosedOnTelemetryFailure,
  readPageExpectedIdentity,
} from "./teach-control.js";
import { fetchJsonWithTimeout } from "./teach-network.js";
import { createGripperControlSession } from "./teach-gripper.js";
import {
  createTeachingState,
  gripperBlockReason,
  motionBlockReason,
  reduceTeachingState,
  resolveSynchronousIk,
  safetyStatusMessage,
} from "./teach-state.js";
import { createTeachingViewer } from "./teach-viewer.js";

const byId = (id) => document.getElementById(id);
const TCP_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"];
const MODEL_URL = "/static/teach/models/piper/piper.urdf";

let teachingState = null;
let viewer = null;
let pollInFlight = false;
let baseInfo = null;
let liveFollower = null;
let reviewedExecutionActive = false;
let reviewedAbortController = null;
let reviewedExecutionPromise = null;
let latestTelemetry = null;
let controlLifecycle = null;
let expectedIdentity = null;
let gripperSession = null;
let gripperSpeedMps = null;
let gripperMessage = null;

async function fetchJson(url, options = {}) {
  return fetchJsonWithTimeout(window.fetch.bind(window), url, { timeoutMs: 1000, ...options });
}

function matrixToTcp(matrix) {
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, quaternion, scale);
  const rotation = new THREE.Euler().setFromQuaternion(quaternion, "XYZ");
  return [
    position.x * 1000,
    position.y * 1000,
    position.z * 1000,
    THREE.MathUtils.radToDeg(rotation.x),
    THREE.MathUtils.radToDeg(rotation.y),
    THREE.MathUtils.radToDeg(rotation.z),
  ];
}

function tcpToMatrix(values) {
  if (values.length !== 6 || !values.every(Number.isFinite)) return null;
  const rotation = new THREE.Euler(
    THREE.MathUtils.degToRad(values[3]),
    THREE.MathUtils.degToRad(values[4]),
    THREE.MathUtils.degToRad(values[5]),
    "XYZ",
  );
  return new THREE.Matrix4().compose(
    new THREE.Vector3(values[0] / 1000, values[1] / 1000, values[2] / 1000),
    new THREE.Quaternion().setFromEuler(rotation),
    new THREE.Vector3(1, 1, 1),
  );
}

function dispatch(event) {
  teachingState = reduceTeachingState(teachingState, event);
  render();
}

function renderTcpFromTarget() {
  if (!viewer?.targetRobot || !teachingState.targetJointsDeg) return;
  setRobotJoints(viewer.targetRobot, teachingState.targetJointsDeg);
  const pose = readToolPose(viewer.targetRobot);
  if (!teachingState.requestedTcp) viewer.setRequestedTcp(pose);
  const values = matrixToTcp(teachingState.requestedTcp || pose);
  TCP_KEYS.forEach((key, index) => {
    const input = document.querySelector(`[data-tcp="${key}"]`);
    if (document.activeElement !== input) input.value = values[index].toFixed(2);
  });
}

function render() {
  if (!teachingState) return;
  const blockReason = motionBlockReason(teachingState);
  if (liveFollower && ["live-armed", "live-active"].includes(liveFollower.state)) {
    const liveSafetyReason = motionBlockReason({ ...teachingState, controlState: "live-armed" });
    if (liveSafetyReason) liveFollower.deactivate(liveSafetyReason);
  }
  byId("arm-side").textContent = teachingState.side?.toUpperCase() || "—";
  byId("connection-status").textContent = teachingState.telemetryValid
    ? "Feedback online"
    : teachingState.telemetryFailureCount > 0 && teachingState.serviceState
      ? `Feedback delayed · ${teachingState.serviceState}`
      : "Read only";
  byId("motion-block-reason").textContent = safetyStatusMessage(teachingState);
  byId("motion-block-reason").className = blockReason ? "blocked" : "ready";

  const badge = byId("ik-badge");
  badge.textContent = teachingState.ik.ok ? "IK valid" : "IK invalid";
  badge.className = `badge ${teachingState.ik.ok ? "valid" : "invalid"}`;
  const error = teachingState.modelError || teachingState.controlError;
  byId("error-detail").hidden = !error;
  byId("error-detail").textContent = error || "";

  if (viewer?.actualRobot && teachingState.actualJointsDeg) {
    viewer.setActual(teachingState.actualJointsDeg, teachingState.actualGripperM);
  }
  if (viewer?.targetRobot && teachingState.targetJointsDeg) {
    viewer.setTarget(teachingState.targetJointsDeg, teachingState.targetGripperM);
    viewer.setTargetValidity(teachingState.ik.ok && !teachingState.ik.pending);
    renderTcpFromTarget();
  }

  document.querySelectorAll(".joint-row").forEach((row, index) => {
    const actual = row.querySelector("[data-role=actual]");
    const range = row.querySelector("[data-role=range]");
    const number = row.querySelector("[data-role=number]");
    actual.textContent = Number.isFinite(teachingState.actualJointsDeg?.[index])
      ? `${teachingState.actualJointsDeg[index].toFixed(2)}°`
      : "—";
    const target = teachingState.targetJointsDeg?.[index];
    if (Number.isFinite(target)) {
      if (document.activeElement !== range) range.value = String(target);
      if (document.activeElement !== number) number.value = target.toFixed(2);
    }
  });

  const hasActual = Array.isArray(teachingState.actualJointsDeg);
  byId("reset-target").disabled = !hasActual;
  const lifecycle = reviewedExecutionActive ? "executing" : (liveFollower?.state || teachingState.controlState);
  const gripperOwnsControl = gripperSession && ["acquiring", "active", "stopping"].includes(gripperSession.state);
  const availability = controlAvailability({
    mode: byId("teaching-mode").value,
    blockReason,
    lifecycle: gripperOwnsControl ? "gripper-active" : lifecycle,
  });
  byId("acquire-control").disabled = !availability.acquire;
  byId("execute-target").disabled = !availability.execute;
  byId("live-deadman").disabled = !availability.deadman;
  document.querySelectorAll("#tcp-controls input, #joint-controls input").forEach((input) => {
    input.disabled = Boolean(teachingState.modelError) || !hasActual || gripperOwnsControl;
  });

  const pageOwnsArmLease = gripperOwnsControl
    || reviewedExecutionActive
    || (liveFollower && !["read-only", "disposed"].includes(liveFollower.state));
  const gripperReason = gripperBlockReason(teachingState, Date.now(), { ownsControl: pageOwnsArmLease });
  const config = teachingState.gripperConfig;
  byId("gripper-actual").textContent = config && Number.isFinite(teachingState.actualGripperM)
    ? `${(teachingState.actualGripperM * config.displayScale).toFixed(config.displayDecimals)} ${config.positionLabel}` : "—";
  if (config && Number.isFinite(teachingState.targetGripperM)) {
    const displayedTarget = teachingState.targetGripperM * config.displayScale;
    if (document.activeElement !== byId("gripper-position")) byId("gripper-position").value = String(displayedTarget);
    if (document.activeElement !== byId("gripper-position-number")) byId("gripper-position-number").value = displayedTarget.toFixed(config.displayDecimals);
  }
  const gripperEnabled = Boolean(config) && (!gripperReason || gripperOwnsControl);
  byId("gripper-position").disabled = !gripperEnabled;
  byId("gripper-position-number").disabled = !gripperEnabled;
  byId("gripper-speed").disabled = !gripperEnabled;
  byId("gripper-speed-number").disabled = !gripperEnabled;
  byId("gripper-status").textContent = gripperOwnsControl
    ? (gripperSession.state === "active" ? "LIVE: release to stop and return to HOLD." : "Switching exclusive control…")
    : (gripperMessage || gripperReason || "Press or drag Opening to control the gripper.");
}

function configureGripperControls() {
  const config = teachingState.gripperConfig;
  if (!config) return;
  const positionStep = Math.max((config.maxPosition - config.minPosition) * config.displayScale / 1000, 0.01);
  const speedStep = Math.max(config.maxSpeed * config.displayScale / 500, 0.01);
  for (const input of [byId("gripper-position"), byId("gripper-position-number")]) {
    input.min = String(config.minPosition * config.displayScale);
    input.max = String(config.maxPosition * config.displayScale);
    input.step = String(positionStep);
  }
  for (const input of [byId("gripper-speed"), byId("gripper-speed-number")]) {
    input.min = String(speedStep);
    input.max = String(config.maxSpeed * config.displayScale);
    input.step = String(speedStep);
    input.value = String(config.defaultSpeed * config.displayScale);
  }
  byId("gripper-position-unit").textContent = config.positionLabel;
  byId("gripper-speed-unit").textContent = config.speedLabel;
  byId("gripper-position-number").setAttribute("aria-label", `Gripper opening ${config.positionLabel}`);
  byId("gripper-speed-number").setAttribute("aria-label", `Gripper speed ${config.speedLabel}`);
  gripperSpeedMps = config.defaultSpeed;
}

function editGripperPosition(rawValue) {
  const config = teachingState.gripperConfig;
  if (!config) return;
  const valueM = Number(rawValue) / config.displayScale;
  if (!Number.isFinite(valueM)) return;
  dispatch({
    type: "gripper-target",
    positionM: Math.max(config.minPosition, Math.min(config.maxPosition, valueM)),
  });
}

function editGripperSpeed(rawValue) {
  const config = teachingState.gripperConfig;
  if (!config) return;
  const valueMps = Number(rawValue) / config.displayScale;
  if (!Number.isFinite(valueMps)) return;
  gripperSpeedMps = Math.max(Number.MIN_VALUE, Math.min(config.maxSpeed, valueMps));
  const valueMmS = gripperSpeedMps * config.displayScale;
  if (document.activeElement !== byId("gripper-speed")) byId("gripper-speed").value = String(valueMmS);
  if (document.activeElement !== byId("gripper-speed-number")) byId("gripper-speed-number").value = valueMmS.toFixed(2);
}

async function releaseArmControlForGripper() {
  const interruptedReviewedExecution = Boolean(reviewedExecutionPromise);
  reviewedAbortController?.abort("Switching to gripper control.");
  if (reviewedExecutionPromise) {
    try { await reviewedExecutionPromise; } catch (_error) { /* UI already reports the result. */ }
  }
  if (interruptedReviewedExecution) {
    dispatch({ type: "control-state", controlState: "read-only", message: null });
  }
  await liveFollower?.stopAndWait("Switching to gripper control.");
}

function makeGripperSession() {
  if (gripperSession) return gripperSession;
  gripperSession = createGripperControlSession({
    expectedIdentity,
    getTargetSnapshot: () => ({
      targetPosition: teachingState.targetGripperM,
      speed: gripperSpeedMps,
      editRevision: teachingState.gripperTargetRevision,
    }),
    getTelemetrySnapshot: () => latestTelemetry,
    beforeAcquire: releaseArmControlForGripper,
    onStateChange: (state, message) => {
      if (state === "read-only") {
        gripperSession = null;
        teachingState = reduceTeachingState(teachingState, {
          type: "gripper-control-released",
        });
      }
      if (message) gripperMessage = message;
      render();
    },
  });
  return gripperSession;
}

async function startGripperControl() {
  const pageOwnsArmLease = reviewedExecutionActive
    || (liveFollower && !["read-only", "disposed"].includes(liveFollower.state));
  const reason = gripperBlockReason(teachingState, Date.now(), { ownsControl: pageOwnsArmLease });
  if (reason || (gripperSession && gripperSession.state !== "read-only")) return;
  gripperMessage = null;
  try {
    await makeGripperSession().start();
  } catch (error) {
    gripperMessage = error.message;
    render();
  }
}

function editJoint(index, rawValue) {
  if (!teachingState.targetJointsDeg) return;
  const value = Number(rawValue);
  if (!Number.isFinite(value)) return;
  const [minimum, maximum] = teachingState.limitsDeg[index];
  const target = [...teachingState.targetJointsDeg];
  target[index] = Math.max(minimum, Math.min(maximum, value));
  dispatch({ type: "target-joints", jointsDeg: target });
}

function buildJointControls() {
  const container = byId("joint-controls");
  teachingState.limitsDeg.forEach(([minimum, maximum], index) => {
    const row = document.createElement("div");
    row.className = "joint-row";
    row.innerHTML = `
      <div class="joint-meta">
        <strong>J${index + 1}</strong>
        <span data-role="actual">—</span>
        <small>${minimum.toFixed(1)}° … ${maximum.toFixed(1)}°</small>
      </div>
      <input data-role="range" aria-label="J${index + 1} target" type="range" min="${minimum}" max="${maximum}" step="0.1" disabled>
      <input data-role="number" aria-label="J${index + 1} target degrees" type="number" min="${minimum}" max="${maximum}" step="0.1" disabled>
    `;
    row.querySelectorAll("input").forEach((input) => {
      input.addEventListener("input", () => editJoint(index, input.value));
    });
    container.append(row);
  });
}

function solveRequestedPose(pose) {
  if (!viewer?.targetRobot || !teachingState.targetJointsDeg) return;
  const previous = [...teachingState.targetJointsDeg];
  // IK is synchronous, so resolve request and result before rendering. This
  // avoids a transient pending-state disconnect while still converting every
  // rejection or exception into an invalid state that render stops fail-closed.
  teachingState = resolveSynchronousIk(teachingState, pose, () => (
    solveIk(viewer.targetRobot, pose, previous, teachingState.limitsDeg)
  ));
  render();
  if (!teachingState.ik.ok) viewer.setTarget(previous, teachingState.actualGripperM);
}

function bindControls() {
  controlLifecycle = createControlLifecycle({
    stop: (reason) => {
      reviewedAbortController?.abort(reason);
      if (liveFollower && !["read-only", "disposed", "stopping"].includes(liveFollower.state)) {
        liveFollower.deactivate(reason);
      }
    },
  });
  document.querySelectorAll("#tcp-controls input").forEach((input) => {
    input.addEventListener("change", () => {
      const values = TCP_KEYS.map((key) => Number(document.querySelector(`[data-tcp="${key}"]`).value));
      const pose = tcpToMatrix(values);
      if (pose) solveRequestedPose(pose);
    });
  });
  byId("gizmo-translate").addEventListener("click", () => {
    viewer?.setMode("translate");
    byId("gizmo-translate").setAttribute("aria-pressed", "true");
    byId("gizmo-rotate").setAttribute("aria-pressed", "false");
  });
  byId("gizmo-rotate").addEventListener("click", () => {
    viewer?.setMode("rotate");
    byId("gizmo-translate").setAttribute("aria-pressed", "false");
    byId("gizmo-rotate").setAttribute("aria-pressed", "true");
  });
  byId("reset-camera").addEventListener("click", () => viewer?.resetCamera());
  byId("reset-target").addEventListener("click", () => dispatch({ type: "reset-target" }));
  byId("teaching-mode").addEventListener("change", (event) => {
    window.dispatchEvent(new Event("teaching-mode-change"));
    byId("command-status").textContent = event.target.value === "preview"
      ? "Preview only. Edits do not move the arm."
      : event.target.value === "reviewed"
        ? "Review the target, then execute one immutable snapshot."
        : "Acquire control, then hold the deadman continuously.";
    if (event.target.value === "preview") {
      disposeLiveControl();
      dispatch({ type: "control-state", controlState: "preview", message: null });
      return;
    }
    render();
  });
  byId("acquire-control").addEventListener("click", acquireLiveControl);
  byId("execute-target").addEventListener("click", beginReviewedExecution);
  const deadman = byId("live-deadman");
  deadman.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    activateLiveControl();
  });
  deadman.addEventListener("keydown", (event) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) {
      event.preventDefault();
      activateLiveControl();
    }
  });
  byId("emergency-stop").addEventListener("click", async () => {
    window.dispatchEvent(new Event("emergency-stop"));
    if (!window.confirm("Emergency-stop this arm now?")) return;
    try {
      await fetchJson("/api/emergency_stop");
      byId("command-status").textContent = "Emergency stop accepted.";
      await pollState();
    } catch (error) {
      dispatch({ type: "control-error", message: `Emergency stop failed: ${error.message}` });
    }
  });
  const positionRange = byId("gripper-position");
  const positionNumber = byId("gripper-position-number");
  const speedRange = byId("gripper-speed");
  const speedNumber = byId("gripper-speed-number");
  positionRange.addEventListener("pointerdown", () => { void startGripperControl(); });
  positionRange.addEventListener("keydown", (event) => {
    if (!event.repeat) void startGripperControl();
  });
  positionNumber.addEventListener("focus", () => { void startGripperControl(); });
  positionNumber.addEventListener("blur", () => gripperSession?.stop("Gripper input released."));
  for (const input of [positionRange, positionNumber]) {
    input.addEventListener("input", () => editGripperPosition(input.value));
  }
  for (const input of [speedRange, speedNumber]) {
    input.addEventListener("input", () => editGripperSpeed(input.value));
  }
}

function defaultSpeed() {
  return [...baseInfo.arm.joint_speed_deg_s.default];
}

function disposeLiveControl() {
  if (!liveFollower) return;
  liveFollower.dispose();
  liveFollower = null;
}

function makeLiveFollower() {
  if (liveFollower) return liveFollower;
  liveFollower = createLiveControlSession({
    getTargetSnapshot: () => [...teachingState.targetJointsDeg],
    getTelemetrySnapshot: () => latestTelemetry,
    expectedIdentity,
    speedDegS: defaultSpeed(),
    onStateChange: (controlState, message) => {
      if (controlState === "stopping") {
        dispatch({ type: "control-state", controlState, message: null });
        byId("command-status").textContent = message || "Live following stopped; control lease released.";
      } else if (controlState === "read-only") {
        dispatch({ type: "control-state", controlState, message: null });
        byId("command-status").textContent = message || "Live following stopped; control lease released. Acquire again to resume.";
      } else {
        dispatch({ type: "control-state", controlState, message: null });
      }
    },
  });
  return liveFollower;
}

async function acquireLiveControl() {
  const reason = motionBlockReason(teachingState);
  if (reason) return;
  const follower = makeLiveFollower();
  byId("command-status").textContent = "Acquiring exclusive control…";
  try {
    await follower.acquire();
    byId("command-status").textContent = "Control acquired. Hold the deadman to move.";
  } catch (error) {
    byId("command-status").textContent = error.message;
    dispatch({ type: "control-finished", message: error.message });
    liveFollower?.dispose();
    liveFollower = null;
  }
}

async function activateLiveControl() {
  if (!liveFollower || motionBlockReason(teachingState)) return;
  try {
    await liveFollower.activate();
    byId("command-status").textContent = "LIVE: release the deadman to stop and HOLD.";
  } catch (error) {
    await liveFollower.stopAndWait(error.message);
    dispatch({ type: "control-finished", message: error.message });
  }
}

async function runReviewedExecution() {
  const reason = motionBlockReason(teachingState);
  if (reason || reviewedExecutionActive) return;
  reviewedExecutionActive = true;
  const abortController = new AbortController();
  reviewedAbortController = abortController;
  const target = Object.freeze([...teachingState.targetJointsDeg]);
  dispatch({ type: "control-state", controlState: "executing", message: null });
  byId("command-status").textContent = "Executing reviewed target…";
  try {
    await executeReviewedTarget({
      targetJointsDeg: target,
      speedDegS: defaultSpeed(),
      signal: abortController.signal,
      expectedIdentity,
      onProgress: (message) => { byId("command-status").textContent = message; },
    });
    byId("command-status").textContent = "Reviewed target reached; control released to HOLD.";
    dispatch({ type: "control-state", controlState: "read-only", message: null });
  } catch (error) {
    byId("command-status").textContent = error.message;
    dispatch({ type: "control-finished", message: error.message });
  } finally {
    if (reviewedAbortController === abortController) reviewedAbortController = null;
    reviewedExecutionActive = false;
    render();
  }
}

function beginReviewedExecution() {
  if (reviewedExecutionPromise) return;
  const execution = runReviewedExecution();
  reviewedExecutionPromise = execution;
  void execution.finally(() => {
    if (reviewedExecutionPromise === execution) reviewedExecutionPromise = null;
  });
}

async function pollState() {
  if (pollInFlight || !teachingState) return;
  pollInFlight = true;
  try {
    const payload = await fetchJson("/api/state");
    const receivedAtMs = Date.now();
    latestTelemetry = { payload, receivedAtMs };
    dispatch({ type: "telemetry", payload, receivedAtMs });
    if (liveFollower && ["live-armed", "live-active"].includes(liveFollower.state)) {
      const safetyReason = motionBlockReason({ ...teachingState, controlState: "live-armed" });
      if (safetyReason) liveFollower.deactivate(safetyReason);
    }
  } catch (error) {
    const reason = `Telemetry request failed: ${error.message}`;
    latestTelemetry = null;
    dispatch({ type: "telemetry-error", message: error.message });
    failClosedOnTelemetryFailure({
      reviewedAbortController,
      gripperSession,
      liveFollower,
    }, reason);
  } finally {
    pollInFlight = false;
  }
}

async function initialize() {
  expectedIdentity = readPageExpectedIdentity(document, window.location);
  bindControls();
  try {
    baseInfo = await fetchJson("/api/base_info");
  } catch (error) {
    baseInfo = null;
  }
  const expectedPort = Number(document.body.dataset.serverPort);
  teachingState = createTeachingState(baseInfo, {
    side: document.body.dataset.armSide,
    port: Number.isInteger(expectedPort) ? expectedPort : undefined,
  });
  if (!baseInfo) dispatch({ type: "telemetry-error", message: "Base information could not be loaded." });
  if (teachingState.limitsDeg) buildJointControls();
  configureGripperControls();
  render();

  viewer = createTeachingViewer(byId("viewer-canvas"), MODEL_URL, {
    gripperConfig: teachingState.gripperConfig,
  });
  byId("viewer-canvas").addEventListener("teach-requested-tcp", (event) => solveRequestedPose(event.detail));
  try {
    await viewer.ready;
    byId("viewer-loading")?.remove();
    dispatch({ type: "model-ready" });
  } catch (error) {
    dispatch({ type: "model-error", message: error instanceof Error ? error.message : String(error) });
  }
  await pollState();
  window.setInterval(pollState, 200);
  window.setInterval(render, 200);
}

initialize().catch((error) => {
  if (!teachingState) teachingState = createTeachingState(null);
  dispatch({ type: "control-error", message: error instanceof Error ? error.message : String(error) });
});
