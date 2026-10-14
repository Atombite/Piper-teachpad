// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

"use strict";

let refreshInFlight = false;
let stateActionInFlight = false;
let latestState = null;

const byId = (id) => document.getElementById(id);

function setText(id, value) {
  byId(id).textContent = value;
}

function formatNumber(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function formatTimestamp(values) {
  const seconds = Array.isArray(values) ? Number(values[0]) : Number(values);
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  return `${new Date(seconds * 1000).toLocaleString()} · ${seconds.toFixed(3)}`;
}

function booleanLabel(value, positive, negative) {
  return value === true ? positive : value === false ? negative : "—";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error?.message || `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return payload;
}

function showError(message) {
  const banner = byId("poll-error");
  banner.textContent = `Communication error: ${message}`;
  banner.hidden = false;
}

function clearError() {
  const banner = byId("poll-error");
  banner.textContent = "";
  banner.hidden = true;
}

function renderArm(arm) {
  const card = byId("arm");
  const present = arm && Object.keys(arm).length > 0;
  const write = (field, value) => {
    card.querySelector(`[data-field="${field}"]`).textContent = value;
  };
  write("presence", present ? "Configured" : "Not configured");
  write("connected", present ? booleanLabel(arm.connected, "Yes", "No") : "—");
  write("enabled", present ? booleanLabel(arm.enabled, "Yes", "No") : "—");
  write("feedback", present ? booleanLabel(arm.feedback_fresh, "Fresh", "Stale") : "—");
  write("fault", present ? String(arm.fault_code ?? "—") : "—");
  card.dataset.health = !present ? "absent" : arm.connected && arm.enabled && arm.feedback_fresh && !arm.fault_code ? "good" : "bad";
}

function appendCell(row, value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value;
  cell.className = className;
  row.appendChild(cell);
}

function renderJoints(actual, target) {
  const body = byId("actual-joints");
  body.replaceChildren();
  for (let index = 0; index < 6; index += 1) {
    const row = document.createElement("tr");
    appendCell(row, `J${index + 1}`);
    appendCell(row, formatNumber(actual?.[index]), "numeric");
    appendCell(row, formatNumber(target?.[index]), "numeric");
    body.appendChild(row);
  }
}

function renderGrippers(actual, target) {
  const body = byId("grippers");
  body.replaceChildren();
  const row = document.createElement("tr");
  appendCell(row, "Gripper");
  appendCell(row, formatNumber(actual, 4), "numeric");
  appendCell(row, formatNumber(target, 4), "numeric");
  body.appendChild(row);
}

function renderState(state) {
  latestState = state;
  setText("overall-state", state.state ?? "—");
  const readinessValid = state.control_ready === true && state.control_block_reason === null;
  const blockedValid = state.control_ready === false
    && typeof state.control_block_reason === "string"
    && state.control_block_reason.length > 0;
  setText("control-readiness", readinessValid ? "READY" : "BLOCKED");
  setText("control-block-reason", readinessValid
    ? "Motion is available."
    : (blockedValid ? state.control_block_reason : "Control readiness is unavailable or invalid."));
  setText("emergency-state", booleanLabel(state.emergency_stopped, "ACTIVE", "Clear"));
  setText("first-fault", state.fault_reason || "None");
  setText("ws-occupancy", booleanLabel(state.ws_controller_connected, "Occupied", "Available"));
  setText("control-mode", typeof state.control_mode === "string" ? state.control_mode.toUpperCase() : "—");
  setText("ws-identity", state.ws_controller_identity || "None");
  renderArm(state.arm);
  renderJoints(state.ActualQ, state.TargetQ);
  renderGrippers(state.ActualGripper, state.TargetGripper);
  setText("receive-timestamp", formatTimestamp(state.robot_receive_timestamp));
  setText("robot-timestamp", formatTimestamp(state.robot_timestamp));
  setText("dashboard-timestamp", new Date().toLocaleString());
  document.body.dataset.emergency = state.emergency_stopped ? "true" : "false";
  document.body.dataset.controlReady = readinessValid ? "true" : "false";
  setStateControlsDisabled(stateActionInFlight);
}

async function refreshState() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    renderState(await fetchJson("/api/state"));
    clearError();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setText("control-readiness", "UNKNOWN");
    setText("control-block-reason", `State request failed: ${message}`);
    document.body.dataset.controlReady = "false";
    showError(message);
  } finally {
    refreshInFlight = false;
  }
}

async function loadBaseInfo() {
  try {
    const base = await fetchJson("/api/base_info");
    const side = base.side.toUpperCase();
    setText("base-info", `${side} ARM · :${base.config.server.port}`);
    setText("arm-label", `${side} arm`);
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  }
}

function setStateControlsDisabled(disabled) {
  document.querySelectorAll('[data-action="recover"], [data-action="homing"]').forEach((button) => {
    button.disabled = disabled;
  });
  byId("enter-hold").disabled = disabled || latestState?.state !== "RUNNING";
  const modeDisabled = disabled || latestState?.state !== "HOLD";
  byId("mode-local").disabled = modeDisabled;
  byId("mode-remote").disabled = modeDisabled;
}

async function runStateAction(label, operation) {
  if (stateActionInFlight) return;
  stateActionInFlight = true;
  setStateControlsDisabled(true);
  setText("action-status", `${label} request in progress…`);
  try {
    await operation();
    setText("action-status", `${label} request accepted.`);
    await refreshState();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setText("action-status", `${label} failed: ${message}`);
    showError(message);
  } finally {
    stateActionInFlight = false;
    setStateControlsDisabled(false);
  }
}

async function runEmergencyStop() {
  setText("action-status", "Emergency stop request in progress…");
  try {
    await fetchJson("/api/emergency_stop");
    setText("action-status", "Emergency stop request accepted.");
    await refreshState();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setText("action-status", `Emergency stop failed: ${message}`);
    showError(message);
  }
}

async function changeState(state) {
  await fetchJson("/change_state", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state }),
  });
}

async function changeControlMode(mode) {
  await fetchJson("/control_mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
}

document.querySelector('[data-action="emergency-stop"]').addEventListener("click", () => {
  runEmergencyStop();
});

document.querySelector('[data-action="recover"]').addEventListener("click", () => {
  if (!confirm("Recover from emergency stop? This does not start homing.")) return;
  runStateAction("Recovery", () => changeState("recover"));
});

document.querySelector('[data-action="homing"]').addEventListener("click", () => {
  if (!confirm("Start manual homing? The robot may move to its reset position.")) return;
  runStateAction("Homing", () => changeState("homing"));
});

document.querySelector('[data-action="hold"]').addEventListener("click", () => {
  runStateAction("HOLD", () => changeState("hold"));
});

document.querySelector('[data-action="mode-local"]').addEventListener("click", () => {
  runStateAction("Local mode", () => changeControlMode("local"));
});

document.querySelector('[data-action="mode-remote"]').addEventListener("click", () => {
  runStateAction("Remote mode", () => changeControlMode("remote"));
});

loadBaseInfo();
refreshState();
setInterval(refreshState, 200);
