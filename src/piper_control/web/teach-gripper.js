// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import { PiperWebSocketClient } from "./teach-control.js";

const SEND_PERIOD_MS = 50;
const TELEMETRY_STALE_MS = 1000;

function defaultClock() {
  return {
    setTimeout: window.setTimeout.bind(window),
    clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window),
    clearInterval: window.clearInterval.bind(window),
    now: () => Date.now(),
  };
}

function abortError(message) {
  const error = new Error(message || "Gripper control cancelled.");
  error.name = "AbortError";
  return error;
}

function readMetadata(baseInfo, expected) {
  const gripper = baseInfo?.arm?.gripper;
  const isPiper = gripper?.backend === "piper"
    && baseInfo?.units?.gripper_position === "meter"
    && baseInfo?.units?.gripper_speed === "meter/second";
  const position = gripper?.position_m;
  const physical = gripper?.physical_position_m || position;
  const speed = gripper?.speed_m_s;
  const metadata = {
    minimum: position?.min,
    maximum: position?.max,
    physicalMinimum: physical?.min,
    physicalMaximum: physical?.max,
    defaultSpeed: speed?.default,
    maxSpeed: speed?.max,
  };
  if (baseInfo?.protocol_version !== "3.0"
      || baseInfo?.control_mode !== "local"
      || baseInfo?.side !== expected?.side
      || baseInfo?.arm?.server?.port !== expected?.port) throw new Error("Gripper arm identity does not match this page.");
  if (!Array.isArray(baseInfo?.supported_commands)
      || !["hello", "move_gripper", "get_state", "get_base_info"].every((cmd) => baseInfo.supported_commands.includes(cmd))) {
    throw new Error("Base command metadata does not support gripper control.");
  }
  if (!isPiper) throw new Error("Gripper backend or units are incompatible.");
  if (gripper?.enabled !== true) throw new Error("Gripper is disabled.");
  if (!Number.isFinite(metadata.minimum) || !Number.isFinite(metadata.maximum)
      || metadata.minimum > metadata.maximum
      || !Number.isFinite(metadata.physicalMinimum) || !Number.isFinite(metadata.physicalMaximum)
      || metadata.physicalMinimum > metadata.minimum || metadata.physicalMaximum < metadata.maximum
      || !Number.isFinite(metadata.defaultSpeed)
      || !Number.isFinite(metadata.maxSpeed) || metadata.defaultSpeed <= 0
      || metadata.maxSpeed <= 0 || metadata.defaultSpeed > metadata.maxSpeed) {
    throw new Error("Gripper limits are invalid.");
  }
  return Object.freeze(metadata);
}

function validateTelemetry(snapshot, now, expected, metadata) {
  if (!snapshot || !Number.isFinite(snapshot.receivedAtMs)
      || now < snapshot.receivedAtMs || now - snapshot.receivedAtMs > TELEMETRY_STALE_MS) {
    throw new Error("Robot feedback is stale or unavailable.");
  }
  const payload = snapshot.payload;
  if (payload?.side !== expected.side) throw new Error("Gripper arm identity does not match this page.");
  if (payload?.control_mode !== "local") throw new Error("Teaching requires local control mode.");
  if (payload?.state !== "RUNNING") throw new Error(`Service is ${payload?.state ?? "unavailable"}; RUNNING is required.`);
  if (payload?.emergency_stopped !== false) throw new Error("Emergency stop is active.");
  if (payload?.control_ready !== true || payload?.control_block_reason !== null) {
    if (payload?.control_ready === false && typeof payload?.control_block_reason === "string"
        && payload.control_block_reason.length > 0) throw new Error(payload.control_block_reason);
    throw new Error("Control readiness is unavailable or invalid.");
  }
  if (!Number.isFinite(payload?.ActualGripper)
      || payload.ActualGripper < metadata.physicalMinimum
      || payload.ActualGripper > metadata.physicalMaximum) {
    throw new Error("Robot gripper feedback is malformed or unsafe.");
  }
}

function validateTarget(snapshot, metadata) {
  const position = snapshot?.targetPosition ?? snapshot?.targetPositionM;
  const speed = snapshot?.speed ?? snapshot?.speedMps;
  if (!Number.isFinite(position) || position < metadata.minimum || position > metadata.maximum) {
    throw new Error("Gripper target is outside configured limits.");
  }
  if (!Number.isFinite(speed) || speed <= 0 || speed > metadata.maxSpeed) {
    throw new Error("Gripper speed is outside configured limits.");
  }
  return { position, speed, editRevision: readEditRevision(snapshot) };
}

function readEditRevision(snapshot) {
  if (!Number.isSafeInteger(snapshot?.editRevision) || snapshot.editRevision < 0) {
    throw new Error("Gripper edit state is invalid.");
  }
  return snapshot.editRevision;
}

export function createGripperControlSession(options) {
  const clock = options.clock || defaultClock();
  const eventTarget = options.eventTarget || window;
  const documentTarget = options.documentTarget || document;
  const now = clock.now || (() => Date.now());
  let state = "read-only";
  let client = null;
  let timer = null;
  let pending = false;
  let generation = 0;
  let disposed = false;
  let activationEditRevision = null;
  let userEdited = false;

  const notify = (message = null) => options.onStateChange?.(state, message);
  const finishClose = (closingClient, reason) => {
    const closed = closingClient ? closingClient.whenClosed() : Promise.resolve();
    return closed.then(() => {
      if (disposed || client !== closingClient || state !== "stopping") return;
      client = null;
      state = "read-only";
      notify(reason);
    });
  };
  const stop = (reason = "Gripper control released.") => {
    if (state === "read-only" || state === "stopping" || state === "disposed") return;
    generation += 1;
    state = "stopping";
    notify(reason);
    if (timer !== null) clock.clearInterval(timer);
    timer = null;
    pending = false;
    const closingClient = client;
    closingClient?.abort(reason);
    void finishClose(closingClient, reason);
  };

  const sendLatest = async () => {
    if (state !== "active" || pending) return;
    let target;
    try {
      validateTelemetry(options.getTelemetrySnapshot?.(), now(), options.expectedIdentity, metadata);
      target = validateTarget(options.getTargetSnapshot?.(), metadata);
      if (!userEdited) {
        if (target.editRevision === activationEditRevision) return;
        userEdited = true;
      }
    } catch (error) {
      stop(error.message);
      return;
    }
    pending = true;
    const sendingClient = client;
    const token = generation;
    try {
      await sendingClient.request("move_gripper", {
        target_position: target.position,
        speed: target.speed,
      }, options.requestTimeoutMs);
    } catch (error) {
      if (client === sendingClient && generation === token) stop(error.message);
    } finally {
      if (client === sendingClient && generation === token) pending = false;
    }
  };

  let metadata = null;
  const stopEvents = new Map([
    ["pointerup", () => stop("Gripper pointer released.")],
    ["pointercancel", () => stop("Gripper pointer cancelled.")],
    ["keyup", () => stop("Gripper key released.")],
    ["blur", () => stop("Page focus lost.")],
    ["pagehide", () => stop("Page disposed.")],
    ["emergency-stop", () => stop("Emergency stop requested.")],
    ["teaching-mode-change", () => stop("Teaching mode changed.")],
  ]);
  for (const [name, handler] of stopEvents) eventTarget.addEventListener(name, handler);
  const visibilityHandler = () => { if (documentTarget.hidden) stop("Page is hidden."); };
  documentTarget.addEventListener("visibilitychange", visibilityHandler);

  return {
    get state() { return state; },
    async start() {
      if (disposed) throw new Error("Gripper control session is disposed.");
      if (state !== "read-only") throw new Error(`Cannot acquire gripper control while ${state}.`);
      const token = ++generation;
      activationEditRevision = readEditRevision(options.getTargetSnapshot?.());
      userEdited = false;
      state = "acquiring";
      notify();
      try {
        if (options.beforeAcquire) await options.beforeAcquire();
        if (token !== generation || state !== "acquiring") throw abortError("Gripper control released before acquisition.");
        client = new PiperWebSocketClient({
          socketFactory: options.socketFactory,
          location: options.location,
          clock,
          openTimeoutMs: options.requestTimeoutMs,
        });
        await client.open();
        if (token !== generation || state !== "acquiring") throw abortError("Gripper acquisition cancelled.");
        metadata = readMetadata(await client.request("get_base_info", {}, options.requestTimeoutMs), options.expectedIdentity);
        if (token !== generation || state !== "acquiring") throw abortError("Gripper acquisition cancelled.");
        const initialState = await client.request("get_state", {}, options.requestTimeoutMs);
        if (token !== generation || state !== "acquiring") throw abortError("Gripper acquisition cancelled.");
        validateTelemetry({ payload: initialState, receivedAtMs: now() }, now(), options.expectedIdentity, metadata);
        validateTarget(options.getTargetSnapshot?.(), metadata);
        state = "active";
        notify();
        timer = clock.setInterval(sendLatest, SEND_PERIOD_MS);
      } catch (error) {
        if (token === generation && state === "acquiring") {
          const closingClient = client;
          closingClient?.abort(error.message);
          await (closingClient ? closingClient.whenClosed() : Promise.resolve());
          client = null;
          state = "read-only";
          notify(error.message);
        }
        throw error;
      }
    },
    stop,
    dispose() {
      if (disposed) return;
      stop("Gripper control disposed.");
      disposed = true;
      generation += 1;
      for (const [name, handler] of stopEvents) eventTarget.removeEventListener(name, handler);
      documentTarget.removeEventListener("visibilitychange", visibilityHandler);
      state = "disposed";
    },
  };
}
