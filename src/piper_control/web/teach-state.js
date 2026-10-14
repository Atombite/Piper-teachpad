// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

const JOINT_COUNT = 6;
export const TELEMETRY_STALE_MS = 1000;

function finiteSix(values) {
  return Array.isArray(values) && values.length === JOINT_COUNT && values.every(Number.isFinite);
}

function readLimits(baseInfo) {
  const minimum = baseInfo?.arm?.joint_limits_deg?.min;
  const maximum = baseInfo?.arm?.joint_limits_deg?.max;
  if (!finiteSix(minimum) || !finiteSix(maximum)
      || minimum.some((value, index) => value > maximum[index])) return null;
  return minimum.map((value, index) => [value, maximum[index]]);
}

function validateSpeeds(baseInfo) {
  const defaults = baseInfo?.arm?.joint_speed_deg_s?.default;
  const maxima = baseInfo?.arm?.joint_speed_deg_s?.max;
  return finiteSix(defaults) && finiteSix(maxima)
    && defaults.every((value, index) => value > 0 && value <= maxima[index]);
}

function readGripperConfig(baseInfo) {
  const gripper = baseInfo?.arm?.gripper;
  const isPiper = gripper?.backend === "piper"
    && baseInfo?.units?.gripper_position === "meter"
    && baseInfo?.units?.gripper_speed === "meter/second";
  const position = gripper?.position_m;
  const physical = gripper?.physical_position_m || position;
  const speed = gripper?.speed_m_s;
  const minimum = position?.min;
  const maximum = position?.max;
  const physicalMinimum = physical?.min;
  const physicalMaximum = physical?.max;
  const defaultSpeed = speed?.default;
  const maxSpeed = speed?.max;
  if (!isPiper
      || !Array.isArray(baseInfo?.supported_commands)
      || !baseInfo.supported_commands.includes("move_gripper")
      || typeof gripper?.enabled !== "boolean"
      || !Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum
      || !Number.isFinite(physicalMinimum) || !Number.isFinite(physicalMaximum)
      || physicalMinimum > minimum || physicalMaximum < maximum
      || !Number.isFinite(defaultSpeed) || !Number.isFinite(maxSpeed)
      || defaultSpeed <= 0 || maxSpeed <= 0 || defaultSpeed > maxSpeed) return null;
  return Object.freeze({
    enabled: gripper.enabled,
    backend: "piper",
    positionUnit: "meter",
    speedUnit: "meter/second",
    minPosition: minimum,
    maxPosition: maximum,
    physicalMinPosition: physicalMinimum,
    physicalMaxPosition: physicalMaximum,
    defaultSpeed,
    maxSpeed,
    displayScale: 1000,
    positionLabel: "mm",
    speedLabel: "mm/s",
    displayDecimals: 1,
    minPositionM: minimum,
    maxPositionM: maximum,
    defaultSpeedMps: defaultSpeed,
    maxSpeedMps: maxSpeed,
  });
}

function validateBaseInfo(baseInfo, expected) {
  if (baseInfo?.protocol_version !== "3.0") return "Base protocol must be exactly 3.0.";
  if (baseInfo?.control_mode !== "local") return "Teaching requires local control mode.";
  if (!["left", "right"].includes(baseInfo?.side)) return "Base arm side is invalid.";
  if (expected?.side && baseInfo.side !== expected.side) return "Base arm side does not match this page.";
  const port = baseInfo?.arm?.server?.port;
  if (!Number.isInteger(port) || port < 1 || port > 65535) return "Base server port is invalid.";
  if (Number.isInteger(expected?.port) && port !== expected.port) return "Base server port does not match this page.";
  const commands = baseInfo?.supported_commands;
  if (!Array.isArray(commands)
      || !["hello", "servoJ", "get_state", "get_base_info"].every((command) => commands.includes(command))) {
    return "Base command metadata does not support safe joint execution.";
  }
  if (baseInfo?.units?.joint_position !== "degree"
      || baseInfo?.units?.joint_speed !== "degree/second") {
    return "Base joint unit metadata is incompatible.";
  }
  if (!validateSpeeds(baseInfo)) return "Base joint speed metadata is invalid.";
  if (!readLimits(baseInfo)) return "Base information has invalid joint limits.";
  return null;
}

function canonicalControlBlockReason(payload) {
  if (payload?.control_mode === "remote") return "Remote control mode is active; teaching requires local mode.";
  if (payload?.control_mode !== "local") return "Control mode is unavailable or invalid.";
  if (payload?.state === "HOLD") return "Service is HOLD; switch mode or resume RUNNING before teaching.";
  if (payload?.state !== "RUNNING") return `Service is ${payload?.state ?? "unavailable"}; RUNNING is required.`;
  if (payload?.emergency_stopped !== false) return "Emergency stop is active.";
  if (payload?.control_ready === true && payload?.control_block_reason === null) return null;
  if (payload?.control_ready === false
      && typeof payload?.control_block_reason === "string"
      && payload.control_block_reason.length > 0) return payload.control_block_reason;
  return "Control readiness is unavailable or invalid.";
}

function targetLimitError(jointsDeg, limitsDeg) {
  if (!finiteSix(jointsDeg)) return "Target must contain six finite joint angles.";
  if (!Array.isArray(limitsDeg) || limitsDeg.length !== JOINT_COUNT) return "Joint limits are unavailable.";
  const index = jointsDeg.findIndex((value, joint) => (
    value < limitsDeg[joint][0] || value > limitsDeg[joint][1]
  ));
  return index < 0
    ? null
    : `J${index + 1} target is outside ${limitsDeg[index][0]}°…${limitsDeg[index][1]}°.`;
}

function currentGripperTarget(state) {
  if (!Number.isFinite(state.actualGripperM)) return state.targetGripperM;
  if (!state.gripperConfig) return state.actualGripperM;
  return Math.max(
    state.gripperConfig.minPosition,
    Math.min(state.gripperConfig.maxPosition, state.actualGripperM),
  );
}

export function createTeachingState(baseInfo, expected = {}) {
  const baseInfoError = validateBaseInfo(baseInfo, expected);
  const gripperConfig = readGripperConfig(baseInfo);
  return {
    side: typeof baseInfo?.side === "string" ? baseInfo.side : null,
    protocolVersion: baseInfo?.protocol_version ?? null,
    controlMode: baseInfo?.control_mode ?? null,
    limitsDeg: readLimits(baseInfo),
    baseInfoError,
    actualJointsDeg: null,
    actualGripperM: null,
    targetGripperM: null,
    gripperTargetEdited: false,
    gripperTargetRevision: 0,
    gripperConfig,
    targetJointsDeg: null,
    targetEdited: false,
    requestedTcp: null,
    tcpRequestRevision: 0,
    ik: { ok: true, pending: false, reason: null, validatedRevision: 0 },
    serviceState: null,
    emergencyStopped: false,
    telemetryValid: false,
    telemetryReceivedAtMs: null,
    telemetryError: "Waiting for valid feedback.",
    telemetryFailureCount: 0,
    controllerOccupied: false,
    modelError: null,
    controlState: "read-only",
    controlError: null,
  };
}

export function reduceTeachingState(state, event) {
  switch (event?.type) {
    case "telemetry": {
      const payload = event.payload;
      const joints = payload?.ActualQ;
      if (!finiteSix(joints)) {
        return { ...state, telemetryValid: false, telemetryError: "Feedback must contain six finite joint angles." };
      }
      const telemetryError = canonicalControlBlockReason(payload);
      const readinessValid = telemetryError === null;
      const nextActual = [...joints];
      return {
        ...state,
        actualJointsDeg: nextActual,
        actualGripperM: Number.isFinite(payload.ActualGripper) ? payload.ActualGripper : state.actualGripperM,
        targetGripperM: state.gripperTargetEdited
          ? state.targetGripperM
          : currentGripperTarget({
            ...state,
            actualGripperM: Number.isFinite(payload.ActualGripper)
              ? payload.ActualGripper
              : state.actualGripperM,
          }),
        targetJointsDeg: state.targetEdited ? state.targetJointsDeg : nextActual,
        serviceState: payload.state ?? null,
        controlMode: payload.control_mode ?? null,
        emergencyStopped: payload.emergency_stopped !== false,
        controllerOccupied: payload.ws_controller_connected === true,
        telemetryValid: readinessValid,
        telemetryReceivedAtMs: Number.isFinite(event.receivedAtMs) ? event.receivedAtMs : Date.now(),
        telemetryError,
        telemetryFailureCount: 0,
      };
    }
    case "gripper-target":
      return {
        ...state,
        targetGripperM: event.positionM,
        gripperTargetEdited: true,
        gripperTargetRevision: state.gripperTargetRevision + 1,
      };
    case "gripper-control-released":
      return {
        ...state,
        targetGripperM: currentGripperTarget(state),
        gripperTargetEdited: false,
      };
    case "telemetry-error":
      return {
        ...state,
        telemetryValid: false,
        telemetryError: event.message || "Telemetry request failed.",
        telemetryFailureCount: state.telemetryFailureCount + 1,
      };
    case "target-joints":
      return {
        ...state,
        targetJointsDeg: Array.isArray(event.jointsDeg) ? [...event.jointsDeg] : event.jointsDeg,
        targetEdited: true,
        requestedTcp: null,
        ik: { ok: true, pending: false, reason: null, validatedRevision: state.tcpRequestRevision },
        controlError: null,
      };
    case "requested-tcp": {
      const revision = state.tcpRequestRevision + 1;
      return {
        ...state,
        requestedTcp: event.pose ?? null,
        tcpRequestRevision: revision,
        targetEdited: true,
        ik: { ok: false, pending: true, reason: "IK validation pending.", validatedRevision: null },
      };
    }
    case "ik-result": {
      if (event.requestRevision !== state.tcpRequestRevision) return state;
      const joints = event.result?.jointsDeg;
      const valid = event.result?.ok === true
        && finiteSix(joints)
        && targetLimitError(joints, state.limitsDeg) === null;
      return {
        ...state,
        ik: {
          ok: valid,
          pending: false,
          reason: valid ? null : (event.result?.reason || "IK returned an invalid joint solution."),
          positionErrorM: event.result?.positionErrorM ?? Number.POSITIVE_INFINITY,
          orientationErrorRad: event.result?.orientationErrorRad ?? Number.POSITIVE_INFINITY,
          validatedRevision: valid ? event.requestRevision : null,
        },
        targetJointsDeg: valid ? [...joints] : state.targetJointsDeg,
        targetEdited: true,
      };
    }
    case "reset-target": {
      const revision = state.tcpRequestRevision + 1;
      return {
        ...state,
        targetJointsDeg: state.actualJointsDeg ? [...state.actualJointsDeg] : null,
        targetEdited: false,
        requestedTcp: null,
        tcpRequestRevision: revision,
        ik: { ok: true, pending: false, reason: null, validatedRevision: revision },
        controlError: null,
      };
    }
    case "model-error":
      return { ...state, modelError: event.message || "Robot model failed to load." };
    case "model-ready":
      return state;
    case "control-state":
      return { ...state, controlState: event.controlState, controlError: event.message ?? null };
    case "control-error":
      return { ...state, controlState: "faulted", controlError: event.message || "Control request failed." };
    case "control-finished":
      return { ...state, controlState: "read-only", controlError: null };
    default:
      return state;
  }
}

export function gripperBlockReason(state, nowMs = Date.now(), { ownsControl = false } = {}) {
  const config = state.gripperConfig;
  if (!config) return "Gripper metadata or units are unavailable or invalid.";
  if (!config.enabled) return "Gripper is disabled in the arm configuration.";
  if (!state.telemetryValid) return state.telemetryError || "Robot feedback is unavailable.";
  if (!Number.isFinite(state.telemetryReceivedAtMs)
      || nowMs < state.telemetryReceivedAtMs
      || nowMs - state.telemetryReceivedAtMs > TELEMETRY_STALE_MS) return "Robot feedback is stale.";
  if (state.controllerOccupied && !ownsControl) return "Another controller owns the arm.";
  if (!Number.isFinite(state.actualGripperM)) return "Gripper feedback is unavailable.";
  if (!Number.isFinite(state.targetGripperM)
      || state.targetGripperM < config.minPosition
      || state.targetGripperM > config.maxPosition) {
    return `Gripper target is outside ${config.minPosition}…${config.maxPosition} ${config.positionLabel}.`;
  }
  return null;
}

export function resolveSynchronousIk(state, pose, solve) {
  const pending = reduceTeachingState(state, { type: "requested-tcp", pose });
  let result;
  try {
    result = solve();
  } catch (error) {
    result = {
      ok: false,
      reason: `IK solver failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
  return reduceTeachingState(pending, {
    type: "ik-result",
    requestRevision: pending.tcpRequestRevision,
    result,
  });
}

export function motionBlockReason(state, nowMs = Date.now()) {
  if (state.modelError) return `Robot model error: ${state.modelError}`;
  if (state.baseInfoError || !state.limitsDeg) return state.baseInfoError || "Base information is unavailable.";
  if (!state.telemetryValid) return state.telemetryError || "Robot feedback is unavailable.";
  if (!Number.isFinite(state.telemetryReceivedAtMs)
      || nowMs - state.telemetryReceivedAtMs > TELEMETRY_STALE_MS) return "Robot feedback is stale.";
  if (state.controllerOccupied && !["acquiring", "executing", "live-armed", "live-active"].includes(state.controlState)) {
    return "Another controller owns the arm.";
  }
  const limitError = targetLimitError(state.targetJointsDeg, state.limitsDeg);
  if (limitError) return limitError;
  if (!state.ik?.ok || state.ik.pending
      || (state.requestedTcp !== null && state.ik.validatedRevision !== state.tcpRequestRevision)) {
    return `IK error: ${state.ik?.reason || "target changed since validation"}`;
  }
  if (!["read-only", "preview", "live-armed"].includes(state.controlState)) {
    return state.controlError || "A control request is in progress.";
  }
  if (state.controlError) return state.controlError;
  return null;
}

export function safetyStatusMessage(state, nowMs = Date.now()) {
  const reason = motionBlockReason(state, nowMs);
  if (state.telemetryFailureCount > 0 && state.serviceState) {
    if (state.telemetryFailureCount === 1) {
      return `${state.serviceState} · Feedback delayed; motion paused.`;
    }
    return `${state.serviceState} (last confirmed) · Communication unavailable after ${state.telemetryFailureCount} requests: ${state.telemetryError}`;
  }
  return reason || "Target is valid for motion.";
}
