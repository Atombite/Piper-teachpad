// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

const JOINT_COUNT = 6;
const OPEN_TIMEOUT_MS = 1000;
const REQUEST_TIMEOUT_MS = 1000;
const LIVE_PERIOD_MS = 50;
const TELEMETRY_STALE_MS = 1000;

function defaultClock() {
  return {
    setTimeout: window.setTimeout.bind(window),
    clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window),
    clearInterval: window.clearInterval.bind(window),
  };
}

function controlUrl(locationValue) {
  const protocol = locationValue.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${locationValue.host}/ws/control`;
}

function finiteSix(values) {
  return Array.isArray(values) && values.length === JOINT_COUNT && values.every(Number.isFinite);
}

function frozenSix(values, label) {
  if (!finiteSix(values)) throw new Error(`${label} must contain exactly six finite values.`);
  return Object.freeze([...values]);
}

function protocolError(payload, fallback = "Control request failed.") {
  const error = new Error(payload?.error?.message || fallback);
  error.code = payload?.error?.code || "PROTOCOL_ERROR";
  return error;
}

function abortError(reason = "Control operation aborted.") {
  const error = new Error(reason instanceof Error ? reason.message : String(reason));
  error.name = "AbortError";
  error.code = "ABORTED";
  return error;
}

function throwIfAborted(signal) {
  if (signal?.aborted) throw abortError(signal.reason || "Control operation aborted.");
}

function validateExpectedIdentity(identity) {
  if (!identity || !["left", "right"].includes(identity.side)
      || !Number.isInteger(identity.port) || identity.port < 1 || identity.port > 65535) {
    throw new Error("Expected arm identity must include a valid side and port.");
  }
  return Object.freeze({ side: identity.side, port: identity.port });
}

export function readPageExpectedIdentity(documentValue, locationValue) {
  const side = documentValue?.body?.dataset?.armSide;
  const rawPort = documentValue?.body?.dataset?.serverPort;
  if (!/^[1-9][0-9]*$/.test(rawPort || "")) {
    throw new Error("Page server port identity is missing or invalid.");
  }
  const port = Number(rawPort);
  const identity = validateExpectedIdentity({ side, port });
  const originPort = locationValue?.port
    ? Number(locationValue.port)
    : locationValue?.protocol === "https:" ? 443
      : locationValue?.protocol === "http:" ? 80 : Number.NaN;
  if (!Number.isInteger(originPort) || originPort !== identity.port) {
    throw new Error("Page server port identity does not match the current origin.");
  }
  return identity;
}

function readControlMetadata(baseInfo, expectedIdentityValue) {
  const expectedIdentity = validateExpectedIdentity(expectedIdentityValue);
  if (baseInfo?.protocol_version !== "3.0") throw new Error("Base protocol must be exactly 3.0.");
  if (baseInfo?.control_mode !== "local") throw new Error("Teaching requires local control mode.");
  if (typeof baseInfo?.side !== "string" || baseInfo.side !== expectedIdentity.side) {
    throw new Error("Fresh base arm side does not match the expected page identity.");
  }
  const serverPort = baseInfo?.arm?.server?.port;
  if (!Number.isInteger(serverPort) || serverPort !== expectedIdentity.port) {
    throw new Error("Fresh base server port does not match the expected page identity.");
  }
  if (!Array.isArray(baseInfo?.supported_commands)
      || !["hello", "servoJ", "get_state", "get_base_info"].every((cmd) => baseInfo.supported_commands.includes(cmd))) {
    throw new Error("Base command metadata does not support safe joint execution.");
  }
  if (baseInfo?.units?.joint_position !== "degree" || baseInfo?.units?.joint_speed !== "degree/second") {
    throw new Error("Base joint units are incompatible.");
  }
  const minimum = baseInfo?.arm?.joint_limits_deg?.min;
  const maximum = baseInfo?.arm?.joint_limits_deg?.max;
  const defaultSpeed = baseInfo?.arm?.joint_speed_deg_s?.default;
  const maxSpeed = baseInfo?.arm?.joint_speed_deg_s?.max;
  if (!finiteSix(minimum) || !finiteSix(maximum)
      || minimum.some((value, index) => value > maximum[index])) {
    throw new Error("Base information has invalid joint limits.");
  }
  if (!finiteSix(defaultSpeed) || !finiteSix(maxSpeed)
      || defaultSpeed.some((value, index) => value <= 0 || value > maxSpeed[index])
      || maxSpeed.some((value) => value <= 0)) {
    throw new Error("Base information has invalid speed limits.");
  }
  return Object.freeze({
    identity: expectedIdentity,
    limitsDeg: Object.freeze(minimum.map((value, index) => Object.freeze([value, maximum[index]]))),
    maxSpeedDegS: Object.freeze([...maxSpeed]),
  });
}

function requireControlReadiness(state) {
  if (state?.control_ready === true && state?.control_block_reason === null) return;
  if (state?.control_ready === false && typeof state?.control_block_reason === "string"
      && state.control_block_reason.length > 0) throw new Error(state.control_block_reason);
  throw new Error("Control readiness is unavailable or invalid.");
}

function validateMotion(targetJointsDeg, speedDegS, metadata, state) {
  const target = frozenSix(targetJointsDeg, "Target");
  const speed = frozenSix(speedDegS, "Speed");
  const limitIndex = target.findIndex((value, index) => (
    value < metadata.limitsDeg[index][0] || value > metadata.limitsDeg[index][1]
  ));
  if (limitIndex >= 0) throw new Error(`J${limitIndex + 1} target is outside configured limits.`);
  const speedIndex = speed.findIndex((value, index) => value <= 0 || value > metadata.maxSpeedDegS[index]);
  if (speedIndex >= 0) throw new Error(`J${speedIndex + 1} speed is outside configured limits.`);
  if (typeof state?.side !== "string" || state.side !== metadata.identity.side) {
    throw new Error("Fresh state arm side does not match the expected page identity.");
  }
  requireControlReadiness(state);
  if (!finiteSix(state?.ActualQ)) throw new Error("Robot feedback is stale or unavailable.");
  return { target, speed };
}

function validateTelemetrySnapshot(snapshot, nowMs, staleMs, expectedIdentity) {
  if (!snapshot || !Number.isFinite(snapshot.receivedAtMs)
      || !Number.isFinite(nowMs) || nowMs < snapshot.receivedAtMs
      || nowMs - snapshot.receivedAtMs > staleMs) {
    throw new Error("Robot feedback is stale or unavailable.");
  }
  const payload = snapshot.payload;
  if (payload?.side !== expectedIdentity.side) {
    throw new Error("Fresh state arm side does not match the expected page identity.");
  }
  requireControlReadiness(payload);
  if (!finiteSix(payload?.ActualQ)) {
    throw new Error("Robot feedback is stale, malformed, or unsafe.");
  }
  return payload;
}

function withinTolerance(actual, target, toleranceDeg = 2) {
  return finiteSix(actual) && actual.every((value, index) => Math.abs(value - target[index]) <= toleranceDeg);
}

function reviewedTimeoutMs(actual, target, speed) {
  const travelMs = Math.max(...target.map((value, index) => (
    Math.abs(value - actual[index]) / speed[index]
  ))) * 1000;
  return Math.max(30000, Math.ceil(travelMs) + 5000);
}

export function controlAvailability({ mode, blockReason, lifecycle }) {
  const clear = !blockReason;
  return {
    acquire: clear && mode === "live" && lifecycle === "read-only",
    execute: clear && mode === "reviewed" && lifecycle === "read-only",
    deadman: clear && mode === "live" && lifecycle === "live-armed",
  };
}

export function failClosedOnTelemetryFailure(controls, reason) {
  controls.reviewedAbortController?.abort(reason);
  controls.gripperSession?.stop(reason);
  if (controls.liveFollower
      && ["live-armed", "live-active"].includes(controls.liveFollower.state)) {
    controls.liveFollower.deactivate(reason);
  }
}

export function createControlLifecycle(options) {
  const eventTarget = options.eventTarget || window;
  const documentTarget = options.documentTarget || document;
  let disposed = false;
  const stop = (reason) => {
    if (!disposed) options.stop(reason);
  };
  const handlers = new Map([
    ["emergency-stop", () => stop("Emergency stop requested.")],
    ["teaching-mode-change", () => stop("Teaching mode changed.")],
    ["blur", () => stop("Page focus lost.")],
    ["pagehide", () => stop("Page disposed.")],
  ]);
  for (const [name, handler] of handlers) eventTarget.addEventListener(name, handler);
  const visibilityHandler = () => {
    if (documentTarget.hidden) stop("Page is hidden.");
  };
  documentTarget.addEventListener("visibilitychange", visibilityHandler);
  return {
    dispose({ stop: stopFirst = true } = {}) {
      if (disposed) return;
      if (stopFirst) options.stop("Control lifecycle disposed.");
      disposed = true;
      for (const [name, handler] of handlers) eventTarget.removeEventListener(name, handler);
      documentTarget.removeEventListener("visibilitychange", visibilityHandler);
    },
  };
}

export class PiperWebSocketClient {
  constructor(options = {}) {
    const locationValue = options.location || window.location;
    this.url = options.url || controlUrl(locationValue);
    this.socketFactory = options.socketFactory || ((url) => new WebSocket(url));
    this.clock = options.clock || defaultClock();
    this.openTimeoutMs = options.openTimeoutMs ?? OPEN_TIMEOUT_MS;
    this.socket = null;
    this.pending = null;
    this.nextId = 1;
    this.openPromise = null;
    this.openReject = null;
    this.openCleanup = null;
    this.terminalListeners = new Set();
    this.closeListeners = new Set();
    this.closed = false;
    this.explicitClose = false;
  }

  open() {
    if (this.openPromise) return this.openPromise;
    if (this.socket?.readyState === 1) return Promise.resolve(this);
    this.explicitClose = false;
    const socket = this.socketFactory(this.url);
    this.socket = socket;
    this.closed = false;
    socket.addEventListener("message", (event) => this.#handleMessage(event));
    socket.addEventListener("error", () => this.#handleTerminal("connection loss", new Error("WebSocket connection error.")));
    socket.addEventListener("close", (event) => {
      this.closed = true;
      for (const listener of this.closeListeners) listener(event);
      this.closeListeners.clear();
      const conflict = event.code === 4409
        ? Object.assign(new Error(event.reason || "controller already connected"), { code: "CLIENT_CONFLICT" })
        : new Error(event.reason || "WebSocket connection closed.");
      this.#handleTerminal(event.code === 4409 ? "controller conflict" : "connection loss", conflict);
    });
    this.openPromise = new Promise((resolve, reject) => {
      this.openReject = reject;
      const timeout = this.clock.setTimeout(() => {
        if (socket.readyState !== 1) {
          this.abort(`WebSocket open timed out after ${this.openTimeoutMs} ms.`);
        }
      }, this.openTimeoutMs);
      const opened = () => {
        this.openCleanup?.();
        this.request("hello", { identity: "local" }, this.openTimeoutMs)
          .then((data) => {
            if (data?.identity !== "local") {
              throw new Error("Hello response did not confirm local identity.");
            }
            this.openReject = null;
            resolve(this);
          })
          .catch((error) => {
            this.#rejectOpen(error);
            this.explicitClose = true;
            if (socket.readyState < 2) socket.close(4003, "identity handshake failed");
          });
      };
      socket.addEventListener("open", opened);
      this.openCleanup = () => {
        this.clock.clearTimeout(timeout);
        socket.removeEventListener("open", opened);
        this.openCleanup = null;
      };
    });
    return this.openPromise;
  }

  request(cmd, fields = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    if (this.socket?.readyState !== 1) return Promise.reject(new Error("WebSocket is not open."));
    if (this.pending) return Promise.reject(new Error("A control request is already in flight."));
    const id = `teach-${this.nextId++}`;
    return new Promise((resolve, reject) => {
      const timeout = this.clock.setTimeout(() => {
        if (this.pending?.id !== id) return;
        this.pending = null;
        reject(Object.assign(new Error(`${cmd} request timed out after ${timeoutMs} ms.`), { code: "TIMEOUT" }));
        this.close(4000, "request timeout");
      }, timeoutMs);
      this.pending = { id, cmd, resolve, reject, timeout };
      try {
        this.socket.send(JSON.stringify({ id, cmd, ...fields }));
      } catch (error) {
        this.clock.clearTimeout(timeout);
        this.pending = null;
        reject(error);
        if (!this.openReject) this.close(4001, "send failed");
      }
    });
  }

  onTerminal(listener) {
    this.terminalListeners.add(listener);
    return () => this.terminalListeners.delete(listener);
  }

  whenClosed() {
    if (this.closed || !this.socket || this.socket.readyState === 3) return Promise.resolve();
    return new Promise((resolve) => this.closeListeners.add(resolve));
  }

  close(code = 1000, reason = "teaching control complete") {
    this.explicitClose = true;
    this.#rejectOpen(new Error(reason));
    this.#failPending(new Error(reason));
    const socket = this.socket;
    if (socket && socket.readyState < 2) socket.close(code, reason);
  }

  abort(reason = "Control operation aborted.") {
    const error = abortError(reason);
    this.explicitClose = true;
    this.#rejectOpen(error);
    this.#failPending(error);
    const socket = this.socket;
    if (socket && socket.readyState < 2) socket.close(4000, error.message);
  }

  #handleMessage(event) {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (_error) {
      this.#failPending(new Error("WebSocket returned malformed JSON."));
      this.close(4002, "malformed response");
      return;
    }
    if (payload?.ok === false && payload?.error?.code === "CLIENT_CONFLICT") {
      this.#failPending(protocolError(payload, "controller already connected"));
      return;
    }
    const pending = this.pending;
    if (!pending) return;
    if (payload?.id !== pending.id || payload?.cmd !== pending.cmd) {
      this.#failPending(Object.assign(new Error("WebSocket response correlation mismatch."), { code: "CORRELATION_ERROR" }));
      this.close(4002, "response correlation mismatch");
      return;
    }
    this.clock.clearTimeout(pending.timeout);
    this.pending = null;
    if (payload.ok !== true) pending.reject(protocolError(payload));
    else pending.resolve(payload.data);
  }

  #failPending(error) {
    if (!this.pending) return;
    this.clock.clearTimeout(this.pending.timeout);
    const { reject } = this.pending;
    this.pending = null;
    reject(error);
  }

  #handleTerminal(reason, error) {
    this.#rejectOpen(error);
    this.#failPending(error);
    if (!this.explicitClose) {
      for (const listener of this.terminalListeners) listener(reason, error);
    }
  }


  #rejectOpen(error) {
    if (!this.openReject) return;
    const reject = this.openReject;
    this.openReject = null;
    this.openCleanup?.();
    reject(error);
  }
}

export async function executeReviewedTarget(options) {
  const target = frozenSix(options.targetJointsDeg, "Target");
  const speed = frozenSix(options.speedDegS, "Speed");
  const expectedIdentity = validateExpectedIdentity(options.expectedIdentity);
  const clock = options.clock || defaultClock();
  const client = new PiperWebSocketClient({
    socketFactory: options.socketFactory,
    location: options.location,
    clock,
    openTimeoutMs: options.requestTimeoutMs,
  });
  const now = options.now || clock.now || (() => Date.now());
  const pollIntervalMs = options.pollIntervalMs ?? 100;
  const signal = options.signal;
  const abortHandler = () => client.abort(signal.reason || "Reviewed execution aborted.");
  signal?.addEventListener("abort", abortHandler, { once: true });
  try {
    throwIfAborted(signal);
    await client.open();
    throwIfAborted(signal);
    const baseInfo = await client.request("get_base_info", {}, options.requestTimeoutMs);
    throwIfAborted(signal);
    const metadata = readControlMetadata(baseInfo, expectedIdentity);
    let state = await client.request("get_state", {}, options.requestTimeoutMs);
    throwIfAborted(signal);
    const validated = validateMotion(target, speed, metadata, state);
    const timeoutMs = options.timeoutMs ?? reviewedTimeoutMs(
      state.ActualQ, validated.target, validated.speed,
    );
    const startedAt = now();
    await client.request("servoJ", {
      target_joints: validated.target,
      speed: validated.speed,
    }, options.requestTimeoutMs);
    throwIfAborted(signal);
    for (;;) {
      if (withinTolerance(state.ActualQ, validated.target, options.toleranceDeg ?? 2)) {
        return { targetJointsDeg: validated.target, actualJointsDeg: Object.freeze([...state.ActualQ]) };
      }
      if (now() - startedAt >= timeoutMs) throw new Error(`Reviewed execution timed out after ${timeoutMs} ms.`);
      await abortableDelay(clock, pollIntervalMs, signal);
      throwIfAborted(signal);
      state = await client.request("get_state", {}, options.requestTimeoutMs);
      throwIfAborted(signal);
      validateMotion(validated.target, validated.speed, metadata, state);
    }
  } finally {
    signal?.removeEventListener("abort", abortHandler);
    client.close();
    await client.whenClosed();
  }
}

function abortableDelay(clock, delayMs, signal) {
  throwIfAborted(signal);
  return new Promise((resolve, reject) => {
    const timer = clock.setTimeout(() => {
      signal?.removeEventListener("abort", aborted);
      resolve();
    }, delayMs);
    const aborted = () => {
      clock.clearTimeout(timer);
      signal.removeEventListener("abort", aborted);
      reject(abortError(signal.reason || "Control operation aborted."));
    };
    signal?.addEventListener("abort", aborted, { once: true });
  });
}

export function createLiveFollower(options) {
  const clock = options.clock || defaultClock();
  const eventTarget = options.eventTarget || window;
  const documentTarget = options.documentTarget || document;
  let state = "read-only";
  let client = null;
  let metadata = null;
  let timer = null;
  let requestPending = false;
  let disposed = false;
  let removeTerminal = null;
  let generation = 0;
  const now = options.now || clock.now || (() => Date.now());
  const telemetryStaleMs = options.telemetryStaleMs ?? TELEMETRY_STALE_MS;
  const expectedIdentity = validateExpectedIdentity(options.expectedIdentity);

  const notify = (message = null) => options.onStateChange?.(state, message);

  const stop = (reason = "Live following stopped.") => {
    if (state === "stopping" || state === "disposed") return;
    generation += 1;
    state = "stopping";
    notify(reason);
    if (timer !== null) clock.clearInterval(timer);
    timer = null;
    const stoppingClient = client;
    stoppingClient?.abort(reason);
    const closed = stoppingClient ? stoppingClient.whenClosed() : Promise.resolve();
    closed.then(() => {
      if (disposed || state !== "stopping" || client !== stoppingClient) return;
      options.onLeaseReleased?.(reason);
    });
  };

  const pointerHandlers = new Map([
    ["pointerup", () => stop("Deadman released.")],
    ["pointercancel", () => stop("Deadman cancelled.")],
    ["keyup", () => stop("Deadman key released.")],
  ]);
  for (const [name, handler] of pointerHandlers) eventTarget.addEventListener(name, handler);
  const lifecycle = createControlLifecycle({ eventTarget, documentTarget, stop });

  const sendLatest = async () => {
    if (state !== "live-active") return;
    let validated;
    try {
      const telemetry = validateTelemetrySnapshot(
        options.getTelemetrySnapshot?.(), now(), telemetryStaleMs, expectedIdentity,
      );
      if (requestPending) return;
      const target = frozenSix(options.getTargetSnapshot(), "Target");
      validated = validateMotion(target, options.speedDegS, metadata, telemetry);
    } catch (error) {
      stop(error.message);
      return;
    }
    requestPending = true;
    try {
      await client.request("servoJ", { target_joints: validated.target, speed: validated.speed }, options.requestTimeoutMs);
    } catch (error) {
      stop(error.message);
    } finally {
      requestPending = false;
    }
  };

  const follower = {
    get state() { return state; },
    async acquire() {
      if (disposed) throw new Error("Live follower is disposed.");
      if (state !== "read-only") throw new Error(`Cannot acquire control while ${state}.`);
      state = "acquiring";
      const token = ++generation;
      notify();
      client = new PiperWebSocketClient({
        socketFactory: options.socketFactory,
        location: options.location,
        clock,
        openTimeoutMs: options.requestTimeoutMs,
      });
      removeTerminal = client.onTerminal((reason, error) => {
        if (state === "acquiring" && error?.code === "CLIENT_CONFLICT") return;
        stop(error?.message || reason);
      });
      const ensureCurrent = () => {
        if (token !== generation || state !== "acquiring" || disposed) {
          throw abortError(state === "disposed" ? "Live follower disposed." : "Live acquisition aborted.");
        }
      };
      try {
        await client.open();
        ensureCurrent();
        const baseInfo = await client.request("get_base_info", {}, options.requestTimeoutMs);
        ensureCurrent();
        metadata = readControlMetadata(baseInfo, expectedIdentity);
        const initialState = await client.request("get_state", {}, options.requestTimeoutMs);
        ensureCurrent();
        validateMotion(options.getTargetSnapshot(), options.speedDegS, metadata, initialState);
        ensureCurrent();
        state = "live-armed";
        notify();
      } catch (error) {
        if (token !== generation || state === "stopping" || state === "disposed") {
          await (client ? client.whenClosed() : Promise.resolve());
          throw error;
        }
        removeTerminal?.();
        removeTerminal = null;
        const closingClient = client;
        closingClient.close();
        await closingClient.whenClosed();
        client = null;
        metadata = null;
        state = "read-only";
        notify(error.message);
        throw error;
      }
    },
    async activate() {
      if (state !== "live-armed") throw new Error("Live control must be acquired before activation.");
      state = "live-active";
      notify();
      timer = clock.setInterval(sendLatest, LIVE_PERIOD_MS);
    },
    deactivate: stop,
    dispose() {
      if (disposed) return;
      stop("Live follower disposed.");
      disposed = true;
      generation += 1;
      removeTerminal?.();
      removeTerminal = null;
      for (const [name, handler] of pointerHandlers) eventTarget.removeEventListener(name, handler);
      lifecycle.dispose({ stop: false });
      state = "disposed";
    },
  };
  return follower;
}

export function createLiveControlSession(options) {
  let follower = null;
  let disposed = false;
  let releaseWaiters = [];

  const resolveReleased = () => {
    const waiters = releaseWaiters;
    releaseWaiters = [];
    for (const resolve of waiters) resolve();
  };

  const notify = (state, message) => options.onStateChange?.(state, message);
  const ensureFollower = () => {
    if (disposed) throw new Error("Live control session is disposed.");
    if (follower) return follower;
    let created = null;
    created = createLiveFollower({
      ...options,
      onLeaseReleased: (message) => {
        if (follower !== created) return;
        created.dispose();
        follower = null;
        notify("read-only", message || "Live following stopped; control lease released.");
        resolveReleased();
      },
      onStateChange: (state, message) => {
        notify(state, message);
      },
    });
    follower = created;
    return created;
  };

  return {
    get state() { return follower?.state || (disposed ? "disposed" : "read-only"); },
    acquire() { return ensureFollower().acquire(); },
    activate() {
      if (!follower) return Promise.reject(new Error("Live control must be acquired before activation."));
      return follower.activate();
    },
    deactivate(reason) { follower?.deactivate(reason); },
    stopAndWait(reason) {
      if (!follower || ["read-only", "disposed"].includes(follower.state)) return Promise.resolve();
      const released = new Promise((resolve) => releaseWaiters.push(resolve));
      follower.deactivate(reason);
      return released;
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      follower?.dispose();
      follower = null;
      resolveReleased();
    },
  };
}
