// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

(() => {
  const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

  async function waitFor(predicate, description, timeoutMs = 30000) {
    const deadline = performance.now() + timeoutMs;
    let lastError = null;
    while (performance.now() < deadline) {
      try {
        const value = await predicate();
        if (value) return value;
      } catch (error) {
        lastError = error;
      }
      await sleep(50);
    }
    throw new Error(`Timed out waiting for ${description}${lastError ? `: ${lastError.message}` : ""}`);
  }

  function dispatchInput(input, value, eventType = "input") {
    input.value = String(value);
    input.dispatchEvent(new Event(eventType, { bubbles: true }));
  }

  window.__teachAcceptance = {
    async waitReady(expectedSide) {
      await waitFor(
        () => document.querySelectorAll("#joint-controls .joint-row").length === 6,
        "six joint controls",
      );
      await waitFor(
        () => document.querySelector("#viewer-loading") === null,
        "URDF model",
      );
      await waitFor(
        () => document.querySelector("#connection-status")?.textContent === "Feedback online",
        "valid robot feedback",
      );
      const side = document.querySelector("#arm-side")?.textContent;
      if (side !== `${expectedSide} ARM` && side !== expectedSide) {
        throw new Error(`Expected ${expectedSide} identity, got ${side}`);
      }
      await waitFor(
        () => !document.querySelector("#reset-target")?.disabled,
        "initial target reset",
      );
      return {
        side,
        actual: [...document.querySelectorAll(".joint-row [data-role=actual]")].map((node) => node.textContent),
        target: [...document.querySelectorAll(".joint-row [data-role=number]")].map((node) => Number(node.value)),
      };
    },

    setJointTarget(jointsDeg) {
      const inputs = [...document.querySelectorAll(".joint-row [data-role=number]")];
      if (inputs.length !== 6 || jointsDeg.length !== 6) throw new Error("Expected six joint inputs.");
      inputs.forEach((input, index) => dispatchInput(input, jointsDeg[index]));
      return inputs.map((input) => Number(input.value));
    },

    selectMode(mode) {
      const select = document.querySelector("#teaching-mode");
      dispatchInput(select, mode, "change");
    },

    click(selector) {
      const element = document.querySelector(selector);
      if (!element || element.disabled) throw new Error(`${selector} is unavailable.`);
      element.click();
    },

    async waitForAvailable(selector) {
      await waitFor(
        () => {
          const element = document.querySelector(selector);
          return element && !element.disabled;
        },
        `${selector} to become available`,
      );
    },

    async waitForReviewedCompletion() {
      await waitFor(
        () => document.querySelector("#command-status")?.textContent.includes("target reached"),
        "reviewed execution completion",
        40000,
      );
      return document.querySelector("#command-status").textContent;
    },

    async waitForLiveArmed() {
      await waitFor(
        () => document.querySelector("#command-status")?.textContent.includes("Control acquired"),
        "live control acquisition",
      );
      await waitFor(
        () => !document.querySelector("#live-deadman")?.disabled,
        "enabled deadman",
      );
    },

    holdDeadman() {
      const button = document.querySelector("#live-deadman");
      button.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, pointerId: 1, isPrimary: true }));
    },

    releaseDeadman() {
      window.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, pointerId: 1, isPrimary: true }));
    },

    async waitForGripperAvailable() {
      await waitFor(
        () => !document.querySelector("#gripper-position")?.disabled,
        "gripper control to become available",
      );
    },

    async dragGripper(positionMm, speedMmS) {
      const position = document.querySelector("#gripper-position");
      const speed = document.querySelector("#gripper-speed-number");
      dispatchInput(speed, speedMmS);
      position.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, pointerId: 2, isPrimary: true }));
      dispatchInput(position, positionMm);
      await waitFor(
        () => document.querySelector("#gripper-status")?.textContent.includes("LIVE"),
        "active gripper control",
      );
    },

    releaseGripper() {
      window.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, pointerId: 2, isPrimary: true }));
    },

    async waitForLeaseReleased() {
      return waitFor(async () => {
        const state = await this.fetchState();
        return state.ws_controller_connected === false ? state : null;
      }, "released controller lease");
    },

    moveTcpXBy(deltaMm) {
      const input = document.querySelector('[data-tcp="x"]');
      dispatchInput(input, Number(input.value) + deltaMm, "change");
      return [...document.querySelectorAll(".joint-row [data-role=number]")]
        .map((jointInput) => Number(jointInput.value));
    },

    async fetchState() {
      return (await fetch("/api/state")).json();
    },

    async waitForEmergencyStop() {
      return waitFor(async () => {
        const state = await this.fetchState();
        return state.emergency_stopped === true && state.ws_controller_connected === false
          ? state : null;
      }, "emergency stop and released controller lease");
    },
  };
})();
