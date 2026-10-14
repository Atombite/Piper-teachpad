// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import { assert, scheduleTests, test } from "./teach-test-harness.js";
import { fetchJsonWithTimeout } from "../../src/piper_control/web/teach-network.js";

test("hung telemetry fetch aborts and rejects within its explicit timeout", async () => {
  let aborted = false;
  const hungFetch = (_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener("abort", () => {
      aborted = true;
      reject(new DOMException("Aborted", "AbortError"));
    });
  });
  let message = "";
  try {
    await fetchJsonWithTimeout(hungFetch, "/api/state", { timeoutMs: 5 });
  } catch (error) {
    message = error.message;
  }
  assert(aborted, "hung fetch was not aborted");
  assert(/timed out/i.test(message), `timeout error was unclear: ${message}`);
});

scheduleTests();
