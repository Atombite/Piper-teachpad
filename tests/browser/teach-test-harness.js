// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

const tests = [];
let scheduled = false;

export function test(name, operation) {
  tests.push({ name, operation });
}

export function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export function assertEqual(actual, expected, message) {
  assert(Object.is(actual, expected), `${message}: expected ${expected}, got ${actual}`);
}

export function assertDeepEqual(actual, expected, message) {
  assert(
    JSON.stringify(actual) === JSON.stringify(expected),
    `${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`,
  );
}

export function scheduleTests() {
  if (scheduled) return;
  scheduled = true;
  queueMicrotask(async () => {
    const output = document.querySelector("#results");
    const failures = [];
    for (const { name, operation } of tests) {
      try {
        await operation();
        output.textContent += `\nPASS ${name}`;
      } catch (error) {
        const detail = error?.stack || String(error);
        failures.push(`${name}\n${detail}`);
        output.textContent += `\nFAIL ${name}\n${detail}`;
      }
    }
    document.body.dataset.status = failures.length === 0 ? "passed" : "failed";
    if (failures.length > 0) document.body.dataset.failures = String(failures.length);
  });
}
