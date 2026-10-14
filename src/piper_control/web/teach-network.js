// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

export async function fetchJsonWithTimeout(fetchImpl, url, options = {}) {
  const { timeoutMs = 1000, ...fetchOptions } = options;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchImpl(url, {
      cache: "no-store",
      ...fetchOptions,
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error?.message || `${response.status} ${response.statusText}`);
    return payload;
  } catch (error) {
    if (controller.signal.aborted) throw new Error(`Request timed out after ${timeoutMs} ms.`);
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}
