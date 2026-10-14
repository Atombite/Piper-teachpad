// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import * as esbuild from "esbuild";


await esbuild.build({
  stdin: {
    contents: [
      'import "./tests/browser/teach-kinematics.test.js";',
      'import "./tests/browser/teach-state.test.js";',
      'import "./tests/browser/teach-viewer.test.js";',
      'import "./tests/browser/teach-network.test.js";',
      'import "./tests/browser/teach-control.test.js";',
      'import "./tests/browser/teach-gripper.test.js";',
    ].join("\n"),
    resolveDir: process.cwd(),
    sourcefile: "teach-tests-entry.js",
  },
  bundle: true,
  format: "iife",
  platform: "browser",
  target: ["es2022"],
  outfile: "tests/browser/teach-tests.bundle.js",
});
