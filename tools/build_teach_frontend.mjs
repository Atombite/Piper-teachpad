// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import { mkdir, writeFile } from "node:fs/promises";

import * as esbuild from "esbuild";


await esbuild.build({
  entryPoints: ["src/piper_control/web/teach-main.js"],
  bundle: true,
  format: "iife",
  platform: "browser",
  target: ["es2022"],
  legalComments: "none",
  minifyWhitespace: true,
  outfile: "src/piper_control/web/teach.bundle.js",
});

const notices = `Piper Teaching Console third-party notices

three 0.186.0
License: MIT
Upstream: https://github.com/mrdoob/three.js
Full text: THREE-LICENSE.txt

urdf-loader 0.13.1
License: Apache-2.0
Upstream: https://github.com/gkjohnson/urdf-loaders
Full text and copyright notice: URDF-LOADER-LICENSE.txt
Note: the installed npm package omits its repository-level LICENSE file; its
package metadata declares Apache-2.0 and its packaged README supplies the
Caltech/JPL copyright notice reproduced with the full Apache-2.0 text.

AgileX Piper model assets from MuJoCo Menagerie
License: MIT
Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper
Full text and copyright notice: PIPER-MODEL-LICENSE.txt
`;

await mkdir("src/piper_control/web/vendor", { recursive: true });
await writeFile(
  "src/piper_control/web/vendor/THIRD_PARTY_NOTICES.txt",
  notices,
  "utf8",
);
