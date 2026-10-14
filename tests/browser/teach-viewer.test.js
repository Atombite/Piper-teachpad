// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";

import {
  assert,
  assertEqual,
  scheduleTests,
  test,
} from "./teach-test-harness.js";
import { createTeachingViewer } from "../../src/piper_control/web/teach-viewer.js";
import {
  createTeachingState,
  motionBlockReason,
  reduceTeachingState,
} from "../../src/piper_control/web/teach-state.js";

const BASE_INFO = {
  protocol_version: "3.0",
  control_mode: "local",
  side: "left",
  supported_commands: ["hello", "servoJ", "get_state", "get_base_info"],
  units: { joint_position: "degree", joint_speed: "degree/second" },
  arm: {
    server: { port: 8765 },
    joint_limits_deg: { min: [-150, 0, -154, -100, -69, -120], max: [150, 179, 0, 100, 69, 120] },
    joint_speed_deg_s: { default: [10, 10, 10, 10, 10, 10], max: [30, 30, 30, 30, 30, 30] },
  },
};

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function disposableRobot(counters) {
  const robot = new THREE.Group();
  robot.addDisposableMesh = () => {
    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshBasicMaterial();
    const texture = new THREE.Texture();
    material.map = texture;
    const geometryDispose = geometry.dispose.bind(geometry);
    geometry.dispose = () => { counters.geometry += 1; geometryDispose(); };
    const materialDispose = material.dispose.bind(material);
    material.dispose = () => { counters.material += 1; materialDispose(); };
    const textureDispose = texture.dispose.bind(texture);
    texture.dispose = () => { counters.texture += 1; textureDispose(); };
    robot.add(new THREE.Mesh(geometry, material));
  };
  return robot;
}

test("viewer renders independent actual and target poses", async () => {
  const container = document.createElement("div");
  container.style.width = "640px";
  container.style.height = "480px";
  document.body.append(container);
  const viewer = createTeachingViewer(
    container,
    "/static/teach/models/piper/piper.urdf",
    {
      gripperConfig: {
        backend: "piper",
        minPosition: 0.01,
        maxPosition: 0.05,
        physicalMinPosition: 0,
        physicalMaxPosition: 0.07,
      },
    },
  );
  await viewer.ready;
  viewer.setActual([0, 90, -77, 0, 0, 0], 0.02);
  viewer.setTarget([5, 80, -70, 1, 2, 3], 0.03);
  assertEqual(viewer.actualRobot.joints.joint1.jointValue[0], 0, "actual joint did not update");
  assert(Math.abs(viewer.targetRobot.joints.joint1.jointValue[0] - 5 * Math.PI / 180) < 1e-10, "target joint did not update");
  viewer.setActual([0, 90, -77, 0, 0, 0], 0.035);
  assert(Math.abs(viewer.actualRobot.joints.joint7.jointValue[0] - 0.0175) < 1e-10,
    "Piper opening used command bounds instead of physical metre bounds");
  viewer.setTarget([5, 80, -70, 1, 2, 3], 0.07);
  assertEqual(viewer.targetRobot.joints.joint7.jointValue[0], 0.035, "full physical opening was not rendered");
  viewer.setGripperRange({ physicalMinPosition: 0, physicalMaxPosition: 0.08 });
  viewer.setActual([0, 90, -77, 0, 0, 0], 0.04);
  assert(Math.abs(viewer.actualRobot.joints.joint7.jointValue[0] - 0.0175) < 1e-10,
    "updated Piper physical bounds were not rendered");
  viewer.setMode("rotate");
  viewer.setTargetValidity(false);
  assertEqual(container.dataset.targetValidity, "invalid", "invalid target was not styled");
  viewer.setTargetValidity(true);
  assertEqual(container.dataset.targetValidity, "valid", "valid target styling did not clear");

  let geometryDisposals = 0;
  let materialDisposals = 0;
  viewer.actualRobot.traverse((object) => {
    if (!object.isMesh) return;
    if (object.geometry && !object.geometry.userData.disposeSpied) {
      object.geometry.userData.disposeSpied = true;
      const dispose = object.geometry.dispose.bind(object.geometry);
      object.geometry.dispose = () => { geometryDisposals += 1; dispose(); };
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!material || material.userData.disposeSpied) continue;
      material.userData.disposeSpied = true;
      const dispose = material.dispose.bind(material);
      material.dispose = () => { materialDisposals += 1; dispose(); };
    }
  });
  viewer.dispose();
  assert(geometryDisposals > 0, "viewer did not release URDF geometries");
  assert(materialDisposals > 0, "viewer did not release URDF materials");
  container.remove();
});

test("missing mesh names the exact asset and reducer fails closed", async () => {
  const container = document.createElement("div");
  container.style.width = "320px";
  container.style.height = "240px";
  document.body.append(container);
  const viewer = createTeachingViewer(container, "/fixture/missing-mesh.urdf");
  let message = "";
  try {
    await viewer.ready;
  } catch (error) {
    message = error instanceof Error ? error.message : String(error);
  }
  assert(/link4\.stl/.test(message), `missing asset not identified: ${message}`);
  let state = createTeachingState(BASE_INFO);
  state = reduceTeachingState(state, { type: "model-error", message });
  assert(/model/i.test(motionBlockReason(state)), "model rejection did not fail closed");
  assert(/link4\.stl/.test(motionBlockReason(state)), "block reason lost exact asset");
  viewer.dispose();
  container.remove();
});

test("dispose before ready harvests late resources exactly once", async () => {
  const container = document.createElement("div");
  document.body.append(container);
  const controls = [deferred(), deferred()];
  const counters = [
    { geometry: 0, material: 0, texture: 0 },
    { geometry: 0, material: 0, texture: 0 },
  ];
  const robots = counters.map(disposableRobot);
  let call = 0;
  const viewer = createTeachingViewer(container, "/unused.urdf", {
    loadRobot(_slotName, _url, onParsed) {
      const index = call++;
      onParsed(robots[index]);
      return controls[index].promise;
    },
  });

  viewer.dispose();
  robots.forEach((robot) => robot.addDisposableMesh());
  controls.forEach((control, index) => control.resolve(robots[index]));
  await viewer.ready;
  viewer.dispose();

  counters.forEach((count, index) => {
    assertEqual(count.geometry, 1, `slot ${index} geometry disposal count`);
    assertEqual(count.material, 1, `slot ${index} material disposal count`);
    assertEqual(count.texture, 1, `slot ${index} texture disposal count`);
  });
  container.remove();
});

test("one failed slot still harvests a sibling delayed success without unhandled rejection", async () => {
  const container = document.createElement("div");
  document.body.append(container);
  const controls = [deferred(), deferred()];
  const counters = [
    { geometry: 0, material: 0, texture: 0 },
    { geometry: 0, material: 0, texture: 0 },
  ];
  const robots = counters.map(disposableRobot);
  let call = 0;
  let unhandled = 0;
  const onUnhandled = (event) => { unhandled += 1; event.preventDefault(); };
  window.addEventListener("unhandledrejection", onUnhandled);
  const viewer = createTeachingViewer(container, "/unused.urdf", {
    loadRobot(_slotName, _url, onParsed) {
      const index = call++;
      onParsed(robots[index]);
      return controls[index].promise;
    },
  });

  robots[0].addDisposableMesh();
  controls[0].reject(new Error("actual failed"));
  let rejected = false;
  try {
    await viewer.ready;
  } catch (error) {
    rejected = /actual failed/.test(error.message);
  }
  robots[1].addDisposableMesh();
  controls[1].resolve(robots[1]);
  await Promise.resolve();
  await new Promise((resolve) => window.setTimeout(resolve, 0));

  assert(rejected, "viewer.ready did not preserve the first slot failure");
  counters.forEach((count, index) => {
    assertEqual(count.geometry, 1, `slot ${index} leaked or double-disposed geometry`);
    assertEqual(count.material, 1, `slot ${index} leaked or double-disposed material`);
    assertEqual(count.texture, 1, `slot ${index} leaked or double-disposed texture`);
  });
  assertEqual(unhandled, 0, "viewer emitted an unhandled rejection");
  viewer.dispose();
  window.removeEventListener("unhandledrejection", onUnhandled);
  container.remove();
});

scheduleTests();
