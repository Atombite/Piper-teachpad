// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import URDFLoader from "urdf-loader";

import { setRobotJoints } from "./teach-kinematics.js";

function paintRobot(robot, color, opacity) {
  const replacedMaterials = new Set();
  robot.traverse((object) => {
    if (!object.isMesh) return;
    const source = Array.isArray(object.material) ? object.material : [object.material];
    const materials = source.map((material) => {
      const clone = material.clone();
      replacedMaterials.add(material);
      clone.color.set(color);
      clone.userData.teachColor = color;
      clone.transparent = opacity < 1;
      clone.opacity = opacity;
      clone.depthWrite = opacity >= 1;
      return clone;
    });
    object.material = Array.isArray(object.material) ? materials : materials[0];
  });
  replacedMaterials.forEach((material) => material?.dispose?.());
}

function disposeObject(root, disposedResources = new WeakSet()) {
  if (!root) return;
  root.traverse((object) => {
    if (object.geometry && !disposedResources.has(object.geometry)) {
      disposedResources.add(object.geometry);
      object.geometry.dispose?.();
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!material || disposedResources.has(material)) continue;
      for (const value of Object.values(material)) {
        if (value?.isTexture && !disposedResources.has(value)) {
          disposedResources.add(value);
          value.dispose();
        }
      }
      disposedResources.add(material);
      material.dispose?.();
    }
  });
}

function loadRobot(modelUrl, onParsed) {
  return new Promise((resolve, reject) => {
    let parsedRobot = null;
    let failure = null;
    const manager = new THREE.LoadingManager();
    manager.onError = (url) => {
      failure ||= new Error(`Robot model asset failed to load: ${url}`);
    };
    manager.onLoad = () => {
      if (failure) reject(failure);
      else if (parsedRobot) resolve(parsedRobot);
      else reject(new Error(`Robot model failed to load: ${modelUrl}`));
    };
    const loader = new URDFLoader(manager);
    loader.load(
      modelUrl,
      (robot) => {
        parsedRobot = robot;
        onParsed(robot);
      },
      undefined,
      (error) => {
        failure ||= new Error(`Robot model failed to load: ${error?.target?.responseURL || modelUrl}`);
      },
    );
  });
}

function setGripper(robot, position, minimum, maximum) {
  if (!Number.isFinite(position) || !Number.isFinite(minimum)
      || !Number.isFinite(maximum) || maximum <= minimum) return;
  const fraction = Math.max(0, Math.min(1, (position - minimum) / (maximum - minimum)));
  const half = fraction * 0.035;
  robot.joints?.joint7?.setJointValue(half);
  robot.updateMatrixWorld(true);
}

export function createTeachingViewer(container, modelUrl, options = {}) {
  let gripperMinimum = 0;
  let gripperMaximum = 0.07;
  const configuredMinimum = options.gripperConfig?.physicalMinPosition;
  const configuredMaximum = options.gripperConfig?.physicalMaxPosition;
  if (Number.isFinite(configuredMinimum) && Number.isFinite(configuredMaximum)
      && configuredMaximum > configuredMinimum) {
    gripperMinimum = configuredMinimum;
    gripperMaximum = configuredMaximum;
  }
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 20);
  camera.position.set(0.85, -0.85, 0.65);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.replaceChildren(renderer.domElement);

  scene.add(new THREE.HemisphereLight(0xd8f7ff, 0x172126, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.8);
  key.position.set(1, -1, 2);
  scene.add(key);
  const grid = new THREE.GridHelper(1.5, 30, 0x426671, 0x253b43);
  grid.rotation.x = Math.PI / 2;
  const axes = new THREE.AxesHelper(0.22);
  scene.add(grid, axes);

  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 0, 0.28);
  orbit.update();
  const requestedTool = new THREE.Object3D();
  scene.add(requestedTool);
  const transform = new TransformControls(camera, renderer.domElement);
  transform.setSize(0.75);
  transform.attach(requestedTool);
  transform.addEventListener("dragging-changed", (event) => { orbit.enabled = !event.value; });
  transform.addEventListener("objectChange", () => {
    requestedTool.updateMatrixWorld(true);
    container.dispatchEvent(new CustomEvent("teach-requested-tcp", {
      detail: requestedTool.matrixWorld.clone(),
    }));
  });
  scene.add(transform.getHelper());

  let disposed = false;
  let frame = 0;
  let actualRobot = null;
  let targetRobot = null;
  let targetValid = true;

  function setGizmoValidity(valid) {
    transform.getHelper().traverse((object) => {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        if (!material?.color) continue;
        if (material.userData.teachValidColor === undefined) {
          material.userData.teachValidColor = material.color.getHex();
        }
        material.color.setHex(valid ? material.userData.teachValidColor : 0xffb020);
      }
    });
  }

  function applyTargetValidity() {
    container.dataset.targetValidity = targetValid ? "valid" : "invalid";
    targetRobot?.traverse((object) => {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        if (!material?.color) continue;
        material.color.setHex(targetValid ? (material.userData.teachColor ?? 0xe45c52) : 0xffb020);
      }
    });
    setGizmoValidity(targetValid);
  }

  function resize() {
    const width = Math.max(1, container.clientWidth || 640);
    const height = Math.max(1, container.clientHeight || 480);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  }
  function render() {
    if (disposed) return;
    resize();
    renderer.render(scene, camera);
    frame = requestAnimationFrame(render);
  }
  render();

  const disposedResources = new WeakSet();
  const slots = {
    actual: { robot: null, settled: false, released: false, color: 0x43ddf2, opacity: 0.38 },
    target: { robot: null, settled: false, released: false, color: 0xe45c52, opacity: 1 },
  };
  let loadFailed = false;

  function releaseSlot(slot) {
    if (!slot.settled || slot.released || !slot.robot) return;
    disposeObject(slot.robot, disposedResources);
    slot.released = true;
  }

  function beginSlot(name) {
    const slot = slots[name];
    const injectedLoad = options.loadRobot;
    const operation = injectedLoad
      ? injectedLoad(name, modelUrl, (robot) => { slot.robot = robot; })
      : loadRobot(modelUrl, (robot) => { slot.robot = robot; });
    return Promise.resolve(operation).then(
      (robot) => {
        slot.robot = robot || slot.robot;
        slot.settled = true;
        if (disposed || loadFailed) {
          releaseSlot(slot);
          return slot.robot;
        }
        paintRobot(slot.robot, slot.color, slot.opacity);
        return slot.robot;
      },
      (error) => {
        slot.settled = true;
        loadFailed = true;
        releaseSlot(slot);
        Object.values(slots).forEach(releaseSlot);
        throw error;
      },
    );
  }

  const actualPromise = beginSlot("actual");
  const targetPromise = beginSlot("target");
  let viewer;
  const ready = Promise.all([actualPromise, targetPromise]).then(([actual, target]) => {
      actualRobot = actual;
      targetRobot = target;
      viewer.actualRobot = actualRobot;
      viewer.targetRobot = targetRobot;
      if (disposed || loadFailed) return viewer;
      scene.add(actualRobot, targetRobot);
      applyTargetValidity();
      return viewer;
    });
  ready.catch(() => {});

  viewer = {
    ready,
    actualRobot: null,
    targetRobot: null,
    setActual(jointsDeg, gripperM) {
      if (!actualRobot) return;
      setRobotJoints(actualRobot, jointsDeg);
      setGripper(actualRobot, gripperM, gripperMinimum, gripperMaximum);
    },
    setTarget(jointsDeg, gripperM) {
      if (!targetRobot) return;
      setRobotJoints(targetRobot, jointsDeg);
      setGripper(targetRobot, gripperM, gripperMinimum, gripperMaximum);
    },
    setGripperRange(config) {
      const minimum = config?.physicalMinPosition;
      const maximum = config?.physicalMaxPosition;
      if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || maximum <= minimum) return;
      gripperMinimum = minimum;
      gripperMaximum = maximum;
    },
    setRequestedTcp(pose) {
      if (!(pose instanceof THREE.Matrix4) || !pose.elements.every(Number.isFinite)) return;
      pose.decompose(requestedTool.position, requestedTool.quaternion, requestedTool.scale);
      requestedTool.updateMatrixWorld(true);
    },
    setMode(mode) {
      transform.setMode(mode === "rotate" ? "rotate" : "translate");
    },
    setTargetValidity(valid) {
      targetValid = valid === true;
      applyTargetValidity();
    },
    resetCamera() {
      camera.position.set(0.85, -0.85, 0.65);
      orbit.target.set(0, 0, 0.28);
      orbit.update();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      cancelAnimationFrame(frame);
      Object.values(slots).forEach(releaseSlot);
      disposeObject(grid, disposedResources);
      disposeObject(axes, disposedResources);
      transform.dispose();
      orbit.dispose();
      renderer.dispose();
      container.replaceChildren();
    },
  };
  return viewer;
}
