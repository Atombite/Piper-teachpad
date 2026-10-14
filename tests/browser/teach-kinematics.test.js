// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";
import URDFLoader from "urdf-loader";

import {
  readToolPose,
  setRobotJoints,
  solveIk,
} from "../../src/piper_control/web/teach-kinematics.js";
import {
  assert,
  assertEqual,
  scheduleTests,
  test,
} from "./teach-test-harness.js";


const LIMITS = [
  [-150, 150],
  [0, 179],
  [-154, 0],
  [-100, 100],
  [-69, 69],
  [-120, 120],
];

function degToRad(value) {
  return value * Math.PI / 180;
}

function loadFixtureRobot() {
  return new Promise((resolve, reject) => {
    const loader = new URDFLoader();
    loader.load(
      "/static/teach/models/piper/piper.urdf",
      resolve,
      undefined,
      reject,
    );
  });
}

function assertFiniteMatrix(matrix, message) {
  assert(
    matrix.elements.every(Number.isFinite),
    `${message}: matrix contains a non-finite element`,
  );
}

function robotJointsDeg(robot) {
  return Array.from({ length: 6 }, (_, index) => (
    THREE.MathUtils.radToDeg(robot.joints[`joint${index + 1}`].jointValue[0])
  ));
}

function independentPoseError(actual, requested) {
  const actualPosition = new THREE.Vector3().setFromMatrixPosition(actual);
  const requestedPosition = new THREE.Vector3().setFromMatrixPosition(requested);
  const actualRotation = new THREE.Quaternion().setFromRotationMatrix(actual);
  const requestedRotation = new THREE.Quaternion().setFromRotationMatrix(requested);
  const quaternionDot = Math.min(1, Math.abs(actualRotation.dot(requestedRotation)));
  return {
    position: actualPosition.distanceTo(requestedPosition),
    orientation: 2 * Math.acos(quaternionDot),
  };
}

function makeLinearFixtureRobot() {
  const jointValues = new Array(6).fill(0);
  const tool = { matrixWorld: new THREE.Matrix4() };
  const robot = {
    joints: {},
    links: { tool0: tool },
    updateMatrixWorld() {
      tool.matrixWorld.makeTranslation(jointValues[0], 0, 0);
    },
  };
  jointValues.forEach((_, index) => {
    robot.joints[`joint${index + 1}`] = {
      jointValue: [0],
      setJointValue(value) {
        jointValues[index] = value;
        this.jointValue[0] = value;
      },
    };
  });
  return robot;
}

test("joint setters drive the six-joint chain and expose a finite tool pose", async () => {
  const robot = await loadFixtureRobot();
  const joints = [10, 70, -90, 5, 10, -15];

  setRobotJoints(robot, joints);
  const pose = readToolPose(robot);

  joints.forEach((value, index) => {
    const actual = robot.joints[`joint${index + 1}`].jointValue[0];
    assert(Math.abs(actual - degToRad(value)) < 1e-10, `joint${index + 1} was not set`);
  });
  assert(pose instanceof THREE.Matrix4, "tool pose must be a THREE.Matrix4");
  assertFiniteMatrix(pose, "tool pose");
});

test("IK converges continuously inside logical limits", async () => {
  const robot = await loadFixtureRobot();
  const seed = [10, 70, -90, 5, 10, -15];
  setRobotJoints(robot, seed);
  const requested = readToolPose(robot).clone();
  requested.elements[12] += 0.015;

  const result = solveIk(robot, requested, seed, LIMITS);

  assertEqual(result.ok, true, `IK should converge (${result.reason})`);
  assert(result.positionErrorM <= 0.001, "position error exceeds 1 mm");
  assert(result.orientationErrorRad <= degToRad(1), "orientation error exceeds 1 degree");
  result.jointsDeg.forEach((value, index) => {
    assert(value >= LIMITS[index][0] && value <= LIMITS[index][1], `joint${index + 1} left its limit`);
    assert(Math.abs(value - seed[index]) <= 20, `joint${index + 1} made a discontinuous jump`);
  });
});

test("IK reaches a real combined translation and rotation target", async () => {
  const robot = await loadFixtureRobot();
  const seed = [10, 70, -90, 5, 10, -15];
  const targetJoints = [12, 69, -89, 7, 11, -13];
  setRobotJoints(robot, targetJoints);
  const requested = readToolPose(robot).clone();
  setRobotJoints(robot, seed);

  const result = solveIk(robot, requested, seed, LIMITS);
  setRobotJoints(robot, result.jointsDeg);
  const independentlyMeasured = independentPoseError(readToolPose(robot), requested);

  assertEqual(result.ok, true, `combined-pose IK should converge (${result.reason})`);
  assert(independentlyMeasured.position <= 0.001, "independent position error exceeds 1 mm");
  assert(
    independentlyMeasured.orientation <= degToRad(1),
    "independent orientation error exceeds 1 degree",
  );
});

test("IK rejects unreachable and non-finite requests", async () => {
  const robot = await loadFixtureRobot();
  const seed = [0, 90, -77.263, 0, 0, 0];
  setRobotJoints(robot, seed);
  const farAway = readToolPose(robot).clone();
  farAway.elements[12] += 10;
  const nonFinite = readToolPose(robot).clone();
  nonFinite.elements[14] = Number.NaN;

  const unreachable = solveIk(robot, farAway, seed, LIMITS);
  const invalid = solveIk(robot, nonFinite, seed, LIMITS);

  assertEqual(unreachable.ok, false, "far-away target must fail");
  assert(
    ["unreachable", "joint limit"].includes(unreachable.reason),
    `far-away failure reason: got ${unreachable.reason}`,
  );
  assertEqual(invalid.ok, false, "non-finite target must fail");
  assertEqual(invalid.reason, "non-finite target", "non-finite failure reason");
});

test("IK rejects requests whose updates remain pinned at a logical joint limit", async () => {
  const robot = await loadFixtureRobot();
  const seed = [0, 90, -77.263, 0, 0, 0];
  setRobotJoints(robot, seed);
  const requested = readToolPose(robot).clone();
  requested.elements[12] += 0.015;
  const pinnedLimits = seed.map((value) => [value, value]);

  const result = solveIk(robot, requested, seed, pinnedLimits);

  assertEqual(result.ok, false, "limit-pinned target must fail");
  assertEqual(result.reason, "joint limit", "limit-pinned failure reason");
});

test("IK failure reports the robot's returned joints and their measured pose error", async () => {
  const robot = await loadFixtureRobot();
  const seed = [0, 90, -77.263, 0, 0, 0];
  setRobotJoints(robot, seed);
  const requested = readToolPose(robot).clone();
  requested.elements[12] += 10;

  const result = solveIk(robot, requested, seed, LIMITS);
  const actualJoints = robotJointsDeg(robot);
  const independentlyMeasured = independentPoseError(readToolPose(robot), requested);

  assertEqual(result.ok, false, "far target must fail");
  result.jointsDeg.forEach((value, index) => {
    assert(Math.abs(value - actualJoints[index]) < 1e-9, `joint${index + 1} result/state mismatch`);
  });
  assert(
    Math.abs(result.positionErrorM - independentlyMeasured.position) < 1e-9,
    "failure position error does not describe returned joints",
  );
  assert(
    Math.abs(result.orientationErrorRad - independentlyMeasured.orientation) < 1e-9,
    "failure orientation error does not describe returned joints",
  );
});

test("IK accepts convergence produced by the eightieth bounded update", () => {
  const robot = makeLinearFixtureRobot();
  const requested = new THREE.Matrix4().makeTranslation(degToRad(240), 0, 0);
  const limits = [[-300, 300], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]];

  const result = solveIk(robot, requested, [0, 0, 0, 0, 0, 0], limits);
  const independentlyMeasured = independentPoseError(readToolPose(robot), requested);

  assertEqual(result.ok, true, `final bounded update should converge (${result.reason})`);
  assert(independentlyMeasured.position <= 0.001, "final update position error exceeds tolerance");
});

test("IK validation failures preserve and return the entering robot state", async () => {
  const robot = await loadFixtureRobot();
  const entering = [10, 70, -90, 5, 10, -15];
  const alternateSeed = [0, 90, -77.263, 0, 0, 0];
  setRobotJoints(robot, entering);
  const invalidTarget = readToolPose(robot).clone();
  invalidTarget.elements[12] = Number.NaN;

  const results = [
    solveIk(robot, invalidTarget, alternateSeed, LIMITS),
    solveIk(robot, new THREE.Matrix4(), alternateSeed, [[-1, 1]]),
    solveIk(robot, new THREE.Matrix4(), [200, 90, -77.263, 0, 0, 0], LIMITS),
  ];

  for (const result of results) {
    result.jointsDeg.forEach((value, index) => {
      assert(Math.abs(value - entering[index]) < 1e-9, `validation changed joint${index + 1}`);
    });
    const actual = robotJointsDeg(robot);
    actual.forEach((value, index) => {
      assert(Math.abs(value - entering[index]) < 1e-9, `robot joint${index + 1} changed`);
    });
  }
});

test("joint and IK inputs reject malformed or non-finite values", async () => {
  const robot = await loadFixtureRobot();
  const pose = new THREE.Matrix4();

  for (const joints of [[0, 0], [0, 0, 0, 0, 0, Number.NaN]]) {
    let rejected = false;
    try {
      setRobotJoints(robot, joints);
    } catch (error) {
      rejected = error instanceof TypeError;
    }
    assert(rejected, "malformed joints must throw TypeError");
  }

  const badSeed = solveIk(robot, pose, [0, 0], LIMITS);
  const badLimits = solveIk(robot, pose, [0, 90, -77, 0, 0, 0], [[-1, 1]]);
  assertEqual(badSeed.ok, false, "malformed seed must fail safely");
  assertEqual(badSeed.reason, "non-finite input", "malformed seed reason");
  assertEqual(badLimits.ok, false, "malformed limits must fail safely");
  assertEqual(badLimits.reason, "non-finite input", "malformed limits reason");
});

scheduleTests();
