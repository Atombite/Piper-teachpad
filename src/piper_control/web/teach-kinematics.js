// Copyright © 2026-2026 atombite.ai
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";


const JOINT_COUNT = 6;
const JACOBIAN_STEP_RAD = 1e-5;
const DAMPING = 0.03;
const MAX_JOINT_STEP_RAD = THREE.MathUtils.degToRad(3);
const MAX_ITERATIONS = 80;
const POSITION_TOLERANCE_M = 0.001;
const ORIENTATION_TOLERANCE_RAD = THREE.MathUtils.degToRad(1);
const LIMIT_PUSH_ITERATIONS = 3;

function isFiniteSix(values) {
  return Array.isArray(values)
    && values.length === JOINT_COUNT
    && values.every(Number.isFinite);
}

function isFiniteLimits(limits) {
  return Array.isArray(limits)
    && limits.length === JOINT_COUNT
    && limits.every((limit) => (
      Array.isArray(limit)
      && limit.length === 2
      && Number.isFinite(limit[0])
      && Number.isFinite(limit[1])
      && limit[0] <= limit[1]
    ));
}

function getJoint(robot, index) {
  const name = `joint${index + 1}`;
  const joint = robot?.joints?.[name] ?? robot?.getObjectByName?.(name);
  if (!joint || typeof joint.setJointValue !== "function") {
    throw new TypeError(`robot is missing controllable ${name}`);
  }
  return joint;
}

function getTool(robot) {
  const tool = robot?.links?.tool0 ?? robot?.getObjectByName?.("tool0");
  if (!tool?.matrixWorld) {
    throw new TypeError("robot is missing tool0");
  }
  return tool;
}

export function setRobotJoints(robot, jointsDeg) {
  if (!isFiniteSix(jointsDeg)) {
    throw new TypeError("jointsDeg must contain six finite values");
  }
  jointsDeg.forEach((value, index) => {
    getJoint(robot, index).setJointValue(THREE.MathUtils.degToRad(value));
  });
  robot.updateMatrixWorld(true);
}

export function readToolPose(robot) {
  robot?.updateMatrixWorld?.(true);
  const pose = getTool(robot).matrixWorld.clone();
  if (!pose.elements.every(Number.isFinite)) {
    throw new TypeError("tool0 pose contains a non-finite value");
  }
  return pose;
}

function readRobotJoints(robot) {
  return Array.from({ length: JOINT_COUNT }, (_, index) => {
    const value = getJoint(robot, index).jointValue?.[0];
    if (!Number.isFinite(value)) {
      throw new TypeError(`joint${index + 1} contains a non-finite value`);
    }
    return THREE.MathUtils.radToDeg(value);
  });
}

function poseParts(matrix) {
  const position = new THREE.Vector3();
  const orientation = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, orientation, scale);
  return { position, orientation: orientation.normalize() };
}

function rotationVector(from, to) {
  const difference = to.clone().multiply(from.clone().invert()).normalize();
  if (difference.w < 0) {
    difference.x *= -1;
    difference.y *= -1;
    difference.z *= -1;
    difference.w *= -1;
  }
  const vectorLength = Math.hypot(difference.x, difference.y, difference.z);
  if (vectorLength < 1e-12) {
    return new THREE.Vector3(0, 0, 0);
  }
  const angle = 2 * Math.atan2(vectorLength, Math.max(-1, Math.min(1, difference.w)));
  return new THREE.Vector3(difference.x, difference.y, difference.z)
    .multiplyScalar(angle / vectorLength);
}

function poseError(currentPose, requestedPose) {
  const current = poseParts(currentPose);
  const requested = poseParts(requestedPose);
  const translation = requested.position.clone().sub(current.position);
  const rotation = rotationVector(current.orientation, requested.orientation);
  return {
    vector: [translation.x, translation.y, translation.z, rotation.x, rotation.y, rotation.z],
    positionErrorM: translation.length(),
    orientationErrorRad: rotation.length(),
  };
}

function solveLinearSystem(matrix, vector) {
  const size = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < size; column += 1) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivotRow][column])) {
        pivotRow = row;
      }
    }
    const pivot = augmented[pivotRow][column];
    if (!Number.isFinite(pivot) || Math.abs(pivot) < 1e-12) {
      return null;
    }
    [augmented[column], augmented[pivotRow]] = [augmented[pivotRow], augmented[column]];
    for (let row = column + 1; row < size; row += 1) {
      const factor = augmented[row][column] / augmented[column][column];
      if (!Number.isFinite(factor)) {
        return null;
      }
      for (let entry = column; entry <= size; entry += 1) {
        augmented[row][entry] -= factor * augmented[column][entry];
      }
    }
  }

  const result = new Array(size).fill(0);
  for (let row = size - 1; row >= 0; row -= 1) {
    let value = augmented[row][size];
    for (let column = row + 1; column < size; column += 1) {
      value -= augmented[row][column] * result[column];
    }
    result[row] = value / augmented[row][row];
    if (!Number.isFinite(result[row])) {
      return null;
    }
  }
  return result;
}

function numericalJacobian(robot, jointsRad, basePose) {
  const base = poseParts(basePose);
  const jacobian = Array.from({ length: 6 }, () => new Array(6).fill(0));
  for (let jointIndex = 0; jointIndex < JOINT_COUNT; jointIndex += 1) {
    const perturbed = [...jointsRad];
    perturbed[jointIndex] += JACOBIAN_STEP_RAD;
    setRobotJoints(robot, perturbed.map(THREE.MathUtils.radToDeg));
    const sample = poseParts(readToolPose(robot));
    const translation = sample.position.clone().sub(base.position)
      .multiplyScalar(1 / JACOBIAN_STEP_RAD);
    const rotation = rotationVector(base.orientation, sample.orientation)
      .multiplyScalar(1 / JACOBIAN_STEP_RAD);
    const column = [translation.x, translation.y, translation.z, rotation.x, rotation.y, rotation.z];
    column.forEach((value, rowIndex) => {
      jacobian[rowIndex][jointIndex] = value;
    });
  }
  setRobotJoints(robot, jointsRad.map(THREE.MathUtils.radToDeg));
  return jacobian;
}

function dampedLeastSquares(jacobian, error) {
  const normal = Array.from({ length: 6 }, (_, row) => (
    Array.from({ length: 6 }, (_, column) => {
      let value = row === column ? DAMPING * DAMPING : 0;
      for (let joint = 0; joint < 6; joint += 1) {
        value += jacobian[row][joint] * jacobian[column][joint];
      }
      return value;
    })
  ));
  const intermediate = solveLinearSystem(normal, error);
  if (!intermediate) {
    return null;
  }
  return Array.from({ length: 6 }, (_, joint) => {
    let value = 0;
    for (let row = 0; row < 6; row += 1) {
      value += jacobian[row][joint] * intermediate[row];
    }
    return value;
  });
}

function failure(reason, jointsDeg, error = null) {
  return {
    ok: false,
    jointsDeg: [...jointsDeg],
    positionErrorM: error?.positionErrorM ?? Number.POSITIVE_INFINITY,
    orientationErrorRad: error?.orientationErrorRad ?? Number.POSITIVE_INFINITY,
    reason,
  };
}

function isConverged(error) {
  return error.positionErrorM <= POSITION_TOLERANCE_M
    && error.orientationErrorRad <= ORIENTATION_TOLERANCE_RAD;
}

function success(jointsDeg, error) {
  return {
    ok: true,
    jointsDeg: [...jointsDeg],
    positionErrorM: error.positionErrorM,
    orientationErrorRad: error.orientationErrorRad,
    reason: null,
  };
}

function evaluateJoints(robot, requestedPose, jointsDeg) {
  setRobotJoints(robot, jointsDeg);
  const actualJointsDeg = readRobotJoints(robot);
  const error = poseError(readToolPose(robot), requestedPose);
  return { actualJointsDeg, error };
}

export function solveIk(robot, requestedPose, seedDeg, limitsDeg, options = {}) {
  void options;
  const enteringJointsDeg = readRobotJoints(robot);
  if (!(requestedPose instanceof THREE.Matrix4)
      || !requestedPose.elements.every(Number.isFinite)) {
    return failure("non-finite target", enteringJointsDeg);
  }
  if (!isFiniteSix(seedDeg) || !isFiniteLimits(limitsDeg)) {
    return failure(
      "non-finite input",
      enteringJointsDeg,
      poseError(readToolPose(robot), requestedPose),
    );
  }
  if (seedDeg.some((value, index) => value < limitsDeg[index][0] || value > limitsDeg[index][1])) {
    return failure(
      "joint limit",
      enteringJointsDeg,
      poseError(readToolPose(robot), requestedPose),
    );
  }

  const limitsRad = limitsDeg.map(([lower, upper]) => [
    THREE.MathUtils.degToRad(lower),
    THREE.MathUtils.degToRad(upper),
  ]);
  let jointsDeg = [...seedDeg];
  let limitPushes = 0;
  let lastError = null;

  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration += 1) {
    const current = evaluateJoints(robot, requestedPose, jointsDeg);
    jointsDeg = current.actualJointsDeg;
    let jointsRad = jointsDeg.map(THREE.MathUtils.degToRad);
    const currentPose = readToolPose(robot);
    lastError = current.error;
    if (!lastError.vector.every(Number.isFinite)) {
      return failure("unreachable", jointsDeg, lastError);
    }
    if (isConverged(lastError)) {
      return success(jointsDeg, lastError);
    }

    const jacobian = numericalJacobian(robot, jointsRad, currentPose);
    if (!jacobian.flat().every(Number.isFinite)) {
      return failure("unreachable", jointsDeg, lastError);
    }
    let delta = dampedLeastSquares(jacobian, lastError.vector);
    if (!delta || !delta.every(Number.isFinite)) {
      return failure("unreachable", jointsDeg, lastError);
    }
    const largestStep = Math.max(...delta.map(Math.abs));
    if (largestStep > MAX_JOINT_STEP_RAD) {
      delta = delta.map((value) => value * MAX_JOINT_STEP_RAD / largestStep);
    }

    let pushedLimit = false;
    let moved = false;
    const requestedJointsRad = jointsRad.map((value, index) => {
      const requested = value + delta[index];
      const bounded = Math.max(limitsRad[index][0], Math.min(limitsRad[index][1], requested));
      pushedLimit ||= Math.abs(requested - bounded) > 1e-12;
      moved ||= Math.abs(bounded - value) > 1e-12;
      return bounded;
    });
    const updated = evaluateJoints(
      robot,
      requestedPose,
      requestedJointsRad.map(THREE.MathUtils.radToDeg),
    );
    jointsDeg = updated.actualJointsDeg;
    lastError = updated.error;
    if (!lastError.vector.every(Number.isFinite)) {
      return failure("unreachable", jointsDeg, lastError);
    }
    if (isConverged(lastError)) {
      return success(jointsDeg, lastError);
    }
    limitPushes = pushedLimit ? limitPushes + 1 : 0;
    if (limitPushes >= LIMIT_PUSH_ITERATIONS || (!moved && pushedLimit)) {
      return failure("joint limit", jointsDeg, lastError);
    }
    if (!moved) {
      return failure("unreachable", jointsDeg, lastError);
    }
  }

  if (isConverged(lastError)) {
    return success(jointsDeg, lastError);
  }
  return failure(limitPushes > 0 ? "joint limit" : "unreachable", jointsDeg, lastError);
}
