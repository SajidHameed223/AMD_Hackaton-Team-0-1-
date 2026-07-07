"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

type Decor = {
  group: THREE.Group;
  x: number;
  y: number;
  z: number;
  scale: number;
  spin: THREE.Vector3;
  bob: number;
  phase: number;
};

type AnimalFactory = (body: number, accent: number) => THREE.Group;

const FOV = 48;
const CAMERA_Z = 9;
const DARK = 0x10333e;
const ORANGE = 0xff7a55;

function material(color: number, opacity = 0.9) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.48,
    metalness: 0.04,
    transparent: true,
    opacity,
  });
}

function addPart(
  group: THREE.Group,
  geometry: THREE.BufferGeometry,
  mat: THREE.Material,
  position: [number, number, number],
  scale: [number, number, number] = [1, 1, 1],
  rotation: [number, number, number] = [0, 0, 0],
) {
  const mesh = new THREE.Mesh(geometry, mat);
  mesh.position.set(...position);
  mesh.scale.set(...scale);
  mesh.rotation.set(...rotation);
  group.add(mesh);
  return mesh;
}

function addEye(group: THREE.Group, x: number, y: number, z = 0.22) {
  addPart(
    group,
    new THREE.SphereGeometry(0.028, 10, 10),
    material(DARK, 0.95),
    [x, y, z],
  );
}

function disposeObject(root: THREE.Object3D) {
  root.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh)) return;
    obj.geometry.dispose();
    const mat = obj.material;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else mat.dispose();
  });
}

function fish(body: number, accent: number) {
  const group = new THREE.Group();
  const bodyMat = material(body, 0.78);
  const accentMat = material(accent, 0.85);
  addPart(group, new THREE.SphereGeometry(0.34, 24, 16), bodyMat, [0, 0, 0], [
    1.25,
    0.72,
    0.62,
  ]);
  addPart(
    group,
    new THREE.ConeGeometry(0.22, 0.38, 3),
    accentMat,
    [-0.48, 0, 0],
    [1, 0.78, 1],
    [0, 0, Math.PI / 2],
  );
  addPart(
    group,
    new THREE.ConeGeometry(0.12, 0.24, 3),
    accentMat,
    [0.02, 0.28, 0],
    [1, 1, 1],
    [0, 0, Math.PI],
  );
  addEye(group, 0.28, 0.08);
  return group;
}

function bird(body: number, accent: number) {
  const group = new THREE.Group();
  const bodyMat = material(body, 0.82);
  const accentMat = material(accent, 0.9);
  addPart(group, new THREE.SphereGeometry(0.28, 24, 16), bodyMat, [0, 0, 0], [
    1.08,
    0.86,
    0.7,
  ]);
  addPart(group, new THREE.SphereGeometry(0.18, 18, 12), bodyMat, [0.28, 0.18, 0]);
  addPart(
    group,
    new THREE.ConeGeometry(0.08, 0.22, 24),
    material(ORANGE, 0.95),
    [0.48, 0.18, 0],
    [1, 1, 1],
    [0, 0, -Math.PI / 2],
  );
  addPart(
    group,
    new THREE.SphereGeometry(0.16, 18, 12),
    accentMat,
    [-0.04, 0.02, 0.18],
    [1.25, 0.42, 0.28],
    [0.2, 0.1, -0.52],
  );
  addEye(group, 0.36, 0.23);
  return group;
}

function bunny(body: number, accent: number) {
  const group = new THREE.Group();
  const bodyMat = material(body, 0.82);
  const accentMat = material(accent, 0.76);
  addPart(group, new THREE.SphereGeometry(0.28, 24, 16), bodyMat, [0, -0.08, 0], [
    1,
    1.05,
    0.72,
  ]);
  addPart(group, new THREE.SphereGeometry(0.22, 20, 14), bodyMat, [0.2, 0.24, 0]);
  addPart(
    group,
    new THREE.CapsuleGeometry(0.055, 0.32, 8, 14),
    bodyMat,
    [0.11, 0.58, 0],
    [1, 1, 1],
    [0, 0, -0.2],
  );
  addPart(
    group,
    new THREE.CapsuleGeometry(0.055, 0.32, 8, 14),
    bodyMat,
    [0.32, 0.56, 0],
    [1, 1, 1],
    [0, 0, 0.26],
  );
  addPart(group, new THREE.SphereGeometry(0.09, 14, 10), accentMat, [-0.24, -0.08, 0.12]);
  addEye(group, 0.29, 0.3);
  return group;
}

function cat(body: number, accent: number) {
  const group = new THREE.Group();
  const bodyMat = material(body, 0.82);
  const accentMat = material(accent, 0.82);
  addPart(group, new THREE.SphereGeometry(0.26, 24, 16), bodyMat, [0, -0.04, 0], [
    1.08,
    0.86,
    0.7,
  ]);
  addPart(group, new THREE.SphereGeometry(0.22, 20, 14), bodyMat, [0.24, 0.22, 0]);
  addPart(
    group,
    new THREE.ConeGeometry(0.09, 0.18, 3),
    bodyMat,
    [0.11, 0.42, 0],
    [1, 1, 1],
    [0, 0, 0.3],
  );
  addPart(
    group,
    new THREE.ConeGeometry(0.09, 0.18, 3),
    bodyMat,
    [0.36, 0.42, 0],
    [1, 1, 1],
    [0, 0, -0.3],
  );
  addPart(
    group,
    new THREE.TorusGeometry(0.18, 0.035, 8, 22, Math.PI * 1.45),
    accentMat,
    [-0.28, 0.06, 0],
    [1, 1, 1],
    [0.2, 0.1, -0.8],
  );
  addEye(group, 0.18, 0.27);
  addEye(group, 0.32, 0.27);
  return group;
}

function turtle(body: number, accent: number) {
  const group = new THREE.Group();
  const shellMat = material(body, 0.82);
  const limbMat = material(accent, 0.78);
  addPart(group, new THREE.SphereGeometry(0.3, 24, 16), shellMat, [0, 0, 0], [
    1.18,
    0.74,
    0.62,
  ]);
  addPart(group, new THREE.SphereGeometry(0.14, 18, 12), limbMat, [0.38, 0.05, 0]);
  addPart(group, new THREE.SphereGeometry(0.08, 14, 10), limbMat, [-0.18, 0.24, 0]);
  addPart(group, new THREE.SphereGeometry(0.08, 14, 10), limbMat, [-0.18, -0.24, 0]);
  addPart(group, new THREE.SphereGeometry(0.07, 14, 10), limbMat, [0.12, 0.27, 0]);
  addPart(group, new THREE.SphereGeometry(0.07, 14, 10), limbMat, [0.12, -0.27, 0]);
  addEye(group, 0.43, 0.09);
  return group;
}

function puppy(body: number, accent: number) {
  const group = new THREE.Group();
  const bodyMat = material(body, 0.8);
  const accentMat = material(accent, 0.82);
  addPart(group, new THREE.SphereGeometry(0.28, 24, 16), bodyMat, [0, -0.02, 0], [
    1.06,
    0.82,
    0.7,
  ]);
  addPart(group, new THREE.SphereGeometry(0.2, 20, 14), bodyMat, [0.3, 0.22, 0]);
  addPart(group, new THREE.SphereGeometry(0.075, 14, 10), accentMat, [0.45, 0.17, 0.02]);
  addPart(
    group,
    new THREE.CapsuleGeometry(0.055, 0.18, 8, 12),
    accentMat,
    [0.14, 0.24, 0.03],
    [1, 1, 1],
    [0, 0, 0.7],
  );
  addPart(
    group,
    new THREE.CapsuleGeometry(0.055, 0.18, 8, 12),
    accentMat,
    [0.34, 0.4, 0.03],
    [1, 1, 1],
    [0, 0, -0.55],
  );
  addEye(group, 0.34, 0.28);
  return group;
}

const factories: AnimalFactory[] = [fish, bird, bunny, cat, turtle, puppy];

export function JoyDecorations() {
  const rootRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const root = rootRef.current;
    if (!canvas || !root) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        premultipliedAlpha: false,
        preserveDrawingBuffer: true,
        antialias: true,
        powerPreference: "low-power",
      });
    } catch {
      root.classList.add("o1-joy--fallback");
      return;
    }
    root.classList.remove("o1-joy--fallback");
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    const rendererCanvas = renderer.domElement;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 40);
    camera.position.set(0, 0, CAMERA_Z);

    scene.add(new THREE.AmbientLight(0xffffff, 1.85));
    const key = new THREE.DirectionalLight(0xffffff, 2.35);
    key.position.set(-3, 4, 7);
    scene.add(key);

    const palette = [
      [0x22cbe8, 0xffd166],
      [0xff7a55, 0x67e0f2],
      [0x6a55c9, 0xffa284],
      [0x2e8b3c, 0xffd166],
      [0xd6558e, 0x22cbe8],
      [0xdf9a00, 0x06aece],
    ] as const;

    const spots = [
      [-0.06, 0.08, 0.78],
      [0.15, 0.1, 0.58],
      [0.38, 0.08, 0.48],
      [0.64, 0.09, 0.46],
      [0.9, 0.16, 0.56],
      [1.05, 0.38, 0.72],
      [0.08, 0.58, 0.62],
      [0.23, 0.86, 0.52],
      [0.52, 0.9, 0.44],
      [0.79, 0.78, 0.5],
      [0.96, 0.62, 0.58],
      [0.72, 0.38, 0.34],
    ] as const;

    const decors: Decor[] = spots.map(([x, y, scale], i) => {
      const [body, accent] = palette[i % palette.length];
      const group = factories[i % factories.length](body, accent);
      group.rotation.set(i * 0.45, i * 0.22, i * 0.17);
      scene.add(group);
      return {
        group,
        x,
        y,
        z: -1.2 - (i % 4) * 0.3,
        scale,
        spin: new THREE.Vector3(
          0.05 + (i % 3) * 0.015,
          0.08 + (i % 4) * 0.012,
          (i % 2 === 0 ? 1 : -1) * (0.09 + (i % 5) * 0.01),
        ),
        bob: 0.09 + (i % 4) * 0.018,
        phase: i * 0.72,
      };
    });

    function resizeAndLayout() {
      const width = rendererCanvas.clientWidth;
      const height = rendererCanvas.clientHeight;
      if (!width || !height) return;

      const drawingWidth = Math.floor(width * renderer.getPixelRatio());
      const drawingHeight = Math.floor(height * renderer.getPixelRatio());
      if (
        rendererCanvas.width !== drawingWidth ||
        rendererCanvas.height !== drawingHeight
      ) {
        renderer.setSize(width, height, false);
      }

      camera.aspect = width / height;
      camera.updateProjectionMatrix();

      const viewH =
        2 * Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * CAMERA_Z;
      const viewW = viewH * camera.aspect;
      decors.forEach((decor) => {
        decor.group.position.x = (decor.x - 0.5) * viewW;
        decor.group.position.y = (0.5 - decor.y) * viewH;
        decor.group.position.z = decor.z;
        const mobileScale = width < 720 ? 0.66 : 1;
        decor.group.scale.setScalar(decor.scale * mobileScale);
      });
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let frame = 0;

    const render = (time = 0) => {
      resizeAndLayout();
      const t = time * 0.001;
      decors.forEach((decor) => {
        decor.group.rotation.x += decor.spin.x * 0.016;
        decor.group.rotation.y += decor.spin.y * 0.016;
        decor.group.rotation.z += decor.spin.z * 0.016;
        decor.group.position.y += Math.sin(t * 0.9 + decor.phase) * decor.bob * 0.014;
      });
      renderer.render(scene, camera);
      if (!reducedMotion.matches) frame = requestAnimationFrame(render);
    };

    render();
    if (!reducedMotion.matches) frame = requestAnimationFrame(render);
    window.addEventListener("resize", resizeAndLayout);

    return () => {
      window.removeEventListener("resize", resizeAndLayout);
      if (frame) cancelAnimationFrame(frame);
      decors.forEach((decor) => disposeObject(decor.group));
      renderer.dispose();
    };
  }, []);

  return (
    <div ref={rootRef} className="o1-joy" aria-hidden>
      <canvas ref={canvasRef} className="o1-joy__canvas" />
      <div className="o1-joy__fallback">
        <span className="o1-joy__paw o1-joy__paw--a" />
        <span className="o1-joy__paw o1-joy__paw--b" />
        <span className="o1-joy__paw o1-joy__paw--c" />
      </div>
    </div>
  );
}
