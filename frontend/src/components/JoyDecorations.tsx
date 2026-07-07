"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

type AnimalKind = "fish" | "bird" | "bunny" | "cat" | "turtle" | "dog";

type Decor = {
  mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  x: number;
  y: number;
  z: number;
  scale: number;
  tilt: number;
  drift: number;
  bob: number;
  phase: number;
};

type DrawingContext = CanvasRenderingContext2D;

const FOV = 48;
const CAMERA_Z = 9;
const SPRITE_W = 128;
const SPRITE_H = 96;
const DARK = 0x10333e;
const WHITE = 0xfffbf1;
const ORANGE = 0xff8a3d;

const animals: Array<[AnimalKind, number, number, number]> = [
  ["fish", -0.04, 0.13, 1.02],
  ["bird", 0.14, 0.1, 0.86],
  ["bunny", 0.38, 0.09, 0.82],
  ["cat", 0.63, 0.1, 0.82],
  ["turtle", 0.9, 0.17, 0.9],
  ["dog", 1.04, 0.39, 1.0],
  ["fish", 0.08, 0.58, 0.86],
  ["cat", 0.23, 0.86, 0.82],
  ["bunny", 0.52, 0.9, 0.78],
  ["bird", 0.78, 0.78, 0.82],
  ["turtle", 0.96, 0.62, 0.9],
  ["dog", 0.72, 0.38, 0.72],
  ["bird", 0.34, 0.5, 0.64],
  ["cat", -0.03, 0.78, 0.76],
];

const palette = [
  [0x22cbe8, 0xffd166],
  [0xff7a55, 0x67e0f2],
  [0x6a55c9, 0xffa284],
  [0x2e8b3c, 0xffd166],
  [0xd6558e, 0x22cbe8],
  [0xdf9a00, 0x06aece],
] as const;

function hex(color: number) {
  return `#${color.toString(16).padStart(6, "0")}`;
}

function rgba(color: number, alpha: number) {
  const r = (color >> 16) & 255;
  const g = (color >> 8) & 255;
  const b = color & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function ellipse(
  ctx: DrawingContext,
  x: number,
  y: number,
  rx: number,
  ry: number,
  fill: string,
  stroke = hex(DARK),
  lineWidth = 3,
) {
  ctx.beginPath();
  ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function circle(
  ctx: DrawingContext,
  x: number,
  y: number,
  radius: number,
  fill: string,
  stroke = hex(DARK),
  lineWidth = 3,
) {
  ellipse(ctx, x, y, radius, radius, fill, stroke, lineWidth);
}

function triangle(
  ctx: DrawingContext,
  points: Array<[number, number]>,
  fill: string,
  stroke = hex(DARK),
  lineWidth = 3,
) {
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function line(ctx: DrawingContext, from: [number, number], to: [number, number]) {
  ctx.beginPath();
  ctx.moveTo(...from);
  ctx.lineTo(...to);
  ctx.stroke();
}

function curvedTail(
  ctx: DrawingContext,
  x: number,
  y: number,
  radius: number,
  start: number,
  end: number,
) {
  ctx.beginPath();
  ctx.arc(x, y, radius, start, end);
  ctx.lineWidth = 7;
  ctx.strokeStyle = hex(DARK);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, radius, start, end);
  ctx.lineWidth = 4;
  ctx.strokeStyle = hex(WHITE);
  ctx.stroke();
}

function eye(ctx: DrawingContext, x: number, y: number) {
  circle(ctx, x, y, 4.2, hex(WHITE), hex(DARK), 2);
  circle(ctx, x + 1, y + 0.8, 1.7, hex(DARK), hex(DARK), 1);
}

function smile(ctx: DrawingContext, x: number, y: number, radius = 6) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0.15 * Math.PI, 0.82 * Math.PI);
  ctx.lineWidth = 2.2;
  ctx.strokeStyle = hex(DARK);
  ctx.stroke();
}

function drawFish(ctx: DrawingContext, body: string, accent: string) {
  triangle(ctx, [[22, 48], [5, 29], [5, 67]], accent);
  triangle(ctx, [[52, 31], [42, 14], [71, 26]], accent);
  triangle(ctx, [[54, 65], [41, 80], [75, 69]], accent);
  ellipse(ctx, 64, 48, 35, 23, body);
  triangle(ctx, [[93, 47], [115, 37], [115, 57]], accent);
  eye(ctx, 80, 41);
  smile(ctx, 91, 48, 6);
  ctx.lineWidth = 2;
  ctx.strokeStyle = rgba(DARK, 0.5);
  line(ctx, [52, 30], [52, 66]);
}

function drawBird(ctx: DrawingContext, body: string, accent: string) {
  triangle(ctx, [[94, 41], [119, 49], [94, 57]], hex(ORANGE));
  ellipse(ctx, 58, 55, 34, 23, body);
  circle(ctx, 84, 41, 18, body);
  ellipse(ctx, 53, 54, 18, 12, accent);
  triangle(ctx, [[26, 51], [8, 39], [17, 61]], accent);
  eye(ctx, 89, 36);
  ctx.lineWidth = 3;
  ctx.strokeStyle = hex(DARK);
  line(ctx, [67, 75], [62, 86]);
  line(ctx, [79, 75], [80, 86]);
  line(ctx, [57, 86], [67, 86]);
  line(ctx, [75, 86], [86, 86]);
}

function drawBunny(ctx: DrawingContext, body: string, accent: string) {
  ellipse(ctx, 49, 57, 29, 24, body);
  ellipse(ctx, 75, 43, 22, 21, body);
  ellipse(ctx, 64, 21, 8, 24, body);
  ellipse(ctx, 84, 19, 8, 25, body);
  ellipse(ctx, 64, 22, 3.5, 15, accent, rgba(DARK, 0.55), 1.6);
  ellipse(ctx, 84, 20, 3.5, 16, accent, rgba(DARK, 0.55), 1.6);
  circle(ctx, 22, 55, 10, hex(WHITE));
  eye(ctx, 81, 40);
  circle(ctx, 92, 47, 3, accent, hex(DARK), 1.8);
  ctx.lineWidth = 1.9;
  ctx.strokeStyle = hex(DARK);
  line(ctx, [95, 49], [109, 45]);
  line(ctx, [95, 52], [109, 54]);
  smile(ctx, 86, 50, 5);
}

function drawCat(ctx: DrawingContext, body: string, accent: string) {
  curvedTail(ctx, 36, 55, 21, 0.86 * Math.PI, 2.05 * Math.PI);
  ellipse(ctx, 56, 61, 27, 20, body);
  triangle(ctx, [[62, 28], [71, 9], [80, 31]], body);
  triangle(ctx, [[92, 30], [103, 13], [106, 37]], body);
  triangle(ctx, [[66, 27], [72, 16], [77, 29]], accent, rgba(DARK, 0.55), 1.7);
  triangle(ctx, [[96, 31], [102, 21], [104, 34]], accent, rgba(DARK, 0.55), 1.7);
  circle(ctx, 85, 43, 24, body);
  eye(ctx, 78, 39);
  eye(ctx, 92, 39);
  circle(ctx, 85, 47, 2.8, accent, hex(DARK), 1.5);
  ctx.lineWidth = 1.8;
  ctx.strokeStyle = hex(DARK);
  line(ctx, [77, 50], [59, 45]);
  line(ctx, [78, 54], [60, 56]);
  line(ctx, [92, 50], [112, 45]);
  line(ctx, [92, 54], [112, 56]);
  smile(ctx, 82, 50, 5);
}

function drawTurtle(ctx: DrawingContext, body: string, accent: string) {
  circle(ctx, 93, 47, 13, accent);
  ellipse(ctx, 58, 52, 35, 25, body);
  circle(ctx, 31, 35, 9, accent);
  circle(ctx, 30, 69, 9, accent);
  circle(ctx, 70, 30, 9, accent);
  circle(ctx, 70, 74, 9, accent);
  triangle(ctx, [[20, 52], [8, 43], [9, 62]], accent);
  eye(ctx, 98, 43);
  ctx.beginPath();
  ctx.ellipse(58, 52, 23, 15, 0, 0, Math.PI * 2);
  ctx.lineWidth = 2;
  ctx.strokeStyle = rgba(DARK, 0.45);
  ctx.stroke();
  line(ctx, [58, 28], [58, 76]);
  line(ctx, [36, 43], [80, 63]);
  line(ctx, [80, 43], [36, 63]);
}

function drawDog(ctx: DrawingContext, body: string, accent: string) {
  curvedTail(ctx, 32, 49, 17, 1.14 * Math.PI, 1.94 * Math.PI);
  ellipse(ctx, 55, 61, 28, 20, body);
  circle(ctx, 84, 42, 24, body);
  ellipse(ctx, 68, 45, 10, 22, accent);
  ellipse(ctx, 102, 45, 10, 22, accent);
  ellipse(ctx, 88, 50, 13, 9, hex(WHITE), hex(DARK), 2.2);
  circle(ctx, 93, 47, 3.2, hex(DARK), hex(DARK), 1);
  eye(ctx, 78, 38);
  eye(ctx, 93, 38);
  ctx.lineWidth = 3;
  ctx.strokeStyle = hex(DARK);
  line(ctx, [44, 78], [44, 87]);
  line(ctx, [66, 78], [66, 87]);
  smile(ctx, 84, 50, 5);
}

function drawAnimal(
  ctx: DrawingContext,
  kind: AnimalKind,
  body: string,
  accent: string,
) {
  ctx.save();
  ctx.translate(0, 1);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = "rgba(16, 51, 62, 0.15)";
  ctx.shadowBlur = 5;
  ctx.shadowOffsetY = 4;

  switch (kind) {
    case "fish":
      drawFish(ctx, body, accent);
      break;
    case "bird":
      drawBird(ctx, body, accent);
      break;
    case "bunny":
      drawBunny(ctx, body, accent);
      break;
    case "cat":
      drawCat(ctx, body, accent);
      break;
    case "turtle":
      drawTurtle(ctx, body, accent);
      break;
    case "dog":
      drawDog(ctx, body, accent);
      break;
  }
  ctx.restore();
}

function makeAnimalTexture(kind: AnimalKind, bodyColor: number, accentColor: number) {
  const canvas = document.createElement("canvas");
  const scale = 3;
  canvas.width = SPRITE_W * scale;
  canvas.height = SPRITE_H * scale;

  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");

  ctx.scale(scale, scale);
  ctx.clearRect(0, 0, SPRITE_W, SPRITE_H);
  drawAnimal(ctx, kind, hex(bodyColor), hex(accentColor));

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  texture.needsUpdate = true;
  return texture;
}

function disposeDecor(decor: Decor) {
  decor.mesh.geometry.dispose();
  decor.mesh.material.map?.dispose();
  decor.mesh.material.dispose();
}

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

    const geometry = new THREE.PlaneGeometry(1.62, 1.22);
    const decors: Decor[] = animals.map(([kind, x, y, scale], i) => {
      const [body, accent] = palette[i % palette.length];
      const texture = makeAnimalTexture(kind, body, accent);
      const material = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0.94,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry.clone(), material);
      mesh.rotation.set(0, 0, (i % 2 === 0 ? -1 : 1) * (0.09 + i * 0.006));
      scene.add(mesh);

      return {
        mesh,
        x,
        y,
        z: -1.4 - (i % 4) * 0.22,
        scale,
        tilt: (i % 2 === 0 ? 1 : -1) * (0.06 + (i % 5) * 0.012),
        drift: (i % 2 === 0 ? 1 : -1) * (0.05 + (i % 4) * 0.012),
        bob: 0.08 + (i % 4) * 0.018,
        phase: i * 0.68,
      };
    });
    geometry.dispose();

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
        decor.mesh.position.x = (decor.x - 0.5) * viewW;
        decor.mesh.position.y = (0.5 - decor.y) * viewH;
        decor.mesh.position.z = decor.z;
        const mobileScale = width < 720 ? 0.68 : 1;
        decor.mesh.scale.setScalar(decor.scale * mobileScale);
      });
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let frame = 0;

    const render = (time = 0) => {
      resizeAndLayout();
      const t = time * 0.001;
      decors.forEach((decor) => {
        decor.mesh.rotation.x = Math.sin(t * 0.5 + decor.phase) * decor.tilt;
        decor.mesh.rotation.y = Math.cos(t * 0.45 + decor.phase) * decor.drift;
        decor.mesh.rotation.z += decor.drift * 0.004;
        decor.mesh.position.y +=
          Math.sin(t * 0.9 + decor.phase) * decor.bob * 0.014;
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
      decors.forEach(disposeDecor);
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
