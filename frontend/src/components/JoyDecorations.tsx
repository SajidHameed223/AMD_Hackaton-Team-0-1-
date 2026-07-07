"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

type Decor = {
  mesh: THREE.Mesh;
  x: number;
  y: number;
  z: number;
  scale: number;
  spin: THREE.Vector3;
  bob: number;
  phase: number;
};

const FOV = 48;
const CAMERA_Z = 9;

function disposeMesh(mesh: THREE.Mesh) {
  mesh.geometry.dispose();
  const material = mesh.material;
  if (Array.isArray(material)) material.forEach((m) => m.dispose());
  else material.dispose();
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

    scene.add(new THREE.AmbientLight(0xffffff, 1.7));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(-3, 4, 7);
    scene.add(key);

    const material = (color: number, opacity = 0.34) =>
      new THREE.MeshStandardMaterial({
        color,
        roughness: 0.42,
        metalness: 0.06,
        transparent: true,
        opacity,
      });

    const specs = [
      {
        geometry: new THREE.TorusGeometry(0.42, 0.075, 12, 36),
        material: material(0x06aece, 0.28),
        x: -0.03,
        y: 0.06,
        z: -1.2,
        scale: 0.86,
        spin: new THREE.Vector3(0.12, 0.28, 0.22),
        bob: 0.08,
      },
      {
        geometry: new THREE.TorusGeometry(0.52, 0.065, 12, 36),
        material: material(0xff7a55, 0.2),
        x: 0.95,
        y: 0.8,
        z: -1.7,
        scale: 1.25,
        spin: new THREE.Vector3(-0.1, 0.18, -0.16),
        bob: 0.1,
      },
      {
        geometry: new THREE.TetrahedronGeometry(0.36),
        material: material(0xffd166, 0.36),
        x: 0.82,
        y: 0.68,
        z: -1.1,
        scale: 0.9,
        spin: new THREE.Vector3(0.24, -0.18, 0.12),
        bob: 0.13,
      },
      {
        geometry: new THREE.IcosahedronGeometry(0.32, 0),
        material: material(0x6a55c9, 0.24),
        x: 0.14,
        y: 0.82,
        z: -1.4,
        scale: 0.86,
        spin: new THREE.Vector3(-0.16, 0.12, 0.2),
        bob: 0.12,
      },
      {
        geometry: new THREE.BoxGeometry(0.34, 0.34, 0.1),
        material: material(0x67e0f2, 0.26),
        x: 0.64,
        y: 0.12,
        z: -2.0,
        scale: 0.78,
        spin: new THREE.Vector3(0.08, 0.16, 0.5),
        bob: 0.07,
      },
      {
        geometry: new THREE.BoxGeometry(0.3, 0.3, 0.1),
        material: material(0xed6a45, 0.22),
        x: 0.34,
        y: 0.9,
        z: -1.8,
        scale: 0.7,
        spin: new THREE.Vector3(-0.14, 0.2, -0.44),
        bob: 0.08,
      },
    ];

    const decors: Decor[] = specs.map((spec, i) => {
      const mesh = new THREE.Mesh(spec.geometry, spec.material);
      mesh.rotation.set(i * 0.6, i * 0.35, i * 0.2);
      scene.add(mesh);
      return {
        mesh,
        x: spec.x,
        y: spec.y,
        z: spec.z,
        scale: spec.scale,
        spin: spec.spin,
        bob: spec.bob,
        phase: i * 0.83,
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
        decor.mesh.position.x = (decor.x - 0.5) * viewW;
        decor.mesh.position.y = (0.5 - decor.y) * viewH;
        decor.mesh.position.z = decor.z;
        const mobileScale = width < 720 ? 0.72 : 1;
        decor.mesh.scale.setScalar(decor.scale * mobileScale);
      });
    }

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let frame = 0;

    const render = (time = 0) => {
      resizeAndLayout();
      const t = time * 0.001;
      decors.forEach((decor) => {
        decor.mesh.rotation.x += decor.spin.x * 0.016;
        decor.mesh.rotation.y += decor.spin.y * 0.016;
        decor.mesh.rotation.z += decor.spin.z * 0.016;
        decor.mesh.position.y += Math.sin(t * 0.8 + decor.phase) * decor.bob * 0.018;
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
      decors.forEach((decor) => disposeMesh(decor.mesh));
      renderer.dispose();
    };
  }, []);

  return (
    <div ref={rootRef} className="o1-joy" aria-hidden>
      <canvas ref={canvasRef} className="o1-joy__canvas" />
      <div className="o1-joy__fallback">
        <span className="o1-joy__ring o1-joy__ring--a" />
        <span className="o1-joy__ring o1-joy__ring--b" />
        <span className="o1-joy__chip o1-joy__chip--a" />
        <span className="o1-joy__chip o1-joy__chip--b" />
      </div>
    </div>
  );
}
