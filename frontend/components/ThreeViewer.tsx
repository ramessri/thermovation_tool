'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { PLYLoader } from 'three-stdlib';
import { OrbitControls } from 'three-stdlib';

function AwaitingPlaceholder() {
  return (
    <div
      style={{ width: '100%', height: '24rem', background: '#0f172a' }}
      className="flex items-center justify-center"
    >
      <span className="text-sm text-slate-500">Awaiting reconstruction…</span>
    </div>
  );
}

function PLYCanvas({ plyUrl }: { plyUrl: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);

    // Camera
    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.001,
      10000
    );
    camera.position.set(0, 0, 5);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Ambient light (not needed for point clouds with vertex colors but harmless)
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));

    // Load PLY
    const loader = new PLYLoader();
    let animationId: number;
    let pointsMesh: THREE.Points | null = null;

    loader.load(
      plyUrl,
      (geometry) => {
        geometry.computeBoundingBox();
        const bbox = geometry.boundingBox!;
        const center = new THREE.Vector3();
        bbox.getCenter(center);
        const diagonal = bbox.min.distanceTo(bbox.max);

        // Center the geometry at origin
        geometry.translate(-center.x, -center.y, -center.z);

        const material = new THREE.PointsMaterial({
          vertexColors: true,
          size: 0.3,
          sizeAttenuation: true,
        });

        pointsMesh = new THREE.Points(geometry, material);
        scene.add(pointsMesh);

        // Position camera
        camera.position.set(0, 0, diagonal * 1.5);
        camera.lookAt(0, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();
      },
      undefined,
      (err) => {
        console.error('PLY load error:', err);
        setLoadError('Could not load point cloud. The file may still be uploading or the URL is invalid.');
      }
    );

    // Animation loop
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationId);
      controls.dispose();
      if (pointsMesh) {
        pointsMesh.geometry.dispose();
        (pointsMesh.material as THREE.Material).dispose();
      }
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [plyUrl]);

  if (loadError) {
    return (
      <div
        style={{ width: '100%', height: '24rem', background: '#0f172a' }}
        className="flex items-center justify-center"
      >
        <span className="text-sm text-red-400 px-6 text-center">{loadError}</span>
      </div>
    );
  }

  return <div ref={containerRef} style={{ width: '100%', height: '24rem' }} />;
}

export function ThreeViewer({ plyUrl }: { plyUrl: string | null }) {
  if (!plyUrl) {
    return <AwaitingPlaceholder />;
  }
  return <PLYCanvas plyUrl={plyUrl} />;
}
