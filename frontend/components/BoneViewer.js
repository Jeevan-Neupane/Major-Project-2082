"use client";

import { useRef, useMemo, useState, useEffect, useCallback } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, GizmoHelper, GizmoViewport } from "@react-three/drei";
import * as THREE from "three";

/* ------------------------------------------------------------------ *
 *  BoneMesh – creates a BufferGeometry from marching-cubes output     *
 * ------------------------------------------------------------------ */
function BoneMesh({ vertices, faces, normals, wireframe }) {
    const geometry = useMemo(() => {
        const geo = new THREE.BufferGeometry();

        // Flatten vertices → Float32Array
        const posArr = new Float32Array(vertices.length * 3);
        for (let i = 0; i < vertices.length; i++) {
            posArr[i * 3] = vertices[i][0];
            posArr[i * 3 + 1] = vertices[i][1];
            posArr[i * 3 + 2] = vertices[i][2];
        }
        geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));

        // Flatten faces → index buffer
        const idxArr = new Uint32Array(faces.length * 3);
        for (let i = 0; i < faces.length; i++) {
            idxArr[i * 3] = faces[i][0];
            idxArr[i * 3 + 1] = faces[i][1];
            idxArr[i * 3 + 2] = faces[i][2];
        }
        geo.setIndex(new THREE.BufferAttribute(idxArr, 1));

        // Use provided normals or recompute
        if (normals && normals.length === vertices.length) {
            const nrmArr = new Float32Array(normals.length * 3);
            for (let i = 0; i < normals.length; i++) {
                nrmArr[i * 3] = normals[i][0];
                nrmArr[i * 3 + 1] = normals[i][1];
                nrmArr[i * 3 + 2] = normals[i][2];
            }
            geo.setAttribute("normal", new THREE.BufferAttribute(nrmArr, 3));
        } else {
            geo.computeVertexNormals();
        }

        // Auto-scale so the bone fits in a unit sphere
        geo.computeBoundingSphere();
        const r = geo.boundingSphere.radius || 1;
        const scale = 50 / r;
        geo.scale(scale, scale, scale);
        geo.center();

        return geo;
    }, [vertices, faces, normals]);

    return (
        <mesh geometry={geometry} castShadow receiveShadow>
            <meshPhysicalMaterial
                color="#E8D4B8"
                roughness={0.55}
                metalness={0.05}
                clearcoat={0.25}
                clearcoatRoughness={0.4}
                wireframe={wireframe}
                side={THREE.DoubleSide}
            />
        </mesh>
    );
}

/* ------------------------------------------------------------------ *
 *  Lighting rig – three-point setup for medical visualisation         *
 * ------------------------------------------------------------------ */
function Lights() {
    return (
        <>
            <ambientLight intensity={0.5} color="#f0f0ff" />
            {/* Key light */}
            <directionalLight
                position={[60, 80, 60]}
                intensity={1.6}
                color="#fffaf0"
                castShadow
                shadow-mapSize-width={1024}
                shadow-mapSize-height={1024}
            />
            {/* Fill light */}
            <directionalLight position={[-40, 20, -30]} intensity={0.7} color="#e0e8ff" />
            {/* Rim / back light */}
            <directionalLight position={[0, -30, -60]} intensity={0.9} color="#fff5ee" />
        </>
    );
}

/* ------------------------------------------------------------------ *
 *  Turntable – gentle auto-rotation                                   *
 * ------------------------------------------------------------------ */
function Turntable({ children, enabled }) {
    const ref = useRef();
    useFrame((_, delta) => {
        if (enabled && ref.current) {
            ref.current.rotation.y += delta * 0.15;
        }
    });
    return <group ref={ref}>{children}</group>;
}

/* ------------------------------------------------------------------ *
 *  BoneViewer – public component                                      *
 * ------------------------------------------------------------------ */
export default function BoneViewer({ meshData }) {
    const [wireframe, setWireframe] = useState(false);
    const [autoRotate, setAutoRotate] = useState(true);
    const [bgColor, setBgColor] = useState("#1a1a2e");
    const [isFullscreen, setIsFullscreen] = useState(false);

    // Close fullscreen on Escape key
    useEffect(() => {
        if (!isFullscreen) return;
        const handler = (e) => { if (e.key === "Escape") setIsFullscreen(false); };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [isFullscreen]);

    // Lock body scroll in fullscreen
    useEffect(() => {
        document.body.style.overflow = isFullscreen ? "hidden" : "";
        return () => { document.body.style.overflow = ""; };
    }, [isFullscreen]);

    if (!meshData || !meshData.vertices || meshData.vertices.length === 0) {
        return (
            <div className="w-full h-[500px] bg-gray-900 rounded-xl border border-gray-800 flex items-center justify-center text-gray-600">
                <div className="text-center">
                    <svg className="w-16 h-16 mx-auto mb-3 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9" />
                    </svg>
                    <p className="text-sm">3D preview will appear here</p>
                </div>
            </div>
        );
    }

    const toolbar = (
        <div className="flex items-center gap-3 px-4 py-2 bg-gray-800 text-white text-xs flex-shrink-0">
            <span className="font-semibold mr-auto">3D Bone Viewer</span>
            <span className="text-gray-400">
                {meshData.vertex_count?.toLocaleString()} verts &middot;{" "}
                {meshData.face_count?.toLocaleString()} faces
            </span>
            <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                    type="checkbox"
                    checked={wireframe}
                    onChange={(e) => setWireframe(e.target.checked)}
                    className="accent-primary-500"
                />
                Wireframe
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                    type="checkbox"
                    checked={autoRotate}
                    onChange={(e) => setAutoRotate(e.target.checked)}
                    className="accent-primary-500"
                />
                Rotate
            </label>
            <select
                value={bgColor}
                onChange={(e) => setBgColor(e.target.value)}
                className="bg-gray-700 rounded px-2 py-1 text-xs"
            >
                <option value="#1a1a2e">Dark Blue</option>
                <option value="#111111">Black</option>
                <option value="#f1f5f9">Light</option>
                <option value="#0f172a">Navy</option>
            </select>
            {/* Fullscreen toggle */}
            <button
                onClick={() => setIsFullscreen((v) => !v)}
                title={isFullscreen ? "Exit fullscreen (Esc)" : "Fullscreen"}
                className="ml-1 p-1 rounded hover:bg-gray-700 transition-colors text-gray-300 hover:text-white"
            >
                {isFullscreen ? (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 9V4H4m0 5h5M15 4h5v5m0-5h-5M9 20H4v-5m5 5v-5M20 15v5h-5m5 0h-5" />
                    </svg>
                ) : (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5v-4m0 4h-4m4 0l-5-5" />
                    </svg>
                )}
            </button>
        </div>
    );

    const canvas3d = (
        <Canvas
            camera={{ position: [0, 0, 120], fov: 50, near: 0.1, far: 1000 }}
            shadows
            dpr={[1, 2]}
            gl={{ antialias: true, alpha: false, toneMapping: THREE.ACESFilmicToneMapping }}
        >
            <color attach="background" args={[bgColor]} />
            <Lights />
            <Turntable enabled={autoRotate}>
                <BoneMesh
                    vertices={meshData.vertices}
                    faces={meshData.faces}
                    normals={meshData.normals}
                    wireframe={wireframe}
                />
            </Turntable>
            <OrbitControls
                enableDamping
                dampingFactor={0.08}
                minDistance={30}
                maxDistance={300}
            />
            <Environment preset="studio" intensity={0.3} />
            <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
                <GizmoViewport labelColor="white" axisHeadScale={0.6} />
            </GizmoHelper>
        </Canvas>
    );

    if (isFullscreen) {
        return (
            <div className="fixed inset-0 z-50 flex flex-col" style={{ background: bgColor }}>
                {toolbar}
                <div className="flex-1 min-h-0">
                    {canvas3d}
                </div>
            </div>
        );
    }

    return (
        <div className="w-full rounded-xl overflow-hidden border border-gray-800 shadow-lg">
            {toolbar}
            <div className="w-full h-[500px]" style={{ background: bgColor }}>
                {canvas3d}
            </div>
        </div>
    );
}
