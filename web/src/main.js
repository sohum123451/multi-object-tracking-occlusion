import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { animate, createTimeline } from 'animejs';

// Application State
let metaData = null;
let sortTracks = [];
let deepsortTracks = [];
let idSwitches = [];

let currentFrame = 1;
const maxFrames = 300;
let isPlaying = false;
let playInterval = null;
let is2DView = false;

// Color Palette generator for track IDs
const colorCache = new Map();
function getTrackColor(id) {
  if (colorCache.has(id)) return colorCache.get(id);
  const hue = ((id * 137.508) % 360) / 360;
  const color = new THREE.Color().setHSL(hue, 0.85, 0.55);
  colorCache.set(id, color);
  return color;
}

// Coordinate mapping: image plane (1920x1080) to space-time cube
const CUBE_W = 20.0;
const CUBE_H = 11.25;
const CUBE_D = 20.0;

function toSpaceTime(footX, footY, frame) {
  const x = (footX / 1920.0 - 0.5) * CUBE_W;
  const y = (0.5 - footY / 1080.0) * CUBE_H;
  const z = ((frame - 1) / Math.max(1, maxFrames - 1)) * CUBE_D;
  return new THREE.Vector3(x, y, z);
}

// Viewport setup
function createViewport(containerId) {
  const container = document.getElementById(containerId);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d0d);

  const camera = new THREE.PerspectiveCamera(
    45,
    container.clientWidth / container.clientHeight,
    0.1,
    200
  );
  camera.position.set(18, -14, 32);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 10);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  // Ambient & Directional Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);

  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight1.position.set(20, 20, 40);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
  dirLight2.position.set(-20, -20, 0);
  scene.add(dirLight2);

  // Space-time bounding box wireframe
  const boxGeom = new THREE.BoxGeometry(CUBE_W, CUBE_H, CUBE_D);
  const boxEdges = new THREE.EdgesGeometry(boxGeom);
  const boxMat = new THREE.LineBasicMaterial({ color: 0x333333, linewidth: 1 });
  const boxWireframe = new THREE.LineSegments(boxEdges, boxMat);
  boxWireframe.position.set(0, 0, CUBE_D / 2.0);
  scene.add(boxWireframe);

  // Time Plane setup
  const planeGeom = new THREE.PlaneGeometry(CUBE_W, CUBE_H);
  const textureLoader = new THREE.TextureLoader();
  const backdropTex = textureLoader.load('./backdrop.jpg');
  const planeMat = new THREE.MeshBasicMaterial({
    map: backdropTex,
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const timePlane = new THREE.Mesh(planeGeom, planeMat);
  timePlane.position.set(0, 0, 0);
  scene.add(timePlane);

  // Active track position markers group
  const activeMarkersGroup = new THREE.Group();
  scene.add(activeMarkersGroup);

  // Switch markers group
  const switchMarkersGroup = new THREE.Group();
  scene.add(switchMarkersGroup);

  // Tubes group
  const tubesGroup = new THREE.Group();
  scene.add(tubesGroup);

  return {
    container,
    scene,
    camera,
    renderer,
    controls,
    timePlane,
    planeMat,
    activeMarkersGroup,
    switchMarkersGroup,
    tubesGroup,
    tubesMap: new Map(),
  };
}

const vpSort = createViewport('container-sort');
const vpDeepSort = createViewport('container-deepsort');

// Synchronize OrbitControls between viewports
let isSyncing = false;
function syncControls(sourceVp, targetVp) {
  if (isSyncing) return;
  isSyncing = true;
  targetVp.camera.position.copy(sourceVp.camera.position);
  targetVp.camera.quaternion.copy(sourceVp.camera.quaternion);
  targetVp.controls.target.copy(sourceVp.controls.target);
  targetVp.controls.update();
  isSyncing = false;
}

vpSort.controls.addEventListener('change', () => syncControls(vpSort, vpDeepSort));
vpDeepSort.controls.addEventListener('change', () => syncControls(vpDeepSort, vpSort));

// Build Track Tubes
function buildTrackTubes(vp, tracks) {
  tracks.forEach(track => {
    const rawPoints = track.frames.map((f, i) =>
      toSpaceTime(track.foot_x[i], track.foot_y[i], f)
    );

    let points = rawPoints;
    if (points.length === 1) {
      // Single frame track: create short segment
      const p2 = points[0].clone();
      p2.z += 0.08;
      points = [points[0], p2];
    }

    const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal');
    const tubularSegments = Math.max(Math.min(points.length * 3, 300), 12);
    const geom = new THREE.TubeGeometry(curve, tubularSegments, 0.08, 8, false);

    const baseColor = getTrackColor(track.id);
    const mat = new THREE.MeshStandardMaterial({
      color: baseColor,
      roughness: 0.35,
      metalness: 0.15,
      transparent: true,
      opacity: 0.85,
    });

    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData = {
      trackId: track.id,
      frames: track.frames,
      footX: track.foot_x,
      footY: track.foot_y,
      baseColor: baseColor,
    };

    vp.tubesGroup.add(mesh);
    vp.tubesMap.set(track.id, mesh);
  });
}

// Build ID Switch Markers
function buildSwitchMarkers(vp, trackerName) {
  const switches = idSwitches.filter(s => s.tracker === trackerName);
  const ringGeom = new THREE.RingGeometry(0.2, 0.35, 18);

  switches.forEach(sw => {
    const pos = toSpaceTime(sw.foot_x, sw.foot_y, sw.frame);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xff4d4f,
      side: THREE.DoubleSide,
      depthTest: false,
    });
    const marker = new THREE.Mesh(ringGeom, ringMat);
    marker.position.copy(pos);
    marker.userData = sw;
    vp.switchMarkersGroup.add(marker);
  });
}

// Keyframe texture cache
const keyframeTextures = new Map();
const textureLoader = new THREE.TextureLoader();

function getKeyframeTexture(frame) {
  const step = (metaData && metaData.keyframe_step) || 5;
  const nearest = Math.max(1, Math.min(maxFrames, Math.round((frame - 1) / step) * step + 1));
  const path = `./keyframes/frame_${String(nearest).padStart(4, '0')}.jpg`;

  if (!keyframeTextures.has(path)) {
    const tex = textureLoader.load(path);
    keyframeTextures.set(path, tex);
  }
  return keyframeTextures.get(path);
}

// Update Active Frame State
function updateFrame(frame) {
  currentFrame = Math.max(1, Math.min(maxFrames, frame));
  document.getElementById('frame-slider').value = currentFrame;
  document.getElementById('frame-indicator').textContent = `${String(currentFrame).padStart(3, '0')} / ${maxFrames}`;

  const zPos = ((currentFrame - 1) / Math.max(1, maxFrames - 1)) * CUBE_D;
  const tex = getKeyframeTexture(currentFrame);

  [vpSort, vpDeepSort].forEach(vp => {
    vp.timePlane.position.z = zPos;
    if (tex && vp.planeMat.map !== tex) {
      vp.planeMat.map = tex;
      vp.planeMat.needsUpdate = true;
    }

    // Update active tracks lighting and active position dots
    while (vp.activeMarkersGroup.children.length > 0) {
      const child = vp.activeMarkersGroup.children[0];
      vp.activeMarkersGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }

    let activeCount = 0;
    vp.tubesMap.forEach((mesh, trackId) => {
      const frames = mesh.userData.frames;
      const idx = frames.indexOf(currentFrame);
      if (idx !== -1) {
        activeCount++;
        mesh.material.opacity = 1.0;
        mesh.material.emissive = mesh.userData.baseColor;
        mesh.material.emissiveIntensity = 0.35;

        // Add position dot on the time plane
        const footX = mesh.userData.footX[idx];
        const footY = mesh.userData.footY[idx];
        const dotPos = toSpaceTime(footX, footY, currentFrame);

        const sphereGeom = new THREE.SphereGeometry(0.18, 12, 12);
        const sphereMat = new THREE.MeshBasicMaterial({ color: mesh.userData.baseColor });
        const sphere = new THREE.Mesh(sphereGeom, sphereMat);
        sphere.position.copy(dotPos);
        vp.activeMarkersGroup.add(sphere);
      } else {
        mesh.material.opacity = 0.35;
        mesh.material.emissiveIntensity = 0.0;
      }
    });

    const statId = vp === vpSort ? 'stat-sort' : 'stat-deepsort';
    document.getElementById(statId).textContent = `Active: ${activeCount} / Total: ${vp.tubesMap.size}`;
  });
}

// Raycasting for clicking switch markers
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function setupRaycasting(vp, trackerName) {
  vp.renderer.domElement.addEventListener('click', (event) => {
    const rect = vp.container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, vp.camera);
    const intersects = raycaster.intersectObjects(vp.switchMarkersGroup.children);
    if (intersects.length > 0) {
      const sw = intersects[0].object.userData;
      triggerSwitchEvent(sw);
    }
  });
}

// Highlight tubes and fly camera to switch event
let highlightedMeshes = [];
function clearHighlights() {
  highlightedMeshes.forEach(mesh => {
    mesh.material.emissive = mesh.userData.baseColor;
    mesh.material.emissiveIntensity = 0.0;
    mesh.scale.set(1, 1, 1);
  });
  highlightedMeshes = [];
}

function triggerSwitchEvent(sw) {
  clearHighlights();

  // Update frame scrubber
  updateFrame(sw.frame);

  // Update details card
  document.getElementById('detail-badge').className = 'badge-active';
  document.getElementById('detail-badge').textContent = `${sw.tracker.toUpperCase()} Switch Detected`;
  document.getElementById('det-tracker').textContent = sw.tracker.toUpperCase();
  document.getElementById('det-frame').textContent = sw.frame;
  document.getElementById('det-gt').textContent = `GT ID #${sw.gt_id}`;
  document.getElementById('det-pred').textContent = `#${sw.old_pred_id} -> #${sw.new_pred_id}`;
  document.getElementById('det-causes').textContent = sw.causes || 'other';
  document.getElementById('det-vis').textContent = sw.visibility !== undefined ? sw.visibility.toFixed(2) : '-';
  document.getElementById('det-iou').textContent = sw.max_gt_iou !== undefined ? sw.max_gt_iou.toFixed(2) : '-';
  document.getElementById('det-speed').textContent = sw.speed !== undefined ? `${sw.speed.toFixed(1)} px/f` : '-';
  document.getElementById('det-cossim').textContent = sw.cos_sim !== undefined ? sw.cos_sim.toFixed(3) : '-';

  // Highlight both involved tubes in corresponding viewport
  const targetVp = sw.tracker === 'sort' ? vpSort : vpDeepSort;
  const tubeOld = targetVp.tubesMap.get(sw.old_pred_id);
  const tubeNew = targetVp.tubesMap.get(sw.new_pred_id);

  [tubeOld, tubeNew].forEach(mesh => {
    if (mesh) {
      mesh.material.opacity = 1.0;
      mesh.material.emissive = new THREE.Color(0xff4d4f);
      mesh.material.emissiveIntensity = 0.8;
      highlightedMeshes.push(mesh);
    }
  });

  // Camera fly-to using anime.js v4 timeline (900ms, outExpo)
  const swPos = toSpaceTime(sw.foot_x, sw.foot_y, sw.frame);
  const targetCamPos = new THREE.Vector3(
    swPos.x + 4.5,
    swPos.y - 3.5,
    swPos.z + 5.5
  );

  const tl = createTimeline({
    defaults: { duration: 900, ease: 'outExpo' },
  });

  tl.add(vpSort.camera.position, {
    x: targetCamPos.x,
    y: targetCamPos.y,
    z: targetCamPos.z,
  }, 0);

  tl.add(vpSort.controls.target, {
    x: swPos.x,
    y: swPos.y,
    z: swPos.z,
  }, 0);

  tl.add(vpDeepSort.camera.position, {
    x: targetCamPos.x,
    y: targetCamPos.y,
    z: targetCamPos.z,
  }, 0);

  tl.add(vpDeepSort.controls.target, {
    x: swPos.x,
    y: swPos.y,
    z: swPos.z,
  }, 0);
}

// Toggle between 3D Space-Time and Top-Down 2D View
function toggleView() {
  is2DView = !is2DView;
  const btn = document.getElementById('btn-view-toggle');
  btn.textContent = is2DView ? '3D Space-Time View' : 'Top-Down 2D View';

  const tl = createTimeline({
    defaults: { duration: 900, ease: 'outExpo' },
  });

  if (is2DView) {
    // Top-down 2D image plane view
    tl.add(vpSort.camera.position, { x: 0, y: 0, z: 32 }, 0);
    tl.add(vpSort.controls.target, { x: 0, y: 0, z: 10 }, 0);
    tl.add(vpDeepSort.camera.position, { x: 0, y: 0, z: 32 }, 0);
    tl.add(vpDeepSort.controls.target, { x: 0, y: 0, z: 10 }, 0);
  } else {
    // 3D perspective isometric view
    tl.add(vpSort.camera.position, { x: 18, y: -14, z: 32 }, 0);
    tl.add(vpSort.controls.target, { x: 0, y: 0, z: 10 }, 0);
    tl.add(vpDeepSort.camera.position, { x: 18, y: -14, z: 32 }, 0);
    tl.add(vpDeepSort.controls.target, { x: 0, y: 0, z: 10 }, 0);
  }
}

// Populate switch dropdown
function populateSwitchDropdown(switches) {
  const select = document.getElementById('switch-select');
  select.innerHTML = '<option value="">-- Select Switch Event --</option>';

  switches.forEach((sw, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = `[F${String(sw.frame).padStart(3, '0')}] ${sw.tracker.toUpperCase()} GT#${sw.gt_id} (P#${sw.old_pred_id}->#${sw.new_pred_id}) - ${sw.causes}`;
    select.appendChild(opt);
  });

  select.addEventListener('change', () => {
    const val = select.value;
    if (val !== '') {
      triggerSwitchEvent(switches[parseInt(val, 10)]);
    }
  });

  document.getElementById('btn-prev-switch').addEventListener('click', () => {
    if (select.selectedIndex > 1) {
      select.selectedIndex -= 1;
      triggerSwitchEvent(switches[parseInt(select.value, 10)]);
    }
  });

  document.getElementById('btn-next-switch').addEventListener('click', () => {
    if (select.selectedIndex < switches.length) {
      select.selectedIndex += 1;
      triggerSwitchEvent(switches[parseInt(select.value, 10)]);
    }
  });
}

// Scrubber Controls
const slider = document.getElementById('frame-slider');
slider.addEventListener('input', (e) => {
  updateFrame(parseInt(e.target.value, 10));
});

const btnPlay = document.getElementById('btn-play');
btnPlay.addEventListener('click', () => {
  isPlaying = !isPlaying;
  btnPlay.textContent = isPlaying ? 'Pause' : 'Play';
  if (isPlaying) {
    playInterval = setInterval(() => {
      let nextF = currentFrame + 1;
      if (nextF > maxFrames) nextF = 1;
      updateFrame(nextF);
    }, 40);
  } else {
    clearInterval(playInterval);
  }
});

document.getElementById('btn-view-toggle').addEventListener('click', toggleView);

document.getElementById('btn-reset-cam').addEventListener('click', () => {
  const tl = createTimeline({ defaults: { duration: 800, ease: 'outExpo' } });
  tl.add(vpSort.camera.position, { x: 18, y: -14, z: 32 }, 0);
  tl.add(vpSort.controls.target, { x: 0, y: 0, z: 10 }, 0);
  tl.add(vpDeepSort.camera.position, { x: 18, y: -14, z: 32 }, 0);
  tl.add(vpDeepSort.controls.target, { x: 0, y: 0, z: 10 }, 0);
});

// Resize handler
window.addEventListener('resize', () => {
  [vpSort, vpDeepSort].forEach(vp => {
    const w = vp.container.clientWidth;
    const h = vp.container.clientHeight;
    vp.camera.aspect = w / h;
    vp.camera.updateProjectionMatrix();
    vp.renderer.setSize(w, h);
  });
});

// Animation Loop
function animateLoop() {
  requestAnimationFrame(animateLoop);
  vpSort.controls.update();
  vpDeepSort.controls.update();
  vpSort.renderer.render(vpSort.scene, vpSort.camera);
  vpDeepSort.renderer.render(vpDeepSort.scene, vpDeepSort.camera);
}
animateLoop();

// Initialize and Load Data
async function initApp() {
  try {
    const [metaRes, sortRes, deepsortRes, switchRes] = await Promise.all([
      fetch('./meta.json').then(r => r.json()),
      fetch('./tracks_sort.json').then(r => r.json()),
      fetch('./tracks_deepsort.json').then(r => r.json()),
      fetch('./id_switches.json').then(r => r.json()),
    ]);

    metaData = metaRes;
    sortTracks = sortRes;
    deepsortTracks = deepsortRes;
    idSwitches = switchRes;

    buildTrackTubes(vpSort, sortTracks);
    buildTrackTubes(vpDeepSort, deepsortTracks);

    buildSwitchMarkers(vpSort, 'sort');
    buildSwitchMarkers(vpDeepSort, 'deepsort');

    setupRaycasting(vpSort, 'sort');
    setupRaycasting(vpDeepSort, 'deepsort');

    populateSwitchDropdown(idSwitches);
    updateFrame(1);
  } catch (err) {
    console.error('Failed to load tracking visualizer data:', err);
  }
}

initApp();
