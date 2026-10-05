import * as THREE from "https://esm.sh/three@0.160.0";
import { OrbitControls } from "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "https://esm.sh/three@0.160.0/examples/jsm/loaders/PLYLoader.js";
import URDFLoader from "https://esm.sh/urdf-loader@0.12.3?deps=three@0.160.0";

const DEFAULT_MATERIAL = new THREE.MeshStandardMaterial({
  color: 0xd8dde5,
  roughness: 0.58,
  metalness: 0.02,
  side: THREE.DoubleSide,
});

THREE.Cache.enabled = true;
const DEFAULT_ROOT_ROTATION_X_RAD = Math.PI;

class GHandUrdfViewer {
  constructor({ containerId, urdfUrl, rootRotationXRad = DEFAULT_ROOT_ROTATION_X_RAD }) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      throw new Error(`URDF viewer container not found: ${containerId}`);
    }

    this.urdfUrl = new URL(urdfUrl, window.location.href).toString();
    const parsedRootRotation = Number(rootRotationXRad);
    this.rootRotationXRad = Number.isFinite(parsedRootRotation)
      ? parsedRootRotation
      : DEFAULT_ROOT_ROTATION_X_RAD;
    this.robot = null;
    this.meshErrors = [];

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf8fafc);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.001, 100);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(1, 1);

    const existingStatus = this.container.querySelector(".ghand-urdf-viewer-status");
    this.container.innerHTML = "";
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    this.status = existingStatus || document.createElement("div");
    this.status.className = "ghand-urdf-viewer-status";
    this.setStatus("Loading URDF...");
    this.container.appendChild(this.status);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x64748b, 2.0));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
    keyLight.position.set(0.3, -0.5, 0.8);
    this.scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xffffff, 1.2);
    fillLight.position.set(-0.4, 0.4, 0.5);
    this.scene.add(fillLight);

    const grid = new THREE.GridHelper(0.24, 12, 0xcbd5e1, 0xe2e8f0);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
    this.animate();
  }

  async load() {
    this.robot = await this.loadRobot();
    this.robot.name = this.robot.robotName || "ghand";
    this.robot.rotation.x = this.rootRotationXRad;
    this.scene.add(this.robot);

    this.prepareRobotMeshes();
    this.robot.updateMatrixWorld(true);
    this.frameRobot();
    this.setStatus(this.meshErrors.length ? `Loaded with ${this.meshErrors.length} mesh error(s)` : "Measured pose");
  }

  loadRobot() {
    return new Promise((resolve, reject) => {
      let loadedRobot = null;
      const manager = new THREE.LoadingManager();
      const loader = new URDFLoader(manager);

      manager.onProgress = (_url, loaded, total) => {
        if (total > 0) {
          this.setStatus(`Loading model ${loaded}/${total}`);
        }
      };
      manager.onError = (url) => {
        this.meshErrors.push(url);
      };
      manager.onLoad = () => {
        if (loadedRobot) {
          resolve(loadedRobot);
        }
      };

      loader.loadMeshCb = (path, meshManager, done) => this.loadMesh(path, meshManager, done);
      loader.load(
        this.urdfUrl,
        (robot) => {
          loadedRobot = robot;
        },
        undefined,
        reject,
      );
    });
  }

  loadMesh(path, manager, done) {
    if (!/\.ply(?:$|[?#])/i.test(path)) {
      done(null, new Error(`Unsupported mesh format: ${path}`));
      return;
    }

    const loader = new PLYLoader(manager);
    loader.load(
      path,
      (geometry) => {
        if (!geometry.attributes.normal) {
          geometry.computeVertexNormals();
        }
        geometry.computeBoundingBox();
        done(new THREE.Mesh(geometry, DEFAULT_MATERIAL.clone()));
      },
      undefined,
      (error) => done(null, error),
    );
  }

  prepareRobotMeshes() {
    this.robot.traverse((object) => {
      if (!object.isMesh) return;

      object.castShadow = false;
      object.receiveShadow = false;
      if (object.geometry && !object.geometry.attributes.normal) {
        object.geometry.computeVertexNormals();
      }

      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        if (!material) continue;
        material.side = THREE.DoubleSide;
        material.needsUpdate = true;
      }
    });
  }

  setJointValues(values) {
    if (!this.robot) return;

    const resolved = {};
    for (const [name, angle] of Object.entries(values || {})) {
      resolved[name] = Number(angle) || 0;
    }

    if (typeof this.robot.setJointValues === "function") {
      this.robot.setJointValues(resolved);
    } else {
      for (const [name, angle] of Object.entries(resolved)) {
        this.robot.setJointValue?.(name, angle);
      }
    }
    this.robot.updateMatrixWorld(true);
  }

  frameRobot() {
    const box = new THREE.Box3().setFromObject(this.robot);
    if (box.isEmpty()) {
      this.camera.position.set(0.24, 0.36, 0.24);
      this.controls.target.set(0, 0, 0);
      return;
    }

    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);
    const radius = Math.max(size.x, size.y, size.z, 0.08);
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(radius * 1.4, radius * -2.0, radius * 1.3));
    this.camera.near = Math.max(radius / 200, 0.0005);
    this.camera.far = Math.max(radius * 20, 2.0);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resize() {
    const rect = this.container.getBoundingClientRect();
    const width = Math.max(Math.floor(rect.width), 1);
    const height = Math.max(Math.floor(rect.height), 1);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  setStatus(text) {
    this.status.textContent = text;
  }
}

window.GHandUrdfViewer = {
  async create(options) {
    const viewer = new GHandUrdfViewer(options);
    await viewer.load();
    return viewer;
  },
};
