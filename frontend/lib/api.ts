import axios from "axios";

/**
 * Derive API base URL at runtime from the browser's current hostname.
 * This makes the same build work from any host (localhost, LAN IP, domain)
 * without needing a rebuild or env-var change.
 *
 * Falls back to NEXT_PUBLIC_API_URL (build-time) for SSR, then to localhost.
 */
function getApiBase(): string {
  if (typeof window !== "undefined") {
    // API is always on port 8000 of the same host the user is browsing from
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
}

function getWsBase(): string {
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
}

export const API_BASE = getApiBase();

export function storageUrl(key: string) {
  return `${getApiBase()}/files/${key}`;
}

const api = axios.create({
  baseURL: getApiBase(),
});

// ── Projects ──────────────────────────────────────────────────────────────────

export const createProject = (data: {
  name: string;
  description?: string;
  scene_type?: "indoor_room" | "outdoor" | "object";
}) => api.post("/api/projects/", data).then((r) => r.data);

export const listProjects = () =>
  api.get("/api/projects/").then((r) => r.data);

export const getProject = (id: string) =>
  api.get(`/api/projects/${id}`).then((r) => r.data);

/** Upload any media file (video or image) to a project. */
export const uploadFile = (
  projectId: string,
  file: File,
  onProgress?: (pct: number) => void,
) => {
  const form = new FormData();
  form.append("file", file);
  return api
    .post(`/api/projects/${projectId}/uploads`, form, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (e) => {
        if (onProgress && e.total)
          onProgress(Math.round((e.loaded / e.total) * 100));
      },
    })
    .then((r) => r.data as { upload_id: string; storage_key: string; filename: string; size_bytes: number });
};

/** @deprecated Use uploadFile */
export const uploadVideo = uploadFile;

export const uploadCalibrationPhoto = (projectId: string, file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api
    .post(`/api/projects/${projectId}/calibration_photo`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((r) => r.data as {
      focal_length_px?: number;
      focal_length_source?: string;
      make?: string;
      model?: string;
    });
};

export const launchPipeline = (projectId: string, uploadId: string, mode: 'standard' | 'scout' = 'standard') =>
  api
    .post(`/api/projects/${projectId}/launch`, null, {
      params: { upload_id: uploadId, mode },
    })
    .then((r) => r.data);

// ── Supplemental pipeline ─────────────────────────────────────────────────────

export const launchSupplemental = (projectId: string, uploadId: string) =>
  api
    .post(`/api/projects/${projectId}/launch_supplemental`, null, {
      params: { upload_id: uploadId },
    })
    .then((r) => r.data);

export const renameProject = (id: string, name: string, description?: string) =>
  api.patch(`/api/projects/${id}`, { name, description }).then((r) => r.data);

export interface ProjectUpload {
  id: string;
  filename: string;
  storage_key: string;
  mime_type: string;
  size_bytes: number;
  uploaded_at: string | null;
}

export const listProjectUploads = (id: string) =>
  api.get<ProjectUpload[]>(`/api/projects/${id}/uploads`).then((r) => r.data);

// ── Project management ────────────────────────────────────────────────────────

export const deleteProject = (id: string) =>
  api.delete(`/api/projects/${id}`).then((r) => r.data);

export const cancelPipeline = (id: string) =>
  api.post(`/api/projects/${id}/cancel`).then((r) => r.data);

export const bulkDeleteProjects = (ids: string[]) =>
  api.post("/api/projects/bulk-delete", { ids }).then((r) => r.data);

export const deleteFailedProjects = () =>
  api.delete("/api/projects/").then((r) => r.data);

// ── Jobs ──────────────────────────────────────────────────────────────────────

export const getJobStatus = (taskId: string) =>
  api.get(`/api/jobs/${taskId}`).then((r) => r.data);

export async function getJobResult(taskId: string): Promise<{
  task_id: string;
  status: string;
  result: {
    frame_keys?: string[];
    image_keys?: string[];
    sparse_cloud_key?: string;
    camera_poses_key?: string;
    dense_cloud_key?: string;
    scaled_cloud_key?: string;
    confirmed_scale_factor?: number;
    gravity_up_world?: number[];
    scale_diagnostics?: { scale_factor?: number; scale_std?: number; scale_n_inliers?: number; [key: string]: unknown };
    exports?: Array<{ label: string; key: string; mime_type: string }>;
    coverage_cloud_key?: string;
    coverage_score?: number;
    suggestions?: any[];
    n_points?: number;
    n_low_coverage?: number;
    point_density?: number;
    scene_volume?: number;
    // SfM stage metrics
    registered_images?: number;
    mean_reprojection_error?: number;
    num_points3D?: number;
    // MVS stage metrics
    dense_point_count?: number;
    runtime_seconds?: number;
  } | null;
}> {
  const res = await fetch(`${API_BASE}/api/jobs/${taskId}`);
  if (!res.ok) throw new Error(`Job fetch failed: ${res.status}`);
  return res.json();
}

// ── WebSocket progress ────────────────────────────────────────────────────────

interface ProgressMessage {
  stage: string;
  progress: number;
  message: string;
}

export function connectProgress(
  projectId: string,
  onMessage: (data: ProgressMessage) => void,
  onError?: (error: string) => void,
  onClose?: () => void,
): () => void {
  const wsUrl = `${getWsBase()}/ws/projects/${projectId}/progress`;
  let ws: WebSocket | null = null;
  let reconnectAttempts = 0;
  const maxReconnectAttempts = 3;
  let isClosed = false;

  function connect() {
    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        reconnectAttempts = 0;
        console.log("[WS] Connected to progress stream");
      };

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data) as ProgressMessage;
          onMessage(data);
        } catch (err) {
          console.error("[WS] Failed to parse message:", err);
        }
      };

      ws.onerror = () => {
        const msg = `WebSocket connection error`;
        console.error("[WS] Error:", msg);
        if (onError) onError(msg);
      };

      ws.onclose = () => {
        if (isClosed) return;
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts - 1), 10000);
          console.log(
            `[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`,
          );
          setTimeout(connect, delay);
        } else {
          console.error("[WS] Max reconnect attempts reached");
          if (onClose) onClose();
        }
      };
    } catch (err) {
      console.error("[WS] Connection failed:", err);
      if (onError) onError("Failed to connect");
    }
  }

  connect();

  return () => {
    isClosed = true;
    if (ws) ws.close();
  };
}
