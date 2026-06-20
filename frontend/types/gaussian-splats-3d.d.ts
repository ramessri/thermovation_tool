declare module '@mkkellogg/gaussian-splats-3d' {
  export class Viewer {
    constructor(options: {
      rootElement?: HTMLElement;
      cameraUp?: [number, number, number];
      initialCameraPosition?: [number, number, number];
      initialCameraLookAt?: [number, number, number];
      sharedMemoryForWorkers?: boolean;
      [k: string]: unknown;
    });
    threeScene: any;
    addSplatScene(url: string, opts?: Record<string, unknown>): Promise<void>;
    start(): void;
    stop(): void;
    dispose?(): void;
  }
}
