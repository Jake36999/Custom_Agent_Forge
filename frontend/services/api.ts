// api.ts
// API service for backend integration (skeleton)
export async function startTraining(mode: string, files: File[], link?: string): Promise<{ job_id: string }> {
  // Implement POST /train
  return { job_id: "mock-job-id" };
}

export async function pollStatus(job_id: string): Promise<{ logs: string[]; status: string; outputPath?: string; stats?: any; error?: string }> {
  // Implement GET /status/{job_id}
  return { logs: ["Pipeline ready."], status: "pending" };
}

export async function downloadOutput(outputPath: string): Promise<Blob> {
  // Implement output download logic
  return new Blob();
}
