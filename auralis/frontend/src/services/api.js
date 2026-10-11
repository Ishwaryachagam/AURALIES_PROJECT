const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export async function checkHealth() {
  const res = await fetch(`${BACKEND_URL}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export { BACKEND_URL };
