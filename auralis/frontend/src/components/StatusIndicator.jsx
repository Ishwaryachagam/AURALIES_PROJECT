export default function StatusIndicator({ connectionState, status }) {
  const connected = connectionState === "connected";
  const statusLabel =
    status === "speaking" ? "🔊 AI Speaking" : status === "listening" ? "🎤 Listening" : "Idle";

  return (
    <div className="status-bar">
      <div className="status-item">
        <span className={`dot ${connected ? "active" : ""}`} />
        Connection: {connectionState || "disconnected"}
      </div>
      <div className="status-item">
        <span className={`dot ${status === "speaking" ? "speaking" : status === "listening" ? "active" : ""}`} />
        Status: {statusLabel}
      </div>
    </div>
  );
}
