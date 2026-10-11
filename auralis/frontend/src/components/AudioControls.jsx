export default function AudioControls({ connected, onStart, onStop, onInterrupt, aiSpeaking }) {
  return (
    <div className="controls">
      <button className="start-btn" onClick={onStart} disabled={connected}>
        Start Conversation
      </button>
      <button className="stop-btn" onClick={onStop} disabled={!connected}>
        Stop Conversation
      </button>
      <button className="interrupt-btn" onClick={onInterrupt} disabled={!connected || !aiSpeaking}>
        Interrupt
      </button>
    </div>
  );
}
