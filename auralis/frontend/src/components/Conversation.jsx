export default function Conversation({ liveTranscript, liveResponse, history }) {
  return (
    <div className="conversation-box">
      <div className="live-turn">
        <div className="user-line">
          <span className="label">USER</span>
          <p>{liveTranscript || "—"}</p>
        </div>
        <div className="ai-line">
          <span className="label">AURALIS</span>
          <p>{liveResponse || "—"}</p>
        </div>
      </div>

      {history.length > 0 && (
        <div className="history">
          <h4>History</h4>
          {history
            .slice()
            .reverse()
            .map((turn, i) => (
              <div key={i} className="history-entry">
                <div className="history-user">You: {turn.user}</div>
                <div className="history-ai">
                  Auralis ({turn.emotion || "unknown"}): {turn.assistant}
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
