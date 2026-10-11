export default function EmotionDisplay({ emotion, confidence }) {
  if (!emotion) {
    return (
      <div className="emotion-box placeholder">
        Emotion: — &nbsp; Confidence: —
      </div>
    );
  }
  return (
    <div className="emotion-box">
      <span>
        Emotion: <strong>{emotion}</strong>
      </span>
      <span>
        Confidence: <strong>{confidence}</strong>
      </span>
    </div>
  );
}
