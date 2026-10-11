export default function LatencyDisplay({ latency }) {
  if (!latency || Object.keys(latency).length === 0) {
    return null;
  }
  return (
    <div className="latency-box">
      {latency.stt_ms !== undefined && <span>STT: {latency.stt_ms}ms</span>}
      {latency.ttft_ms !== undefined && <span>TTFT: {latency.ttft_ms}ms</span>}
      {latency.llm_ms !== undefined && <span>LLM: {latency.llm_ms}ms</span>}
      {latency.tts_ms !== undefined && <span>TTS: {latency.tts_ms}ms</span>}
      {latency.first_audio_ms !== undefined && <span>First audio: {latency.first_audio_ms}ms</span>}
      {latency.total_ms !== undefined && <span>Total: {latency.total_ms}ms</span>}
    </div>
  );
}
