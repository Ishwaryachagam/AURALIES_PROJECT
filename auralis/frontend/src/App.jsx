import { useState, useRef, useCallback } from "react";
import { AuralisClient } from "./services/webrtc";
import StatusIndicator from "./components/StatusIndicator";
import EmotionDisplay from "./components/EmotionDisplay";
import Conversation from "./components/Conversation";
import LatencyDisplay from "./components/LatencyDisplay";
import Waveform from "./components/Waveform";
import AudioControls from "./components/AudioControls";

export default function App() {
  const [connectionState, setConnectionState] = useState("disconnected");
  const [status, setStatus] = useState("idle"); // idle | listening | speaking
  const [liveTranscript, setLiveTranscript] = useState("");
  const [liveResponse, setLiveResponse] = useState("");
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [latency, setLatency] = useState({});
  const [history, setHistory] = useState([]);
  const [localStream, setLocalStream] = useState(null);
  const [remoteStream, setRemoteStream] = useState(null);
  const [error, setError] = useState(null);

  const clientRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const pendingUserTextRef = useRef("");
  const pendingEmotionRef = useRef("");

  const handleMessage = useCallback((data) => {
    switch (data.type) {
      case "status":
        setStatus(data.status);
        break;
      case "speech_start":
        setLiveTranscript("Listening...");
        setLiveResponse("");
        break;
      case "speech_end":
        setLiveTranscript((t) => (t === "Listening..." ? "Transcribing..." : t));
        break;
      case "transcript":
        if (data.note === "empty_or_unclear") {
          setLiveTranscript("(couldn't understand - try again)");
        } else {
          setLiveTranscript(data.text);
          pendingUserTextRef.current = data.text;
        }
        break;
      case "emotion":
        setEmotion(data.emotion);
        setConfidence(data.confidence);
        pendingEmotionRef.current = data.emotion;
        break;
      case "llm_response":
        setLiveResponse(data.text);
        setHistory((h) => [
          ...h,
          { user: pendingUserTextRef.current, assistant: data.text, emotion: pendingEmotionRef.current },
        ]);
        break;
      case "latency":
        setLatency(data);
        break;
      case "interrupted":
        setStatus("listening");
        break;
      case "error":
        setError(data.message);
        break;
      default:
        console.log("Unhandled message:", data);
    }
  }, []);

  const handleStart = async () => {
    setError(null);
    const client = new AuralisClient({
      onMessage: handleMessage,
      onRemoteStream: (stream) => {
        setRemoteStream(stream);
        if (remoteAudioRef.current) {
          remoteAudioRef.current.srcObject = stream;
        }
      },
      onConnectionStateChange: setConnectionState,
    });

    try {
      await client.connect();
      clientRef.current = client;
      setLocalStream(client.localStream);
      setStatus("listening");
    } catch (err) {
      setError(err.message);
    }
  };

  const handleStop = () => {
    clientRef.current?.disconnect();
    clientRef.current = null;
    setConnectionState("disconnected");
    setStatus("idle");
    setLocalStream(null);
    setRemoteStream(null);
  };

  const handleInterrupt = () => {
    clientRef.current?.sendInterrupt();
  };

  return (
    <div className="app">
      <header>
        <h1>AURALIS</h1>
        <p className="subtitle">Real-Time Voice-to-Voice AI Engine</p>
      </header>

      <StatusIndicator connectionState={connectionState} status={status} />
      <EmotionDisplay emotion={emotion} confidence={confidence} />

      {localStream && <Waveform stream={localStream} color="#35d07f" label="You" />}
      {remoteStream && <Waveform stream={remoteStream} color="#7fb8ff" label="Auralis" />}

      <Conversation liveTranscript={liveTranscript} liveResponse={liveResponse} history={history} />

      <LatencyDisplay latency={latency} />

      <AudioControls
        connected={connectionState === "connected"}
        onStart={handleStart}
        onStop={handleStop}
        onInterrupt={handleInterrupt}
        aiSpeaking={status === "speaking"}
      />

      {error && <div className="error-box">Error: {error}</div>}

      <audio ref={remoteAudioRef} autoPlay />
    </div>
  );
}
