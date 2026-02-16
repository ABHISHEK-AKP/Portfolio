import React, { useState, useRef, useEffect } from "react";

const ChatBotWidget = ({ setTypingState, speakText }) => {
  const [expanded, setExpanded] = useState(false);
  const [chatLog, setChatLog] = useState([]);
  const inputRef = useRef();
  const BACKEND_URL = "http://127.0.0.1:5000";

  const addChatMessage = (sender, text) => {
    setChatLog((prev) => [...prev, { sender, text }]);
  };

  const sendMessage = () => {
    const msg = inputRef.current.value.trim();
    if (!msg) return;

    addChatMessage("You", msg);
    inputRef.current.value = "";

    if (setTypingState) setTypingState(true);

    // --- SSE Streaming ---
    const botMsgIndex = chatLog.length + 1;
    addChatMessage("Bot", ""); // placeholder for streaming

    const eventSource = new EventSource(`${BACKEND_URL}/stream_chat?message=${encodeURIComponent(msg)}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.done) {
        eventSource.close();
        if (setTypingState) setTypingState(false);
        return;
      }
      if (data.token) {
        setChatLog((prev) => {
          const newLog = [...prev];
          newLog[botMsgIndex] = {
            sender: "Bot",
            text: (newLog[botMsgIndex]?.text || "") + data.token,
          };
          return newLog;
        });
        if (speakText) speakText(data.token); // optional TTS streaming
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      if (setTypingState) setTypingState(false);
      setChatLog((prev) => {
        const newLog = [...prev];
        newLog[botMsgIndex] = {
          sender: "Bot",
          text: "Sorry, something went wrong while streaming.",
        };
        return newLog;
      });
    };
  };

  return (
    <div
      id="chat-widget"
      className={expanded ? "expanded" : ""}
      style={{
        position: "fixed",
        bottom: "20px",
        right: "20px",
        width: expanded ? "400px" : "200px",
        height: expanded ? "600px" : "40px",
        borderRadius: "15px",
        boxShadow: "0 4px 8px rgba(0,0,0,0.3)",
        overflow: "hidden",
        cursor: "pointer",
        transition: "all 0.3s ease",
        zIndex: 9999,
        display: "flex",
        flexDirection: "column",
        color: "white",
        backgroundColor: "#111"
      }}
    >
      {/* Header */}
      <div
        id="chat-header"
        style={{
          width: "100%",
          height: "40px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontWeight: "bold",
        }}
        onClick={() => setExpanded(!expanded)}
      >
        💬 Ask about me
      </div>

      {/* Body */}
      {expanded && (
        <div
          id="chat-body"
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            padding: "10px",
          }}
        >
          <div
            id="chat-log"
            style={{ flex: 1, overflowY: "auto", marginBottom: "10px" }}
          >
            {chatLog.map((msg, idx) => (
              <div key={idx} style={{ marginBottom: "5px" }}>
                <strong>{msg.sender}:</strong> {msg.text}
              </div>
            ))}
          </div>
          <input
            type="text"
            id="user-input"
            ref={inputRef}
            placeholder="Type a message..."
            style={{
              padding: "8px",
              marginBottom: "5px",
              boxSizing: "border-box",
              backgroundColor: "#222",
              color: "white",
              border: "1px solid #333",
              borderRadius: "5px"
            }}
          />
          <button
            id="send-btn"
            onClick={sendMessage}
            style={{
              width: "100%",
              padding: "8px",
              color: "white",
              backgroundColor: "#444",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Send
          </button>
        </div>
      )}
    </div>
  );
};

export default ChatBotWidget;
