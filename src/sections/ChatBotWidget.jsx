import React, { useState, useRef } from "react";

const ChatBotWidget = ({ setTypingState, speakText }) => {
  const [expanded, setExpanded] = useState(false);
  const [chatLog, setChatLog] = useState([]);
  const inputRef = useRef();
  const BACKEND_URL = "http://127.0.0.1:8000";
  const addChatMessage = (sender, text) => {
    
    setChatLog((prev) => [...prev, { sender, text }]);
  };

  const sendMessage = async () => {
    const msg = inputRef.current.value.trim().toLowerCase();
    if (!msg) return;

    addChatMessage("You", msg);
    inputRef.current.value = "";

    if (setTypingState) setTypingState(true); // avatar reacts

    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      const data = await response.json();

      if (setTypingState) setTypingState(false);
      addChatMessage("Bot", data.reply);

      if (speakText) speakText(data.reply); // TTS
    } catch (err) {
      console.error("Chat error:", err);
      if (setTypingState) setTypingState(false);
      addChatMessage("Bot", "Sorry, something went wrong.");
    }
  };

  return (
    <div
      id="chat-widget"
      className={expanded ? "expanded, bg-black-200" : "bg-black-200"}
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
        color : "white"
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
          color: "white",
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
          className="bg-black-200"
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            padding: "10px",
          }}
        >
          <div
            id="chat-log"
            style={{ flex: 1, overflowY: "auto", marginBottom: "10px", overflowX: "scroll" }}
          >
            {chatLog.map((msg, idx) => (
              <div key={idx}>
                <strong>{msg.sender}:</strong> {msg.text}
              </div>
            ))}
          </div>
          <input
           className="bg-black-200"
            type="text"
            id="user-input"
            ref={inputRef}
            placeholder="Type a message..."
            style={{
              padding: "8px",
              marginBottom: "5px",
              boxSizing: "border-box",
            }}
          />
          <button
            id="send-btn"
            onClick={sendMessage}
            className="bg-black-500"
            style={{
              width: "100%",
              padding: "8px",
              color: "white",
              border: "none",
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
