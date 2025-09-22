import React from "react";
import { exp } from "three/tsl";
const ChatBotWidget = () => {
    return (
        <div id="chat-widget">
        <div id="chat-header">💬 Chat</div>
        <div id="chat-body">
            <div id="chat-log"></div>
            <input type="text" id="user-input" placeholder="Type a message..." />
            <button id="send-btn">Send</button>
        </div>
      </div>
    ) 
}
export default ChatBotWidget;