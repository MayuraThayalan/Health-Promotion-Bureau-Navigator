import { useState } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = () => {
    if (!input.trim()) return;

    // Add user message right side
    setMessages([...messages, { text: input, isUser: true }]);
    setInput("");
    setLoading(true);

    setTimeout(() => {
      const reply = `You asked: "${input}"\n\n`;
      setMessages((prev) => [...prev, { text: reply, isUser: false }]);
      setLoading(false);
    }, 900);
  };

  return (
    <div className="chat-page">
      <div className="chat-header">Health Promotion Bureau Navigator</div>

      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.isUser ? "user-message" : "bot-message"}`}
          >
            {msg.text}
          </div>
        ))}

        {loading && <div className="message bot-message">Typing...</div>}
      </div>

      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask about health, symptoms, prevention..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
