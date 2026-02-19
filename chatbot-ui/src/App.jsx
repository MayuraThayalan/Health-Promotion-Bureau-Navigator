import { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:5000/chat";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, isUser: true };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
      if (!response.ok) throw new Error("Server responded with an error");
      
      const data = await response.json();
      setMessages((prev) => [...prev, { text: data.answer, isUser: false }]);
      
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { text: "Error connecting to server.", isUser: false },
      ]);
    } finally {
      setLoading(false);
    }
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
