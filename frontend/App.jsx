import { useState } from "react";

function App() {
  const [message, setMessage] = useState("");

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 flex">
      {/* Sidebar */}
      <aside className="w-64 border-r bg-white p-4 hidden md:block">
        <h1 className="text-xl font-semibold mb-6">DhronAI</h1>

        <button className="w-full rounded-lg border px-4 py-2 text-left hover:bg-gray-50">
          + New conversation
        </button>

        <div className="mt-6 text-sm text-gray-500">
          Conversations
        </div>
      </aside>

      {/* Chat Panel */}
      <main className="flex-1 flex flex-col">
        <header className="border-b bg-white px-6 py-4">
          <h2 className="font-semibold">Legal Assistant</h2>
        </header>

        <section className="flex-1 flex items-center justify-center p-6">
          <div className="max-w-2xl w-full text-center">
            <h2 className="text-3xl font-semibold mb-3">
              Ask DhronAI
            </h2>

            <p className="text-gray-500 mb-8">
              Ask questions about the Bharatiya Nagarik Suraksha Sanhita.
            </p>

            <div className="grid gap-3 sm:grid-cols-2 text-left">
              <button className="border rounded-lg p-4 bg-white hover:bg-gray-50">
                What are the provisions for arrest?
              </button>

              <button className="border rounded-lg p-4 bg-white hover:bg-gray-50">
                What is the procedure for issuing a warrant?
              </button>

              <button className="border rounded-lg p-4 bg-white hover:bg-gray-50">
                What are the provisions related to bail?
              </button>

              <button className="border rounded-lg p-4 bg-white hover:bg-gray-50">
                Explain the relevant section in simple terms.
              </button>
            </div>
          </div>
        </section>

        <div className="border-t bg-white p-4">
          <div className="max-w-3xl mx-auto flex gap-2">
            <input
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Ask a question..."
              className="flex-1 rounded-lg border px-4 py-3 outline-none focus:ring-2 focus:ring-gray-300"
            />

            <button
              className="rounded-lg bg-gray-900 text-white px-5 py-3"
              disabled={!message.trim()}
            >
              Send
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;