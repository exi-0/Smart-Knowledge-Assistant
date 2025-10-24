from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pandas as pd
import os

from smart_knowledge_assistant import (
    load_all_data,
    embed_and_index_texts,
    retrieve_relevant_docs,
    generate_response,
    embedding_model,
    kb_index
)

app = FastAPI()
chat_memory = []

DATA_PATH = "data/employee_data.xlsx"

# === Load and index all data at startup ===
@app.on_event("startup")
def startup_event():
    print("Loading and embedding data...")
    all_texts = load_all_data()
    embed_and_index_texts(all_texts)
    print("Assistant is ready!")

# === Chat Endpoint ===
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(chat: ChatRequest):
    global chat_memory
    memory_context = chat_memory[-3:]
    docs = retrieve_relevant_docs(chat.query)
    context = memory_context + docs
    answer = generate_response(chat.query, context)

    chat_memory.append(chat.query)
    chat_memory.append(answer)
    chat_memory = chat_memory[-6:]

    return {"answer": answer}

# === Employee Input Model ===
class Employee(BaseModel):
    employee_id: str
    name: str
    email: str
    department: str
    joining_date: str
    role: str

@app.post("/add-employee")
def add_employee(employee: Employee):
    file_path = DATA_PATH

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=["employee_id", "name", "email", "department", "joining_date", "role"])

    new_entry = {
        "employee_id": employee.employee_id,
        "name": employee.name,
        "email": employee.email,
        "department": employee.department,
        "joining_date": employee.joining_date,
        "role": employee.role
    }

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_excel(file_path, index=False)

    # Re-index new entry
    text = " | ".join(new_entry.values())
    embedding = embedding_model.encode([text])[0].tolist()
    kb_index.upsert([
        {
            "id": f"emp_{employee.employee_id}",
            "values": embedding,
            "metadata": {"text": text}
        }
    ])

    return {"message": "Employee added and indexed successfully."}

# === Web UI ===
@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return """
    <html>
        <head>
            <title>Smart Knowledge Assistant</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(to right, #ece9e6, #ffffff);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: start;
                    height: 100vh;
                    padding-top: 20px;
                    margin: 0;
                }
                h2, h3 { color: #333; margin: 10px 0; }
                form {
                    display: flex;
                    gap: 10px;
                    margin: 10px;
                    flex-wrap: wrap;
                    justify-content: center;
                }
                input[type="text"], input[type="email"], input[type="date"] {
                    padding: 8px;
                    font-size: 14px;
                    border: 2px solid #aaa;
                    border-radius: 6px;
                    width: 200px;
                }
                button {
                    padding: 10px 20px;
                    font-size: 14px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                }
                button:hover { background-color: #45a049; }
                pre {
                    background: #f4f4f4;
                    padding: 15px;
                    border-radius: 10px;
                    width: 80%;
                    max-width: 600px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }
            </style>
        </head>
        <body>
            <h2>Ask the Smart Knowledge Assistant</h2>
            <form onsubmit="submitChatForm(event)">
                <input type="text" id="query" placeholder="Type your question..." required />
                <button type="submit">Ask</button>
            </form>
            <pre id="response">Your answer will appear here...</pre>

            <h3>Add New Employee</h3>
            <form onsubmit="submitEmployeeForm(event)">
                <input type="text" id="employee_id" placeholder="Employee ID" required />
                <input type="text" id="name" placeholder="Name" required />
                <input type="email" id="email" placeholder="Email" required />
                <input type="text" id="department" placeholder="Department" required />
                <input type="date" id="joining_date" required />
                <input type="text" id="role" placeholder="Role" required />
                <button type="submit">Add Employee</button>
            </form>
            <pre id="emp_response">Employee status will appear here...</pre>

            <script>
                async function submitChatForm(event) {
                    event.preventDefault();
                    const query = document.getElementById('query').value;
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query})
                    });
                    const data = await response.json();
                    document.getElementById('response').innerText = "Assistant: " + data.answer;
                }

                async function submitEmployeeForm(event) {
                    event.preventDefault();
                    const payload = {
                        employee_id: document.getElementById('employee_id').value,
                        name: document.getElementById('name').value,
                        email: document.getElementById('email').value,
                        department: document.getElementById('department').value,
                        joining_date: document.getElementById('joining_date').value,
                        role: document.getElementById('role').value
                    };

                    const response = await fetch('/add-employee', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(payload)
                    });

                    const result = await response.json();
                    document.getElementById('emp_response').innerText = result.message || "Added!";
                }
            </script>
        </body>
    </html>
    """

# uvicorn main:app --reload
