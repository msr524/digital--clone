import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import os, time
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Digital Ramesh", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
.block-container { padding-top: 2rem; max-width: 820px; }
.hero { background: linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
        border:1px solid rgba(99,102,241,0.3); border-radius:20px;
        padding:28px; margin-bottom:24px; text-align:center; }
.hero h1 { font-size:2rem; font-weight:700;
           background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
           background-clip:text; margin:0 0 6px; }
.hero p  { color:#94a3b8; font-size:0.88rem; margin:0; }
.pills   { display:flex; justify-content:center; gap:10px; margin-top:12px; flex-wrap:wrap; }
.pill    { background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.35);
           color:#6ee7b7; border-radius:999px; padding:4px 14px; font-size:0.75rem; }
.pill.m  { background:rgba(99,102,241,0.12); border-color:rgba(99,102,241,0.35); color:#a5b4fc; }
[data-testid="stChatMessage"] {
    background:rgba(255,255,255,0.04)!important;
    border:1px solid rgba(255,255,255,0.08)!important;
    border-radius:16px!important; margin-bottom:10px!important; padding:14px 18px!important; }
[data-testid="stChatInputContainer"] {
    background:rgba(255,255,255,0.05)!important;
    border:1px solid rgba(255,255,255,0.12)!important;
    border-radius:14px!important; }
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0f0c29,#1a1a2e)!important;
    border-right:1px solid rgba(99,102,241,0.2)!important; }
[data-testid="stMetric"] {
    background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
    border-radius:12px; padding:10px 14px; }
[data-testid="stMetricLabel"] { color:#94a3b8!important; font-size:0.75rem!important; }
[data-testid="stMetricValue"] { color:#818cf8!important; font-size:1.1rem!important; }
.stButton > button {
    background:linear-gradient(135deg,#4f46e5,#7c3aed)!important;
    color:white!important; border:none!important;
    border-radius:10px!important; font-weight:500!important; width:100%; }
hr { border-color:rgba(255,255,255,0.08)!important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:rgba(99,102,241,0.4); border-radius:99px; }
</style>
""", unsafe_allow_html=True)

if not GROQ_API_KEY:
    st.error("🔑 Add GROQ_API_KEY to your .env file")
    st.stop()

client_ai = Groq(api_key=GROQ_API_KEY)

@st.cache_resource(show_spinner="📂 Loading memory bank...")
def init_db():
    try:
        local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        db = chromadb.PersistentClient(path="ramesh_memory")
        return db.get_or_create_collection(
            name="ramesh_chat",
            embedding_function=local_ef
        )
    except Exception as e:
        st.error(f"DB Error: {e}")
        st.stop()

collection = init_db()

for k, v in [("messages",[]),("queries",0),("context",[])]:
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<div class="hero">
  <h1>🧠 Digital Ramesh</h1>
  <p>Personal AI Clone • RAG + Groq</p>
  <div class="pills">
    <span class="pill">🟢 Online</span>
    <span class="pill">🔗 Memory Loaded</span>
    <span class="pill m">⚡ Llama 3.3 70B</span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧠 Digital Ramesh")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: st.metric("💬 Chats", st.session_state.queries)
    with c2:
        try: mc = collection.count()
        except: mc = 0
        st.metric("🗂️ Memories", mc)
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    models = {
        "llama-3.3-70b-versatile": "Llama 3.3 70B ⭐",
        "llama-3.1-8b-instant":    "Llama 3.1 8B ⚡",
        "gemma2-9b-it":            "Gemma 2 9B 🔥"
    }
    selected_model = st.selectbox(
        "🤖 Model", list(models.keys()), format_func=lambda x: models[x]
    )
    temp  = st.slider("🌡️ Creativity", 0.0, 1.0, 0.80, 0.05)
    n_mem = st.slider("🔍 Memories", 3, 15, 8)
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.queries  = 0
        st.session_state.context  = []
        st.rerun()
    if st.session_state.context:
        with st.expander("📄 Retrieved Memories"):
            for i, m in enumerate(st.session_state.context, 1):
                st.caption(f"**{i}.** {m[:200]}...")
    st.markdown("---")
    st.markdown(
        "<p style='color:#475569;font-size:0.72rem;text-align:center;'>"
        "Digital Clone • RAG + Groq + ChromaDB</p>",
        unsafe_allow_html=True
    )

if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center;padding:48px 24px;'>
        <div style='font-size:3.5rem;margin-bottom:16px;'>👋</div>
        <h3 style='color:#818cf8;margin:0 0 8px;'>Hey, it's Ramesh!</h3>
        <p style='color:#64748b;margin:0;font-size:0.9rem;'>
            Talk to me like you'd WhatsApp Ramesh bro.
        </p>
    </div>""", unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"],
                             avatar="👤" if msg["role"] == "user" else "🧠"):
            st.markdown(msg["content"])


# ══════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════

def retrieve_memories(query: str, n: int):
    try:
        res  = collection.query(query_texts=[query], n_results=n)
        docs = res.get("documents", [[]])[0]
        return (docs, "\n\n".join(docs)) if docs else ([], "")
    except Exception:
        return [], ""


def extract_qa_pairs(memories: list) -> str:
    """
    Extract clean QA pairs — Murali message → Ramesh reply.
    No pipe characters. Each reply on its own line.
    """
    pairs = []
    for doc in memories:
        lines     = doc.strip().split("\n")
        murali_q  = ""
        ramesh_rs = []
        for line in lines:
            line = line.strip()
            if line.startswith("MURALI:"):
                if murali_q and ramesh_rs:
                    pairs.append(
                        f"Murali: {murali_q}\n"
                        f"Ramesh: {chr(10).join(ramesh_rs)}"
                    )
                murali_q  = line.replace("MURALI:", "").strip()
                ramesh_rs = []
            elif line.startswith("RAMESH:"):
                clean = line.replace("RAMESH:", "").strip()
                if clean:
                    ramesh_rs.append(clean)
        if murali_q and ramesh_rs:
            pairs.append(
                f"Murali: {murali_q}\n"
                f"Ramesh: {chr(10).join(ramesh_rs)}"
            )
    return "\n\n".join(pairs[:6])


def extract_ramesh_replies(memories: list) -> list:
    """Extract only Ramesh's reply lines for style learning."""
    replies = []
    for doc in memories:
        for line in doc.split("\n"):
            line = line.strip()
            if line.startswith("RAMESH:"):
                clean = line.replace("RAMESH:", "").strip()
                if clean and len(clean) > 2:
                    replies.append(clean)
    return replies


def get_recent_chat(messages: list) -> str:
    """Last 10 messages as readable transcript."""
    recent = messages[-10:]
    lines  = []
    for m in recent:
        speaker = "Murali" if m["role"] == "user" else "Ramesh"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def get_last_ramesh_replies(messages: list) -> str:
    """Last 5 Ramesh replies to enforce no repetition."""
    replies = [m["content"] for m in messages if m["role"] == "assistant"]
    last5   = replies[-5:]
    return "\n".join(f"- {r}" for r in last5) if last5 else "None yet."


def detect_echo(prompt: str, last_replies: list) -> bool:
    """
    Detect if the model might echo Murali's message.
    Triggers fallback if prompt words appear heavily in last replies.
    """
    prompt_words = set(prompt.lower().split())
    for reply in last_replies[-2:]:
        reply_words = set(reply.lower().split())
        overlap = prompt_words & reply_words
        if len(overlap) > 2:
            return True
    return False


def build_system_prompt(qa_pairs: str,
                        ramesh_replies: list,
                        recent_chat: str,
                        last_replies: str) -> str:

    reply_samples = "\n".join(f"• {r}" for r in ramesh_replies[:8]) \
                    if ramesh_replies else "• short casual Tenglish replies"

    return f"""You are Ramesh Baa — texting your closest friend Murali on WhatsApp.

━━━ REAL PAST CONVERSATIONS ━━━
These are real exchanges between Murali and Ramesh.
Study who says what — understand Ramesh's response style:

{qa_pairs if qa_pairs else "No close match — use personality rules below."}

━━━ RAMESH'S REAL REPLY SAMPLES ━━━
{reply_samples}

━━━ CURRENT CONVERSATION ━━━
{recent_chat}

━━━ YOUR LAST 5 REPLIES — NEVER REPEAT THESE ━━━
{last_replies}

━━━ BANNED PHRASES — NEVER USE ━━━
These are overused — they are now completely banned:
- "nee bontha ra"
- "cheppu bava properly"
- "cheppu bava"
- "enti ra cheppu properly"
- "em chepali ra"
React differently every single time.

━━━ HOW TO REPLY ━━━

IF memory matches current topic:
→ Reply in Ramesh's style from those examples
→ Stay on that topic naturally

IF no memory matches (new/unseen topic):
→ Reply from Ramesh's core personality:
   - Warm casual best friend energy
   - Ask a natural question to continue conversation
   - "ha cheppu ra", "em aindhi", "seri baa", "adhi ela aindhi"
   - Keep it short and real — 1 line

IF Murali says something vague (enti, em, a, o):
→ Do NOT echo it back
→ React naturally: "ha ra", "ento", "cheppu bava" (use sparingly)
→ Or reference something from earlier in the conversation

━━━ BANTER RULES ━━━
Friendly insults → fire back with wit, NEVER agree:
- "ne bontha"  → "nee bontha ra, nenu em chesanu"
- "ne mohamu"  → "nee mohamu chudara mirror lo"
- "pagal"      → "nuvve pagal ra, nenu normal"
Use each comeback ONCE only — vary them every time.

━━━ HARD RULES ━━━
1. ONE reply — 1 line only, max 2 short lines
2. Tenglish — mix Telugu and English naturally
3. NEVER echo or repeat Murali's words back
4. NEVER invent places, events, or stories
5. NEVER repeat from last 5 replies
6. NEVER use banned phrases above
7. NEVER say you are an AI
8. NEVER use formal English"""


BLOCKED = ["sex", "cock", "suck my", "nude", "penis", "fuck", "boobs"]

def is_inappropriate(text: str) -> bool:
    return any(w in text.lower() for w in BLOCKED)


def stream_response(payload: list, temp: float):
    stream = client_ai.chat.completions.create(
        model=selected_model,
        messages=payload,
        temperature=temp,
        max_tokens=60,
        stream=True,
        frequency_penalty=1.5,  # strongly penalise repeated words
        presence_penalty=1.0,   # encourage fresh content
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── CHAT INPUT ──
if prompt := st.chat_input("Message Ramesh..."):

    if is_inappropriate(prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant",
                                          "content": "Enti ra pagal ayyava, waste ga matladakura"})
        st.session_state.queries += 1
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        try:
            # Step 1: Retrieve relevant memories
            raw_docs, _ = retrieve_memories(prompt, n_mem)
            st.session_state.context = raw_docs

            # Step 2: Extract clean QA pairs — no pipes
            qa_pairs = extract_qa_pairs(raw_docs)

            # Step 3: Extract Ramesh reply style samples
            ramesh_replies = extract_ramesh_replies(raw_docs)

            # Step 4: Conversation transcript for continuity
            recent_chat = get_recent_chat(st.session_state.messages)

            # Step 5: Last 5 replies for anti-repetition
            last_replies_str = get_last_ramesh_replies(st.session_state.messages)

            # Step 6: Build prompt
            sys_prompt = build_system_prompt(
                qa_pairs, ramesh_replies, recent_chat, last_replies_str
            )

            # Step 7: Clean payload
            payload = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": prompt},
            ]

            # Step 8: Stream response
            box, full, start = st.empty(), "", time.time()
            for token in stream_response(payload, temp):
                full += token
                box.markdown(full + "▌")

            # Step 9: Clean up any accidental pipe characters in reply
            full = full.replace(" | ", " ").replace("|", "")
            box.markdown(full)

            st.caption(
                f"⏱️ {time.time()-start:.1f}s • "
                f"{len(raw_docs)} memories • "
                f"{selected_model.split('-versatile')[0]}"
            )

            st.session_state.messages.append({"role": "assistant", "content": full})
            st.session_state.queries += 1

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                st.warning("⏳ Rate limit — switch to Llama 3.1 8B in sidebar or wait 30s.")
            elif "api_key" in err.lower():
                st.error("🔑 Invalid Groq API Key.")
            else:
                st.error(f"⚠️ {err}")