import re
import chromadb
from chromadb.utils import embedding_functions

CHAT_FILE   = "chat.txt"
DB_PATH     = "ramesh_memory"
COLLECTION  = "ramesh_chat"

# ✅ All name variations Ramesh uses in the chat
RAMESH_NAMES = [
    "ramesh baa",
    "ramesh btech",
    "ramesh h",
    "ramesh :",
    "ramesh",
]
MURALI_NAME = "muralisuryaramesh"

# ✅ Content to skip completely
SKIP_CONTENT = [
    "<media omitted>",
    "this message was deleted",
    "you deleted this message",
    "messages and calls are end-to-end encrypted",
    "null",
    "<this message was edited>",
]

SKIP_PREFIXES = [
    "http://", "https://", "location:",
    "here's a", "here is a", "introduction:",
    "conclusion:", "definition:", "explanation:",
    "✅", "📢", "🎓", "⏳", "📅", "📘",
]

def is_ramesh(sender: str) -> bool:
    s = sender.lower().strip()
    return any(r in s for r in RAMESH_NAMES)

def is_murali(sender: str) -> bool:
    return MURALI_NAME.lower() in sender.lower()

def should_skip(content: str) -> bool:
    c = content.lower().strip()
    if c in SKIP_CONTENT:
        return True
    if any(c.startswith(p.lower()) for p in SKIP_PREFIXES):
        return True
    if len(content.strip()) < 2:
        return True
    # Skip very long messages (study material, assignments — not conversation)
    if len(content) > 300:
        return True
    # Skip messages that are just emojis or symbols
    if re.match(r'^[\U0001F000-\U0001FFFF\s]+$', content):
        return True
    return False

def parse_chat(filepath: str) -> list:
    """Parse chat.txt — format: 'Name: message' per line."""
    pattern  = re.compile(r'^([^:]+):\s*(.+)$')
    messages = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                continue
            sender  = m.group(1).strip()
            content = m.group(2).strip()

            if not (is_ramesh(sender) or is_murali(sender)):
                continue
            if should_skip(content):
                continue

            messages.append({
                "sender":  "Ramesh" if is_ramesh(sender) else "Murali",
                "message": content
            })

    return messages

def build_pairs(messages: list) -> tuple:
    """
    Build (Murali message → Ramesh reply) pairs.
    Also collect all Ramesh-only lines for style learning.
    """
    pairs = []
    style = []

    for i, msg in enumerate(messages):
        if msg["sender"] == "Ramesh":
            style.append(msg["message"])

        if msg["sender"] == "Murali":
            replies = []
            for j in range(i + 1, min(i + 5, len(messages))):
                if messages[j]["sender"] == "Ramesh":
                    replies.append(messages[j]["message"])
                elif messages[j]["sender"] == "Murali":
                    break
            if replies:
                doc  = f"MURALI: {msg['message']}\n"
                doc += "\n".join(f"RAMESH: {r}" for r in replies)
                pairs.append({"doc": doc})

    return pairs, style

def main():
    print("=" * 48)
    print("  Digital Clone — Ingest Pipeline")
    print("=" * 48)

    print(f"\n Reading: {CHAT_FILE}")
    messages = parse_chat(CHAT_FILE)

    ramesh_c = sum(1 for m in messages if m["sender"] == "Ramesh")
    murali_c = sum(1 for m in messages if m["sender"] == "Murali")
    print(f" Total messages  : {len(messages)}")
    print(f" Ramesh messages : {ramesh_c}")
    print(f" Murali messages : {murali_c}")

    if len(messages) == 0:
        print("\n❌ 0 messages parsed — check name spelling in chat.txt")
        return

    pairs, style = build_pairs(messages)
    print(f"\n QA pairs built  : {len(pairs)}")
    print(f" Style lines     : {len(style)}")

    # Preview 3 sample pairs
    print("\n Sample QA pairs:")
    for p in pairs[:3]:
        print(f"  {p['doc'][:100]}")
        print()

    # ── ChromaDB ──
    local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    db = chromadb.PersistentClient(path=DB_PATH)
    try:
        db.delete_collection(COLLECTION)
        print(" Cleared old collection.")
    except Exception:
        pass

    col = db.get_or_create_collection(
        name=COLLECTION,
        embedding_function=local_ef,
        metadata={"hnsw:space": "cosine"}
    )

    docs, ids = [], []

    # Store QA pairs
    for i, p in enumerate(pairs):
        docs.append(p["doc"])
        ids.append(f"qa_{i}")

    # Store style chunks (every 5 Ramesh lines grouped)
    for i in range(0, len(style), 5):
        chunk = "\n".join(style[i:i+5])
        docs.append(chunk)
        ids.append(f"style_{i}")

    print(f"\n Storing {len(docs)} documents...")
    for s in range(0, len(docs), 500):
        col.add(
            documents=docs[s:s+500],
            ids=ids[s:s+500]
        )
        print(f"  Batch {s}–{min(s+500, len(docs))} stored ✅")

    print(f"\n✅ Done! {col.count()} total documents in DB.")
    print(" Now run: streamlit run main.py\n")

if __name__ == "__main__":
    main()