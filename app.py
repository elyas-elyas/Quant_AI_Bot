import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# --- Page Configuration ---
st.set_page_config(
    page_title="Master IRFA & Quant Bot",
    page_icon="📘",
    layout="wide" # Layout wide pour mieux voir les sources
)

# --- Global Settings ---
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
llm = Ollama(model="llama3.1", request_timeout=120.0)

Settings.embed_model = embed_model
Settings.llm = llm

# --- Load Data (Cached) ---
@st.cache_resource(show_spinner=False)
def load_data():
    storage_dir = "./storage"
    if not os.path.exists(storage_dir):
        return None
    
    with st.spinner("Chargement de la base de connaissances (Cours IRFA + Quant)..."):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        return index

index = load_data()

# --- Chat Engine Initialization ---
if "chat_engine" not in st.session_state and index is not None:
    # On utilise le mode 'condense_plus_context' qui est très bon pour garder le fil de la conversation
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", 
        system_prompt=(
            "Tu es un assistant expert pour le Master IRFA et la Finance Quantitative. "
            "Réponds aux questions en te basant STRICTEMENT sur le contexte fourni (les cours). "
            "Si la réponse n'est pas dans les documents, dis-le clairement. "
            "Réponds toujours dans la langue de l'utilisateur (Français ou Anglais)."
        ),
        similarity_top_k=3 # Il va chercher les 3 meilleures pages pour répondre
    )

# --- UI Layout ---
st.title("📘 Assistant Master IRFA")
st.markdown("Pose une question technique. Je chercherai la réponse directement dans tes PDFs.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si le message a des sources attachées, on les affiche
        if "sources" in message:
            with st.expander("📚 Sources & Pages consultées"):
                for source in message["sources"]:
                    st.markdown(f"**Document :** `{source['file']}` (Page {source['page']})")
                    st.caption(f"...{source['text']}...")
                    st.divider()

# --- Chat Input & Response ---
if prompt := st.chat_input("Ex: Rappelle-moi la définition d'une martingale selon le cours"):
    
    # 1. User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Assistant response
    if index is None:
        st.error("⚠️ Erreur : Base de données introuvable. Lance 'ingest_data.py' d'abord.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyse de tes cours en cours..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.markdown(response.response)
                
                # Gestion des sources (Citations)
                sources_data = []
                if response.source_nodes:
                    with st.expander("📚 Voir les sources dans les cours"):
                        for node in response.source_nodes:
                            # Récupération des métadonnées (Nom du fichier, Page)
                            meta = node.metadata
                            file_name = meta.get('file_name', 'Inconnu')
                            page_label = meta.get('page_label', '?')
                            # Nettoyage du texte pour l'affichage
                            text_snippet = node.node.get_content()[:300].replace("\n", " ")
                            
                            st.markdown(f"**📄 {file_name}** - *Page {page_label}*")
                            st.caption(f"Extrait : \"{text_snippet}...\"")
                            st.divider()
                            
                            sources_data.append({
                                "file": file_name,
                                "page": page_label,
                                "text": text_snippet
                            })

        # 3. Save to history with sources
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.response,
            "sources": sources_data
        })