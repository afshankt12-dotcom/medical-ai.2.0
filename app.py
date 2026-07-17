import streamlit as st
from src.chatbot import MedicalChatbot

import base64
import time
import re
import os
import tempfile

from langdetect import detect
from deep_translator import GoogleTranslator


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Medical AI",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# IMAGE TO BASE64
# --------------------------------------------------

def get_base64(path):

    with open(path, "rb") as img:

        return base64.b64encode(
            img.read()
        ).decode()


# --------------------------------------------------
# LOADING SCREEN
# --------------------------------------------------

def loading_screen():

    logo = get_base64("assets/logo.png")

    loader = st.empty()

    loader.markdown(
f"""
<style>

@keyframes spin {{
0% {{
transform: rotate(0deg);
}}

100% {{
transform: rotate(360deg);
}}
}}

.loading {{

height:90vh;

display:flex;

justify-content:center;

align-items:center;

flex-direction:column;

}}

.logo {{

width:180px;

animation:spin 2s linear infinite;

}}

.loading-text {{

margin-top:25px;

font-size:32px;

font-weight:bold;

color:#C62828;

}}

</style>

<div class="loading">

<img
class="logo"
src="data:image/png;base64,{logo}">

<div class="loading-text">

Loading Medical AI...

</div>

</div>

""",
unsafe_allow_html=True
    )

    time.sleep(8)

    loader.empty()


if "loaded" not in st.session_state:

    loading_screen()

    st.session_state.loaded = True


# --------------------------------------------------
# BACKGROUND
# --------------------------------------------------

background = get_base64(
    "assets/background.png"
)
# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown(
f"""
<style>

.stApp {{

background-image:url(
"data:image/png;base64,{background}"
);

background-size:cover;

background-position:center;

background-attachment:fixed;

}}

.main .block-container {{

max-width:950px;

margin:auto;

padding:30px;

background:rgba(255,255,255,.90);

border-radius:20px;

box-shadow:0 6px 20px rgba(0,0,0,.25);

}}

.stTextInput input {{

height:55px !important;

border-radius:30px !important;

font-size:18px !important;

padding-left:20px !important;

background:white !important;

color:black !important;

border:2px solid #E53935 !important;

}}

.stButton button {{

width:50px;

height:50px;

border-radius:50%;

font-size:22px;

background:#E53935;

color:white !important;

}}

.user-box {{

background:#E53935;

padding:15px;

border-radius:15px;

margin-top:15px;

color:white !important;

}}

.user-box * {{

color:white !important;

}}

.bot-box {{

background:white;

padding:15px;

border-radius:15px;

border-left:6px solid #E53935;

box-shadow:0 4px 12px rgba(0,0,0,.2);

color:black !important;

}}

.bot-box * {{

color:black !important;

}}

p,
span,
label,
div {{

color:black !important;

}}

footer {{

visibility:hidden;

}}

div[data-testid="stAudioInput"],
div[data-testid="stAudioInput"] > div,
div[data-testid="stAudioInput"] div {{
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

div[data-testid="stAudioInput"] button {{
    background-color: #E53935 !important;
    border-radius: 50% !important;
    width: 50px !important;
    height: 50px !important;
    min-width: 50px !important;
    min-height: 50px !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: transform 0.2s !important;
}}

div[data-testid="stAudioInput"] button:hover {{
    transform: scale(1.05) !important;
    background-color: #D32F2F !important;
}}

div[data-testid="stAudioInput"] button svg {{
    display: none !important;
}}

div[data-testid="stAudioInput"] button::before {{
    content: "" !important;
    display: inline-block !important;
    width: 24px !important;
    height: 24px !important;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z"/></svg>') !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
}}

div[data-testid="stAudioInput"] canvas,
div[data-testid="stAudioInput"] audio {{
    display: none !important;
}}

div[data-testid="stAudioInput"] p {{
    display: none !important;
}}

div[data-testid="stAudioInput"] span {{
    color: transparent !important;
    font-size: 0px !important;
    line-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}
</style>
""",
unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_chatbot():
    return MedicalChatbot()

chatbot = load_chatbot()
# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.image(
    "assets/girl.png",
    width=120
)

st.markdown(
"""
<h1 style="
text-align:center;
color:#C62828;
font-size:48px;
margin-bottom:10px;
">
🩺 Medical AI
</h1>

<p style="
text-align:center;
font-size:20px;
color:black;
margin-bottom:30px;
">
Ask medical questions and get AI-powered answers.
</p>
""",
unsafe_allow_html=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""

if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None

if "debug_info" not in st.session_state:
    st.session_state.debug_info = None


# --------------------------------------------------
# CLEAR CHAT BUTTON
# --------------------------------------------------

col1, col2, col3 = st.columns([1,7,1])

with col1:

    if st.button(
        "🗑️",
        use_container_width=True,
        key="clear"
    ):

        st.session_state.messages = []
        st.session_state.voice_question = ""
        st.session_state.last_processed_audio = None
        st.session_state.debug_info = None
        st.rerun()


st.write("")


# --------------------------------------------------
# INPUT BAR
# --------------------------------------------------

input_col = st.container()

with input_col:
    text_question = st.text_input(
        "",
        value=st.session_state.voice_question,
        placeholder="Ask anything...",
        label_visibility="collapsed",
        key="question_input"
    )

# --------------------------------------------------
# GET QUESTION
# --------------------------------------------------

question = ""


if st.session_state.voice_question.strip():

    question = st.session_state.voice_question


elif text_question.strip():

    question = text_question



# --------------------------------------------------
# LANGUAGE DETECTION FUNCTION
# --------------------------------------------------

def detect_language(text):
    """
    Automatically detects the language of the user input.
    If the text is written in English characters but contains Malayalam symptom/conversational keywords,
    it is classified as 'manglish' to route through transliteration.
    Otherwise, if it contains only basic English letters/punctuation, it defaults to English ('en')
    to prevent short words (like 'no', 'yes') from being misclassified.
    `langdetect` is only called when the script is non-Latin or foreign language indicators exist.
    """
    text_lower = text.lower()
    
    # Keywords that indicate Malayalam written in Roman characters (Manglish)
    manglish_keywords = {
        "enikku", "enik", "enikk", "njan", "njangal", "nee", "ningal", "avan", "aval", "avar", "ithu", "athu", "enth", "entha", "enthanu", "enthundu", "enthund",
        "und", "undu", "illa", "illath", "alla", "allath", "poyi", "poya", "vannu", "vanna", "varum", "pogum", "pokum", "chey", "cheyya", "cheyyu", "cheyyuka",
        "kazhi", "kazhikkan", "kazhichu", "kudi", "kudikkan", "kudichu", "kanuka", "kandu", "kannan", "parayu", "parayan", "paranju", "ariyuka", "ariya", "ariyilla",
        "pani", "vedana", "chum", "chuma", "vayar", "thala", "sheenam", "ksheenam", "shardi", "chumma", "neeru",
        "sugamano", "sugamanu", "sukham", "evide", "evideya", "eppol", "eppozha", "engane", "enganeya", "nalla", "nanni", "oru"
    }
    
    # Extract lowercase words
    words = re.findall(r"\b[a-z]+\b", text_lower)
    has_manglish_keyword = any(word in manglish_keywords for word in words)
    
    # Check if the text consists strictly of English letters, digits, and common punctuation
    is_latin = bool(re.match(r"^[A-Za-z0-9\s,.!?'\"\-()]+$", text))
    
    if is_latin:
        if has_manglish_keyword:
            return "manglish"
            
        # Treat normal English text as English by default to prevent langdetect false positives
        # (e.g. short words like 'no' being misclassified as Norwegian).
        # We only query langdetect if there's high uncertainty or it might be another language.
        try:
            from langdetect import detect_langs
            predictions = detect_langs(text)
            best_pred = predictions[0]
            
            # Common English words that are often misclassified by langdetect on short inputs
            common_english_words = {
                "no", "yes", "chest", "pain", "fever", "cough", "headache", "throat", "stomach", "cold", "flu",
                "what", "is", "how", "to", "do", "i", "have", "a", "the", "in", "of", "for", "on", "with", "at", "who", "when", "why"
            }
            has_common_english = any(word in common_english_words for word in words)
            
            # Only classify as non-English if langdetect is highly confident (> 0.85) AND it doesn't contain common English terms
            if best_pred.lang != "en" and best_pred.prob > 0.85 and not has_common_english:
                return best_pred.lang
        except Exception:
            pass
            
        # Default fallback for Latin script is English
        return "en"
        
    # Non-Latin characters (Devanagari, Arabic, Malayalam script, etc.) -> use langdetect
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"


# --------------------------------------------------
# MANGLISH CONVERTER
# --------------------------------------------------

def convert_manglish_to_malayalam(text):
    """
    Transliterates Manglish (Malayalam written in English script) to Malayalam script
    using a rule-based mapping for medical symptoms and common terms.
    """
    replacements = {
        "vayar vedana": "വയറുവേദന",
        "vayarvedana": "വയറുവേദന",
        "thala vedana": "തലവേദന",
        "thalavedana": "തലവേദന",
        "chum undu": "ചുമയുണ്ട്",
        "chumundu": "ചുമയുണ്ട്",
        "pani undu": "പനിയുണ്ട്",
        "paniundu": "പനിയുണ്ട്",
        "enikku": "എനിക്ക്",
        "enik": "എനിക്ക്",
        "enikk": "എനിക്ക്",
        "pani": "പനി",
        "undu": "ഉണ്ട്",
        "und": "ഉണ്ട്",
        "chum": "ചുമ",
        "chuma": "ചുമ",
        "vayar": "വയർ",
        "vedana": "വേദന",
        "thala": "തല",
        "sheenam": "ക്ഷീണം",
        "ksheenam": "ക്ഷീണം",
        "shardi": "ഛർദ്ദി",
        "alla": "അല്ല",
        "sugamano": "സുഖമാണോ",
        "sugamanu": "സുഖമാണ്",
        "sukham": "സുഖം",
        "njan": "ഞാൻ",
        "entha": "എന്താ",
        "enthanu": "എന്താണ്",
        "evide": "എവിടെ",
        "evideya": "എവിടെയാ",
        "engane": "എങ്ങനെ",
        "enganeya": "എങ്ങനെയാ",
        "nanni": "നന്ദി",
        "oru": "ഒരു"
    }

    text_lower = text.lower()

    # Transliterate known multi-word phrases first
    for phrase in ["vayar vedana", "thala vedana", "vayarvedana", "thalavedana", "chum undu", "chumundu", "pani undu", "paniundu"]:
        if phrase in text_lower:
            text_lower = text_lower.replace(phrase, replacements[phrase])

    words = text_lower.split()
    result = []

    for word in words:
        if word in replacements:
            result.append(replacements[word])
        else:
            result.append(word)

    return " ".join(result)


# --------------------------------------------------
# CHATBOT (AUTO LANGUAGE)
# --------------------------------------------------

if question.strip():

    original_input = question
    
    # 1. Detect language automatically
    try:
        user_language = detect_language(question)
    except Exception:
        # Fall back to English on detection failure
        user_language = "en"

    manglish_converted = None
    english_question = question

    # 2. Process user input into English based on language type
    if user_language == "manglish":
        # Transliterate Manglish to Malayalam script
        manglish_converted = convert_manglish_to_malayalam(question)
        try:
            # Translate Malayalam script to English
            english_question = GoogleTranslator(
                source="ml",
                target="en"
            ).translate(manglish_converted)
        except Exception:
            # Fall back to original input text if translation fails
            english_question = question
            
    elif user_language != "en":
        try:
            # Translate foreign language to English automatically
            english_question = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(question)
        except Exception:
            # Fall back to original input text if translation fails
            english_question = question

    # --------------------------------------------------
    # AI THINKING LOADING
    # --------------------------------------------------

    with st.spinner(
        "🤖 Medical AI is thinking..."
    ):

        time.sleep(2)

        try:
            # Query ONLY the English question to the chatbot
            result = chatbot.ask(
                english_question
            )
            raw_answer = result["answer"]
            confidence = result["confidence"]
        except Exception as e:
            # Prevent crashes if model inference or retrieval fails
            raw_answer = "I'm not confident enough to answer that question. Please rephrase it or consult a healthcare professional."
            confidence = 0.0

    # Confidence fallback check (threshold 0.25)
    if confidence < 0.25:
        processed_answer = "I'm not confident enough to answer that question. Please rephrase it or consult a healthcare professional."
    else:
        processed_answer = raw_answer

    # 3. Determine target language for translating answer back
    # For Manglish, we translate to standard Malayalam script
    target_lang = "ml" if user_language == "manglish" else user_language

    # Output debugging logs to standard output
    print(f"[DEBUG LOG] Original question: '{original_input}'")
    print(f"[DEBUG LOG] Detected language: '{user_language}'")
    print(f"[DEBUG LOG] English question sent to chatbot: '{english_question}'")
    print(f"[DEBUG LOG] Target language for translation: '{target_lang}'")

    if target_lang != "en":
        try:
            # Translate answer back to original language
            final_answer = GoogleTranslator(
                source="en",
                target=target_lang
            ).translate(processed_answer)
        except Exception as e:
            # Fall back to original English answer if translation back fails
            print(f"[DEBUG LOG] Translation back to '{target_lang}' failed: {e}")
            final_answer = processed_answer
    else:
        final_answer = processed_answer

    # Save to debug info
    st.session_state.debug_info = {
        "original_input": original_input,
        "detected_language": user_language,
        "manglish_converted": manglish_converted,
        "english_translation": english_question,
        "raw_answer": raw_answer,
        "confidence": confidence,
        "final_answer": final_answer
    }

    # Save message
    st.session_state.messages.append(
        {
            "question": original_input,

            "answer": final_answer,

            "confidence": confidence
        }
    )

    # Clear voice input
    st.session_state.voice_question = ""



# --------------------------------------------------
# CHAT DISPLAY
# --------------------------------------------------

for chat in reversed(
    st.session_state.messages
):

    st.markdown(
        f"""
<div class="user-box">

<b>👤 You</b>

<br><br>

{chat["question"]}

</div>
""",
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
<div class="bot-box">

<b>🤖 Medical AI</b>

<br><br>

{chat["answer"]}

</div>
""",
        unsafe_allow_html=True
    )


    # Confidence

    confidence = float(
        chat.get(
            "confidence",
            0.0
        )
    )


    # Keep value between 0 and 1

    confidence = max(
        0.0,
        min(
            confidence,
            1.0
        )
    )


    st.progress(
        confidence
    )


    st.caption(
        f"Confidence: {confidence * 100:.1f}%"
    )



# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()


st.markdown(
"""
<div style="
text-align:center;
color:black;
font-size:15px;
padding:10px;
">

© 2026 Medical AI

</div>
""",
unsafe_allow_html=True
)