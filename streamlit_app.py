import streamlit as st
from emotion_classifier import EmotionClassifier
from rag_pipeline import generate_response

st.set_page_config(page_title="Mental Health Companion", page_icon="🧠")

st.title("🧠 Mental Health Companion")
st.markdown("Talk about how you're feeling. The assistant will listen and respond with empathy.")

# Initialize the EmotionClassifier once
classifier = EmotionClassifier()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
user_input = st.chat_input("How are you feeling today?")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Emotion detection
    emotion_label, confidence = classifier.predict_emotion(user_input)

    # Generate assistant response
    with st.spinner("Thinking..."):
        reply = generate_response(user_input, emotion_label)

    # Save assistant reply
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "emotion": emotion_label
    })

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(f"**Detected emotion:** _{msg['emotion'].capitalize()}_")
            st.write(msg["content"])
