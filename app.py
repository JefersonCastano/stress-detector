import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Fine-tuned model ID
MODEL_ID = st.secrets["MODEL_ID"]

# Format the message
def format_test(post):
    prompt = f'Consider this post: "{post}" Question: The answer to the question "Does the poster suffers from stress?" is '
    return [{"role": "user", "content": prompt}]

# Call the model
def predict(test_messages, fine_tuned_model_id):
    response = client.chat.completions.create(
        model=fine_tuned_model_id,
        messages=test_messages,
        temperature=0,
        max_tokens=150
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("🧠 Stress Detector - Fine-Tuned Model")
st.write("Enter a post to analyze whether it shows signs of stress.")

user_input = st.text_area("✍️ User post:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            formatted = format_test(user_input)
            result = predict(formatted, MODEL_ID)

        # Normalize and split
        result_cleaned = result.replace("REASONING:", "Reasoning:").replace("reasoning:", "Reasoning:")
        parts = result_cleaned.split("Reasoning:")

        answer_full = parts[0].strip()
        reasoning = parts[1].strip() if len(parts) > 1 else ""

        st.success("✅ Analysis Complete")
        st.markdown("---")

        st.markdown("**😫 Does the poster suffers from stress?**")

        # Show full response and color-highlight if it contains "yes" or "no"
        if "yes" in answer_full.lower():
            st.error(f"🟠 {answer_full}")
        elif "no" in answer_full.lower():
            st.success(f"🟢 {answer_full}")
        else:
            st.info(f"🔵 {answer_full}")

        if reasoning:
            st.markdown("**🧾 Reasoning:**")
            st.info(reasoning)
