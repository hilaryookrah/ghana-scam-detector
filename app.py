import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Scam message checker", page_icon="📱")

@st.cache_resource
def load_model():
    return joblib.load("results/model_mixed.joblib")

model = load_model()
tfidf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]
terms = np.array(tfidf.get_feature_names_out())

def explain(text, k=5):
    vec = tfidf.transform([text])
    contrib = vec.multiply(clf.coef_[0]).toarray()[0]
    order = np.argsort(contrib)
    toward_scam = [(terms[i], contrib[i]) for i in order[::-1][:k] if contrib[i] > 0]
    toward_genuine = [(terms[i], contrib[i]) for i in order[:k] if contrib[i] < 0]
    return toward_scam, toward_genuine

st.title("Scam message checker")
st.caption("A learning project, not a security product. It can be wrong, so never share "
           "your PIN or OTP with anyone, whatever the model says.")

text = st.text_area("Paste a text message", height=140,
                    placeholder="e.g. Please reverse the money I sent you by mistake, urgent")

if st.button("Check message") and text.strip():
    prob = float(model.predict_proba([text])[0, 1])
    if prob >= 0.5:
        st.error(f"Looks like a scam ({prob:.0%} scam score)")
    else:
        st.success(f"Looks genuine ({prob:.0%} scam score)")

    scam_terms, genuine_terms = explain(text)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pushed toward scam")
        if scam_terms:
            for term, c in scam_terms:
                st.write(f"- {term}")
        else:
            st.write("nothing stood out")
    with col2:
        st.subheader("Pushed toward genuine")
        if genuine_terms:
            for term, c in genuine_terms:
                st.write(f"- {term}")
        else:
            st.write("nothing stood out")

st.divider()
st.caption("Trained on a public English SMS spam dataset plus a small set of "
           "Ghana-style messages written for this project. See the README for limits.")