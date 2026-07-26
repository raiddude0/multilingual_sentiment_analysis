import streamlit as st
import sys
import os
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.infer import predict, predict_batch, classifier


st.set_page_config(
    page_title="Multilingual Sentiment Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [role="tablist"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("🌍 Multilingual Sentiment Analysis")
st.markdown("""
Analyze sentiment in **multiple languages** using XLM-RoBERTa fine-tuned on multilingual data.
Supports: English, French, Arabic, Spanish, German, and 100+ other languages!
""")


with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** XLM-RoBERTa-base (fine-tuned)
    
    **Languages:** 100+ supported
    
    **Classes:** 
    - 😞 Negative
    - 😐 Neutral  
    - 😊 Positive
    
    **Confidence:** Probability score (0-100%)
    """)
    
    st.divider()
    st.markdown("**Examples:**")
    st.code("""
    "Great product!" → Positive
    "Service was awful" → Negative
    "It's okay" → Neutral
    """)


tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📚 Batch Analysis", "🧪 Demo"])


with tab1:
    st.subheader("Analyze a Single Text")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type something in any language...",
            height=120,
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("")  
        analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")
    
    if analyze_button and text_input.strip():
        try:
            with st.spinner("Analyzing..."):
                result = predict(text_input.strip())
            
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sentiment", result['label'].upper())
            
            with col2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            with col3:
               
                sentiment_colors = {
                    "positive": "🟢",
                    "neutral": "🟡",
                    "negative": "🔴"
                }
                emoji = sentiment_colors.get(result['label'], "⚪")
                st.metric("Status", emoji)
            
            
            st.progress(result['confidence'], text=f"Confidence: {result['confidence']*100:.1f}%")
            
            
            with st.expander("📊 Detailed Results"):
                st.json({
                    "text": text_input,
                    "sentiment": result['label'],
                    "confidence": round(result['confidence'], 4),
                    "confidence_percentage": f"{result['confidence']*100:.2f}%"
                })
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze")


with tab2:
    st.subheader("Analyze Multiple Texts")
    
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        placeholder="Line 1: First text...\nLine 2: Second text...\nLine 3: Third text...",
        height=200,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        batch_button = st.button("🚀 Batch Analyze", use_container_width=True, type="primary")
    
    if batch_button and batch_input.strip():
        texts = [line.strip() for line in batch_input.strip().split('\n') if line.strip()]
        
        if texts:
            try:
                with st.spinner(f"Analyzing {len(texts)} texts..."):
                    results = predict_batch(texts)
                
               
                st.success(f"✅ Analyzed {len(texts)} texts")
                
                results_data = []
                for text, result in zip(texts, results):
                    results_data.append({
                        "Text": text[:50] + "..." if len(text) > 50 else text,
                        "Sentiment": result['label'].upper(),
                        "Confidence": f"{result['confidence']*100:.1f}%"
                    })
                
                st.dataframe(
                    results_data,
                    use_container_width=True,
                    hide_index=True
                )
                
                
                sentiments = [r['label'] for r in results]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    positive_count = sentiments.count('positive')
                    st.metric("😊 Positive", positive_count)
                
                with col2:
                    neutral_count = sentiments.count('neutral')
                    st.metric("😐 Neutral", neutral_count)
                
                with col3:
                    negative_count = sentiments.count('negative')
                    st.metric("😞 Negative", negative_count)
                
                
                import json
                csv_data = "Text,Sentiment,Confidence\n"
                for text, result in zip(texts, results):
                    csv_data += f'"{text}",{result["label"]},{result["confidence"]:.4f}\n'
                
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter at least one text")


with tab3:
    st.subheader("🧪 Try Demo Examples")
    
    demo_examples = {
        "🌟 Positive Examples": [
            "This product is amazing! Best purchase ever.",
            "Merci beaucoup, service excellent!",
            "الطعام لذيذ جداً والخدمة رائعة",
        ],
        "😐 Neutral Examples": [
            "The movie was okay, nothing special.",
            "C'est un produit ordinaire.",
            "إنه مقبول جداً",
        ],
        "😞 Negative Examples": [
            "Terrible experience, never coming back.",
            "Très déçu par la qualité.",
            "خدمة سيئة جداً وسعر مرتفع",
        ]
    }
    
    for category, examples in demo_examples.items():
        with st.expander(category, expanded=False):
            for idx, example in enumerate(examples):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"*{example}*")
                
                with col2:
                    if st.button("Analyze", key=f"{category}_{idx}"):
                        try:
                            result = predict(example)
                            sentiment_emoji = {
                                "positive": "🟢",
                                "neutral": "🟡",
                                "negative": "🔴"
                            }
                            with col3:
                                st.success(f"{sentiment_emoji.get(result['label'])} {result['label'].upper()}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")


st.divider()
st.markdown("""
<div style='text-align: center'>
    <small>Built with ❤️ using Streamlit • XLM-RoBERTa • Multilingual Sentiment Analysis</small>
</div>
""", unsafe_allow_html=True)
