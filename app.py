import gradio as gr
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.infer import predict, predict_batch


# ============ Single Text Analysis ============
def analyze_single(text):
    if not text or not text.strip():
        return "⚠️ Please enter some text", "", ""
    
    try:
        result = predict(text.strip())
        sentiment = result['label'].upper()
        confidence = f"{result['confidence']*100:.1f}%"
        
        sentiment_emoji = {
            "positive": "🟢",
            "neutral": "🟡",
            "negative": "🔴"
        }
        emoji = sentiment_emoji.get(result['label'], "⚪")
        
        return f"{emoji} {sentiment}", confidence, text
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

# ============ Batch Analysis ============
def analyze_batch(batch_text):
    if not batch_text or not batch_text.strip():
        return "⚠️ Please enter texts (one per line)", []
    
    texts = [line.strip() for line in batch_text.strip().split('\n') if line.strip()]
    
    if not texts:
        return "⚠️ Please enter at least one text", []
    
    try:
        results = predict_batch(texts)
        
        output_data = []
        for text, result in zip(texts, results):
            sentiment_emoji = {
                "positive": "🟢",
                "neutral": "🟡",
                "negative": "🔴"
            }
            emoji = sentiment_emoji.get(result['label'], "⚪")
            
            output_data.append([
                text[:70] + "..." if len(text) > 70 else text,
                f"{emoji} {result['label'].upper()}",
                f"{result['confidence']*100:.1f}%"
            ])
        
        # Summary
        sentiments = [r['label'] for r in results]
        pos = sentiments.count('positive')
        neu = sentiments.count('neutral')
        neg = sentiments.count('negative')
        
        summary = f"✅ Analyzed {len(texts)} texts | 😊 Positive: {pos} | 😐 Neutral: {neu} | 😞 Negative: {neg}"
        
        return summary, output_data
    except Exception as e:
        return f"❌ Error: {str(e)}", []

# ============ Create Gradio Interface ============
with gr.Blocks(title="🌍 Multilingual Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 Multilingual Sentiment Analysis")
    gr.Markdown("""
    Analyze sentiment in **100+ languages** using XLM-RoBERTa.
    Supports: English, French, Arabic, Spanish, German, and many more!
    """)
    
    with gr.Tabs():
        # ============ Single Text Tab ============
        with gr.TabItem("📝 Single Text"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Enter text to analyze",
                        placeholder="Type something in any language...",
                        lines=4
                    )
                with gr.Column(scale=1):
                    analyze_btn = gr.Button("🔍 Analyze", size="lg", variant="primary")
            
            with gr.Row():
                sentiment_output = gr.Textbox(label="Sentiment", interactive=False)
                confidence_output = gr.Textbox(label="Confidence", interactive=False)
            
            gr.Textbox(label="Analyzed Text", interactive=False, show_label=True)
            
            analyze_btn.click(
                analyze_single,
                inputs=[text_input],
                outputs=[sentiment_output, confidence_output]
            )
        
        # ============ Batch Analysis Tab ============
        with gr.TabItem("📚 Batch Analysis"):
            batch_input = gr.Textbox(
                label="Enter multiple texts (one per line)",
                placeholder="Text 1...\nText 2...\nText 3...",
                lines=8
            )
            batch_btn = gr.Button("🚀 Batch Analyze", size="lg", variant="primary")
            
            batch_summary = gr.Textbox(label="Summary", interactive=False)
            batch_results = gr.Dataframe(
                headers=["Text", "Sentiment", "Confidence"],
                label="Results",
                interactive=False
            )
            
            batch_btn.click(
                analyze_batch,
                inputs=[batch_input],
                outputs=[batch_summary, batch_results]
            )
        
        # ============ Demo Tab ============
        with gr.TabItem("🧪 Demo Examples"):
            gr.Markdown("### Try these examples!")
            
            demo_examples = {
                "🌟 Positive": [
                    "This product is amazing! Best purchase ever.",
                    "Merci beaucoup, service excellent!",
                    "الطعام لذيذ جداً والخدمة رائعة",
                ],
                "😐 Neutral": [
                    "The movie was okay, nothing special.",
                    "C'est un produit ordinaire.",
                    "إنه مقبول جداً",
                ],
                "😞 Negative": [
                    "Terrible experience, never coming back.",
                    "Très déçu par la qualité.",
                    "خدمة سيئة جداً وسعر مرتفع",
                ]
            }
            
            for category, examples in demo_examples.items():
                with gr.Group():
                    gr.Markdown(f"#### {category}")
                    for example in examples:
                        with gr.Row():
                            gr.Textbox(value=example, interactive=False, show_label=False)
                            btn = gr.Button("Analyze", size="sm")
                            output = gr.Textbox(interactive=False, show_label=False)
                            
                            btn.click(
                                lambda text=example: analyze_single(text)[0],
                                outputs=[output]
                            )
    
    gr.Markdown("""
    ---
    Built with ❤️ using Gradio • XLM-RoBERTa • Multilingual Sentiment Analysis
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
