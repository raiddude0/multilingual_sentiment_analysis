import gradio as gr
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.infer import predict, predict_batch


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
        
        
        sentiments = [r['label'] for r in results]
        pos = sentiments.count('positive')
        neu = sentiments.count('neutral')
        neg = sentiments.count('negative')
        
        summary = f"✅ Analyzed {len(texts)} texts | 😊 Positive: {pos} | 😐 Neutral: {neu} | 😞 Negative: {neg}"
        
        return summary, output_data
    except Exception as e:
        return f"❌ Error: {str(e)}", []


custom_theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="slate",
).set(
    body_background_fill="#000000",
    body_text_color="#00FFFF",
    button_primary_background_fill="#00FFFF",
    button_primary_text_color="#000000",
    button_primary_background_fill_hover="#00DDDD",
    block_title_text_color="#00FFFF",
    block_label_text_color="#00FFFF",
    input_background_fill="#111111",
    input_border_color="#00FFFF",
    input_placeholder_color="#666666",
    border_color_primary="#00FFFF",
)

with gr.Blocks(title="🌍 Multilingual Sentiment Analysis", theme=custom_theme) as demo:
    gr.Markdown("# 🌍 Multilingual Sentiment Analysis")
    gr.Markdown("""
    Analyze sentiment in **100+ languages** using XLM-RoBERTa.
    Supports: English, French, Arabic, Spanish, German, and many more!
    """)
    
    with gr.Tabs():
       
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
            
            analyze_btn.click(
                analyze_single,
                inputs=[text_input],
                outputs=[sentiment_output, confidence_output]
            )
        
       
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
        
    gr.Markdown("""
    ---
    Built with ❤️ using Gradio • XLM-RoBERTa • Multilingual Sentiment Analysis
    """)

if __name__ == "__main__":
    demo.launch()
