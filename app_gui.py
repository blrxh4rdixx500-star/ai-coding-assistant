import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# المسار الافتراضي للنموذج المدرب
DEFAULT_MODEL_PATH = "./gpt2_finetuned_on_zip"

def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            return None, None, f"Error: Model path {model_path} not found."
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return tokenizer, model, "Model loaded successfully!"
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

def generate_code(prompt, model_path, max_length, temperature):
    tokenizer, model, status = load_model(model_path)
    if model is None:
        return status
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# بناء واجهة Gradio
with gr.Blocks(title="AI Coding Assistant GUI") as demo:
    gr.Markdown("# 🤖 AI Coding Assistant")
    gr.Markdown("تطبيق واجهة رسومية لتجربة نموذج الذكاء الاصطناعي الخاص بك للبرمجة.")
    
    with gr.Tab("تجربة النموذج (Inference)"):
        with gr.Row():
            with gr.Column():
                model_input = gr.Textbox(label="مسار النموذج", value=DEFAULT_MODEL_PATH)
                prompt_input = gr.Textbox(label="أدخل بداية الكود (Prompt)", lines=5, placeholder="e.g., def calculate_fibonacci(n):")
                max_len_slider = gr.Slider(minimum=50, maximum=1024, value=256, step=1, label="أقصى طول للكود")
                temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="درجة الإبداع (Temperature)")
                btn = gr.Button("توليد الكود", variant="primary")
            
            with gr.Column():
                output_display = gr.Code(label="الكود المولد", language="python", lines=15)
        
        btn.click(fn=generate_code, inputs=[prompt_input, model_input, max_len_slider, temp_slider], outputs=output_display)

    with gr.Tab("تعليمات التشغيل"):
        gr.Markdown("""
        ### كيفية استخدام الواجهة:
        1. **مسار النموذج**: تأكد من وضع المسار الصحيح للمجلد الذي يحتوي على النموذج المدرب.
        2. **البرومبت**: اكتب بداية الكود أو وصفاً للمهمة البرمجية.
        3. **توليد الكود**: اضغط على الزر وانتظر حتى يقوم النموذج بإكمال الكود.
        
        ### متطلبات النظام:
        - نظام لينكس (Ubuntu/Debian موصى به).
        - بطاقة رسومية (GPU) تدعم CUDA للحصول على أفضل أداء.
        """)

if __name__ == "__main__":
    # تشغيل الواجهة الرسومية
    demo.launch(share=True)
