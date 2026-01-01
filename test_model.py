import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_code(prompt, model_path="./gpt2_finetuned_on_zip"):
    print(f"-> Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except:
        # إذا لم يكتمل التدريب المكثف بعد، نستخدم النموذج الأساسي للتجربة
        print("-> Model not found in output directory, using base gpt2 for demonstration.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

    # إعداد المدخلات
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # توليد الكود
    print("-> Generating...")
    outputs = model.generate(
        **inputs, 
        max_length=150, 
        num_return_sequences=1, 
        temperature=0.7, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

if __name__ == "__main__":
    # مثال لتجربة النموذج
    test_prompt = "### File: mingpt/model.py\n\nclass GPT(nn.Module):"
    print("\n" + "="*30)
    print("TESTING MODEL GENERATION")
    print("="*30)
    print(f"Prompt: {test_prompt}")
    
    generated_code = generate_code(test_prompt)
    
    print("\n" + "="*30)
    print("GENERATED CODE:")
    print("="*30)
    print(generated_code)
