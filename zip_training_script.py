import os
import zipfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import io

# ==============================================================================
# 1. إعدادات المشروع (استخدام نموذج صغير جداً للتجربة في الـ Sandbox)
# ==============================================================================
MODEL_ID = "gpt2" # نموذج صغير جداً وسريع للتحميل
ZIP_FILE_PATH = "/home/ubuntu/source_code.zip"
OUTPUT_DIR = "./gpt2_finetuned_on_zip"

# ==============================================================================
# 2. دالة قراءة الملفات من ZIP وتحويلها لبيانات تدريب
# ==============================================================================
def load_data_from_zip(zip_path):
    data = []
    print(f"-> Reading files from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_info in z.infolist():
            # قراءة ملفات الكود فقط (مثلاً .py و .md)
            if file_info.filename.endswith(('.py', '.md')) and not file_info.is_dir():
                with z.open(file_info) as f:
                    try:
                        content = f.read().decode('utf-8')
                        if len(content.strip()) > 50: # تجاهل الملفات الفارغة جداً
                            data.append({
                                "text": f"### File: {file_info.filename}\n\n{content}"
                            })
                    except:
                        continue
    print(f"-> Loaded {len(data)} code files from ZIP.")
    return Dataset.from_list(data)

# ==============================================================================
# 3. تشغيل التدريب
# ==============================================================================
def main():
    # 3.1. تحميل النموذج والتوكنايزر
    print(f"-> Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    # 3.2. تحميل البيانات من ZIP
    dataset = load_data_from_zip(ZIP_FILE_PATH)

    # 3.3. ترميز البيانات (Tokenization)
    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 3.4. إعداد وسائط التدريب (مكثفة قليلاً للتجربة)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=10, # زيادة عدد الدورات
        logging_steps=5,
        save_strategy="epoch", # حفظ النموذج بعد كل دورة
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
    )

    # 3.5. بدء التدريب
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("-> Starting intensive training (10 epochs) on the ZIP data...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"-> Intensive training successful! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
