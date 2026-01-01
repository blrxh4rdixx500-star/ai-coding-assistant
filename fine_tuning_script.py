import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, IterableDataset
from typing import Dict, Any

# ==============================================================================
# 1. إعدادات المشروع
# ==============================================================================

# اسم النموذج الأساسي (DeepSeek-Coder هو خيار ممتاز للبرمجة)
MODEL_ID = "deepseek-ai/deepseek-coder-7b-base"
# المسار الذي تم إعداد البيانات فيه في السكريبت السابق
DATA_PATH = "/home/ubuntu/raw_data/*.jsonl"
# مسار حفظ النموذج المدرب
OUTPUT_DIR = "./deepseek_coder_finetuned"

# ==============================================================================
# 2. إعدادات التدريب (يمكن تعديلها حسب الموارد المتاحة)
# ==============================================================================

# إعدادات QLoRA (لتقليل استهلاك الذاكرة)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"] # وحدات الانتباه

# إعدادات التدريب
BATCH_SIZE = 4 # يجب أن يكون صغيراً جداً عند استخدام QLoRA
GRADIENT_ACCUMULATION_STEPS = 8 # لزيادة حجم الدفعة الفعال (Effective Batch Size = 4 * 8 = 32)
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1 # لتدريب أولي، يمكن زيادته لاحقاً
MAX_SEQ_LENGTH = 1024 # طول التسلسل الأقصى (يمكن زيادته إذا كانت الموارد تسمح)

# ==============================================================================
# 3. تحميل البيانات بالتدفق (Streaming)
# ==============================================================================

def load_streaming_dataset(data_path: str) -> IterableDataset:
    """
    تحميل البيانات الضخمة باستخدام وضع التدفق (streaming=True).
    (نفس الدالة المستخدمة في سكريبت المعالجة)
    """
    print(f"-> Loading dataset from: {data_path} with streaming...")
    
    # يجب أن تكون ملفات البيانات جاهزة ومُنسقة كـ JSONL
    dataset = load_dataset(
        "json",
        data_files={"train": data_path},
        split="train",
        streaming=True,
    )
    
    # تطبيق دالة التنسيق (يجب أن تكون مطابقة لما تم في سكريبت المعالجة)
    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        instruction = example.get("instruction", "")
        code_solution = example.get("code_solution", "")
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{code_solution}"
        return {"text": text}

    processed_dataset = dataset.map(
        format_example,
        remove_columns=list(dataset.features.keys()),
        batched=False
    )
    
    return processed_dataset

# ==============================================================================
# 4. دالة الترميز (Tokenization)
# ==============================================================================

def tokenize_function(example):
    """
    تقوم بتحويل النص إلى رموز (Tokens) يمكن للنموذج فهمها.
    """
    # استخدام التوكنايزر المعرف عالمياً
    global tokenizer
    
    # إضافة علامة نهاية التسلسل (EOS)
    output = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    
    # يجب أن تكون التسميات (labels) هي نفس المدخلات (input_ids) لتدريب Causal LM
    output["labels"] = output["input_ids"].copy()
    return output

# ==============================================================================
# 5. دالة التدريب الرئيسية
# ==============================================================================

def main():
    # 5.1. تحميل النموذج والتوكنايزر
    print(f"-> Loading model and tokenizer: {MODEL_ID}")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # تعيين رمز الحشو ليكون رمز نهاية التسلسل

    # تحميل النموذج بـ 4-bit (QLoRA)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16, # استخدام bfloat16 لتحسين الأداء
        device_map="auto",
    )
    
    # إعداد النموذج لتدريب QLoRA
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    # إعداد إعدادات LoRA
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    
    # تطبيق إعدادات LoRA على النموذج
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5.2. تحميل ومعالجة البيانات
    streaming_dataset = load_streaming_dataset(DATA_PATH)
    
    # تطبيق الترميز على البيانات المتدفقة
    tokenized_dataset = streaming_dataset.map(
        tokenize_function,
        remove_columns=["text"], # إزالة عمود النص بعد الترميز
        batched=False,
    )
    
    # 5.3. إعداد وسائط التدريب
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False, # استخدام bfloat16 بدلاً من fp16
        bf16=True,
        optim="paged_adamw_8bit", # مُحسِّن مُحسّن للذاكرة
        # إعدادات إضافية للتدريب الموزع (DeepSpeed/Accelerate)
        # يمكن تفعيلها عند التشغيل عبر accelerate launch
        # deepspeed="ds_config.json", 
    )

    # 5.4. إنشاء المدرب (Trainer) وبدء التدريب
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("-> Starting training...")
    trainer.train()

    # 5.5. حفظ النموذج النهائي
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"-> Training complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    # يجب تشغيل هذا السكريبت باستخدام accelerate launch
    # مثال: accelerate launch fine_tuning_script.py
    # تأكد من تثبيت accelerate و peft و bitsandbytes
    main()
