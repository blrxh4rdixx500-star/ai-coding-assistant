import os
from datasets import load_dataset, IterableDataset
from typing import Iterator, Dict, Any

# ==============================================================================
# 1. إعداد المسارات والتهيئة
# ==============================================================================

# المسار المتوقع لملفات البيانات الخام (JSONL)
# يجب على المستخدم وضع جميع ملفات الـ 300GB+ هنا.
RAW_DATA_PATH = "/home/ubuntu/raw_data/*.jsonl"

# اسم النموذج الذي سيتم تدريبه (يستخدم لتحديد قالب التنسيق)
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-base"

# ==============================================================================
# 2. دالة لتنسيق البيانات (Prompt Formatting)
# ==============================================================================

def format_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    تقوم بتنسيق مثال واحد من البيانات ليناسب نموذج اللغة الكبير (LLM).
    يجب أن يكون الناتج نصاً واحداً يحتوي على التعليمات والحل.

    افتراض: البيانات الخام تحتوي على عمودين: 'instruction' و 'code_solution'.
    """
    instruction = example.get("instruction", "")
    code_solution = example.get("code_solution", "")

    # قالب تنسيق بسيط (يمكن تعديله ليناسب نموذج DeepSeek أو Llama)
    # هذا التنسيق يحاكي محادثة بين مستخدم ونموذج.
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{code_solution}"
    
    return {"text": text}

# ==============================================================================
# 3. دالة تحميل البيانات بالتدفق (Streaming)
# ==============================================================================

def load_streaming_dataset(data_path: str) -> IterableDataset:
    """
    تحميل البيانات الضخمة باستخدام وضع التدفق (streaming=True) لتقليل استهلاك الذاكرة.
    
    :param data_path: مسار الملفات (يمكن أن يكون نمط glob مثل: /path/*.jsonl)
    :return: IterableDataset جاهز للمعالجة
    """
    print(f"-> Loading dataset from: {data_path} with streaming...")
    
    # استخدام 'json' كـ builder إذا كانت الملفات بصيغة JSONL
    # يمكن تغييرها إلى 'parquet' إذا كانت الملفات بصيغة Parquet
    dataset = load_dataset(
        "json",
        data_files={"train": data_path},
        split="train",
        streaming=True,  # المفتاح لتدفق البيانات الضخمة
    )
    
    # تطبيق دالة التنسيق على البيانات المتدفقة
    # يتم تطبيق الدالة على كل مثال عند قراءته، مما يحافظ على كفاءة الذاكرة.
    processed_dataset = dataset.map(
        format_example,
        remove_columns=list(dataset.features.keys()), # إزالة الأعمدة الأصلية
        batched=False # المعالجة مثالاً بمثال
    )
    
    return processed_dataset

# ==============================================================================
# 4. مثال على الاستخدام
# ==============================================================================

if __name__ == "__main__":
    # ملاحظة: هذا الجزء لن يعمل إلا إذا كان لديك ملفات JSONL في المسار المحدد.
    # يتم استخدامه هنا فقط لتوضيح كيفية استخدام الدالة.
    
    # لغرض التجربة، سنفترض وجود ملف واحد وهمي
    if not os.path.exists(os.path.dirname(RAW_DATA_PATH.split('*')[0])):
        print(f"Directory for raw data not found. Please create it and place your data.")
        # إنشاء دليل وهمي
        os.makedirs(os.path.dirname(RAW_DATA_PATH.split('*')[0]), exist_ok=True)
        # إنشاء ملف وهمي صغير للتجربة
        dummy_data = [
            {"instruction": "Write a Python function to calculate the factorial of a number.", "code_solution": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"},
            {"instruction": "Create a JavaScript function to reverse a string.", "code_solution": "const reverseString = (str) => str.split('').reverse().join('');"}
        ]
        import json
        with open("/home/ubuntu/raw_data/dummy_shard_000.jsonl", "w", encoding="utf-8") as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")
        
        print("Created a dummy data file for demonstration: /home/ubuntu/raw_data/dummy_shard_000.jsonl")

    try:
        # تحميل البيانات
        streaming_data = load_streaming_dataset(RAW_DATA_PATH)
        
        print("\n-> First 5 formatted examples (streaming):")
        # طباعة أول 5 أمثلة دون تحميل كل الـ 300GB
        for i, example in enumerate(streaming_data):
            if i >= 5:
                break
            print("-" * 20)
            print(example["text"])
            
    except Exception as e:
        print(f"\nAn error occurred during data loading: {e}")
        print("Please ensure your data files are correctly placed and formatted as JSONL.")

# ملاحظة للمستخدم:
# يجب أن تكون كل سطر في ملفات JSONL عبارة عن كائن JSON صالح.
# يجب أن تحتوي كائنات JSON على مفتاحي 'instruction' و 'code_solution'.
