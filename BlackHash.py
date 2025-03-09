import hashlib
import argparse
import itertools
import requests
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

# تمكين الكاش لتحسين أداء توليد الهاش
@lru_cache(maxsize=None)
def generate_hash(algorithm, text):
    if algorithm in hashlib.algorithms_guaranteed:
        return hashlib.new(algorithm, text.encode('utf-8')).hexdigest()
    raise ValueError(f"خوارزمية غير مدعومة: {algorithm}")

# مولد القاموس لتحسين استهلاك الذاكرة
def wordlist_generator(wordlist_path):
    try:
        with open(wordlist_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()
    except FileNotFoundError:
        raise FileNotFoundError("[❌] ملف القاموس غير موجود!")

# كسر الهاش باستخدام قاموس كلمات مع معالجة متوازية
def crack_hash_with_dict(hash_value, algorithm, wordlist, workers=None):
    # حساب عدد الكلمات لتحديد التقدم
    try:
        with open(wordlist, "r", encoding="utf-8") as f:
            total_words = sum(1 for _ in f)
    except FileNotFoundError:
        return "[❌] ملف القاموس غير موجود!"
    
    # تحديد عدد العمليات المتوازية
    workers = min(4, workers or os.cpu_count() or 1)
    
    # حجم القطعة الأمثل
    chunk_size = max(1, total_words // (workers * 4))
    
    # مولد القطع
    def chunk_producer():
        chunk = []
        for word in wordlist_generator(wordlist):
            chunk.append(word)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for chunk in chunk_producer():
            future = executor.submit(check_chunk, hash_value, algorithm, chunk)
            futures[future] = len(chunk)

        with tqdm(total=total_words, desc="[🌀] فحص الكلمات", unit=" كلمة") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(futures[future])
                if result:
                    executor.shutdown(cancel_futures=True)
                    return f"[✔] كلمة المرور: {result}"
    
    return "[❌] لم يتم العثور على تطابق!"

# وظيفة مساعدة للهجوم بالقاموس
def check_chunk(hash_value, algorithm, chunk):
    for word in chunk:
        if generate_hash(algorithm, word) == hash_value:
            return word
    return None

# وظيفة مساعدة للهجوم بالقوة الغاشمة
def brute_worker(params):
    hash_value, algorithm, charset, length, start, end = params
    total_chars = len(charset)
    for i in range(start, end):
        word = ''.join([charset[(i // (total_chars ** j)) % total_chars] for j in range(length)])
        if generate_hash(algorithm, word) == hash_value:
            return word
    return None

# كسر الهاش بالقوة الغاشمة مع معالجة متوازية
def crack_hash_brute_force(hash_value, algorithm, charset, max_length, workers=None):
    charset = ''.join(sorted(set(charset)))
    total_chars = len(charset)
    
    # تحديد عدد العمليات المتوازية
    workers = min(4, workers or os.cpu_count() or 1)
    
    for length in range(1, max_length + 1):
        total = total_chars ** length
        chunk_size = max(1, total // (workers * 2))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(0, total, chunk_size):
                end_idx = min(i + chunk_size, total)
                params = (hash_value, algorithm, charset, length, i, end_idx)
                futures.append(executor.submit(brute_worker, params))

            with tqdm(total=total, desc=f"[🌀] فحص الطول {length}", unit=" توليفة") as pbar:
                for future in as_completed(futures):
                    pbar.update(chunk_size)
                    result = future.result()
                    if result:
                        executor.shutdown(cancel_futures=True)
                        return f"[✔] كلمة المرور: {result}"
    
    return "[❌] لم يتم العثور على تطابق!"

# كسر الهاش باستخدام خدمات الإنترنت
def crack_hash_online(hash_value):
    try:
        response = requests.post(
            "https://api.crackstation.net/crack",
            json={"hash": hash_value},
            headers={"Content-Type": "application/json"}
        )
        data = response.json()
        if data.get("found"):
            return f"[✔] كلمة المرور: {data['password']}"
        return "[❌] لم يتم العثور على تطابق في قاعدة البيانات!"
    except Exception as e:
        return f"[❌] خطأ في الاتصال: {str(e)}"

# واجهة الأوامر الرئيسية
def main():
    parser = argparse.ArgumentParser(description="BlackHash - أداة متقدمة لكسر الهاشات")
    parser.add_argument("-m", "--mode", choices=["generate", "crack-dict", "crack-brute", "crack-online"], required=True, help="وضع التشغيل")
    parser.add_argument("-a", "--algorithm", choices=hashlib.algorithms_guaranteed, help="خوارزمية الهاش")
    parser.add_argument("-t", "--text", help="النص المراد توليد الهاش منه")
    parser.add_argument("-hv", "--hash_value", help="الهاش المراد كسره")
    parser.add_argument("-w", "--wordlist", help="مسار ملف القاموس")
    parser.add_argument("-c", "--charset", default="abcdefghijklmnopqrstuvwxyz0123456789", help="مجموعة الأحرف للهجوم")
    parser.add_argument("-l", "--max_length", type=int, default=6, help="الحد الأقصى لطول كلمة المرور")
    parser.add_argument("--workers", type=int, help="عدد العمليات المتوازية (الافتراضي: عدد الأنوية)")

    args = parser.parse_args()

    if args.mode == "generate":
        if not args.text or not args.algorithm:
            print("[❌] يلزم تحديد النص والخوارزمية!")
            return
        print(f"[🔹] الهاش ({args.algorithm}): {generate_hash(args.algorithm, args.text)}")

    elif args.mode == "crack-dict":
        if not all([args.hash_value, args.algorithm, args.wordlist]):
            print("[❌] يلزم تحديد الهاش والخوارزمية وملف القاموس!")
            return
        print(crack_hash_with_dict(args.hash_value, args.algorithm, args.wordlist, args.workers))

    elif args.mode == "crack-brute":
        if not args.hash_value or not args.algorithm:
            print("[❌] يلزم تحديد الهاش والخوارزمية!")
            return
        print(crack_hash_brute_force(args.hash_value, args.algorithm, args.charset, args.max_length, args.workers))

    elif args.mode == "crack-online":
        if not args.hash_value:
            print("[❌] يلزم تحديد الهاش!")
            return
        print(crack_hash_online(args.hash_value))

if __name__ == "__main__":
    main()
