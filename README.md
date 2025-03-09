# BlackHash Pro - Ultra-Fast Hash Cracking Tool

![BlackHash Pro](https://img.shields.io/badge/Status-Active-brightgreen.svg)  
**Developed by [gen] (Member of Black Wolf Team)**  
📅 **All Rights Reserved © 2025**  

## 🚀 مقدمة

**BlackHash Pro** هي أداة متقدمة ومبتكرة لكسر التشفير (Hash Cracking) بسرعة فائقة باستخدام أحدث التقنيات، مثل:
- المعالجة المتوازية **(Parallel Processing)**
- تسريع العتاد **(GPU Acceleration - CUDA)**
- تحسينات **FPGA** للأداء العالي
- دعم **Bloom Filters** للتحقق السريع
- قاعدة بيانات **RocksDB** للحسابات المسبقة

تم تصميم الأداة خصيصًا للباحثين في الأمن السيبراني، فرق اختبار الاختراق، والمطورين الذين يحتاجون إلى أداة قوية وعملية.

---

## ✨ الميزات الرئيسية
✅ **دعم واسع للخوارزميات**: `MD5`, `SHA-1`, `SHA-256`, `SHA-512`, `Blake2`, `XXHash`, `Argon2` وغيرها.  
✅ **تسريع باستخدام المعالج الرسومي (GPU)**: تحسين الأداء باستخدام CUDA.  
✅ **نظام موزع (Distributed Cracking)**: دعم التكامل مع **Ray Cluster** و **FPGA** لزيادة السرعة.  
✅ **مرشح بلوم (Bloom Filter)**: تقليل عدد عمليات البحث وزيادة سرعة الفحص.  
✅ **نظام قاعدة بيانات مسبقة (Precomputed DB)**: تسريع فك التشفير من خلال **RocksDB**.  
✅ **دعم قوائم الكلمات الضخمة** مع المعالجة المتوازية **(Multi-threading & SIMD)**.  
✅ **وضع الهجوم بالقوة الغاشمة (Brute-Force Attack)**: تجربة جميع الاحتمالات مع تحسينات الأداء.  
✅ **دعم التحقق من كلمات المرور باستخدام Argon2**.  

---

## 🔥 المتطلبات

- **Python 3.8+**
- مكتبات إضافية يمكن تثبيتها بالأمر التالي:

```sh
pip install -r requirements.txt
