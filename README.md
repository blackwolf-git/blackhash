# BlackHash Pro - Ultra-Fast Hash Cracking Tool

### Developed by [gen] (Member of Black Wolf Team)  
**All Rights Reserved © 2025**

## 📌 Overview
**BlackHash Pro** is an ultra-fast and highly optimized hash cracking tool designed for both CPU and GPU acceleration. It leverages **SIMD, CUDA, Ray Distributed Computing, and FPGA integration** to provide unmatched performance in hash cracking.

This tool supports a wide range of hashing algorithms and integrates **Bloom filters, precomputed databases, and RocksDB storage** to enhance efficiency. 

---

## 🚀 Features

✔ **Multi-Algorithm Support**: MD5, SHA1, SHA256, SHA512, Blake2, xxHash, Argon2, and more.  
✔ **Parallel Processing**: Uses multiprocessing and `ProcessPoolExecutor` for faster execution.  
✔ **GPU Acceleration**: CUDA-optimized kernel for high-speed hash computation.  
✔ **Bloom Filters**: Efficiently store and check previously computed hashes.  
✔ **Database Optimization**: Uses RocksDB for precomputed hashes.  
✔ **Distributed Cracking**: Implements `ray` for multi-node processing.  
✔ **FPGA Support**: Experimental FPGA acceleration for extreme performance.  
✔ **Error Handling & Logging**: Prevents failures and ensures a smooth experience.  

---

## ⚡ Supported Hashing Algorithms

| Algorithm  | Supported |
|------------|-----------|
| MD5        | ✅ |
| SHA-1      | ✅ |
| SHA-224    | ✅ |
| SHA-256    | ✅ |
| SHA-384    | ✅ |
| SHA-512    | ✅ |
| Blake2b    | ✅ |
| Blake2s    | ✅ |
| xxHash     | ✅ |
| Argon2     | ✅ |

---

## 🔥 Installation

### Prerequisites
- **Python 3.8+**
- **CUDA (for GPU acceleration)**
- **RocksDB**
- **Ray (for distributed processing)**

### Install Dependencies
```bash
pip install -r requirements.txt

Required System Libraries (Linux)

sudo apt-get install librocksdb-dev lz4


---

⚙️ Usage

Crack a Hash

python blackhash.py -a sha256 -H 5d41402abc4b2a76b9719d911017c592 -w wordlist.txt

Generate Precomputed Database

python blackhash.py --precompute -a sha256 -w wordlist.txt

Enable GPU Acceleration

python blackhash.py -a sha256 -H 5d41402abc4b2a76b9719d911017c592 --gpu

Use Distributed Cracking

ray start --head
python blackhash.py -a sha256 -H 5d41402abc4b2a76b9719d911017c592 --distributed


---

🛠️ Performance Optimization

🚀 SIMD Optimization

The tool processes 1M hashes per batch using optimized CPU parallelism.

🔥 GPU Acceleration

Utilizes CUDA with optimized memory coalescing for ultra-fast hash calculations.

🌍 Distributed Processing

Crack large-scale hash databases by running multiple nodes with Ray.


---

🖥️ Benchmark Results


---

⚠️ Legal Disclaimer

This tool is intended for ethical security research and password recovery only. The author and the Black Wolf Team are not responsible for any misuse of this software.


---

🏆 Credits

Developed by gen (Member of Black Wolf Team).
For updates and support, visit Black Wolf Team.
