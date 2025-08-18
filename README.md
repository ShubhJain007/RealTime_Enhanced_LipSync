# Real-Time Enhanced Lip-Sync Avatar

This repository implements a **real-time lip-syncing avatar system** for online meetings.  
It integrates **lip-sync generation**, **low-light enhancement**, and **face restoration** into an optimized pipeline, ensuring **high-quality, realistic, and robust avatars** even under challenging conditions.

---

## 🚀 Key Features
- **Low-Light & Exposure Correction** → Enhances video frames using **LIME**, **DUAL**, and **Zero-DCE** methods.  
- **Lip-Sync Generation** → Powered by **MuseTalk**, a GAN-based model for accurate, expressive lip movement.  
- **Face Restoration** → Uses **CodeFormer** to fix blurry lips and sharpen facial details.  
- **Real-Time Ready** → Achieves up to **29 FPS** in streaming mode.  

---

## 🛠️ Pipeline
1. **Preprocessing** → Corrects illumination in low-light/uneven lighting videos.  
2. **Lip-Sync Generation (MuseTalk)** → Aligns lips with input audio.  
3. **Enhancement (CodeFormer)** → Restores and sharpens facial details.  


---

## 📊 Results
- **Raw Input** → Low-light, static face  
- **MuseTalk** → Lip-synced but blurry  
- **CodeFormer** → Clear, sharp, and realistic  

**Performance**:  
- Real-Time Mode: **29 FPS**  
- Disk-Save Mode: **10 FPS** (I/O bottleneck)
- Result videos are present in the results folder

---

## ⚡ Installation
```bash
# Clone repo
git clone https://github.com/ShubhJain007/RealTime_Enhanced_LipSync.git
cd RealTime_Enhanced_LipSync

# Install dependencies
pip install -r requirements.txt

# Usage
python main.py --video input.mp4 --audio speech.wav --output result.mp4


