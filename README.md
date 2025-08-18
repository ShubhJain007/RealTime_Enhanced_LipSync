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

<p align="center">
  <img src="docs/pipeline_diagram.png" alt="Pipeline Diagram" width="700">
</p>

---

## 📊 Results
- **Raw Input** → Low-light, static face  
- **MuseTalk** → Lip-synced but blurry  
- **CodeFormer** → Clear, sharp, and realistic  

**Performance**:  
- Real-Time Mode: **29 FPS**  
- Disk-Save Mode: **10 FPS** (I/O bottleneck)  

<p align="center">
  <img src="docs/results_comparison.png" alt="Results Comparison" width="700">
</p>

---

## 📂 Project Structure
