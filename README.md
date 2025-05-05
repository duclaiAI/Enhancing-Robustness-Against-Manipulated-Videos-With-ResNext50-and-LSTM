# 🧠 Enhancing Robustness Against Manipulated Videos (Deepfake Detection using ResNeXt-50 + LSTM)

🚀 A practical project developed for the **DAP391m course**. This system integrates **ResNeXt-50** for spatial feature extraction and **LSTM** for temporal modeling to detect manipulated (deepfake) videos with up to **90% accuracy**.

![image](https://github.com/user-attachments/assets/5a9ea0cc-2a11-45ac-ae86-01afbbe38c76)

🎥 **[Watch Demo & Presentation](https://www.canva.com/design/DAGhezuF-Yc/mnfiQEyuJDMSvpWe3xdgbg/edit?utm_content=DAGhezuF-Yc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)**

---

## 📁 Project Structure

```
Enhancing-Robustness-Against-Manipulated-Videos-With-ResNext50-and-LSTM    
│
├── LatexReport/          # Technical report in Springer format
├── app.py                # Web-based UI for deepfake prediction
├── train.py              # Model training script
├── requirements.txt      # Required Python packages
```

📊 **Project Flowchart:**
![Project Flow](https://github.com/user-attachments/assets/5ad6f394-15aa-4f20-8a86-da8b98b82b95)

---

## 📚 Dataset

We used a curated dataset of real and manipulated videos containing thousands of frames, balanced across classes.
![Dataset Overview](https://github.com/user-attachments/assets/6d372bd9-2b4f-4309-bdc2-d1c29dd3b03b)

---

## 🏗️ Model Architecture

The model combines:

* **ResNeXt-50**: Extracts spatial features from video frames
* **LSTM**: Models temporal consistency across sequences of frames

📌 **Main Parameters:**

![Model Diagram](https://github.com/user-attachments/assets/ecd2a9bf-3e77-4822-9f2e-e38f1975a45a)
![Model Hyperparameters](https://github.com/user-attachments/assets/fcfc3ec6-8b36-40a5-a357-c2c4ab09efc6)

---

## ✅ Results

Achieved **90% classification accuracy** on the validation set with robust performance on unseen deepfake samples.
![Results](https://github.com/user-attachments/assets/329c29fe-40b1-44b9-89ca-4e0ea4c47639)

---

## 👥 Contributors

* [Lai Le Dinh Duc](https://github.com/duclaiAI)
* [Le Minh Khoi](https://github.com/FPT-KhoiLe)

---

## 📄 Report

Read the full technical report in `LatexReport/`, formatted for **Springer publication standards**.

---

## 🛠️ Setup with Python 3.10.8

```bash
# Clone the repo
git clone https://github.com/your-repo/deepfake-detection.git
cd deepfake-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```
Download Trained Model [Here](https://drive.google.com/file/d/1JSMl_gM1ZYoZgAH45U8PzJsu-qXh2duO/view?usp=sharing) and put it in the sameple folder with app.py.
