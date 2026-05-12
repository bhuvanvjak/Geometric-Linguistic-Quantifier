# Geometric Linguistic Quantifier (GLQ)

The **Geometric Linguistic Quantifier** is an advanced NLP tool designed to analyze and quantify code-mixed text (e.g., text that mixes English, Hindi, and Telugu). Instead of traditional token counting, it uses deep learning (`Sentence-Transformers` and `PyTorch`) to convert sentences into high-dimensional geometric vectors and calculates the exact linguistic composition based on cosine similarity and contrast-enhanced softmax calibration.

---

## 📂 Project Structure

- `web_quantifier.py` : The main Streamlit web application.
- `generate_anchors.py` : The backend script used to generate the baseline language vectors from raw data.
- `data/baselines/` : The directory where the mathematical "anchors" (`.pt` files) for each language are stored.
- `Roadmap.txt` : The original project development roadmap.

---

## ⚙️ Prerequisites

Before running the project, ensure you have Python installed and your virtual environment/conda environment active. You will need to install the following libraries:

```bash
pip install streamlit torch sentence-transformers pandas altair numpy
```

---

## 🚀 How to Build from Scratch

If you want to generate the mathematical language baselines yourself instead of using the pre-provided ones:

1. Open your terminal and navigate to this folder.
2. Ensure you have the raw data corpora available (the script looks for raw text data to encode).
3. Run the generator script:
   ```bash
   python generate_anchors.py
   ```
4. This script will load the `SentenceTransformer` model, encode the text for English, Hindi, and Telugu, calculate their mean vectors, normalize them, and save them as `.pt` (PyTorch tensor) files inside the `data/baselines/` directory.

---

## ▶️ How to Run the Interface

Once the baselines are generated (or if you are using the pre-existing ones provided in `data/baselines/`), you can start the application:

1. Open your terminal in this folder (`GLQ Final`).
2. Run the Streamlit command:
   ```bash
   streamlit run web_quantifier.py
   ```
3. Your web browser will automatically open the application at `http://localhost:8501`.
4. Type any code-mixed text into the input box to see the dynamic charts and language density percentages!
