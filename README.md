# 🎙️ Audio Information Retrieval System

An end-to-end **content-based audio retrieval system** that compares multiple audio representation techniques — **MFCC, YAMNet, and Wav2Vec2** — and uses **FAISS vector search** to retrieve acoustically similar audio samples.

The system was developed using the **AudioMNIST dataset (30,000 spoken-digit recordings)** and evaluated using both dataset samples and independently recorded audio queries.

A **Streamlit application** was also developed to demonstrate the complete pipeline:

**Audio Query → Preprocessing → Embedding → FAISS Search → Top-K Similar Audio**

---

##  Project Objective

Traditional Information Retrieval systems primarily operate on text. Audio retrieval introduces an additional challenge: raw audio signals must first be transformed into meaningful numerical representations before they can be efficiently searched.

The goal of this project was therefore to investigate:

> **Which audio representation produces the most effective similarity-based retrieval?**

Three different representation approaches were explored:

- MFCC — handcrafted acoustic features
- YAMNet — pretrained general-purpose audio embeddings
- Wav2Vec2 — pretrained speech representations

Their retrieval performance was then compared using a common **FAISS-based vector search pipeline**.

---

## System Architecture

The system consists of two main pipelines.

### Offline Indexing Pipeline

```text
AudioMNIST Dataset
        │
        ▼
Audio Preprocessing
        │
        ▼
Feature Extraction
        │
        ├── MFCC 13D
        ├── MFCC 30D
        ├── YAMNet 1024D
        └── Wav2Vec2 768D
        │
        ▼
Vector Normalization
        │
        ▼
     FAISS Index
```

### Query Pipeline

```text
User Audio Query
        │
        ▼
Preprocessing
        │
        ▼
Wav2Vec2 Embedding
        │
        ▼
FAISS Similarity Search
        │
        ▼
Top-K Similar Audio Samples
```

The same representation pipeline is used for both the indexed audio collection and incoming queries so that they can be compared in the same embedding space.

---

##  Dataset

The project uses the **AudioMNIST** dataset.

The dataset contains:

- **30,000 audio recordings**
- Spoken digits from **0–9**
- **3,000 recordings per digit**
- Speakers with different ages, genders, and accents

Exploratory analysis showed that the dataset is relatively clean, with consistent recording duration and loudness.

However, the dataset also contains demographic and recording-domain differences that become important when evaluating the system using independently recorded queries.

---

##  Exploratory Data Analysis

Before building the retrieval system, I analysed several characteristics of the audio collection:

- Zero Crossing Rate (ZCR)
- RMS energy
- Audio duration
- Waveforms
- Spectrograms
- MFCC representations
- Speaker gender distribution
- Speaker accent distribution

Most AudioMNIST recordings were approximately **0.5–0.8 seconds long**, with a peak around 0.65 seconds.

The dataset recordings were generally clean, with minimal background noise and relatively consistent loudness.

---

##  Query Preprocessing

The AudioMNIST recordings required relatively little preprocessing because of their consistent recording conditions.

Real-world recorded queries, however, differed considerably from the dataset.

The following preprocessing pipeline was therefore applied to user queries:

1. **Resampling** to 16 kHz
2. **Silence trimming**
3. **RMS normalization**
4. **Duration standardization** to one second
5. **Light spectral noise reduction**

Audio augmentation was also investigated, but it reduced retrieval performance and was therefore excluded from the final pipeline.

---

#  Audio Representation Experiments

One of the main goals of the project was to compare traditional audio features with pretrained deep-learning representations.

## 1. MFCC

**Mel-Frequency Cepstral Coefficients (MFCCs)** were used as a traditional audio representation baseline.

Two configurations were evaluated:

- MFCC — **13 dimensions**
- MFCC — **30 dimensions**

Frame-level MFCC features were averaged over time to obtain one fixed-dimensional representation for each recording.

---

## 2. YAMNet

**YAMNet** is a pretrained audio-event classification model based on MobileNetV1 and trained on AudioSet.

Instead of using its final classification output, intermediate embeddings were extracted and averaged over time.

This produced a:

**1024-dimensional audio representation**

YAMNet captures general acoustic properties well, but the embedding-space analysis showed considerable overlap between spoken-digit classes.

---

## 3. Wav2Vec2

**Wav2Vec2** learns speech representations directly from raw audio using self-supervised pretraining.

The final hidden representations were averaged over time to create a:

**768-dimensional embedding**

Compared with MFCC and YAMNet, Wav2Vec2 produced much clearer separation between spoken-digit classes.

This was also reflected in the retrieval results.

---

#  Vector Search with FAISS

After feature extraction, the embeddings were stored in separate **FAISS indexes**.

FAISS enables efficient nearest-neighbour search over high-dimensional vectors.

The project experimented with:

- Euclidean distance
- Cosine similarity

For cosine-based retrieval, vectors were L2-normalized and searched using a FAISS inner-product index.

The best-performing final configuration was:

> **Wav2Vec2 embeddings + FAISS cosine similarity**

---

#  Evaluation

To evaluate the system beyond querying with samples from the original dataset, I recorded **10 independent audio queries — one for each digit from 0 to 9**.

Each query was processed using the same feature extraction pipeline and searched against the 30,000-sample AudioMNIST index.

Three Information Retrieval metrics were used:

### Precision@10

Measures the proportion of relevant samples among the first 10 retrieved results.

### Mean Precision

Average retrieval precision across all queries.

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant result appears in the ranked retrieval results.

---

##  Experimental Results

| Representation | Precision@10 Dataset / User | Mean Precision Dataset / User | MRR Dataset / User |
|---|---:|---:|---:|
| MFCC 13D | 0.9027 / 0.3100 | 0.9445 / 0.3125 | 0.9690 / 0.3125 |
| MFCC 30D | 0.9560 / 0.3000 | 0.9797 / 0.3143 | 0.9916 / 0.3167 |
| YAMNet 1024D | 0.6235 / 0.2600 | 0.7562 / 0.3145 | 0.8363 / 0.3510 |
| **Wav2Vec2** | **0.9932 / 0.7000** | **0.9962 / 0.7333** | **0.9973 / 0.7333** |

**Wav2Vec2 significantly outperformed the other representations**, particularly when retrieving results for independently recorded user queries.

---

##  Embedding Space Analysis

To better understand the differences between representation methods, I visualized their embeddings using **t-SNE**.

The experiments showed:

**MFCC**

Some grouping between digits was visible, but several classes overlapped.

**YAMNet**

The digit classes were distributed broadly throughout the embedding space with substantial overlap.

**Wav2Vec2**

The digit classes formed much clearer and more compact clusters with relatively little overlap.

This provided a useful visual explanation for Wav2Vec2's stronger retrieval performance.

---

##  Dataset vs Real-World Query Gap

One particularly interesting result was the large difference between retrieval using AudioMNIST samples and retrieval using independently recorded queries.

For example, Wav2Vec2 achieved:

```text
Precision@10

AudioMNIST query       0.9932
Recorded user query    0.7000
```

To investigate this, I projected both dataset and query embeddings into the same space.

The recorded query embeddings appeared shifted relative to the main AudioMNIST clusters.

Possible causes include:

- different recording devices;
- background noise;
- speaker characteristics;
- accent differences;
- different recording environments.

This experiment highlighted an important limitation of embedding-based retrieval systems:

> **Strong performance on in-domain data does not necessarily translate directly to real-world queries from a different domain.**

---

#  Streamlit Application

To demonstrate the complete retrieval pipeline, I developed a small **Streamlit application**.

The user can:

1. Record or upload an audio query
2. Preprocess the audio
3. Extract its Wav2Vec2 embedding
4. Search the FAISS index
5. Retrieve the most similar AudioMNIST recordings

```text
🎙️ Record / Upload Audio
          ↓
     Preprocessing
          ↓
       Wav2Vec2
          ↓
    FAISS Vector Search
          ↓
   🔊 Top-K Audio Results
```

---

#  Demo

>  **Demo video:** Coming soon

---

# 🛠️ Tech Stack

**Language**

- Python

**Audio Processing**

- Librosa
- NumPy

**Representation Learning**

- MFCC
- YAMNet
- Wav2Vec2
- PyTorch / Transformers

**Information Retrieval**

- FAISS
- Cosine Similarity
- Nearest-Neighbour Search

**Analysis & Visualization**

- Pandas
- Matplotlib
- t-SNE

**Application**

- Streamlit

---

# 📑 Project Report

The complete project report contains the methodology, exploratory analysis, embedding visualizations, retrieval experiments, evaluation and discussion of the results.

👉 **[View Full Project Report](./Report_IR.pdf)**

---

#  Repository Structure

```text
Multimedia-Information-Retrieval/
│
├── MIR_Audio_Preprocessing.ipynb   # EDA and audio preprocessing
├── MIR_notebook.ipynb              # Embedding, indexing and evaluation
├── UI.py                            # Streamlit application
├── Report_IR.pdf                    # Complete project report
└── README.md
```

---

#  Running the Project

Clone the repository:

```bash
git clone https://github.com/ydv1412/Multimedia-Information-Retrieval.git
cd Multimedia-Information-Retrieval
```

Install the required dependencies and run the Streamlit interface:

```bash
streamlit run UI.py
```

> The exact environment and model dependencies used in the notebooks should be installed before running the complete retrieval pipeline.

---

#  Key Findings

The experiments produced three main findings:

1. **Wav2Vec2 representations were considerably more effective for spoken-digit retrieval than MFCC and YAMNet embeddings.**

2. **Wav2Vec2 + FAISS cosine similarity produced the best overall retrieval system.**

3. A substantial performance gap appeared between in-domain AudioMNIST queries and independently recorded audio, highlighting the importance of **domain shift** in real-world retrieval systems.

---

#  Future Work

The system could be extended in several directions:

- Text-to-audio retrieval
- Combined text + audio queries
- Larger and more diverse audio collections
- Improved robustness to accents and recording environments
- Domain adaptation for user-recorded audio
- Approximate FAISS indexes for larger-scale collections
- Learned reranking of retrieved audio samples

A particularly interesting extension would be developing a **multimodal retrieval system** capable of searching the same collection using either natural-language descriptions or audio examples.

---

#  Author

**Shri Prakash Yadav**  
M.Sc. Data Science  
University of Naples Federico II
