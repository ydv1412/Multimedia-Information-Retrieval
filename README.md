# Audio Information Retrieval System

This repository contains the implementation of an audio information retrieval system developed using the AudioMNIST dataset.

The project investigates multiple audio feature extraction techniques, including MFCCs, YAMNet, and Wav2Vec2, and compares their effectiveness for similarity-based audio retrieval. Feature embeddings are indexed using FAISS, and retrieval is performed using cosine similarity.

The system supports audio-based queries and is evaluated using standard information retrieval metrics such as Precision@K and Mean Reciprocal Rank (MRR). Finally a small Streamlit interface is built to demonstrate the end-to-end audio query and retrieval pipeline.
