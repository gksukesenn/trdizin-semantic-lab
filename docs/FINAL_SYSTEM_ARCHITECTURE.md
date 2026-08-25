# Final Sistem Mimarisi

## Veri akışı

```text
TR Dizin API
      │
      ▼
50.000 Türkçe PAPER
      │
      ▼
JSONL Dataset
      │
      ├──────────────► Metadata
      │
      ▼
TR-MTEB
768D normalized embeddings
      │
      ├─────────────────────────────┐
      │                             │
      ▼                             ▼
UMAP 10D                       Qdrant
      │                             │
      ▼                             ▼
HDBSCAN Leaf                 Semantic Search
      │                             │
      ▼                             ▼
Topic Clusters                Top-N Articles
      │                             │
      └──────────────┬──────────────┘
                     ▼
              Search Result
          + Primary Topic
          + Secondary Topic
          + Cluster
          + Metadata


Arama katmanı
                    User Query
                        │
                        ▼
                     TR-MTEB
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       Abstract Qdrant       Title Qdrant
              │                   │
              └────────┬──────────┘
                       │
                 Experimental
                       RRF
                       ▲
                       │
                  BM25 Sparse


Demo tarafında **çok basit ama temiz** bir ekran yapacağız:

```text
┌────────────────────────────────────────────────────────────┐
│ TR Dizin Semantic Explorer                 50.000 Makale   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  [ Yapay zekânın eğitimde kullanılması................. ] │
│                                               [ Ara ]       │
│                                                            │
│  Mod:  ● Semantic     ○ Hybrid Experimental                │
│                                                            │
│  Yıl: [2020] - [2025]      Database: [SOCIAL ▼]           │
│                                                            │
├────────────────────────────────────────────────────────────┤
│ 1. Makale Başlığı                              Score .71   │
│                                                            │
│ Primary: Eğitim Araştırmaları                              │
│ Secondary: Yapay Zeka                                     │
│ 2024 • SOCIAL • Direct                                    │
│                                                            │
│ Abstract başlangıcı...                                    │
├────────────────────────────────────────────────────────────┤
│ 2. ...                                                    │
└────────────────────────────────────────────────────────────┘