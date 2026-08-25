import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hdbscan
import joblib
import numpy as np
import torch
import umap
from hdbscan.prediction import approximate_predict
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score

from day24_compare_noise_assignment_methods import (
    build_cluster_representations,
    build_subject_information,
    get_output_directory,
    load_articles,
    load_embeddings,
    load_h01_assignments,
    validate_alignment,
)
from day25_build_h01_centroid_pipeline import (
    classify_relative_confidence,
    classify_topic_structure,
)


MODEL_ID = "trmteb/turkish-embedding-model-fine-tuned"
MODEL_TOKEN_LIMIT = 512

INPUT_FILENAME = "day26_new_abstract.txt"
RUNTIME_FILENAME = "day26_h01_runtime.joblib"
RESULT_FILENAME = "day26_new_abstract_result.json"

UMAP_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
RANDOM_SEED = 42

HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5
HDBSCAN_SELECTION_METHOD = "eom"


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def get_input_path() -> Path:
    """Yeni abstract dosyasının yolunu döndürür."""

    return (
        get_project_root()
        / "data"
        / "input"
        / INPUT_FILENAME
    )


def get_runtime_path() -> Path:
    """Kaydedilmiş UMAP + HDBSCAN çalışma zamanını döndürür."""

    runtime_directory = (
        get_output_directory()
        / "day26_runtime"
    )

    runtime_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        runtime_directory
        / RUNTIME_FILENAME
    )


def get_result_path() -> Path:
    """Yeni abstract sonucunun kaydedileceği yolu döndürür."""

    return (
        get_output_directory()
        / RESULT_FILENAME
    )


def load_new_abstract() -> str:
    """Dosyadan yeni Türkçe abstractı okur."""

    input_path = get_input_path()

    if not input_path.exists():
        raise FileNotFoundError(
            "Yeni abstract dosyası bulunamadı:\n"
            f"{input_path}"
        )

    abstract_text = " ".join(
        input_path.read_text(
            encoding="utf-8"
        ).split()
    ).strip()

    if not abstract_text:
        raise ValueError(
            "Yeni abstract dosyası boş."
        )

    if len(abstract_text) < 100:
        print(
            "\nUYARI: Girilen metin oldukça kısa. "
            "Tam abstract kullanılması daha güvenilir olur."
        )

    return abstract_text


def build_label_mapping(
    new_labels: np.ndarray,
    reference_labels: np.ndarray,
) -> Dict[int, int]:
    """
    Yeniden eğitilen HDBSCAN cluster ID'lerini,
    Day 17'de kaydedilen H01 cluster ID'leriyle eşleştirir.

    HDBSCAN cluster numaralarının kendi başına anlamı yoktur.
    Bu nedenle kümeler, ortak makale sayısı en yüksek olacak
    biçimde Hungarian algoritmasıyla eşleştirilir.
    """

    new_cluster_ids = sorted(
        {
            int(label)
            for label in new_labels
            if int(label) >= 0
        }
    )

    reference_cluster_ids = sorted(
        {
            int(label)
            for label in reference_labels
            if int(label) >= 0
        }
    )

    if len(new_cluster_ids) != len(
        reference_cluster_ids
    ):
        raise ValueError(
            "Yeniden oluşturulan HDBSCAN cluster sayısı "
            "referans H01 sonucuyla uyuşmuyor.\n"
            f"Yeni cluster sayısı     : {len(new_cluster_ids)}\n"
            f"Referans cluster sayısı : {len(reference_cluster_ids)}"
        )

    overlap_matrix = np.zeros(
        (
            len(new_cluster_ids),
            len(reference_cluster_ids),
        ),
        dtype=np.int32,
    )

    for new_position, new_cluster_id in enumerate(
        new_cluster_ids
    ):
        new_mask = (
            new_labels == new_cluster_id
        )

        for (
            reference_position,
            reference_cluster_id,
        ) in enumerate(reference_cluster_ids):
            overlap_matrix[
                new_position,
                reference_position,
            ] = int(
                np.sum(
                    new_mask
                    & (
                        reference_labels
                        == reference_cluster_id
                    )
                )
            )

    row_indices, column_indices = (
        linear_sum_assignment(
            -overlap_matrix
        )
    )

    mapping: Dict[int, int] = {}

    for row_index, column_index in zip(
        row_indices,
        column_indices,
    ):
        mapping[
            new_cluster_ids[
                int(row_index)
            ]
        ] = reference_cluster_ids[
            int(column_index)
        ]

    if len(mapping) != len(
        new_cluster_ids
    ):
        raise ValueError(
            "Bütün HDBSCAN clusterları referans "
            "clusterlarla eşleştirilemedi."
        )

    return mapping


def apply_label_mapping(
    labels: np.ndarray,
    label_mapping: Dict[int, int],
) -> np.ndarray:
    """Yeni label değerlerini referans H01 ID'lerine dönüştürür."""

    mapped_labels = np.full(
        labels.shape,
        -1,
        dtype=np.int32,
    )

    for row_index, label in enumerate(
        labels
    ):
        integer_label = int(label)

        if integer_label < 0:
            mapped_labels[row_index] = -1
        else:
            mapped_labels[row_index] = (
                label_mapping[
                    integer_label
                ]
            )

    return mapped_labels


def prepare_runtime(
    embeddings: np.ndarray,
    reference_assignments: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    UMAP ve HDBSCAN H01 modelini hazırlar.

    İlk çalıştırmada eğitir ve kaydeder.
    Sonraki çalıştırmalarda kayıtlı nesneleri yükler.
    """

    runtime_path = get_runtime_path()

    if runtime_path.exists():
        print("\nKayıtlı Day 26 runtime yükleniyor...")

        runtime = joblib.load(
            runtime_path
        )

        required_keys = {
            "umap_reducer",
            "hdbscan_clusterer",
            "label_mapping",
            "adjusted_rand_index",
            "mapped_exact_agreement",
        }

        missing_keys = (
            required_keys
            - set(runtime.keys())
        )

        if missing_keys:
            raise ValueError(
                "Runtime dosyasında eksik alanlar var: "
                + ", ".join(
                    sorted(missing_keys)
                )
            )

        return runtime

    print("\n" + "=" * 85)
    print("DAY 26 RUNTIME HAZIRLANIYOR")
    print("=" * 85)

    print(
        "\n1. TR-MTEB embeddingleri "
        "UMAP ile 10 boyuta indiriliyor..."
    )

    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=True,
    )

    reduced_embeddings = (
        reducer.fit_transform(
            embeddings
        )
    )

    print(
        f"UMAP çıktı şekli: "
        f"{reduced_embeddings.shape}"
    )

    print(
        "\n2. HDBSCAN H01 yeniden oluşturuluyor..."
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=(
            HDBSCAN_MIN_CLUSTER_SIZE
        ),
        min_samples=(
            HDBSCAN_MIN_SAMPLES
        ),
        metric="euclidean",
        cluster_selection_method=(
            HDBSCAN_SELECTION_METHOD
        ),
        prediction_data=True,
        gen_min_span_tree=False,
    )

    new_labels = clusterer.fit_predict(
        reduced_embeddings
    )

    reference_labels = np.array(
        [
            row["label"]
            for row in reference_assignments
        ],
        dtype=np.int32,
    )

    label_mapping = build_label_mapping(
        new_labels=new_labels,
        reference_labels=reference_labels,
    )

    mapped_labels = apply_label_mapping(
        labels=new_labels,
        label_mapping=label_mapping,
    )

    adjusted_rand_index = float(
        adjusted_rand_score(
            reference_labels,
            new_labels,
        )
    )

    mapped_exact_agreement = float(
        np.mean(
            mapped_labels
            == reference_labels
        )
    )

    print(
        f"Yeniden bulunan cluster sayısı: "
        f"{len(set(new_labels) - {-1})}"
    )

    print(
        f"Adjusted Rand Index          : "
        f"{adjusted_rand_index:.4f}"
    )

    print(
        f"Eşleme sonrası tam uyum      : "
        f"%{mapped_exact_agreement * 100:.2f}"
    )

    runtime = {
        "umap_reducer": reducer,
        "hdbscan_clusterer": clusterer,
        "label_mapping": label_mapping,
        "adjusted_rand_index": (
            adjusted_rand_index
        ),
        "mapped_exact_agreement": (
            mapped_exact_agreement
        ),
    }

    joblib.dump(
        runtime,
        runtime_path,
    )

    print(
        f"\nRuntime kaydedildi:\n"
        f"{runtime_path}"
    )

    return runtime


def load_thresholds() -> Dict[str, float]:
    """Day 25'te hesaplanan göreli güven eşiklerini okur."""

    summary_path = (
        get_output_directory()
        / "day25_h01_centroid_pipeline_summary.json"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            "Day 25 teknik özet dosyası bulunamadı:\n"
            f"{summary_path}"
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    thresholds = summary.get(
        "relative_thresholds"
    )

    if not isinstance(
        thresholds,
        dict,
    ):
        raise ValueError(
            "Day 25 özetinde relative_thresholds bulunamadı."
        )

    required_thresholds = {
        "direct_margin_q25",
        "direct_margin_q75",
        "noise_margin_q25",
        "noise_margin_q75",
        "direct_probability_q25",
        "direct_probability_q75",
    }

    missing_thresholds = (
        required_thresholds
        - set(thresholds)
    )

    if missing_thresholds:
        raise ValueError(
            "Göreli eşiklerde eksik alanlar var: "
            + ", ".join(
                sorted(missing_thresholds)
            )
        )

    return {
        key: float(value)
        for key, value
        in thresholds.items()
    }


def load_embedding_model() -> Tuple[
    SentenceTransformer,
    str,
]:
    """TR-MTEB modelini CUDA veya CPU üzerinde yükler."""

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 85)
    print("YENİ ABSTRACT EMBEDDING")
    print("=" * 85)

    print(f"\nModel  : {MODEL_ID}")
    print(f"Cihaz  : {device}")

    model = SentenceTransformer(
        MODEL_ID,
        device=device,
    )

    model.max_seq_length = (
        MODEL_TOKEN_LIMIT
    )

    return model, device


def encode_new_abstract(
    model: SentenceTransformer,
    abstract_text: str,
) -> Tuple[np.ndarray, int, bool]:
    """Yeni abstractı normalize edilmiş TR-MTEB vektörüne dönüştürür."""

    token_ids = model.tokenizer.encode(
        abstract_text,
        add_special_tokens=True,
    )

    token_count = len(
        token_ids
    )

    will_truncate = (
        token_count
        > MODEL_TOKEN_LIMIT
    )

    embedding = model.encode(
        [abstract_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding = embedding.astype(
        np.float32,
        copy=False,
    )

    return (
        embedding,
        token_count,
        will_truncate,
    )


def rank_centroids(
    score_row: np.ndarray,
    cluster_ids: List[int],
) -> List[int]:
    """Centroid skorlarını büyükten küçüğe sıralar."""

    ranked_positions = np.argsort(
        score_row
    )[::-1]

    return [
        cluster_ids[
            int(position)
        ]
        for position
        in ranked_positions
    ]


def choose_secondary_cluster(
    primary_cluster: int,
    ranked_cluster_ids: List[int],
) -> int:
    """Birincil clusterdan farklı en yakın clusterı seçer."""

    for cluster_id in ranked_cluster_ids:
        if cluster_id != primary_cluster:
            return cluster_id

    raise RuntimeError(
        "İkincil cluster bulunamadı."
    )


def predict_new_abstract(
    abstract_text: str,
    new_embedding: np.ndarray,
    token_count: int,
    will_truncate: bool,
    runtime: Dict[str, Any],
    cluster_ids: List[int],
    cluster_info: Dict[int, Dict[str, Any]],
    centroid_matrix: np.ndarray,
    thresholds: Dict[str, float],
    device: str,
) -> Dict[str, Any]:
    """Yeni abstract için birincil ve ikincil konu üretir."""

    reducer = runtime[
        "umap_reducer"
    ]

    clusterer = runtime[
        "hdbscan_clusterer"
    ]

    label_mapping = {
        int(key): int(value)
        for key, value
        in runtime[
            "label_mapping"
        ].items()
    }

    reduced_embedding = reducer.transform(
        new_embedding
    )

    raw_labels, strengths = (
        approximate_predict(
            clusterer,
            reduced_embedding,
        )
    )

    raw_label = int(
        raw_labels[0]
    )

    hdbscan_probability = float(
        strengths[0]
    )

    if raw_label >= 0:
        mapped_h01_label = (
            label_mapping[
                raw_label
            ]
        )
    else:
        mapped_h01_label = -1

    centroid_scores = (
        new_embedding
        @ centroid_matrix.T
    )[0]

    ranked_cluster_ids = (
        rank_centroids(
            score_row=centroid_scores,
            cluster_ids=cluster_ids,
        )
    )

    nearest_centroid_cluster = (
        ranked_cluster_ids[0]
    )

    if mapped_h01_label >= 0:
        primary_cluster = (
            mapped_h01_label
        )

        assignment_method = (
            "HDBSCAN approximate_predict"
        )

        centroid_agrees = (
            nearest_centroid_cluster
            == mapped_h01_label
        )
    else:
        primary_cluster = (
            nearest_centroid_cluster
        )

        assignment_method = (
            "HDBSCAN noise → en yakın centroid"
        )

        centroid_agrees = None

    secondary_cluster = (
        choose_secondary_cluster(
            primary_cluster=primary_cluster,
            ranked_cluster_ids=(
                ranked_cluster_ids
            ),
        )
    )

    cluster_position = {
        cluster_id: position
        for position, cluster_id
        in enumerate(cluster_ids)
    }

    primary_similarity = float(
        centroid_scores[
            cluster_position[
                primary_cluster
            ]
        ]
    )

    secondary_similarity = float(
        centroid_scores[
            cluster_position[
                secondary_cluster
            ]
        ]
    )

    similarity_margin = (
        primary_similarity
        - secondary_similarity
    )

    relative_confidence = (
        classify_relative_confidence(
            original_hdbscan_label=(
                mapped_h01_label
            ),
            hdbscan_probability=(
                hdbscan_probability
            ),
            centroid_agrees_with_hdbscan=(
                centroid_agrees
            ),
            similarity_margin=(
                similarity_margin
            ),
            direct_probability_q25=(
                thresholds[
                    "direct_probability_q25"
                ]
            ),
            direct_probability_q75=(
                thresholds[
                    "direct_probability_q75"
                ]
            ),
            direct_margin_q25=(
                thresholds[
                    "direct_margin_q25"
                ]
            ),
            direct_margin_q75=(
                thresholds[
                    "direct_margin_q75"
                ]
            ),
            noise_margin_q25=(
                thresholds[
                    "noise_margin_q25"
                ]
            ),
            noise_margin_q75=(
                thresholds[
                    "noise_margin_q75"
                ]
            ),
        )
    )

    topic_structure = (
        classify_topic_structure(
            original_hdbscan_label=(
                mapped_h01_label
            ),
            centroid_agrees_with_hdbscan=(
                centroid_agrees
            ),
            similarity_margin=(
                similarity_margin
            ),
            direct_margin_q25=(
                thresholds[
                    "direct_margin_q25"
                ]
            ),
            direct_margin_q75=(
                thresholds[
                    "direct_margin_q75"
                ]
            ),
            noise_margin_q25=(
                thresholds[
                    "noise_margin_q25"
                ]
            ),
            noise_margin_q75=(
                thresholds[
                    "noise_margin_q75"
                ]
            ),
        )
    )

    primary_info = cluster_info[
        primary_cluster
    ]

    secondary_info = cluster_info[
        secondary_cluster
    ]

    return {
        "input": {
            "character_count": len(
                abstract_text
            ),
            "token_count": token_count,
            "model_token_limit": (
                MODEL_TOKEN_LIMIT
            ),
            "will_truncate": (
                will_truncate
            ),
            "abstract_text": (
                abstract_text
            ),
        },
        "model": {
            "embedding_model": (
                MODEL_ID
            ),
            "embedding_dimension": int(
                new_embedding.shape[1]
            ),
            "device": device,
        },
        "runtime_validation": {
            "adjusted_rand_index": (
                runtime[
                    "adjusted_rand_index"
                ]
            ),
            "mapped_exact_agreement": (
                runtime[
                    "mapped_exact_agreement"
                ]
            ),
        },
        "prediction": {
            "assignment_method": (
                assignment_method
            ),
            "hdbscan_status": (
                "clusterlandı"
                if mapped_h01_label >= 0
                else "noise / belirsiz"
            ),
            "raw_hdbscan_label": (
                raw_label
            ),
            "mapped_h01_cluster": (
                mapped_h01_label
            ),
            "hdbscan_probability": (
                hdbscan_probability
            ),
            "nearest_centroid_cluster": (
                nearest_centroid_cluster
            ),
            "centroid_agrees_with_hdbscan": (
                centroid_agrees
            ),
            "primary_cluster": (
                primary_cluster
            ),
            "primary_topic": (
                primary_info[
                    "dominant_subject_name"
                ]
            ),
            "primary_centroid_similarity": (
                primary_similarity
            ),
            "primary_medoid_article_id": (
                primary_info[
                    "medoid_article_id"
                ]
            ),
            "primary_medoid_title": (
                primary_info[
                    "medoid_title"
                ]
            ),
            "secondary_cluster": (
                secondary_cluster
            ),
            "secondary_topic": (
                secondary_info[
                    "dominant_subject_name"
                ]
            ),
            "secondary_centroid_similarity": (
                secondary_similarity
            ),
            "similarity_margin": (
                similarity_margin
            ),
            "relative_confidence": (
                relative_confidence
            ),
            "topic_structure": (
                topic_structure
            ),
        },
        "important_note": (
            "Konu isimleri H01 clusterlarındaki baskın "
            "TR Dizin subject metadata alanlarından "
            "türetilen geçici etiketlerdir. Göreli güven "
            "kalibre edilmiş bir olasılık değildir."
        ),
    }


def save_result(
    result: Dict[str, Any],
) -> Path:
    """Yeni abstract sonucunu JSON olarak kaydeder."""

    result_path = get_result_path()

    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result_path


def print_result(
    result: Dict[str, Any],
) -> None:
    """Tahmin sonucunu terminalde okunabilir biçimde gösterir."""

    prediction = result[
        "prediction"
    ]

    input_info = result[
        "input"
    ]

    print("\n" + "=" * 85)
    print("YENİ ABSTRACT KONU TAHMİNİ")
    print("=" * 85)

    print(
        f"\nKarakter sayısı     : "
        f"{input_info['character_count']}"
    )

    print(
        f"Token sayısı        : "
        f"{input_info['token_count']}"
    )

    print(
        f"Token kesilmesi     : "
        f"{'Evet' if input_info['will_truncate'] else 'Hayır'}"
    )

    print(
        f"\nHDBSCAN durumu      : "
        f"{prediction['hdbscan_status']}"
    )

    print(
        f"HDBSCAN olasılığı   : "
        f"{prediction['hdbscan_probability']:.4f}"
    )

    print(
        f"Atama yöntemi       : "
        f"{prediction['assignment_method']}"
    )

    print(
        f"\nBirincil cluster    : "
        f"{prediction['primary_cluster']}"
    )

    print(
        f"Birincil konu       : "
        f"{prediction['primary_topic']}"
    )

    print(
        f"Birincil benzerlik  : "
        f"{prediction['primary_centroid_similarity']:.4f}"
    )

    print(
        f"\nİkincil cluster     : "
        f"{prediction['secondary_cluster']}"
    )

    print(
        f"İkincil konu        : "
        f"{prediction['secondary_topic']}"
    )

    print(
        f"İkincil benzerlik   : "
        f"{prediction['secondary_centroid_similarity']:.4f}"
    )

    print(
        f"\nBenzerlik farkı     : "
        f"{prediction['similarity_margin']:.4f}"
    )

    print(
        f"Göreli güven        : "
        f"{prediction['relative_confidence']}"
    )

    print(
        f"Konu yapısı         : "
        f"{prediction['topic_structure']}"
    )

    print(
        f"\nTemsilci makale     : "
        f"{prediction['primary_medoid_title']}"
    )


def main() -> None:
    print("=" * 85)
    print("DAY 26 — VERİ SETİ DIŞI ABSTRACT TAHMİNİ")
    print("=" * 85)

    articles = load_articles()
    embeddings = load_embeddings()
    reference_assignments = (
        load_h01_assignments()
    )

    validate_alignment(
        articles=articles,
        assignments=(
            reference_assignments
        ),
    )

    (
        article_subject_sets,
        subject_display_names,
    ) = build_subject_information(
        articles
    )

    (
        cluster_ids,
        cluster_info,
        representations,
    ) = build_cluster_representations(
        articles=articles,
        embeddings=embeddings,
        assignments=(
            reference_assignments
        ),
        article_subject_sets=(
            article_subject_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
    )

    centroid_matrix = representations[
        "centroid"
    ]

    runtime = prepare_runtime(
        embeddings=embeddings,
        reference_assignments=(
            reference_assignments
        ),
    )

    thresholds = load_thresholds()

    abstract_text = (
        load_new_abstract()
    )

    model, device = (
        load_embedding_model()
    )

    (
        new_embedding,
        token_count,
        will_truncate,
    ) = encode_new_abstract(
        model=model,
        abstract_text=abstract_text,
    )

    result = predict_new_abstract(
        abstract_text=abstract_text,
        new_embedding=new_embedding,
        token_count=token_count,
        will_truncate=will_truncate,
        runtime=runtime,
        cluster_ids=cluster_ids,
        cluster_info=cluster_info,
        centroid_matrix=centroid_matrix,
        thresholds=thresholds,
        device=device,
    )

    result_path = save_result(
        result
    )

    print_result(
        result
    )

    print("\n" + "=" * 85)
    print("DAY 26 TAMAMLANDI")
    print("=" * 85)

    print(
        f"\nJSON sonuç:\n"
        f"{result_path}"
    )

    print(
        "\nYeni bir abstract denemek için yalnızca şu "
        "dosyanın içeriğini değiştirip komutu yeniden çalıştır:\n"
        f"{get_input_path()}"
    )


if __name__ == "__main__":
    main()