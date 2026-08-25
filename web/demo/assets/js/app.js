    const state = {
        mode: "semantic",
        searching: false,
    };

    const elements = {
        status: document.getElementById("systemStatus"),
        statusText: document.getElementById("statusText"),
        articleCount: document.getElementById("articleCount"),

        semanticMode: document.getElementById("semanticMode"),
        hybridMode: document.getElementById("hybridMode"),
        modeNote: document.getElementById("modeNote"),

        query: document.getElementById("query"),
        yearFrom: document.getElementById("yearFrom"),
        yearTo: document.getElementById("yearTo"),
        database: document.getElementById("database"),
        limit: document.getElementById("limit"),

        searchButton: document.getElementById("searchButton"),
        searchInfo: document.getElementById("searchInfo"),
        results: document.getElementById("results"),
    };


    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }


    function numberFormat(value) {
        return new Intl.NumberFormat(
            "tr-TR"
        ).format(value);
    }


    function shortText(value, maximum = 620) {
        const text = String(value ?? "")
            .replace(/\s+/g, " ")
            .trim();

        if (text.length <= maximum) {
            return text;
        }

        return text.slice(
            0,
            maximum - 3
        ) + "...";
    }


    function setMode(mode) {
        state.mode = mode;

        elements.semanticMode.classList.toggle(
            "active",
            mode === "semantic"
        );

        elements.hybridMode.classList.toggle(
            "active",
            mode === "hybrid"
        );

        if (mode === "semantic") {
            elements.modeNote.textContent =
                "TR-MTEB abstract embedding + cosine similarity";
        } else {
            elements.modeNote.textContent =
                "Abstract dense + Title dense + BM25 sparse → RRF";
        }
    }


    async function loadStatus() {
        try {
            const response = await fetch(
                "/api/status"
            );

            if (!response.ok) {
                throw new Error(
                    "Status endpoint başarısız."
                );
            }

            const data = await response.json();

            elements.status.classList.add(
                "online"
            );

            elements.statusText.textContent =
                `${data.device.toUpperCase()} • ` +
                `${numberFormat(data.article_count)} point`;

            elements.articleCount.textContent =
                numberFormat(
                    data.article_count
                );

        } catch (error) {
            elements.status.classList.remove(
                "online"
            );

            elements.statusText.textContent =
                "Backend bağlantısı yok";
        }
    }


    function badge(value, cssClass = "") {
        if (
            value === undefined ||
            value === null ||
            value === ""
        ) {
            return "";
        }

        return `
            <span class="badge ${cssClass}">
                ${escapeHtml(value)}
            </span>
        `;
    }


    function methodBadge(method) {
        if (method === "direct") {
            return badge(
                "Direct HDBSCAN",
                "direct"
            );
        }

        if (method === "centroid_fallback") {
            return badge(
                "Centroid Fallback",
                "fallback"
            );
        }

        return badge(method);
    }


    function resultDetails(result) {
        const details =
            result.search_details || {};

        if (
            details.mode === "hybrid"
        ) {
            return `
                RRF ${Number(
                    details.rrf_score || 0
                ).toFixed(6)}
                &nbsp;•&nbsp;
                Abstract rank:
                ${details.abstract_rank ?? "—"}
                &nbsp;•&nbsp;
                Title rank:
                ${details.title_rank ?? "—"}
                &nbsp;•&nbsp;
                BM25 rank:
                ${details.bm25_rank ?? "—"}
            `;
        }

        return `
            Abstract cosine score:
            ${Number(
                details.abstract_score || 0
            ).toFixed(4)}
        `;
    }


    function renderResult(result) {
        const score =
            Number(result.score || 0);

        const scoreText =
            state.mode === "semantic"
                ? score.toFixed(4)
                : score.toFixed(6);

        const margin =
            result.similarity_margin;

        const marginBadge =
            margin === undefined ||
            margin === null
                ? ""
                : badge(
                    `Konu marjı ${Number(
                        margin
                    ).toFixed(4)}`
                );

        return `
            <article class="result-card">

                <div class="result-head">

                    <div class="rank">
                        ${result.rank}
                    </div>

                    <div>
                        <h3 class="result-title">
                            ${escapeHtml(
                                result.title_tr
                            )}
                        </h3>
                    </div>

                    <div class="score">
                        <div class="score-value">
                            ${scoreText}
                        </div>

                        <div class="score-label">
                            ${
                                state.mode === "semantic"
                                ? "cosine"
                                : "RRF"
                            }
                        </div>
                    </div>

                </div>

                <div class="meta-row">

                    ${badge(
                        result.publication_year
                    )}

                    ${(result.databases || [])
                        .map(value =>
                            badge(value)
                        )
                        .join("")}

                    ${methodBadge(
                        result.assignment_method
                    )}

                    ${badge(
                        `Cluster ${result.primary_cluster}`
                    )}

                    ${marginBadge}

                </div>

                <div class="topics">

                    <div class="topic">
                        <div class="topic-label">
                            Birincil konu
                        </div>

                        <div class="topic-value">
                            ${escapeHtml(
                                result.primary_topic ||
                                "Belirlenmedi"
                            )}
                        </div>
                    </div>

                    <div class="topic">
                        <div class="topic-label">
                            İkincil konu
                        </div>

                        <div class="topic-value">
                            ${escapeHtml(
                                result.secondary_topic ||
                                "Belirlenmedi"
                            )}
                        </div>
                    </div>

                </div>

                <div class="abstract">
                    ${escapeHtml(
                        shortText(
                            result.abstract_tr
                        )
                    )}
                </div>

                <div class="details">
                    Makale ID:
                    ${escapeHtml(
                        result.article_id
                    )}
                    &nbsp;•&nbsp;
                    ${resultDetails(result)}
                </div>

            </article>
        `;
    }


    async function search() {
        if (state.searching) {
            return;
        }

        const query =
            elements.query.value.trim();

        if (!query) {
            elements.searchInfo.innerHTML =
                `<span class="error">
                    Sorgu boş olamaz.
                </span>`;
            return;
        }

        state.searching = true;

        elements.searchButton.disabled = true;
        elements.searchButton.textContent =
            "Aranıyor...";

        elements.searchInfo.textContent =
            "Sorgu embeddingi üretiliyor ve Qdrant aranıyor...";

        elements.results.innerHTML = "";

        const body = {
            query,
            mode: state.mode,
            limit: Number(
                elements.limit.value
            ),
            year_from:
                elements.yearFrom.value || null,
            year_to:
                elements.yearTo.value || null,
            database:
                elements.database.value || null,
        };

        try {
            const response = await fetch(
                "/api/search",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify(body),
                }
            );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    "Arama başarısız."
                );
            }

            const embeddingMs =
                Number(
                    data.embedding_seconds
                ) * 1000;

            const searchMs =
                Number(
                    data.search_seconds
                ) * 1000;

            elements.searchInfo.innerHTML = `
                <strong>
                    ${data.result_count}
                    sonuç
                </strong>
                &nbsp;•&nbsp;
                query embedding
                ${embeddingMs.toFixed(1)} ms
                &nbsp;•&nbsp;
                search
                ${searchMs.toFixed(1)} ms
                &nbsp;•&nbsp;
                mod:
                ${escapeHtml(data.mode)}
            `;

            elements.results.innerHTML =
                data.results
                    .map(renderResult)
                    .join("");

            if (!data.results.length) {
                elements.results.innerHTML = `
                    <div class="result-card">
                        Filtrelere uyan sonuç bulunamadı.
                    </div>
                `;
            }

        } catch (error) {
            elements.searchInfo.innerHTML = `
                <span class="error">
                    ${escapeHtml(
                        error.message
                    )}
                </span>
            `;
        } finally {
            state.searching = false;

            elements.searchButton.disabled =
                false;

            elements.searchButton.textContent =
                "Ara";
        }
    }


    elements.semanticMode.addEventListener(
        "click",
        () => setMode("semantic")
    );

    elements.hybridMode.addEventListener(
        "click",
        () => setMode("hybrid")
    );

    elements.searchButton.addEventListener(
        "click",
        search
    );

    elements.query.addEventListener(
        "keydown",
        event => {
            if (event.key === "Enter") {
                search();
            }
        }
    );


    loadStatus();
