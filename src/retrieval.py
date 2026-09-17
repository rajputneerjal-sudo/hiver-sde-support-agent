from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedCase:
    intent: str
    customer_text: str
    brand_reply: str
    similarity: float


class ResolvedCaseRetriever:
    """
    Hybrid TF-IDF retriever.

    Combines:
    1. Word-level TF-IDF for semantic phrase overlap.
    2. Character-level TF-IDF for spelling variations and
       slightly different wording.

    The two similarity scores are combined into one ranking score.
    """

    def __init__(self, kb_csv_path: str):
        self.kb = pd.read_csv(kb_csv_path)

        self.kb["customer_message"] = (
            self.kb["customer_message"]
            .fillna("")
            .astype(str)
        )

        self.kb["amazonhelp_response"] = (
            self.kb["amazonhelp_response"]
            .fillna("")
            .astype(str)
        )

        # ====================================================
        # WORD-LEVEL TF-IDF
        # ====================================================

        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b\w[\w']+\b",
        )

        self.word_matrix = self.word_vec.fit_transform(
            self.kb["customer_message"].tolist()
        )

        # ====================================================
        # CHARACTER-LEVEL TF-IDF
        # ====================================================

        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            lowercase=True,
        )

        self.char_matrix = self.char_vec.fit_transform(
            self.kb["customer_message"].tolist()
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        k: int = 3,
        intent_filter: str | None = None,
    ) -> list[RetrievedCase]:

        query = str(query)

        # ====================================================
        # WORD SIMILARITY
        # ====================================================

        q_word = self.word_vec.transform([query])

        word_similarity = cosine_similarity(
            q_word,
            self.word_matrix,
        )[0]

        # ====================================================
        # CHARACTER SIMILARITY
        # ====================================================

        q_char = self.char_vec.transform([query])

        char_similarity = cosine_similarity(
            q_char,
            self.char_matrix,
        )[0]

        # ====================================================
        # HYBRID SCORE
        # ====================================================

        # Word similarity receives more weight because it captures
        # meaningful customer phrases better.
        #
        # Character similarity helps with spelling differences,
        # abbreviations and slightly different wording.

        similarities = (
            0.75 * word_similarity
            + 0.25 * char_similarity
        )

        df = self.kb.copy()

        df["similarity"] = similarities

        # ====================================================
        # OPTIONAL INTENT FILTER
        # ====================================================

        # The current real KB normally does not contain intent
        # labels. If labels are available in the future, use them
        # to restrict retrieval to the predicted intent.

        if intent_filter and "intent" in df.columns:

            filtered = df[df["intent"] == intent_filter]

            if len(filtered) > 0:
                df = filtered

        # ====================================================
        # REMOVE EMPTY / VERY WEAK MATCHES
        # ====================================================

        df = df[df["customer_message"].str.strip() != ""]

        # ====================================================
        # RANK RESULTS
        # ====================================================

        df = (
            df.sort_values(
                "similarity",
                ascending=False,
            )
            .head(k)
        )

        # ====================================================
        # BUILD RESULTS
        # ====================================================

        results = []

        for row in df.itertuples():

            if "intent" in df.columns:
                intent = getattr(
                    row,
                    "intent",
                    "unknown",
                )
            else:
                intent = "unknown"

            results.append(
                RetrievedCase(
                    intent=intent,
                    customer_text=row.customer_message,
                    brand_reply=row.amazonhelp_response,
                    similarity=float(row.similarity),
                )
            )

        return results