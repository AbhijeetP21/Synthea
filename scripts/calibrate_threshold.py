"""One-off: measure cosine-distance distributions for relevant vs off-topic
queries against the ingested patient, to pick RETRIEVAL_MAX_DISTANCE."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Chunk, get_engine
from app.llm import get_embeddings_client

PATIENT_ID = "38c50ee6-b356-4a70-2fd9-44e304a734cd"

QUERIES = {
    "relevant: medications": "What medications is this patient taking?",
    "relevant: diabetes": "Does this patient have diabetes?",
    "relevant: allergies": "Does the patient have any documented allergies?",
    "OFF-TOPIC: capital": "What is the capital of France?",
    "OFF-TOPIC: bread": "How do I bake sourdough bread at home?",
    "OFF-TOPIC: weather": "What will the weather be like tomorrow?",
}


def main() -> None:
    emb = get_embeddings_client()
    engine = get_engine()
    for label, q in QUERIES.items():
        vec = emb.embed_query(q)
        with Session(engine) as s:
            stmt = (
                select(Chunk.embedding.cosine_distance(vec))
                .where(Chunk.patient_id == PATIENT_ID)
                .order_by(Chunk.embedding.cosine_distance(vec))
                .limit(8)
            )
            dists = [float(d) for d in s.execute(stmt).scalars().all()]
        print(f"{label:28s} top1={dists[0]:.3f}  top8={dists[-1]:.3f}  "
              f"mean={sum(dists)/len(dists):.3f}")


if __name__ == "__main__":
    main()
