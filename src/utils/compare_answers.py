from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

semantic_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def exact_match(predicted_answer: str, gold_answer: str) -> bool:
    return predicted_answer.strip().lower() == gold_answer.strip().lower()

def semantic_match(predicted_answer: str, gold_answer: str) -> bool:
    embeddings = semantic_model.encode([predicted_answer, gold_answer])
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])
    return similarity_score >= 0.8



def compare_answers(question: str, wiki_text: str, gold_answer: str) -> str:
    result = qa_pipeline(question=question, context=wiki_text)
    # ({
    #     'context': wiki_text,
    #     'question': question
    # })
    predicted_answer = result['answer']
    if exact_match(predicted_answer, gold_answer):
        return predicted_answer, "e"
    if semantic_match(predicted_answer, gold_answer):
        return predicted_answer, "s"
    return predicted_answer, "n"