"""Estensione §5 #9 — Self-Correction Loop.

Dopo la generazione della risposta iniziale, esegue un secondo passaggio LLM
che verifica se ogni affermazione nella risposta è supportata dai chunks
recuperati. Se trova affermazioni non supportate, le rimuove o le corregge.

Impatto atteso: +Faithfulness (le affermazioni non supportate vengono eliminate),
possibile -Answer Correctness se troppo conservativo. Costo: 1 LLM call aggiuntivo.
"""
import logging

logger = logging.getLogger(__name__)

VERIFICATION_PROMPT = """Sei un verificatore di fatti rigoroso. Ti viene fornita una RISPOSTA generata da un sistema RAG
e il CONTESTO (i chunks di documenti) su cui si basa.

Il tuo compito è:
1. Verificare se ogni affermazione nella RISPOSTA è supportata dal CONTESTO.
2. Rimuovere o correggere le affermazioni NON supportate dal CONTESTO.
3. Mantenere il formato originale [RISPOSTA].
4. Se la risposta è già completamente supportata, restituiscila identica.
5. Se nessuna affermazione è supportata, scrivi "ASTENSIONE: Le informazioni non sono supportate dalle fonti."

NON aggiungere nuove informazioni che non siano nel CONTESTO.

CONTESTO:
{context}

RISPOSTA DA VERIFICARE:
{answer}

RISPOSTA VERIFICATA:"""


def self_correct(answer: str, context: str, llm) -> tuple[str, bool]:
    """Esegue il secondo passaggio di verifica.

    Returns:
        (corrected_answer, was_modified) — was_modified=True se la risposta è cambiata.
    """
    if "ASTENSIONE:" in answer:
        return answer, False

    try:
        prompt = VERIFICATION_PROMPT.format(context=context, answer=answer)
        corrected = llm.invoke(prompt).strip()

        was_modified = corrected.strip() != answer.strip()
        if was_modified:
            logger.info("Self-correction: risposta modificata dal verificatore.")
        else:
            logger.info("Self-correction: risposta confermata (nessuna modifica).")

        return corrected, was_modified
    except Exception as e:
        logger.warning(f"Self-correction fallita: {e}. Mantengo risposta originale.")
        return answer, False
