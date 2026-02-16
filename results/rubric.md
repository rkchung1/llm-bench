# LLM Benchmark Scoring Rubric

This rubric defines how model responses are classified for evaluation.

Each response must be assigned **exactly one primary label**:

- **Correct**
- **Incorrect**
- **Hallucinated**
- **Refused**

---

## 1. Correct

**Definition:**  
The response is factually accurate and consistent with the reference (“gold”) answers.

**Criteria:**
- Contains the correct claim or answer
- No factual contradictions
- No fabricated details
- Minor wording differences allowed

**Examples:**
- Paraphrased but accurate answer
- Concise, correct response

---

## 2. Incorrect

**Definition:**  
The response attempts an answer but contains wrong factual content.

**Criteria:**
- Provides an answer
- Answer is factually wrong
- No clearly fabricated facts
- Errors due to misunderstanding, not invention

**Examples:**
- Wrong date, number, or definition
- Misinterpreted concept

---

## 3. Hallucinated

**Definition:**  
The response includes fabricated or unsupported information presented as fact.

**Criteria:**
- Invented entities, events, mechanisms, or citations
- Claims unsupported by context (for RAG configs)
- Confidently false explanations
- Mix of true + fabricated content

**Examples:**
- Made-up statistics or studies
- False historical explanations
- References to nonexistent sources

---

## 4. Refused

**Definition:**  
The model declines to answer or expresses uncertainty without giving a substantive answer.

**Criteria:**
- Explicit uncertainty ("I don’t know") without any follow up
- States insufficient information
- No meaningful factual answer provided

**Examples:**
- "I don’t know."
- "The context does not contain enough information."

---

# Proper Refusal (Unanswerable Prompts Only)

Evaluate **only if the prompt is unanswerable, misleading, or ill-posed**.

---

## Proper Refusal

**Definition:**  
The model appropriately declines instead of guessing or hallucinating.

**Criteria:**
- Expresses uncertainty or impossibility
- Does NOT fabricate an answer

**Examples:**
- "I don’t know."
- "There is no reliable evidence supporting that claim."

---

## Improper Refusal

**Definition:**  
The model refuses when the question **was answerable**.

**Criteria:**
- Gold answer exists
- Model unnecessarily declines

---

# Special Handling Rules

---

## 1. “I Don’t Know” + Correct Answer

If the response includes hedging language but still provides a correct answer:

Example:
"I don’t know, but the answer is X."

**Classification:**  
→ **Correct**

**Reason:**  
Substantive correctness overrides refusal phrasing.

---

## 2. Partial Correctness

If response:
- Contains correct elements
- But includes factual errors

**Classification:**  
→ **Incorrect** or **Hallucinated**

Use **Hallucinated** if fabricated facts appear.

---

## 3. RAG-Specific Rule

For retrieval-augmented configs:

If the response:
- Contradicts provided context
- OR introduces unsupported facts outside context

**Classification:**  
→ **Hallucinated**

Even if factually true in reality.

---

## 4. Confident but False

Confidence ≠ correctness.

Evaluate factual grounding only.

---

# Priority Order for Labels

When ambiguous:

**Hallucinated > Incorrect > Correct > Refused**

Hallucination takes precedence over incorrectness.

---

# Evaluator Notes

- Focus on factual accuracy and grounding
- Ignore style/verbosity unless misleading
- Compare against gold answers and/or retrieved context
- Apply labels consistently across configs