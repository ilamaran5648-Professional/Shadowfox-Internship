from app.validation.groundedness import GroundednessValidator

def test_groundedness_refusal():
    validator = GroundednessValidator()
    result = validator.validate(
        query="What is the price of Bitcoin?",
        answer="Information not found in the provided documents.",
        context_chunks=[]
    )
    assert result["is_grounded"] is True
    assert result["is_refusal"] is True
    assert result["confidence_level"] == "N/A (Refusal)"

def test_groundedness_high_confidence():
    validator = GroundednessValidator()
    context = [{
        "text": "SHADOWFOX internship duration is from August 1 2026 to September 2 2026.",
        "filename": "internship.txt"
    }]
    answer = "The SHADOWFOX internship duration runs from August 1 2026 to September 2 2026."

    result = validator.validate(
        query="What is the internship duration?",
        answer=answer,
        context_chunks=context
    )
    assert result["is_grounded"] is True
    assert result["confidence_level"] == "HIGH"
    assert result["overlap_score"] > 0.5

def test_groundedness_low_confidence():
    validator = GroundednessValidator()
    context = [{
        "text": "Python is a popular programming language.",
        "filename": "python.txt"
    }]
    answer = "Quantum computing uses qubits and entanglement principles to compute faster than supercomputers."

    result = validator.validate(
        query="Explain quantum computing.",
        answer=answer,
        context_chunks=context
    )
    assert result["is_grounded"] is False
    assert result["confidence_level"] == "LOW"
