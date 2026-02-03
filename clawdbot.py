"""
ClawDBot – Clinical Language & Workflow Assistant

Purpose:
Transforms ML outputs and risk analytics into
clinician-friendly, non-diagnostic CDS explanations.
"""

def explain_risk(risk_type, score, shap_features=None):
    explanation = f"{risk_type} risk score is {score}/100. "

    if shap_features:
        top_factors = ", ".join(shap_features[:3])
        explanation += f"Key contributing factors include: {top_factors}. "

    explanation += (
        "This output is for clinical decision support only "
        "and does not constitute diagnosis or treatment recommendations."
    )

    return explanation


def clinician_note(summary_dict):
    """
    Generates a structured clinician-facing note.
    """
    note = f"""
    Clinical Risk Summary:
    - AKI Risk: {summary_dict.get('aki')}
    - Bleeding Risk: {summary_dict.get('bleeding')}
    - Sepsis Risk: {summary_dict.get('sepsis')}

    Interpretation:
    Elevated risk indicators identified. Review patient context,
    labs, and medications before making clinical decisions.
    """
    return note.strip()
