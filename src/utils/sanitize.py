def anonymize_text(text: str, anonymize: bool = True):
    if not anonymize or text is None:
        return text
    # Replace known names with generic placeholders
    replacements = {
        "CIFAR-10": "DatasetX",
        "CIFAR10": "DatasetX",
        "ResNet-9": "ModelX",
        "ResNet9": "ModelX",
        "ResNet 9": "ModelX",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text