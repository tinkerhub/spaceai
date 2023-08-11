from typing import List
def retrieve_learning_topics(dataset) -> List[str]:
    """
    return learning path topics
    """
    labels = list(dataset)
    labels = [f"learn {label}" for label in labels]
    return labels
    
def retrieve_url(topic, dataset) -> str:
    """
    return learning path url
    """
    url = dataset.get(topic)
    url = "learning path not available" if not url else url
    return url