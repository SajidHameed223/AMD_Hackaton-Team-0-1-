PROFILES = {
    "math": {
        "temperature": 0.1,
        "max_tokens": 120,
        "top_p": 0.7
    },
    "code": {
        "temperature": 0.2,
        "max_tokens": 300,
        "top_p": 0.9
    },
    "summary": {
        "temperature": 0.3,
        "max_tokens": 200,
        "top_p": 0.9
    },
    "creative": {
        "temperature": 0.8,
        "max_tokens": 400,
        "top_p": 0.95
    },
    "default": {
        "temperature": 0.4,
        "max_tokens": 200,
        "top_p": 0.9
    }
}


def get_profile(task_type: str):
    return PROFILES.get(task_type, PROFILES["default"])