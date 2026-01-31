import requests

BASE_URL = "https://api.mail.tm"

def create_temp_email():
    domain = requests.get(f"{BASE_URL}/domains").json()["hydra:member"][0]["domain"]
    email = f"user{requests.utils.random.randint(1000,9999)}@{domain}"
    password = "pass1234"

    requests.post(f"{BASE_URL}/accounts", json={
        "address": email,
        "password": password
    })

    res = requests.post(f"{BASE_URL}/token", json={
        "address": email,
        "password": password
    })

    token = res.json()["token"]
    return email, token


def fetch_messages(token):
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(f"{BASE_URL}/messages", headers=headers)
    messages = res.json()["hydra:member"]

    emails = []

    for msg in messages:
        msg_id = msg["id"]
        msg_data = requests.get(f"{BASE_URL}/messages/{msg_id}", headers=headers).json()
        emails.append(msg_data["text"])

    return emails
