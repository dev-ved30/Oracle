import requests

TOKEN = "token bd55ceb8-4563-4f96-bcfb-d4cec364ac47"


def save_source_to_group(object_id, group_id=1959):
    url = "https://fritz.science/api/source_groups"

    payload = {
        "objId": object_id,
        "inviteGroupIds": [group_id],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": TOKEN
    }

    response = requests.post(url, json=payload, headers=headers)
    print(response.json())


def send_comment(object_id, text, group_id=1959):
    url = f"https://fritz.science/api/sources/{object_id}/comments"

    payload = {
        "text": text,
        "group_ids": [group_id],
    }

    headers = {
        "Authorization": TOKEN,
    }

    response = requests.post(url, json=payload, headers=headers)
    print(response.json())



save_source_to_group("ZTF24aauufci")
send_comment("ZTF24aauufci", "test")