import os
from supabase import create_client, Client
from datetime import datetime

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)


def add_image(user_id: str, image_path: str):
    """Adds the image to the database.

    Parameters:
        user_id (str): user id of the user
        image_path (str): path of the image

    Returns: status code, message
    """
    try:
        current_time = str(datetime.now()).replace(" ", "_")
        filename = f"{user_id}/{current_time}.png"
        supabase.storage.from_("nft-stylegan").upload(
            file=image_path, path=filename, file_options={"content-type": "image/png"}
        )
        image_url = supabase.storage.from_("nft-stylegan").get_public_url(filename)
        supabase.table("users").insert(
            {
                "user_id": user_id,
                "file_name": current_time + ".png",
                "file_type": "image",
                "model_used": "nft-stylegan",
            }
        ).execute()
        data = {"url": image_url}
        return {
            "statusCode": 200,
            "message": "Image uploaded successfully",
            "data": data,
        }
    except Exception as e:
        return e
