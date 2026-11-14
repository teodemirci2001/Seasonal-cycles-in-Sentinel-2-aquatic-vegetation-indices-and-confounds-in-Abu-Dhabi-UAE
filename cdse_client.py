from __future__ import annotations

import os
import time
from typing import Iterator

import requests


# CDSEndpoint

_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_PAGE = 100          # max results per page 


TIMEOUT        = (10, 90)   
MAX_RETRIES    = 6
BACKOFF_BASE   = 2          

def _make_session() -> requests.Session:
    session = requests.Session()
    user = os.environ.get("CDSE_USERNAME")
    pw   = os.environ.get("CDSE_PASSWORD")
    if user and pw:
        try:
            token_url = (
                "https://identity.dataspace.copernicus.eu/auth/realms/"
                "CDSE/protocol/openid-connect/token"
            )
            resp = session.post(
                token_url,
                data={
                    "client_id": "cdse-public",
                    "grant_type": "password",
                    "username": user,
                    "password": pw,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            token = resp.json()["access_token"]
            session.headers["Authorization"] = f"Bearer {token}"
            print("[cdse]   authenticated as", user)
        except Exception as ex:
            print(f"[cdse]   auth failed ({ex}); continuing as anonymous")
    else:
        print("[cdse]   no credentials set — using anonymous access (lower rate limit)")
    return session


def _get_with_retry(session: requests.Session, url: str, params: dict) -> dict:
    delay = BACKOFF_BASE
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT)
            remaining = resp.headers.get("X-RateLimit-Remaining")
            limit      = resp.headers.get("X-RateLimit-Limit")
            if remaining is not None:
                pct = int(remaining) / max(int(limit), 1) * 100
                if pct < 20:
                    print(f"\r[cdse]   rate limit: {remaining}/{limit} requests remaining ({pct:.0f}%)  ",
                          flush=True)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", delay))
                print(f"\r[cdse]   429 rate-limited — waiting {retry_after}s "
                      f"(attempt {attempt}/{MAX_RETRIES}) ...", flush=True)
                time.sleep(retry_after)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code == 503:
                print(f"\r[cdse]   503 service unavailable — waiting {delay}s "
                      f"(attempt {attempt}/{MAX_RETRIES}) ...", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            print(f"\r[cdse]   timeout after {TIMEOUT[1]}s "
                  f"(attempt {attempt}/{MAX_RETRIES}) — retrying in {delay}s ...", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)

        except requests.exceptions.ConnectionError as ex:
            print(f"\r[cdse]   connection error ({ex}) "
                  f"(attempt {attempt}/{MAX_RETRIES}) — retrying in {delay}s ...", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)

    raise RuntimeError(
        f"CDSE query failed after {MAX_RETRIES} attempts. "
        "Check your network / CDSE status: https://dataspace.copernicus.eu/"
    )


def iter_features(
    collection: str,
    filters: dict,
    *,
    session: requests.Session | None = None,
    page_size: int = _PAGE,
) -> Iterator[dict]:
    if session is None:
        session = _make_session()

    _COLLECTION_MAP = {
        "Sentinel2": "SENTINEL-2",
    }
    cdse_collection = _COLLECTION_MAP.get(collection, collection)
    
    clauses = [f"Collection/Name eq '{cdse_collection}'"]
    for key, val in filters.items():
        if key == "geometry":
            clauses.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{val}')")
        elif key == "productType":
            clauses.append(
                f"Attributes/OData.CSC.StringAttribute/any("
                f"att:att/Name eq 'productType' and "
                f"att/OData.CSC.StringAttribute/Value eq '{val}')"
            )
        elif key == "cloudCoverLt":
            clauses.append(
                f"Attributes/OData.CSC.DoubleAttribute/any("
                f"att:att/Name eq 'cloudCover' and "
                f"att/OData.CSC.DoubleAttribute/Value lt {float(val)})"
            )
        elif key == "contentDateStartGt":
            clauses.append(f"ContentDate/Start gt {val}T00:00:00.000Z")
        elif key == "contentDateStartLt":
            clauses.append(f"ContentDate/Start lt {val}T23:59:59.999Z")
        else:
            clauses.append(f"{key} eq '{val}'")

    odata_filter = " and ".join(clauses)

    params = {
        "$filter":  odata_filter,
        "$top":     page_size,
        "$skip":    0,
        "$expand":  "Attributes",
        "$orderby": "ContentDate/Start asc",
    }

    while True:
        data = _get_with_retry(session, _BASE, params)
        items = data.get("value", [])
        for item in items:
            yield item
        if len(items) < page_size:
            break   
        params["$skip"] += page_size


