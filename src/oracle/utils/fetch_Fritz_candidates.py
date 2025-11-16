from requests import get
import numpy as np
import time
import os

FRITZ_API_KEY = os.environ.get("FRITZ_API_KEY")

headers = {
    'Authorization': f'token {FRITZ_API_KEY}',
    'Content-Type': 'application/json',
}


def get_all_candidates(groupid: int, start_date: str, end_date: str):
    """
    Query for ZTF sources that pass a specified Fritz group's filter ("candidates")
    within a specified date range

    Parameters
    ----------
    groupid: int
        ID for the Fritz group
        RCF_groupid = "41"
        RCFDeep_groupid = "280"
    start_date: str
        Start date of the query (YYYY-MM-DD)
    end_date: str
        End date of the query (YYYY-MM-DD)

    Returns
    -------
    ztfids: list
        List of ZTFIDs passing the specified Fritz group's filter

    Adapted from nabeelre/BTSbot/compile_ZTFIDs.query_rejects()
    """
    endpoint = "https://fritz.science/api/candidates"

    objids = []
    page_num = 1
    num_per_page = 500
    total_matches = None
    page_lim = None

    while True:
        print(f"Page {page_num} of candidates queries")

        params = {
            "savedStatus": "notSavedToAnySelected",
            "startDate": start_date,
            "endDate": end_date,
            "groupIDs": str(groupid),
            "numPerPage": num_per_page,
            "pageNumber": page_num
        }
        r = get(endpoint, headers=headers, params=params)

        if r.status_code != 200:
            print(f"Request failed with status code: {r.status_code}")
            print(f"Response text: {r.text}")
            break

        try:
            response_json = r.json()
            data = response_json['data']
            candidates = data['candidates']
            total_matches = data.get('totalMatches', None)
        except Exception as e:
            print("Error occurred in response parsing: ", r.text)
            print(f"Error: {e}")
            break

        if page_num == 1:
            page_lim = int(total_matches / num_per_page) + 1
            print(f"{total_matches} candidates over {page_lim} pages")

        # If no candidates were found on this page, end loop
        if len(candidates) == 0 or candidates is None:
            break

        # Add the current page's candidates to the set of unique candidates
        page_objids = [candidate['id'] for candidate in candidates]
        objids.extend(page_objids)
        print(f"  {len(page_objids)} candidates on page {page_num}, {len(objids)} total")

        # Stop if we've collected all expected candidates
        if total_matches is not None and len(objids) >= total_matches:
            print(f"Collected all {total_matches} expected candidates, ending loop")
            break

        page_num += 1
        # Make sure to not go past the total number of candidates on the next page
        if page_num > page_lim:
            print(f"Reached page limit of {page_lim}, ending loop")
            break

        time.sleep(0.5)

    return list(objids)


if __name__ == "__main__":
    objids = get_all_candidates(
        groupid=280,
        start_date="2021-06-01",  # Last update to RCF Deep Filter
        end_date="2025-11-01",  # ~Today
    )

    np.savetxt("rcfdeep_objids.txt", objids, fmt='%s')
