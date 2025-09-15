import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_subpath_links(start_url, allowed_prefix, delay=0.5):
    visited = set()
    to_visit = [start_url]
    all_links = set()
    print(f"\nStarting crawl from: {start_url}")
    print(f"Only accepting links starting with: {allowed_prefix}\n")
    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)
        try:
            print(f"Crawling: {url}")
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            for link_tag in soup.find_all('a', href=True):
                href = link_tag['href']
                full_url = urljoin(url, href)
                # Clean up URL (remove anchors/fragments)
                full_url = full_url.split("#")[0].split("?")[0]
                if full_url.startswith(allowed_prefix) and full_url not in visited and full_url not in all_links:
                    print(f"Found: {full_url}")
                    all_links.add(full_url)
                    to_visit.append(full_url)
        except Exception as e:
            print(f"Failed to crawl {url}: {e}")

    print(f"\nFinished crawling. Total unique pages collected: {len(all_links)}\n")
    return list(all_links)