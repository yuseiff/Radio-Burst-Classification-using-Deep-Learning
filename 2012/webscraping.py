import requests
from bs4 import BeautifulSoup
import os

# URL of the page to scrape
url = 'https://soleil.i4ds.ch/solarradio/callistoQuicklooks/?date=20120122'

# Directory to save downloaded files
download_dir = 'downloaded_files'
os.makedirs(download_dir, exist_ok=True)

# Send a GET request to fetch the raw HTML content
response = requests.get(url)
response.raise_for_status()  # Check for request errors

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all <a> tags with href attributes
links = soup.find_all('a', href=True)

# Filter and download the files
for link in links:
    href = link['href']
    if href.startswith('../data/'):
        # Construct the full URL
        file_url = f'https://soleil.i4ds.ch/solarradio/{href[3:]}'
        
        # Get the file name
        file_name = href.split('/')[-1]
        file_path = os.path.join(download_dir, file_name)
        
        # Download the file
        print(f'Downloading {file_url}...')
        file_response = requests.get(file_url)
        file_response.raise_for_status()  # Check for request errors
        
        # Save the file
        with open(file_path, 'wb') as file:
            file.write(file_response.content)
        print(f'Saved to {file_path}')

print('Download complete!')
