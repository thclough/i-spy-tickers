import requests
import csv
from bs4 import BeautifulSoup
import utils

def write_mf_csv(csv_path):
    """Create csv of ticker symbols for mutual funds and their names from SEC data
    
    Args:
        csv_path (str) : path of csv to write to 
    """

    # create headers for the csv
    with open(csv_path, "w", newline="") as mf_csv:
        csv_writer = csv.writer(mf_csv)
        csv_writer.writerow(["ticker","mutual_fund_name"])

    source="https://www.sec.gov/files/company_tickers_mf.json"

    json_dict = utils.get_json_dict(source)

    for company_list in json_dict["data"]:
        cik = company_list[2]
        ticker = company_list[3]

        fund_name = get_sec_fund_name(cik)

        if fund_name:
            with open(csv_path, "a", newline="") as mf_csv:
                csv_writer = csv.writer(mf_csv)
                csv_writer.writerow([ticker, fund_name])



def get_sec_fund_name(cik):
    """Gets the name of the mutual fund registered in SEC EDGAR
    based on the central index key (cik)

    Args:
        cik (str) : central index key of target fund in EDGAR

    Returns:
        fund_name (str) : name of the fund
    """

    # url for querying EDGAR mutual fund apu
    query_url = f"https://www.sec.gov/cgi-bin/series?company=&sc=companyseries&ticker=&CIK={cik}&type=N-PX"

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'

    response = requests.get(query_url, headers = {'User-Agent': user_agent})

    soup = BeautifulSoup(response.text, 'html.parser')

    if response.status_code != 200:
        print(response.status_code)
        print("Warning, could not locate cik")
        return False

    tds = soup.findAll('td', attrs={"nowrap": "nowrap"}, recursive=True)

    # name of interest is the fifth table data item 
    fund_name = tds[4].text.title()

    return fund_name

def main():
    write_mf_csv("mf_names.csv")

if __name__ == "__main__":
    main()