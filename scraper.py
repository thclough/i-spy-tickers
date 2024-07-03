#%%
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import private
import requests
import joblib
import csv
import random
import time
import regex as re

def get_links(driver, ticker, criterion=None):
    """uses webdriver to retrieve links under a certain criteria and returns a list of the links
    
    Args:
        driver (webdriver.Chrome object): driver to retrieve links for
        ticker (str): ticker to scrape news for
        criterion (str, default: None): required string to be in link

    Returns:
        filtered_links(list): list of links that meet criteria on page
    """
    # instantiate filtered links list
    filtered_links = []

    # pass_screen if symbol found in yahoo finance directory
    if not pass_yf_screen(ticker):
        return filtered_links
    
    # create the url
    url = create_yfinance_news_url(ticker)

    # load the page
    driver.get(url)

    # wait to load dynamic content, use a random wait time to 
    wait_time = random.uniform(1,2)
    driver.implicitly_wait(wait_time)
    
    # gather all anchors on page
    anchors = driver.find_elements(By.TAG_NAME, value = "a")
    
    # filter through href values of anchors, add hrefs that meet criterion
    for anchor in anchors:
        href = anchor.get_attribute("href")
        if criterion:
            if criterion in href:
                filtered_links.append(href)
        else:
            filtered_links.append(href)

    return filtered_links

def create_yfinance_news_url(ticker):
    """create url for yahoo finance news about ticker"""
    return f"https://finance.yahoo.com/quote/{ticker}/news"

def get_tickers(csv_path="nsdq_screener.csv"):
    """Get a list of tickers from nasdaq screener
    
    Args:
        csv_path (str): path to csv file with nasdaq tickers

    Returns:
        tickers (list): list of tickers
    """
    # instantiate list for tickers
    tickers = []

    with open(csv_path) as file:
        csv_reader = csv.reader(file)
        next(csv_reader) # skip headers
        for line in csv_reader:
            ticker = line[0]
            if ticker.isalpha():
                tickers.append(ticker)

    return tickers

def pass_yf_screen(ticker):
    """See if the tickers loads a valid yahoo finance quote news page
    
    Args:
        ticker (str) : the ticker of interest to load a page for (all caps)

    Returns:
        (bool) : pass status, True if ticker news page, False if not
    """
    # create the url 
    url = create_yfinance_news_url(ticker)

    # make http request
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
    response = requests.get(url, headers = {'User-Agent': user_agent})
    if response.status_code != 200:
        print("Warning")
    
    # parse html
    soup = BeautifulSoup(response.text, 'html.parser')

    # h1 header should contain the ticker
    header = soup.find('h1', recursive=True)
    if type(header) != type(None):
        if ticker in header.text:
            return True
    return False

def set_up_driver(exec_path, headless=True):
    """Function to set up the Selenium Driver
    
    Args:
        exec_path (str) : path of driver executable
        headless (bool, default=True) : headless or headful (browser shows) selection

    Returns:
        driver (webdriver.Chrome) : selenium driver 
    """

    service = Service(executable_path=exec_path)
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def update_news_links(ticker, news_links_to_add, compiled_news_links, error_tickers):
    """Adds news links to master set of news links
    
    Args:
        ticker (str) : ticker related to the news links
        news_links_to_add (list) : list of strings of the news links
        compiled_news_links (set) : set ot add links to 
        error_tickers (list) : list of tickers for which no news links could be found
    """
    # if an error/no headlines (probably an error)
    if news_links_to_add == []:
        error_tickers.append(ticker)
        print(f"Received {len(error_tickers)} blanks")

    # add ticker news links and save to joblib
    for link in news_links_to_add:
        compiled_news_links.add(link)

def load_links(path_to_links="joblib_objects/compiled_news_links"):
    """load in set of links from joblib item or create empty set
    
    Args:
        path_to_links (str) : name of the joblib item
    
    Returns:
        compiled_news_links (set) : links if saved or empty set if one not created yet
    """

    try:
        compiled_news_links = joblib.load(path_to_links)
    except FileNotFoundError:
        compiled_news_links = set()
    
    return compiled_news_links

def load_links_list(path_to_links, path_to_links_list):
    """Load links list or create links list if none. 
    
    Args:
        path_to_links (str) : name of the joblib item
        path_to_links_list (str) : name of the joblib item

    Return:
        links_list (list) : link list
    """
    try:
        links_list = joblib.load(path_to_links_list)
    except FileNotFoundError:
        links_set = load_links(path_to_links)
        links_list = list(links_set)
        joblib.dump(links_list, path_to_links_list)
    return links_list
    
def scrape_links_for_text(news_links, 
                          txt_path="data/news.txt", 
                          delay=3, 
                          tags = ["h1", "h2", "p"], 
                          cool_down=True,
                          cool_down_time=300,
                          consecutive_error_tolerance = 3):
    """Scrapes articles in links for text from certain tags and creates a txt files with the text data

    Args:
        news_links (list) : list of news_links to scrape
        txt_path (str, default="news.txt") : name of text file to create or append to
        delay (int, default = 3) : time to wait between get requests (seconds)
        tags (list, default = ["h1", "h2", "p"]) : list of tags to gather text from (for soup object)
        cool_down (bool, default = True) : whether or not to implement a cool down
            The cool down takes effect if there are a certain number (consecutive_error_tolerance) of consecutive links which do not receive 200 status response 
        consecutive_error_tolerance (int, default = 3) : see cool_down
        cool_down_time (int, default = 300) : number of seconds to wait for cool down
    """
    # headers for 
    headers ={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36"
    }
    
    # set consecutive error counter and tolerance for consecutive errors 
    consecutive_errors = 0
    consecutive_error_tolerance = 3

    # loop through the links
    idx = 0
    error_idxs = []

    while idx < len(news_links):
        print(idx)
        link = news_links[idx]

        # make the request
        response = requests.get(link, headers=headers)

        if response.status_code == 200: # if successful
            # break consecutive error chain and reset error idxs
            consecutive_errors = 0
            error_idxs = []

            # create soup object
            soup = BeautifulSoup(response.text, 'html.parser')

            # loop through tags and finf all
            for tag in tags:
                elements = soup.find_all(tag)
                # loop through elements of a certain kind and add text to text file
                for element in elements:
                    if element.text != chr(160): # not equal to a non-breaking space
                        # write to the text file
                        with open(txt_path, mode="a", newline='') as file:
                            file.write(element.text + "\n")
        else:
            consecutive_errors += 1
            error_idxs.append(idx)
            # if reached error tolerance
        idx += 1
        
        if consecutive_errors == consecutive_error_tolerance:
            # if cool_down is set to True, will try to wait out rate limiting, and start again
            if cool_down:
                print("cooling down...")
                time.sleep(cool_down_time)

                # go back to the first error news link
                idx = error_idxs[0]
                link = news_links[idx]
                response = requests.get(link, headers=headers)

                if response.status_code != 200: # cool down failed
                    raise(Exception("cool down failed..."))
            else:
                raise Exception("Error Tolerance Reached, no cool_down")
            
        time.sleep(delay)

def main(compile_links=False):

    path_to_links = "joblib_objects/compiled_news_links"
    path_to_links_list = "joblib_objects/compiled_news_links_list"
    
    # load joblib item if exists or create an empty set
    compiled_news_links = load_links(path_to_links)

    ## gather links with webdriver
    if compile_links:
        # gather/create list of tickers
        tickers = get_tickers()

        # set up the webdriver
        driver = set_up_driver(private.DRIVER_PATH, headless=True)

        # instantiate set to hold all yahoo fiance urls, and list error tickers
        error_tickers = []

        # loop through news pages for every ticker and add urls to url list
        for ticker in tickers:
            ticker_news_links = get_links(driver=driver,
                                        ticker=ticker,
                                        criterion="https://finance.yahoo.com/news/")
            
            update_news_links(ticker, ticker_news_links, compiled_news_links, error_tickers)

            # uncomment to save news links
            # joblib.dump(compiled_news_links, path_to_links)
            # joblib.dump(error_tickers, "error_tickers")
            time.sleep(3)

        # quit the driver
        driver.quit()

    ## gather the text from the news links
    compiled_news_links_list = load_links_list(path_to_links, path_to_links_list)

    scrape_links_for_text(compiled_news_links_list, delay=5)

if __name__ == "__main__":
    main()
