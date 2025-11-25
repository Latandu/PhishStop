from calendar import c
import email
import re
import logging
from tracemalloc import stop
from urllib.parse import urljoin, urlparse
import jsonpickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, DetectorFactory, LangDetectException
import unicodedata
from bs4 import BeautifulSoup, SoupStrainer
from numpy import empty
import tldextract
from email.utils import parseaddr
import textstat

class HTMLTransformer:
    def __init__(self):
        DetectorFactory.seed = 0
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        nltk.download('omw-1.4')
        self.stop_words = set(stopwords.words('english'))
        self.phone_pattern = re.compile(r"""
            (\+?\d{1,3}[\s.-]?)?     
            (\(?\d{3}\)?[\s.-]?)     
            (\d{3}[\s.-]?\d{4})       
        """, re.VERBOSE)
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')


    def transform_html(self, input_html: str):
        html_dict ={}
        links, link_domains, emails_set, phone_numbers_set = set(), set(), set(), set()
        soup = BeautifulSoup(input_html, 'html.parser')
        for link in BeautifulSoup(input_html, 'html.parser', parse_only=SoupStrainer('a')):
            if link.has_attr('href') and link['href'].strip():
                try:
                    links.add(link['href'])
                    link_domains.add(tldextract.extract(link['href']).registered_domain)
                except Exception as e:
                    logging.error(f"Error extracting link: {e}")
        for link in soup.find_all("a"):
            link.decompose()

        text = soup.get_text(separator=' ', strip=True)
        text = text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        text = unicodedata.normalize('NFKC', text)
        text = text.lower()
        text = re.sub(' +', ' ', text)
        if not isinstance(text, str) and not text.strip() and not links:
            return None
        try:
            language = detect(text)
        except LangDetectException:
            return None
        
        if language != 'en':
            return None
        text_links = re.findall(r'(https?://\S+)', text)

        for link in text_links:
            if link not in links:
                links.add(link)
                domain = tldextract.extract(link).registered_domain
                if domain:
                    link_domains.add(domain)

            text = text.replace(link, '')


        phone_numbers = re.findall(self.phone_pattern, text)
        if phone_numbers:
            phone_numbers_set = set(phone_numbers)
            text = re.sub(self.phone_pattern, 'phonenumber', text)


        emails = re.findall(self.email_pattern, text)
        if emails:
            emails_set = set(emails)
            text = re.sub(self.email_pattern, 'emailaddress', text)
        
        # word_tokens = word_tokenize(text.lower())
        # filtered_tokens = [word for word in word_tokens if word not in self.stop_words]
        # text = ' '.join(filtered_tokens)
        
        readability = textstat.flesch_reading_ease(text)
        total_chars = len(text)
        special_chars = len(re.findall(r'[^\w\s]', text))
        special_char_ratio = special_chars / total_chars if total_chars > 0 else 0


        html_dict = {
            "links": jsonpickle.encode(links, unpicklable=False),
            "link_domains": jsonpickle.encode(link_domains, unpicklable=False),
            "text": jsonpickle.encode(text, unpicklable=False),
            "phone_numbers": jsonpickle.encode(phone_numbers_set, unpicklable=False),
            "emails": jsonpickle.encode(emails_set, unpicklable=False),
            "readability_score": jsonpickle.encode(readability, unpicklable=False),
            "special_char_ratio": jsonpickle.encode(special_char_ratio, unpicklable=False)

        }
        # html_dict = {
        #     "links": links,
        #     "link_domains": link_domains,
        #     "text": text,
        #     "phone_numbers": phone_numbers_set,
        #     "emails": emails_set,
        #     "readability_score": readability,
        #     "special_char_ratio": special_char_ratio

        # }
        return html_dict