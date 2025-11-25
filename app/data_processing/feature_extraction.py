import json
import re
import email
import mailbox
import polars as pl
import logging
from email.header import decode_header, make_header
from .html_transformer import HTMLTransformer
from helpers.transformers import ReceiverTransformer
from helpers.transformers import AuthResultsParser
from helpers.transformers import SenderTransformer
from helpers.keywords import KEYWORDS
import torch
from sentence_transformers import SentenceTransformer
import ast
import os
import glob
import math
import re
from typing import Set, List, Optional
import polars as pl
import numpy as np
import tldextract
from scipy.stats import entropy

class FeatureExtraction():

    def domain_entropy(self, domain: Optional[str]) -> float:
        if not isinstance(domain, str) or not domain:
            return 0.0
        counts = np.array([domain.count(c) for c in set(domain)], dtype=float)
        probs = counts / counts.sum()
        return float(entropy(probs, base=2))

    def extract_tld(self, domain: Optional[str]) -> str:
        if not domain:
            return ""
        ext = tldextract.extract(domain)
        return ext.suffix or ""

    def count_uppercase_words(self, text: Optional[str]) -> int:
        if not isinstance(text, str):
            return 0
        return len(re.findall(r"[A-Z]{2,}", text))

    def count_exclamation_marks(self, text: Optional[str]) -> int:
        if not isinstance(text, str):
            return 0
        return text.count("!")

    def count_malicious_domains(self, link_domains: Optional[str], malicious_set: Set[str]) -> int:
        if not isinstance(link_domains, str):
            return 0
        count = 0
        domains = json.loads(link_domains)
        for d in domains:
            if d.lower() in malicious_set:
                print(f"Malicious domain found: {d.lower()}")
                count += 1
        return count


    def load_parquet_data(self, folder_path: str) -> pl.DataFrame:
        parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {folder_path}")

        dfs = [pl.read_parquet(pf) for pf in parquet_files]
        return pl.concat(dfs, how="diagonal_relaxed")

    def prepare_embeddings(self, df: pl.DataFrame) -> np.ndarray:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        embeddings = model.encode(df['body_subject'].to_list(), show_progress_bar=True, convert_to_numpy=True)
        return embeddings
    
    def load_malicious_domains(self, txt_files: List[str] = None) -> Set[str]:
        """Load malicious domains from text files."""
        try:
            if txt_files is None:
                txt_files = ['data/phishing_domains/full-domains-aa.txt', 'data/phishing_domains/full-domains-ab.txt']
            
            malicious_domains = set()
            
            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        domain = line.strip()
                        if domain: 
                            malicious_domains.add(domain)
            
            print(f"Loaded {len(malicious_domains)} malicious domains from {len(txt_files)} files")
            return malicious_domains
        
        except Exception as e:
            raise e


    def add_features(self, df: pl.DataFrame, malicious_domains: Set[str]) -> pl.DataFrame:
        df = df.with_columns([
            pl.col("sender_domain").map_elements(self.domain_entropy, return_dtype=pl.Float32).alias("sender_domain_entropy"),
            pl.col("sender_domain").map_elements(self.extract_tld, return_dtype=pl.Utf8).alias("sender_domain_tld"),
            pl.col("body_text").map_elements(lambda x: self.count_exclamation_marks(x), return_dtype=pl.Int32)
            .alias("num_exclamation_marks"),
            pl.col("link_domains").map_elements(lambda x: self.count_malicious_domains(x, malicious_domains), return_dtype=pl.Int32)
            .alias("num_malicious_links"),
        ])
        df = df.with_columns([
            (pl.col("num_malicious_links") > 0).cast(pl.Int8).alias("has_malicious_link")
        ])

        df = df.with_columns([
            pl.col("spf_flag").fill_null("").str.to_lowercase().is_in(["fail", "softfail", "none", "temperror"]).cast(pl.Int8).alias("spf_flag_missing"),
            pl.col("dkim_flag").fill_null("").str.to_lowercase().is_in(["fail", "none", "temperror"]).cast(pl.Int8).alias("dkim_flag_missing"),
            ((pl.col("return_path_domain").is_null()) | (pl.col("return_path_domain") == "")).cast(pl.Int8).alias("return_path_domain_missing")
        ])

        df = df.drop(["links", "emails", "phone_numbers"], strict=False)

        cols_to_process = ["num_links", "subject_length", "body_length", "keyword_count", "num_received_headers", "sender_domain_entropy"]
        for col in cols_to_process:
                df = df.with_columns(pl.col(col).fill_null(0))
        df = df.with_columns(pl.col("has_attachment").fill_null(False))
        print(df)
        return df

    def build_preprocessing_pipeline(self):
        """Build preprocessing pipeline for numeric features"""
        from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        
        log_features = [
            "num_exclamation_marks", "num_uppercase_words", "num_malicious_links"
        ]
        
        numeric_features = [
            "sender_domain_entropy",
            "spf_flag_missing", "dkim_flag_missing", "return_path_domain_missing",
            "num_links", "subject_length", "body_length", "keyword_count", "num_received_headers", 
        ]
        
        log_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("log", FunctionTransformer(np.log1p)),
            ("scaler", MinMaxScaler())
        ])
        
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", MinMaxScaler())
        ])
        
        preprocessor = ColumnTransformer([
            ("log_numeric", log_pipeline, log_features),
            ("numeric", numeric_pipeline, numeric_features)
        ], remainder="drop")
        
        all_features = log_features + numeric_features
        return preprocessor, all_features

    def decode_mime_header(self, value: str) -> str:
        if not value:
            return ""
        try:
            decoded = make_header(decode_header(value))
            return str(decoded)
        except Exception:
            return value

    def extract_body(self, msg: mailbox.mboxMessage) -> str:
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        return body

    def _extract_features_from_message(self, msg, html_transformer, source=None, phishing=None):
        features = {}
        
        # Body
        body = self.extract_body(msg)
        body_dict = html_transformer.transform_html(body)
        if body_dict is None:
            return None

        if source is not None:
            features["source"] = source
        if phishing is not None:
            features["phishing"] = phishing


        # Authentication
        auth_parser = AuthResultsParser(msg.get("Authentication-Results", ""))
        auth_features = auth_parser.parse()
        features["spf_flag"] = auth_features.get("spf_result", "")
        features["dkim_flag"] = auth_features.get("dkim_result", "")
        features["d_flag"] = auth_features.get("dmarc_result", "")

        # Received headers
        received_headers = msg.get_all("Received", [])
        features["num_received_headers"] = len(received_headers)

        # Return-Path domain
        return_path = msg.get("Return-Path", "") or ""
        return_path_clean = return_path.strip().strip("<>").replace('"', '')
        match = re.search(r'@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', return_path_clean)
        features["return_path_domain"] = match.group(1).lower() if match else ""

        # Body and links
        features["body_text"] = body_dict.get("text", "") or "unknown"
        text_lower = (body_dict.get("text", "") or "unknown").lower()
        features["keyword_count"] = sum(text_lower.count(k) for k in KEYWORDS)

        has_attachment = any("attachment" in str(part.get("Content-Disposition", "")).lower()
                             for part in msg.walk()) if msg.is_multipart() else False
        features["has_attachment"] = has_attachment

        links = body_dict.get("links", [])
        features["links"] = links
        features["link_domains"] = body_dict.get("link_domains", [])
        features["phone_numbers"] = body_dict.get("phone_numbers", [])
        features["emails"] = body_dict.get("emails", [])
        features["num_links"] = len(links)

        # Receiver
        to_field = self.decode_mime_header(msg.get("To", ""))
        name, email_addr = ReceiverTransformer.transform(to_field)
        features["receiver_name"] = name
        features["receiver_email"] = email_addr

        # Sender
        from_field = self.decode_mime_header(msg.get("From", ""))
        sender_name, sender_email = SenderTransformer.transform(from_field)
        features["sender_name"] = sender_name
        features["sender_email"] = sender_email
        sender_domain = sender_email.split("@")[-1] if "@" in sender_email else ""
        features["sender_domain"] = sender_domain

        # Subject and body lengths
        subject = self.decode_mime_header(msg.get("Subject", ""))
        features["subject"] = subject or "unknown" 
        features["subject_length"] = len(subject or "unknown")
        features["body_length"] = len(body_dict.get("text", "") or "unknown")

        # Readability and special chars
        features["readability_score"] = body_dict.get("readability_score", 0.0)
        features["special_char_ratio"] = body_dict.get("special_char_ratio", 0.0)

        features["body_subject"] = features["body_text"] + "\n\n" + features["subject"]
        return features
    
    def process_eml(self, path: str) -> dict:
        try:
            with open(path, 'rb') as f:
                msg = email.message_from_bytes(f.read())
        except Exception as e:
            raise ValueError(f"Failed to read .eml file: {e}")
        
        html_transformer = HTMLTransformer()
        features = self._extract_features_from_message(msg, html_transformer)
        
        if features is None:
            raise ValueError("Only English emails are supported. The email appears to be in a different language or couldn't be parsed.")
            
        return features
    
    def process_to_dataframe(self, features):
        # Handle both single dict and list of dicts
        if isinstance(features, dict):
            features_list = [features]
        else:
            features_list = features
        df = pl.DataFrame(features_list)
        df = df.filter(
            pl.col("subject").is_not_null() & 
            pl.col("body_text").is_not_null()
        )
        malicious_domains = self.load_malicious_domains()
        df = self.add_features(df, malicious_domains)
    
        
       # embeddings = self.prepare_embeddings(df)
        return df

    def process_eml_to_dataframe(self, path: str) -> pl.DataFrame:
        features = self.process_eml(path)
        return self.process_to_dataframe(features)
        
    def process_text(self, subject: str, body_text: str) -> tuple[pl.DataFrame, np.ndarray]:
        features = {
            "source": "text_input",
            "subject": subject or "",
            "body_text": body_text or "",
            "subject_length": len(subject or ""),
            "body_length": len(body_text or ""),
            "body_subject": (body_text or "") + "\n\n" + (subject or ""),
            "sender_domain": "",
            "num_links": 0,
            "links": [],
            "link_domains": [],
            "phone_numbers": [],
            "emails": [],
            "keyword_count": sum((body_text or "").lower().count(k) for k in KEYWORDS),
            "has_attachment": False,
            "num_received_headers": 0,
            "spf_flag": "",
            "dkim_flag": "",
            "return_path_domain": "",
            "readability_score": 0.0,
            "special_char_ratio": 0.0
        }

        return self.process_to_dataframe(features)

    def process_csv(self, input_csv_path: str, output_parquet_path: str, 
                    body_column: str = "body", label_column: str = "label") -> pl.DataFrame:
        print(f"Processing {input_csv_path}...")
        
        # Read CSV
        df = pl.read_csv(input_csv_path, has_header=True, separator=",", encoding="utf8")
        print(f"Loaded {df.shape[0]} rows from {input_csv_path}")
        
        html_transformer = HTMLTransformer()
        all_features = []
        
        for row in df.iter_rows(named=True):
            html_content = row.get(body_column, "")
            phishing_label = bool(row.get(label_column, 0) == 1)
            
            try:
                # Transform HTML content
                body_dict = html_transformer.transform_html(html_content)
                if body_dict is None:
                    continue
                
                # Get text and subject, skip if either is None
                body_text = body_dict.get("text")
                subject = row.get("subject")
                if body_text is None or subject is None:
                    continue
                
                # Parse sender and receiver
                sender_name, sender_email = SenderTransformer.transform(row.get("sender", ""))
                receiver_name, receiver_email = ReceiverTransformer.transform(row.get("receiver", ""))
                sender_domain = sender_email.split("@")[-1] if "@" in sender_email else ""
                
                features = {
                    "source": input_csv_path,
                    "phishing": phishing_label,
                    "body_subject": body_text + "\n\n" + subject,
                    "body_text": body_text,
                    "subject": subject, 
                    "subject_length": len(subject),
                    "body_length": len(body_text),
                    "sender_domain": sender_domain,
                    "sender_name": sender_name or "",
                    "sender_email": sender_email or "",
                    "receiver_name": receiver_name or "",
                    "receiver_email": receiver_email or "",
                    "links": body_dict.get("links", []),
                    "link_domains": body_dict.get("link_domains", []),
                    "phone_numbers": body_dict.get("phone_numbers", []),
                    "emails": body_dict.get("emails", []),
                    "num_links": len(body_dict.get("links", [])),
                    "keyword_count": sum(body_text.lower().count(k) for k in KEYWORDS),
                    "has_attachment": False,
                    "num_received_headers": 0,
                    "spf_flag": "",
                    "dkim_flag": "",
                    "d_flag": "",
                    "return_path_domain": "",
                    "readability_score": float(body_dict.get("readability_score", 0.0)),
                    "special_char_ratio": float(body_dict.get("special_char_ratio", 0.0)),
                }
                
                all_features.append(features)
                
            except Exception as e:
                logging.warning(f"Error processing row: {e}")
                continue
        
        if not all_features:
            raise ValueError(f"No valid rows found in CSV file: {input_csv_path}")
        
        # Process through the same pipeline as other methods
        transformed_df = self.process_to_dataframe(all_features)
        print(f"Transformed shape: {transformed_df.shape}")
        
        # Write to parquet
        self.write_to_file(transformed_df, output_parquet_path)
        print(f"Saved to {output_parquet_path}\n")
        
        return transformed_df
    
    def process_multiple_csvs(self, csv_configs: list[dict]) -> list[pl.DataFrame]:
        results = []
        
        for config in csv_configs:
            input_csv = config["input_csv"]
            output_parquet = config["output_parquet"]
            body_column = config.get("body_column", "body")
            label_column = config.get("label_column", "label")
            
            df = self.process_csv(
                input_csv_path=input_csv,
                output_parquet_path=output_parquet,
                body_column=body_column,
                label_column=label_column
            )
            results.append(df)
        
        return results

    def process_mbox(self, path: str, mbox_source: str, phishing: bool) -> pl.DataFrame:
        all_features = []

        try:
            mbox = mailbox.mbox(path)
        except Exception as e:
            exit(f"Failed to open mbox file: {e}")

        html_transformer = HTMLTransformer()

        for msg in mbox:
            msg = email.message_from_bytes(msg.as_bytes())

            # Extract features using shared method
            features = self._extract_features_from_message(msg, html_transformer, mbox_source, phishing)
            if features is None:
                continue
            
            all_features.append(features)
        
        if not all_features:
            raise ValueError(f"No valid messages found in mbox file: {path}")
        
        return self.process_to_dataframe(all_features)

    def write_to_file(self, df: pl.DataFrame, path: str):
        df.write_parquet(path)