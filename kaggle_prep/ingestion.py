"""
Data ingestion module for kaggle-prep.
Provides unified adapter interface for Kaggle datasets, local files, databases, and S3 objects.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd


class DataSourceAdapter(ABC):
    """Abstract interface for dataset loading adapters."""

    @abstractmethod
    def load_dataframe(self, sample: Optional[int] = None) -> pd.DataFrame:
        """Load and return the DataFrame (optionally sampled)."""
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return a string descriptor for the dataset."""
        pass


class KaggleSourceAdapter(DataSourceAdapter):
    """Adapter for downloading and loading Kaggle datasets/competitions."""

    def __init__(self, dataset_slug: str, output_dir: str, competition: bool = False, local: bool = False, verbose: bool = False):
        self.dataset_slug = dataset_slug
        self.output_dir = output_dir
        self.competition = competition
        self.local = local
        self.verbose = verbose
        self._df = None

    @property
    def dataset_name(self) -> str:
        return self.dataset_slug

    def load_dataframe(self, sample: Optional[int] = None) -> pd.DataFrame:
        from .cli import create_output_directory, download_dataset, download_competition, load_first_csv
        output_path = Path(self.output_dir)
        data_extensions = ["*.csv", "*.parquet", "*.tsv", "*.xlsx", "*.json"]
        data_exists = any(any(output_path.glob(ext)) for ext in data_extensions)

        if self.local:
            if not data_exists:
                raise FileNotFoundError(f"No data found in '{self.output_dir}' to use with --local!")
            print(f" Using existing local data in: {output_path}")
        else:
            if not data_exists:
                print(f" No local data found in '{output_path}'. Initiating download...")
            else:
                print(f" Downloading/updating data in: {output_path}")
            create_output_directory(self.output_dir)
            api = None
            if self.competition:
                success = download_competition(api, self.dataset_slug, output_path, self.verbose)
            else:
                success = download_dataset(api, self.dataset_slug, output_path, self.verbose)

            if not success and not data_exists:
                raise RuntimeError(f"Could not obtain dataset '{self.dataset_slug}'.")

        df = load_first_csv(output_path, sample=sample)
        if df is None:
            raise ValueError(f"No tabular data files found in '{self.output_dir}' after download/load.")
        return df


class FileSourceAdapter(DataSourceAdapter):
    """Adapter for local tabular files (CSV, Parquet, JSON, Excel, TSV)."""

    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json"}

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Local file not found: {self.file_path}")
        if self.file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{self.file_path.suffix}'. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

    @property
    def dataset_name(self) -> str:
        return self.file_path.stem

    def load_dataframe(self, sample: Optional[int] = None) -> pd.DataFrame:
        suffix = self.file_path.suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(self.file_path, low_memory=False)
            elif suffix == ".tsv":
                df = pd.read_csv(self.file_path, sep="\t", low_memory=False)
            elif suffix == ".parquet":
                df = pd.read_parquet(self.file_path)
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(self.file_path)
            elif suffix == ".json":
                df = pd.read_json(self.file_path)
            else:
                df = pd.read_csv(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to read file '{self.file_path}': {e}") from e

        if sample and sample < len(df):
            print(f" Sampling {sample:,} rows from {len(df):,} rows...")
            df = df.sample(sample, random_state=42).reset_index(drop=True)

        print(f" Loaded local file: {self.file_path.name} ({len(df):,} rows, {len(df.columns)} columns)")
        return df


class DBSourceAdapter(DataSourceAdapter):
    """Adapter for querying databases via SQLAlchemy."""

    def __init__(self, connection_string: str, query: Optional[str] = None, table: Optional[str] = None):
        self.connection_string = connection_string
        self.query = query
        self.table = table

        if not query and not table:
            raise ValueError("Either --query or --table must be specified when using --db.")

    @property
    def dataset_name(self) -> str:
        if self.table:
            return self.table
        return "db_query"

    def load_dataframe(self, sample: Optional[int] = None) -> pd.DataFrame:
        try:
            import sqlalchemy
        except ImportError:
            raise ImportError(
                "SQLAlchemy is required for database ingestion. Install it via `pip install sqlalchemy`."
            )

        try:
            engine = sqlalchemy.create_engine(self.connection_string)
            if self.query:
                sql = self.query
            else:
                sql = f"SELECT * FROM {self.table}"

            with engine.connect() as conn:
                df = pd.read_sql_query(sqlalchemy.text(sql), conn)
        except Exception as e:
            raise ValueError(f"Database query failed: {e}") from e

        if sample and sample < len(df):
            print(f" Sampling {sample:,} rows from {len(df):,} rows...")
            df = df.sample(sample, random_state=42).reset_index(drop=True)

        print(f" Loaded from DB ({self.dataset_name}): ({len(df):,} rows, {len(df.columns)} columns)")
        return df


class S3SourceAdapter(DataSourceAdapter):
    """Adapter for reading S3 objects directly using pandas."""

    def __init__(self, s3_uri: str):
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI '{s3_uri}'. URI must start with 's3://'.")
        self.s3_uri = s3_uri

    @property
    def dataset_name(self) -> str:
        # Extract filename from key
        key = self.s3_uri.split("/")[-1]
        return Path(key).stem if key else "s3_dataset"

    def load_dataframe(self, sample: Optional[int] = None) -> pd.DataFrame:
        suffix = Path(self.s3_uri.split("?")[0]).suffix.lower()
        try:
            if suffix == ".parquet":
                df = pd.read_parquet(self.s3_uri)
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(self.s3_uri)
            elif suffix == ".json":
                df = pd.read_json(self.s3_uri)
            elif suffix == ".tsv":
                df = pd.read_csv(self.s3_uri, sep="\t")
            else:
                df = pd.read_csv(self.s3_uri)
        except Exception as e:
            raise ValueError(f"Failed to load S3 object from '{self.s3_uri}': {e}") from e

        if sample and sample < len(df):
            print(f" Sampling {sample:,} rows from {len(df):,} rows...")
            df = df.sample(sample, random_state=42).reset_index(drop=True)

        print(f" Loaded S3 dataset ({self.s3_uri}): ({len(df):,} rows, {len(df.columns)} columns)")
        return df


def get_source_adapter(
    dataset: Optional[str] = None,
    file_path: Optional[str] = None,
    db_conn: Optional[str] = None,
    query: Optional[str] = None,
    table: Optional[str] = None,
    s3_uri: Optional[str] = None,
    output_dir: str = "data",
    competition: bool = False,
    local: bool = False,
    verbose: bool = False,
) -> DataSourceAdapter:
    """Factory function to build appropriate DataSourceAdapter based on input flags."""
    provided = [
        ("dataset", bool(dataset)),
        ("file", bool(file_path)),
        ("db", bool(db_conn)),
        ("s3", bool(s3_uri)),
    ]
    active = [name for name, is_set in provided if is_set]

    if len(active) > 1:
        raise ValueError(f"Mutually exclusive input sources specified: {', '.join(active)}. Please specify only one.")
    if len(active) == 0:
        raise ValueError("No input source provided. Specify a Kaggle dataset slug, --file, --db, or --s3.")

    if file_path:
        return FileSourceAdapter(file_path)
    elif db_conn:
        return DBSourceAdapter(db_conn, query=query, table=table)
    elif s3_uri:
        return S3SourceAdapter(s3_uri)
    else:
        return KaggleSourceAdapter(dataset, output_dir=output_dir, competition=competition, local=local, verbose=verbose)
