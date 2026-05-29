# Data policy

This repository does not redistribute raw WRDS, CRSP, RavenPack, Compustat, IBES, TAQ, or other vendor data.

Allowed public artifacts:

- source code;
- schema aliases and non-sensitive metadata;
- aggregate result tables;
- generated figures that cannot reconstruct vendor records;
- synthetic toy data generated independently of vendor data;
- documentation and reports.

Forbidden public artifacts:

- raw vendor records;
- full protected Parquet extracts;
- row-level CRSP/RavenPack/Compustat/IBES/TAQ data;
- credentials, `.pgpass`, API keys, or tokens.
