import pandas as pd

# store the length of the individual time series datasets for each KPI to make preprocessing easier
lengths_a = {"02e99bd4f6cfb33f": 241189,
           "09513ae3e75778a3": 239975,
           "18fbb1d5a5dc099d": 240299,
           "1c35dbf57f55f5e4": 240969,
           "71595dd7171f4540": 295337,
           "7c189dd36f048a6c": 295379,
           "8c892e5525f3e491": 294019,
           "9bd90500bfd11edb": 238798,
           "a40b1df87e3f1c87": 275850,
           "a5bf5d65261d859a": 237426,
           "affb01ca2b4f0b45": 295361,
           "c58bfcbacb2822d1":241453,
           "cff6d3c01e6a6bfa": 295258,
           "da403e4e3f87c9e0": 241148,
           "e0770391decc44ce": 294048}

lengths_b = {}



def preprocess_data(dataset):
    x = 1


if __name__ == "__main__":
    for dataset in ["aiops_a", "aiops_b"]:
        preprocess_data(dataset)
