import pandas as pd

lang = "es"

# ori = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/originals/testset-gma/testset_gma/"
# ori_pandas = pd.read_csv(ori + f"test-gma-es2en-health.csv")

cleaned_tr = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/cleaned/scielo-gma/"
cleaned_pandas_tr = pd.read_csv(cleaned_tr + f"es-en-gma-health.csv")

cleaned_ts = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/cleaned/testset_gma/"
cleaned_pandas_ts = pd.read_csv(cleaned_ts + f"test-gma-en2es-health.csv")

# a = set(ori_pandash[lang]).union(set(ori_pandasb[lang]))
a = set(cleaned_pandas_tr[lang])
b = set(cleaned_pandas_ts[lang])

# raw = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq/scielo_health_es_en/raw"
# with open(raw+f"/train.{lang}") as f:
#     raw_train = f.readlines()
#
# with open(raw + f"/val.{lang}") as f:
#     raw_val = f.readlines()
#
# with open(raw + f"/test.{lang}") as f:
#     raw_test = f.readlines()
#
# a = set(cleaned_pandas[lang])
# b = set(raw_train + raw_val)

res1 = b.issubset(a)
res2 = a.intersection(b)

assert res1 == False
assert res2 == set()

asd = 3
