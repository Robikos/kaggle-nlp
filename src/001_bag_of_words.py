import pandas as pd
from tokenizers import CharBPETokenizer

data = pd.read_csv("data/IMDB Dataset.csv")

with open("data/reviews.txt", "w") as f:
    f.write("\n".join(data['review']))

tokenizer = CharBPETokenizer()
tokenizer.train("data/reviews.txt")

sentences = [
    "Point A and Point B are equal.",
    "B is 2000 far from the first point.",
    "A plus B equals 10"
]

with open("data/reviews.txt", "w") as f:
    f.write("\n".join(sentences))

tokenizer = CharBPETokenizer()
tokenizer.train("data/reviews.txt")

x = tokenizer.encode("Point A, Point B, Point C")
