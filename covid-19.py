#Kütüphaneleri içe aktarma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.manifold import MDS

# JSON formatındaki dosyadan veriyi oku (örnek olarak dataset.json kullanıldı.)
data = pd.read_json("dataset.json")

# Stop-words listesini belirle
stop_words = "english"

# TF-IDF vektörlerini çıkar
t_v = TfidfVectorizer(stop_words=stop_words, max_df=5)
t = t_v.fit_transform(data["title"])

a_v = TfidfVectorizer(stop_words=stop_words, max_df=5)
a = a_v.fit_transform(data["abstract"])

r_v = TfidfVectorizer(stop_words=stop_words, max_df=5)
r = r_v.fit_transform([item for sublist in data["references"] for item in sublist])

# Kümeleri ve etiketleri tanımlama
n_clusters = 3
c_c = {0: "#e7298a", 1: "#1b9e77", 2: "#7570b3"}
c_n = {0: "Küme 1", 1: "Küme 2", 2: "Küme 3"}

# Her bir özellik için K-Means kümeleme uygula
for name, feature_matrix in [("title", t), ("abstract", a), ("references", r)]:
    # K-Means kümeleme modelini uygula
    km = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=500, n_init=1).fit(feature_matrix)

    # MDS ile iki boyuta indirgeme
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    dist = 1 - cosine_similarity(feature_matrix)
    pos = mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]

    # Veriyi DataFrame'e dönüştürme
    key = [x for x in data.keys() if x.startswith(name)]
    dat = pd.DataFrame(dict(x=xs, y=ys, label=km.labels_.tolist(), title=data[key[0]] if name != "references" else [item for sublist in data["references"] for item in sublist]))

    # Plot oluşturma
    fig, ax = plt.subplots(figsize=(20, 15))
    for k, i in dat.groupby("label"):
        ax.plot(i.x, i.y, marker="o", ms=10, label=c_n[k], color=c_c[k], mec="none")
        ax.set_aspect("auto")
        ax.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
        ax.tick_params(axis="y", which="both", left="off", top="off", labelleft="off")

    # Küme etiketleri
    ax.legend(numpoints=1, loc="upper left", fontsize="medium")
    ax.axis("off")

    # Referanslar hariç diğerleri için metin etiketleri
    if name != "references":
        for i in range(len(dat)):
            title = dat.loc()[i]["title"]
            title = title if len(title) < 100 else title[0:100]
            ax.text(dat.loc()[i]["x"], dat.loc()[i]["y"], title, size=10)

    # Plot'ı göster
    plt.show()