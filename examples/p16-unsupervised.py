#%%
from dataset_poetry import Xd_train, Xd_vali, y_train, y_vali

X_train = Xd_train["numeric"]
X_vali = Xd_vali["numeric"]

from sklearn.manifold import TSNE


#%%

from matplotlib import pyplot as plot

perplexity = 2
viz = TSNE(perplexity=perplexity)
V = viz.fit_transform(X_train)
poetry = y_train == 1
npoetry = y_train == 0


plot.title("T-SNE(Poetry), perplexity={}".format(perplexity))
plot.scatter(V[npoetry, 0], V[npoetry, 1], c="darkgray", alpha=0.5)
plot.scatter(V[poetry, 0], V[poetry, 1], c="g", alpha=0.5)
plot.savefig("graphs/p16-tsne-p{}.png".format(perplexity))

# %%

from sklearn.decomposition import PCA

perplexity = 250
viz = PCA(n_components=2)
V = viz.fit_transform(X_train)
poetry = y_train == 1
npoetry = y_train == 0


plot.title("PCA(Poetry)")
plot.scatter(V[npoetry, 0], V[npoetry, 1], c="darkgray", alpha=0.5)
plot.scatter(V[poetry, 0], V[poetry, 1], c="g", alpha=0.5)
plot.savefig("graphs/p16-pca.png")
# %%

from sklearn.decomposition import TruncatedSVD

viz = TruncatedSVD(n_components=2)
V = viz.fit_transform(Xd_train["textual"])
poetry = y_train == 1
npoetry = y_train == 0


plot.title("TruncatedSVD(Poetry.Text)/LSA")
plot.scatter(V[npoetry, 0], V[npoetry, 1], c="darkgray", alpha=0.5)
plot.scatter(V[poetry, 0], V[poetry, 1], c="g", alpha=0.5)
plot.savefig("graphs/p16-lsa-text.png")
# %%

viz = TruncatedSVD(n_components=2)
V = viz.fit_transform(Xd_train["merged"])
poetry = y_train == 1
npoetry = y_train == 0


plot.title("TruncatedSVD(Poetry.Merged)/LSA")
plot.scatter(V[npoetry, 0], V[npoetry, 1], c="darkgray", alpha=0.5)
plot.scatter(V[poetry, 0], V[poetry, 1], c="g", alpha=0.5)
plot.savefig("graphs/p16-lsa-merged.png")

# %%

viz = TruncatedSVD(n_components=2)
V = viz.fit_transform(Xd_train["numeric"])
poetry = y_train == 1
npoetry = y_train == 0


plot.title("TruncatedSVD(Poetry.Numeric)/LSA")
plot.scatter(V[npoetry, 0], V[npoetry, 1], c="darkgray", alpha=0.5)
plot.scatter(V[poetry, 0], V[poetry, 1], c="g", alpha=0.5)
plot.savefig("graphs/p16-lsa-numeric.png")

# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=15)
C_train = kmeans.fit_transform(X_train)
print(C_train.shape)

viz = TSNE(n_components=2, perplexity=20)
V = viz.fit_transform(C_train)
poetry = y_train == 1
npoetry = y_train == 0


plot.title("TSNE of KMeans")
plot.scatter(V[npoetry, 0], V[npoetry, 1], c="darkgray", alpha=0.5)
plot.scatter(V[poetry, 0], V[poetry, 1], c="g", alpha=0.5)
plot.savefig("graphs/p16-tsne-kmeans.png")

# %%


# %%
