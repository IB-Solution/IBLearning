from IBLearning.MachineLearning.KMeansCluster import TextClustering

# text to classify
texts = [
    "hello world",
    "hello world",
    "hello world",
    "hello world",
    "this world is beautiful",
    "issou is a meme",
    "welcome to the world",
    "Risitas is the origin of issou",
    "Issou is a city in France"
]

# Create a TextClustering object
textClustering = TextClustering.TextClustering(TextClustering.Language.english)

# Add texts to the TextClustering object
for text in texts:
    textClustering.AddText(text)

# If needed, you can rebase the clusters with a new accuracy (create new clusters)
textClustering.RebaseClusterMean(accuracyMin=0.50)

# Get the clusters
clusters = textClustering.clusters

# Get the first cluster texts
texts = clusters[0].texts

# Get the first cluster words with count
words = clusters[0].words