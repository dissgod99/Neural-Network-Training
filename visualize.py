import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_loss_per_steps(loss:list, steps:list) -> None:
    plt.plot(steps, loss, "-o")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss per Steps")
    plt.show()



"""path = "data/wine_dataset.csv"

df = pd.read_csv(path, sep=",")

print(df.head())

#plot scatterplot of the most two correlated features
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df,
                x="Phenols",
                y="Flavanoids",
                hue="Wine")
plt.title("Phenols accoring to Flavanoids")
plt.show()

#plot confusion matrix
correlation = df.corr()
print(correlation)
plt.figure(figsize=(10, 8))
sns.heatmap(data=correlation, annot=True)
plt.title("Confusion Matrix")
plt.show()"""