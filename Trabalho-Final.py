import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



print('\n\n-----------------------  Carregamento e exloraçao dos dados  ----------------------------------\n')

data = pd.read_csv('gds3715.csv', sep=',', index_col=0).transpose()
meta = pd.read_csv('meta-gds3715.csv', sep=',', index_col=0)

print("Data:   Amostras x Genes ------>", data.shape)
print("Meta:   Amostras x variáveis -->", meta.shape)

print("\nGenes:\n", data.columns)
print("\nAmostras:\n", meta.index)
print("\nVariáveis:\n", meta.columns)


print("\nTipos de dados presentes nas colunas do ficheiro 'Data': ", set(data.dtypes))
print("Valores Null encontrados: ", data.isnull().sum().sum(), "\n")

# Subgrupos da Data dividida segundo estado da doença:
disease_states = meta['disease.state'].value_counts()
print(disease_states)
plt.pie(disease_states, labels=disease_states.index, autopct='%.2f%%', shadow=True, explode=(0.08, 0, 0))
plt.title('Percentagem de doentes por estados de doença')
plt.show()

# Subgrupos da Data dividida segundo tratamento:
agents = meta['agent'].value_counts()
print(agents)
plt.pie(agents, labels=agents.index, autopct='%.2f%%', shadow=True)
plt.title('Percentagem de doentes por tratamentos efetuados')
plt.show()


print('\n\n----------------------------  Pre_Processamento  ----------------------------------------\n')

from sklearn import preprocessing
from sklearn import neighbors
from sklearn.feature_selection import VarianceThreshold

input_data = data.values             # converte em np.array
variancias = input_data.var(axis=0)  # calcula variância nos dados, por cada coluna
var_media  = variancias.mean()       # média das variâncias

# mostra gráfico com a variação no valor de cada gene(para ver o que o filtro flat pattern irá fazer)
plt.bar(np.arange(len(variancias)), variancias, width=30)
plt.title("Variância em cada gene")
plt.xlabel('Genes')
plt.ylabel("Variância")
plt.show()

# Filtro flat pattern (retira genes com pouca variabilidade)
model_flat = VarianceThreshold(threshold=var_media * 2)
input_filtrado1 = model_flat.fit_transform(input_data)
print("Tamanho inicial: ", input_data.shape, "\nTamanho depois do filtro flat pattern: ", input_filtrado1.shape)

# Remoçao de anomalias(pontos que se encontram muito fora do normal)
outlier_model = neighbors.LocalOutlierFactor(n_neighbors=20, contamination=0.1)
remover = outlier_model.fit_predict(input_filtrado1.transpose())
input_filtrado2 = np.delete(input_filtrado1, remover, axis=1)
print("Tamanho depois do filtro de outliers: ", input_filtrado2.shape)

# Normalização dos dados
scaled_input = preprocessing.scale(input_filtrado2)
print("Média: ", scaled_input.mean())
print("Desvio padrão: ", scaled_input.std())


labels_doenca     = meta.values[:, 1]  # insulin sensible, insulin resistant, diabetic
labels_tratamento = meta.values[:, 2]  # insulina vs nao tratados



print('\n\n----------------------------  Análise Estatisticos Multivariada  ----------------------------------------\n')

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

# selecao de componentes principais(que representam a maior parte da variabilidade dos dados)
pca = PCA(n_components=10)
pca_data = pca.fit_transform(scaled_input)
perc_variacao = pca.explained_variance_ratio_
print("Os %s genes selecionados explicam %.1f%% das variaçoes nos dados" % (len(perc_variacao), perc_variacao.sum() * 100))
print('Explicação da variação por gene %s' % str(perc_variacao))

for color, name in zip("rgb", set(labels_tratamento)):
	plt.scatter(pca_data[labels_tratamento == name, 0], pca_data[labels_tratamento == name, 1], c=color, label=name)
plt.legend(loc="best")
plt.title("PCA-Tratamentos")
plt.show()
for color, name in zip("rgb", set(labels_doenca)):
	plt.scatter(pca_data[labels_doenca == name, 0], pca_data[labels_doenca == name, 1], c=color, label=name)
plt.legend(loc="best")
plt.title("PCA-Doenças")
plt.show()



print("Clustering segundo o tratamento do paciente\n")
kmeans_model = KMeans(n_clusters=len(set(labels_tratamento)))
spectral_clust_model = SpectralClustering(n_clusters=len(set(labels_tratamento)), affinity='nearest_neighbors',
										  assign_labels='kmeans')
hierarq_clust_model = AgglomerativeClustering(n_clusters=len(set(labels_tratamento)), affinity="euclidean",
											  linkage="ward")
clustering_algorithms = (
	('Kmeans Algorithm', kmeans_model),
	('Spectral Clustering Algorithm', spectral_clust_model),
	('Hierarchical Clustering Algorithm', hierarq_clust_model),)

plt.figure(figsize=(10, 4))
plt.subplots_adjust(left=.02, right=.98, bottom=.03, top=.85, wspace=.05,
hspace=.01)
plot_num = 1

for name, algorithm in clustering_algorithms:
	plt.subplot(1, len(clustering_algorithms), plot_num)
	algorithm.fit_predict(scaled_input)
	clusters = algorithm.labels_

	print(name)
	print(pd.crosstab(labels_tratamento, clusters), "\n")
	plt.scatter(scaled_input[:, 0], scaled_input[:, 1], c=clusters)
	if name == 'Kmeans Algorithm':
		centers = algorithm.cluster_centers_
		plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.6)
	plt.xticks(())
	plt.yticks(())
	plt.title(name, fontsize=10)
	plot_num += 1

plt.suptitle("Clustering segundo o tratamento do paciente")
plt.show()


print("Clustering segundo a doença:\n")
kmeans_model = KMeans(n_clusters=len(set(labels_doenca)))
spectral_clust_model = SpectralClustering(n_clusters=len(set(labels_doenca)), affinity='nearest_neighbors',
										  assign_labels='kmeans')
hierarq_clust_model = AgglomerativeClustering(n_clusters=len(set(labels_doenca)), affinity="euclidean",
											  linkage="ward")
clustering_algorithms = (
	('Kmeans Algorithm', kmeans_model),
	('Spectral Clustering Algorithm', spectral_clust_model),
	('Hierarchical Clustering Algorithm', hierarq_clust_model),)

plt.figure(figsize=(10, 4))
plt.subplots_adjust(left=.02, right=.98, bottom=.03, top=.85, wspace=.05,
hspace=.01)
plot_num = 1
for name, algorithm in clustering_algorithms:
	plt.subplot(1, len(clustering_algorithms), plot_num)
	algorithm.fit_predict(scaled_input)
	clusters = algorithm.labels_

	print(name)
	print(pd.crosstab(labels_doenca, clusters), "\n")
	plt.scatter(scaled_input[:, 0], scaled_input[:, 1], c=clusters)
	if name == 'Kmeans Algorithm':
		centers = algorithm.cluster_centers_
		plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.6)
	plt.xticks(())
	plt.yticks(())
	plt.title(name, fontsize=10)
	plt.suptitle("Clustering segundo a doença")
	plot_num += 1
plt.show()



print('\n\n----------------------------  Redução de dimensionalidade  ---------------------------------------')

from sklearn.feature_selection import SelectPercentile, f_classif
print("Redução da dimensionalidade segundo os tratamentos")
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(scaled_input, labels_tratamento)
pvalues = selector.pvalues_

ind_keep = np.where(pvalues < 0.0001)[0]  # indices dos valores que serao selecionados
reduced_data_tratamento = scaled_input[:, ind_keep]
print("Data depois de limpas as variaveis dependentes entre si: ", reduced_data_tratamento.shape)


print("\nRedução da dimensionalidade segundo as doenças")
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(scaled_input, labels_doenca)
pvalues = selector.pvalues_

ind_keep = np.where(pvalues < 0.00000000005)[0]
reduced_data_doencas = scaled_input[:, ind_keep]
print("Data depois de limpas as variáveis dependentes entre si: ", reduced_data_doencas.shape)



print('\n\n----------------------------  Aprendizagem Máquina Supervisionada  ---------------------------------------')

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

grupos = (("Tratamento (untreated / insulin)", labels_tratamento, reduced_data_tratamento),
		  ("Doenças (insulin sensitive / insulin resistant / diabetic)", labels_doenca, reduced_data_doencas),
		  )

for nome, label, dados in grupos:
	print("\nPredição de", nome, ":")
	indices = np.random.permutation(len(dados))

	# divisao dos clean_data: 80/100 para treino, 20/100 para avaliação
	div_data = int(len(indices) * 80 / 100)
	train_in = dados[indices[:div_data]]
	train_out = label[indices[:div_data]]
	test_in = dados[indices[div_data:]]
	test_out = label[indices[div_data:]]

	classifiers = (("KNN  ", KNeighborsClassifier()),
				   ("TREE ", tree.DecisionTreeClassifier()),
				   ("SVM  ", svm.SVC(gamma=0.001, C=100)),
				   ("NN   ", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 8, 4), random_state=1)),
				   ("BAYES", GaussianNB())
				   )

	for name, model in classifiers:
		model.fit(train_in, train_out)
		preds = model.predict(test_in)
		score = (preds == test_out).sum() / len(preds)
		print(name, ":", round(score, 3))
