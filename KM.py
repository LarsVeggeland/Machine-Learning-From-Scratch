#---------- Imported libraries ----------

from random import random
from matplotlib import pyplot as plt



#---------- Util functions ----------

def get_data(filename : str, normalize : bool = False) -> list:
    data = []
    with open(filename, "r") as file:
        for line in file.readlines():
            data.append([float(dim) for dim in line.split(" ")])

        # Using min-max normalization
        if normalize:
            for col in range(len(data[0])):
                col_vals = [sample[col] for sample in data]
                col_max = max(col_vals)
                col_min = min(col_vals)
                for i in range(len(data)):
                    data[i][col] = (data[i][col] - col_min)/(col_max - col_min)
    return data


def euclidean(vec1 : list, vec2 : list ) -> float:
    return sum([(vec1[i] - vec2[i])**2 for i in range(len(vec1))])**0.5


def find_nearest_vector(vec : list, vectors : list) -> list:
    # Returns the vector among vectors the closest to vec 
    return min([(euclidean(vec, v), v) for v in vectors], key=lambda x : x[0])[1]


def init_centeroids(data : list, K : int) -> list:
    # Generating K random vectors
    centeroids = [[random() for j in range(len(data[0]))] for i in range(K)]
    # Finds and returns the vectors in the data closest to the randomly generated vectors as the initial centeroids
    return [find_nearest_vector(centeroid, data) for centeroid in centeroids]


def find_nearest_centeroid(vec : list, centeroids : list) -> int:
    # Finds the centeroid closest to vec in the vector space and returns its index in centeroids (Ugly AF fix)
    return min([(euclidean(vec, centeroids[i]), i) if len(centeroids[i]) > 0 else (1000**1000, i) for i in range(len(centeroids))], key = lambda x : x[0])[1]


def find_average_vector(cluster : list) -> list:
    # Rerturns a vector with average values for all dimensions
    return [sum([cluster[vector][dim] for vector in range(len(cluster))])/len(cluster) for dim in range(len(cluster[0]))]


def WCSS(centeroid : list, cluster : list) -> float:
    return sum(euclidean(vec, centeroid) for vec in cluster)



#---------- K-Means ----------

def KM(data : list, K : int, max_iter : int, centeroids : list = None, only_samples_as_centroids : bool = False):

    # Get the initial centeroids
    if centeroids is None: centeroids = init_centeroids(data, K)
    else: centeroids = centeroids[:K]#len(centeroids) # Maybe not necessary...
    # Cluster the data around the initial centeroids
    clusters = [[] for i in range(K)]

    for _ in range(max_iter):

        new_centeroids = [[] for i in range(K)]
        new_clusters = [[] for i in range(K)]

        # Cluster the vectors around the centeroids
        for vec in data: 
            new_clusters[find_nearest_centeroid(vec, centeroids)].append(vec)

        # Find new centeroids by finding the vector closest to the average vector in the cluster
        for i in range(len(clusters)):
            if len(new_clusters[i]) > 0: 
                if only_samples_as_centroids:
                    # As we are only using actual data points as centroids the new centroid is the one the closest to the average of the vector
                    new_centeroids[i] = find_nearest_vector(find_average_vector(new_clusters[i]), new_clusters[i])
                else:
                    # The new centroid is the average vector of the cluster
                    new_centeroids[i] = find_average_vector(new_clusters[i])

        if new_centeroids == centeroids and clusters == new_clusters:
            print(f"Halting as centeroids are unchanged at {_} iterations")
            break
            
        centeroids = new_centeroids
        clusters = new_clusters

    return (clusters, centeroids)



#---------- Plot 'elbow' (None of the remaining functions are used by the K-means implementation) ----------

def plot_performance_per_K_clusters(data : list, centeroids : list, max_iter : int):
    Ks = [K for K in range(2, 16)]
    error = []

    for K in Ks:
        res = KM(data, K, max_iter, centeroids)
        cs, ce = res[0], res[1]
        error.append(sum([WCSS(ce[i], cs[i]) for i in range(len(ce))]))

    print(f"Number of empty clusters at: {sum([len(cluster) == 0 for cluster in cs])}")


    print(error)
    plt.plot(Ks, error)
    plt.scatter(Ks, error)
    plt.ylabel("WCSS")
    plt.xlabel("K-clusters")
    plt.show()


def write_performance_to_file(data : list, centeroids : list, max_iter : int):
    Ks = [K for K in range(2, 16)]
    error = []

    for K in Ks:
        res = KM(data, K, max_iter, centeroids)
        cs, ce = res[0], res[1]
        error.append(sum([WCSS(ce[i], cs[i]) for i in range(len(ce))]))

    with open("/Users/larsveggeland/Documents/Python/assignment3/performance.csv", "a") as csv:
        csv.write("K, WCSS\n")
        for i in range(len(Ks)):
            csv.write(f"{Ks[i]}, {error[i]}\n")

    with open("/Users/larsveggeland/Documents/Python/assignment3/cluster_performance.csv", "a") as csv:
        csv.write("K, Cluster, WCSS\n")
        for i in range(len(Ks)):
            res = KM(data, Ks[i], max_iter, centeroids)
            cs, ce = res[0], res[1]
            for j in range(len(cs)):
                csv.write(f"{Ks[i]}, {j+1}, {WCSS(ce[j], cs[j])}\n")


# Provided centeroids
centeroids = get_data("/Users/larsveggeland/Documents/Python/assignment3/centroids.txt")

samples = get_data("/Users/larsveggeland/Documents/Python/assignment3/test.txt")

# Random centeroids
#centeroids = init_centeroids(samples, 15)

# Simply specify the number of maximum iterations
plot_performance_per_K_clusters(samples, centeroids, max_iter=10)